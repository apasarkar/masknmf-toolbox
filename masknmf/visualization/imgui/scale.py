"""
How big the viewers draw text and panels: the UI scale and the text size File > Options sets, kept in
``~/.config/masknmf/settings.json``.

GLFW reports a content scale of 1 on some HiDPI setups (Hyprland with ``xwayland:force_zero_scaling``, for one, where
imgui_bundle's X11 GLFW runs through XWayland), so the canvas never learns the monitor is scaled and draws 14 px text
on a 2x panel. Left on auto, the UI scale is the desktop's (Hyprland's largest monitor scale, else ``GDK_SCALE``) over
the pixel ratio the canvas already draws at, so a canvas the OS scales properly stays at 1. ``MASKNMF_UI_SCALE``
overrides the saved scale for one run.
"""

import json
import os
import subprocess
from dataclasses import dataclass
from functools import lru_cache, partial
from pathlib import Path
from typing import Callable, Optional

from imgui_bundle import imgui, implot

from masknmf.visualization.imgui.clipboard import connect_clipboard
from masknmf.visualization.imgui.keybinds import DESKTOP_SIZE, LARGER, SMALLER, pressed
from masknmf.visualization.imgui.layout import MIN_RENDER_AREA, collapsed_size, is_notebook_canvas

DIR_CONFIG = Path.home() / ".config" / "masknmf"
FILEPATH_SETTINGS = DIR_CONFIG / "settings.json"
RANGE_SCALE = (0.5, 4.0)
RANGE_TEXT = (0.6, 2.0)
STEP = 1.1
IMPLOT_SIZES = (
    "plot_padding", "label_padding", "legend_padding", "legend_inner_padding", "legend_spacing",
    "mouse_pos_padding", "annotation_padding", "fit_padding", "plot_default_size", "plot_min_size",
    "major_tick_len", "minor_tick_len",
)


@dataclass
class Size:
    """The size settings: ui_scale None follows the desktop."""

    ui_scale: Optional[float] = None
    text_size: float = 1.0


def load() -> dict:
    try:
        values = json.loads(FILEPATH_SETTINGS.read_text())
    except (OSError, ValueError):
        return {}
    return values if isinstance(values, dict) else {}


def save(values: dict) -> None:
    try:
        DIR_CONFIG.mkdir(parents=True, exist_ok=True)
        FILEPATH_SETTINGS.write_text(json.dumps({**load(), **values}, indent=1))
    except OSError:
        pass  # a read-only home only costs remembering the size


def clamp(value: float, bounds: tuple[float, float]) -> float:
    return round(min(max(float(value), bounds[0]), bounds[1]), 2)


def number(value) -> Optional[float]:
    """value as a positive float, None when it is not one."""
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if value > 0 else None


def hyprland_scale() -> Optional[float]:
    """
    The scale Hyprland leaves to its XWayland windows, which is what GLFW windows are here (imgui_bundle loads an X11
    GLFW): the largest monitor scale with ``xwayland:force_zero_scaling`` on, 1 with it off, when Hyprland stretches
    them itself. None when hyprctl does not answer.
    """
    try:
        answers = [
            json.loads(subprocess.run(["hyprctl", *query, "-j"], capture_output=True, text=True, timeout=2).stdout)
            for query in (["getoption", "xwayland:force_zero_scaling"], ["monitors"])
        ]
        zero_scaling, monitors = answers
        # NOTE: Hyprland 0.5x answers {"bool": true}, older ones {"int": 1}
        if not (zero_scaling.get("bool") or zero_scaling.get("int")):
            return 1.0
        return max(float(m.get("scale", 1)) for m in monitors)
    except (OSError, ValueError, TypeError, AttributeError, subprocess.SubprocessError):
        return None


@lru_cache(maxsize=1)
def desktop_scale() -> float:
    """
    The desktop's scale as far as the environment tells: Hyprland's for its XWayland windows, else GDK_SCALE.

    Hyprland comes first: GDK_SCALE is a whole number, 1 at a monitor scale of 1.25.
    """
    if os.environ.get("HYPRLAND_INSTANCE_SIGNATURE"):
        scale = hyprland_scale()
        if scale is not None:
            return scale
    return number(os.environ.get("GDK_SCALE")) or 1.0


def initial_size() -> Size:
    """MASKNMF_UI_SCALE, else the saved settings."""
    saved = load()
    ui_scale = number(os.environ.get("MASKNMF_UI_SCALE")) or number(saved.get("ui_scale"))
    text_size = number(saved.get("text_size")) or 1.0
    return Size(
        ui_scale=None if ui_scale is None else clamp(ui_scale, RANGE_SCALE),
        text_size=clamp(text_size, RANGE_TEXT),
    )


SIZE = initial_size()


def set_ui_scale(scale: Optional[float]) -> None:
    """Set and save the UI scale; None follows the desktop."""
    SIZE.ui_scale = None if scale is None else clamp(scale, RANGE_SCALE)
    save({"ui_scale": SIZE.ui_scale})


def set_text_size(size: float) -> None:
    SIZE.text_size = clamp(size, RANGE_TEXT)
    save({"text_size": SIZE.text_size})


def auto_scale(pixel_ratio: float) -> float:
    """The desktop's scale over what the canvas already draws at, never below 1."""
    return clamp(max(desktop_scale() / max(pixel_ratio, 1e-3), 1.0), RANGE_SCALE)


def ui_scale(pixel_ratio: float) -> float:
    return auto_scale(pixel_ratio) if SIZE.ui_scale is None else SIZE.ui_scale


def handle_keys(scale: float) -> None:
    """Ctrl + / Ctrl - step the UI scale from ``scale``, the one drawn; Ctrl 0 goes back to the desktop's."""
    if pressed(LARGER):
        set_ui_scale(scale * STEP)
    elif pressed(SMALLER):
        set_ui_scale(scale / STEP)
    elif pressed(DESKTOP_SIZE):
        set_ui_scale(None)


def push_font_em(x: float, font: Optional[imgui.ImFont] = None) -> None:
    """Push font (None: the current one) at x times the current font size; pop with imgui.pop_font."""
    style = imgui.get_style()
    # push_font takes a size before the style's font scales, which get_font_size already includes
    imgui.push_font(font, x * imgui.get_font_size() / (style.font_scale_main * style.font_scale_dpi))


class Scaler:
    """
    Keeps one imgui context's style, fonts and, given its figure, edge windows at SIZE: the new UI scale's sizes are
    the default style's times the scale, and edge windows, sized for the text they hold, grow or shrink by the step
    of the UI scale times the text size from the last one.
    """

    implot_scale = 1.0  # implot's style is one for every context

    def __init__(self, figure=None):
        self.figure = figure
        self.applied = 1.0
        self.text_size = 1.0

    def __call__(self) -> None:
        scale = ui_scale(imgui.get_io().display_framebuffer_scale.x)
        handle_keys(scale)
        scale = ui_scale(imgui.get_io().display_framebuffer_scale.x)
        if scale != self.applied:
            self.rescale_style(scale)
        if (scale, SIZE.text_size) != (self.applied, self.text_size):
            self.rescale_windows(scale / self.applied, SIZE.text_size / self.text_size)
            self.applied, self.text_size = scale, SIZE.text_size
        imgui.get_style().font_scale_main = scale * SIZE.text_size
        self.rescale_implot(scale)

    @staticmethod
    def rescale_style(scale: float) -> None:
        style = imgui.get_style()
        scaled = imgui.Style()
        scaled.scale_all_sizes(scale)
        for i in range(imgui.Col_.count):
            scaled.set_color_(i, style.color_(i))
        scaled.font_size_base = style.font_size_base
        scaled.font_scale_dpi = style.font_scale_dpi
        imgui.get_current_context().style = scaled

    def rescale_windows(self, ratio_scale: float, ratio_text: float) -> None:
        """The figure's edge windows and subplot toolbars by both ratios, their handles by the UI scale's alone."""
        ratio = ratio_scale * ratio_text
        if self.figure is None:
            return
        windows = list(self.figure.imgui_windows.values())
        for subplot in self.figure:
            windows.extend(subplot.imgui_windows.values())
        for window in windows:
            if window is None or window.size is None:
                continue
            window._separator_thickness *= ratio_scale
            if getattr(window, "_old_size", None):
                window._old_size = round(window._old_size * ratio)
            wanted = getattr(window, "_masknmf_wanted", None)
            shrunk = wanted is not None and window.size == window._masknmf_fitted
            window.size = max(1, round(window.size * ratio))
            if shrunk:
                # fit_windows holds the window under its size for lack of room: that size follows the scale too
                window._masknmf_wanted, window._masknmf_fitted = round(wanted * ratio), window.size

    def fit_windows(self) -> None:
        """
        Shrink the figure's edge windows, largest first, until MIN_RENDER_AREA px of the canvas, times the UI scale and
        the text size, are left to each row and column of panels; they grow back to the size they were given once
        there is room.

        With less, a subplot's viewport gets a negative size, every draw raises before imgui runs, and the window stops
        updating: a UI scale or text size too big for the window, or a tiling desktop making the window small, froze it.
        """
        figure = self.figure
        # the sizes the panels' toolbars are drawn at, which new ones only become once the gui has run
        room = MIN_RENDER_AREA * self.applied * self.text_size
        area = figure.get_pygfx_render_area()
        rects = [subplot.frame.rect for subplot in figure]  ## (x, y, width, height) each
        for locations, axis in ((("left", "right"), 0), (("top", "bottom"), 1)):
            windows = [figure.imgui_windows[location] for location in locations]
            windows = [w for w in windows if w is not None and w.size is not None]
            # the panels side by side on this axis, from the smallest one's share of the render area
            count = max(round(area[2 + axis] / max(min(rect[2 + axis] for rect in rects), 1.0)), 1)
            # NOTE: the NDWidget's bottom controls set their own height every frame, so shrinking them does not last
            fixed = sum(w.size for w in windows if w.location == "bottom")
            windows = [w for w in windows if w.location != "bottom"]
            # a size other than the one last set here was set by hand, by a collapse or by the UI scale: it is the one wanted
            wanted = {id(w): w._masknmf_wanted if w.size == getattr(w, "_masknmf_fitted", None) else w.size for w in windows}
            excess = sum(wanted.values()) + fixed + room * count - figure.canvas.get_logical_size()[axis]
            for window in sorted(windows, key=lambda w: wanted[id(w)], reverse=True):
                floor = collapsed_size(window, window._separator_thickness)
                cut = max(min(excess, wanted[id(window)] - floor), 0)
                excess -= cut
                window._masknmf_wanted = wanted[id(window)]
                window._masknmf_fitted = window.size = round(wanted[id(window)] - cut)

    @classmethod
    def rescale_implot(cls, scale: float) -> None:
        if scale == cls.implot_scale or implot.get_current_context() is None:
            return
        ratio = scale / cls.implot_scale
        style = implot.get_style()
        for name in IMPLOT_SIZES:
            v = getattr(style, name)
            setattr(style, name, imgui.ImVec2(v.x * ratio, v.y * ratio))
        cls.implot_scale = scale


def draw_scaled(scaler: Scaler, gui: Callable[[], None]) -> None:
    scaler()
    gui()


def scaled(gui: Callable[[], None], figure=None) -> Callable[[], None]:
    """gui, run after its context is brought to SIZE; hand this to ImguiRenderer.set_gui."""
    return partial(draw_scaled, Scaler(figure), gui)


def fit_canvas(canvas) -> None:
    """Grow a desktop canvas made for a UI scale of 1 to the UI scale."""
    scale = ui_scale(canvas.get_pixel_ratio())
    if scale != 1.0:
        width, height = canvas.get_logical_size()
        canvas.set_logical_size(width * scale, height * scale)


def scale_figure(figure) -> None:
    """
    Draw a fastplotlib figure's imgui at SIZE, its window grown to match, with copy and paste through the system
    clipboard. Safe to call twice.
    """
    if getattr(figure, "_masknmf_scaled", False):
        return
    scaler = Scaler(figure)
    figure.imgui_renderer.set_gui(partial(draw_scaled, scaler, figure._draw_imgui))
    # an animation runs before the panels render, which is where a negative viewport raises
    # NOTE: a partial, since fastplotlib hands a figure to any animation that takes an argument, a bound method included
    figure.add_animations(partial(Scaler.fit_windows, scaler))
    connect_clipboard(figure.imgui_renderer.imgui_context)
    figure._masknmf_scaled = True
    if not is_notebook_canvas(figure):
        fit_canvas(figure.canvas)
