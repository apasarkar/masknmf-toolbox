"""File > Options: the settings the viewers share. The size ones are saved (see scale); the rest last the session."""

from dataclasses import dataclass
from typing import Optional

from imgui_bundle import icons_fontawesome_6 as fa
from imgui_bundle import imgui

from masknmf.visualization.imgui import scale
from masknmf.visualization.imgui.layout import is_notebook_canvas
from masknmf.visualization.imgui.theme import THEME, close_button, em, help_mark, popup, to_vec4


@dataclass
class Options:
    """What File > Options sets, for every viewer in the process."""

    separate_image_window: bool = True


OPTIONS = Options()

OPTIONS_LABEL = f"{fa.ICON_FA_GEARS}  options"

# a slider's value while it is dragged: the size applies when it is let go, so the popup does not resize under the mouse
DRAGGED: dict[str, float] = {}


def draw_options_menu() -> bool:
    """A menu bar holding File > options, for a panel with no File menu of its own. True when options is picked."""
    picked = False
    # a child carries the menu bar, so the docked window itself needs no flag
    imgui.begin_child(
        "##menu",
        imgui.ImVec2(0, 0),
        imgui.ChildFlags_.auto_resize_y | imgui.ChildFlags_.always_auto_resize,
        imgui.WindowFlags_.menu_bar,
    )
    if imgui.begin_menu_bar():
        if imgui.begin_menu("File"):
            picked = imgui.menu_item_simple(OPTIONS_LABEL)
            imgui.end_menu()
        imgui.end_menu_bar()
    imgui.end_child()
    return picked


def size_slider(label: str, key: str, value: float, bounds: tuple[float, float], fmt: str) -> Optional[float]:
    """A slider over value; the value it was let go at, else None."""
    imgui.set_next_item_width(em(12))
    _, DRAGGED[key] = imgui.slider_float(label, DRAGGED.get(key, value), bounds[0], bounds[1], fmt)
    if imgui.is_item_deactivated_after_edit():
        return DRAGGED.pop(key)
    if not imgui.is_item_active():
        DRAGGED.pop(key, None)
    return None


def draw_size_options() -> None:
    """The UI scale, on the desktop's or set, and the text size on top of it, for every viewer and saved."""
    pixel_ratio = imgui.get_io().display_framebuffer_scale.x
    imgui.text_disabled("Size")
    changed, auto = imgui.checkbox("Match the desktop", scale.SIZE.ui_scale is None)
    if changed:
        scale.set_ui_scale(None if auto else scale.auto_scale(pixel_ratio))
    help_mark(
        f"On: the UI scale follows the desktop's, {scale.auto_scale(pixel_ratio):.2f}x here.\n"
        "Off: the scale below, e.g. when text is too small or too big on a HiDPI screen."
    )
    imgui.begin_disabled(scale.SIZE.ui_scale is None)
    picked = size_slider("UI scale", "ui_scale", scale.ui_scale(pixel_ratio), scale.RANGE_SCALE, "%.2fx")
    if picked is not None:
        scale.set_ui_scale(picked)
    imgui.end_disabled()
    help_mark("Text, buttons, spacing and the docked panels together. Ctrl + / Ctrl - step it, Ctrl 0 matches the desktop again.")
    picked = size_slider("Text size", "text_size", scale.SIZE.text_size, scale.RANGE_TEXT, "%.2fx")
    if picked is not None:
        scale.set_text_size(picked)
    help_mark("Text only, on top of the UI scale.")
    imgui.text_disabled(f"Saved in {scale.FILEPATH_SETTINGS}")


def draw_options_popup(figure, is_open: bool) -> bool:
    """The Options window of the viewer on ``figure``. Returns the new open state."""
    if not is_open:
        return False
    opened, is_open = popup("Options", is_open)
    if opened:
        imgui.text_colored(to_vec4(THEME.accent), f"{fa.ICON_FA_GEARS}  Options")
        imgui.separator()
        imgui.dummy(imgui.ImVec2(em(24), em(0.3)))
        draw_size_options()
        imgui.dummy(imgui.ImVec2(0, em(0.3)))
        imgui.text_disabled("Static images / Full FOV")
        # a notebook has no windows to open: the images stay a popup inside the viewer
        notebook = is_notebook_canvas(figure)
        imgui.begin_disabled(notebook)
        changed, separate = imgui.checkbox("Open as separate window", OPTIONS.separate_image_window and not notebook)
        if changed:
            OPTIONS.separate_image_window = separate
        imgui.end_disabled()
        help_mark(
            "Separate window: the images open in a window of their own, movable off this one, e.g. to another screen.\n"
            "Off, or in a notebook, they open as a popup inside this window."
        )
        imgui.dummy(imgui.ImVec2(0, em(0.3)))
        imgui.separator()
        if close_button():
            is_open = False
    imgui.end()
    return is_open
