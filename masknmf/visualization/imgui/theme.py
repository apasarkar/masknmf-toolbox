"""
Shared imgui styling for the viewers: a small color theme plus card, section
and popup helpers so panels read as one design.
"""

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Tuple

from imgui_bundle import imgui

Color = Tuple[float, float, float, float]


def to_vec4(color) -> imgui.ImVec4:
    if isinstance(color, imgui.ImVec4):
        return color
    r, g, b = color[:3]
    return imgui.ImVec4(r, g, b, color[3] if len(color) > 3 else 1.0)


@dataclass(frozen=True)
class Theme:
    """RGBA palette (0..1) and rounding used by the viewer panels."""

    accent: Color = (0.40, 0.68, 1.00, 1.0)
    text_dim: Color = (0.62, 0.62, 0.65, 1.0)
    ok: Color = (0.40, 0.90, 0.40, 1.0)
    warn: Color = (1.00, 0.80, 0.20, 1.0)
    err: Color = (1.00, 0.40, 0.40, 1.0)
    code: Color = (0.55, 0.75, 1.00, 1.0)
    border: Color = (0.35, 0.35, 0.37, 0.7)
    danger: Color = (0.75, 0.15, 0.15, 0.8)
    danger_hover: Color = (0.90, 0.20, 0.20, 1.0)
    rounding: float = 6.0
    card_rounding: float = 0.0


THEME = Theme()


def em(x: float = 1.0) -> float:
    """x font heights in pixels; only valid inside a frame."""
    return imgui.get_font_size() * x


@contextmanager
def card(name: str, title: str, height: float, width: float = 0.0, theme: Theme = THEME):
    """
    Bordered child window with an accent title.

    Parameters
    ----------
    height : float
        Fixed height so cards laid out on one row line up.
    width : float
        0 sizes the card to its content.
    """
    flags = imgui.ChildFlags_.borders
    if width == 0:
        flags |= imgui.ChildFlags_.auto_resize_x
    imgui.push_style_color(imgui.Col_.border, to_vec4(theme.border))
    imgui.push_style_var(imgui.StyleVar_.child_rounding, theme.card_rounding)
    imgui.begin_child(
        name,
        imgui.ImVec2(width, height),
        child_flags=flags,
        window_flags=imgui.WindowFlags_.no_scrollbar,
    )
    imgui.text_colored(to_vec4(theme.accent), title)
    try:
        yield
    finally:
        imgui.end_child()
        imgui.pop_style_var()
        imgui.pop_style_color()


def section(title: str, theme: Theme = THEME):
    """Accent heading with a rule under it."""
    imgui.dummy(imgui.ImVec2(0, em(0.3)))
    imgui.text_colored(to_vec4(theme.accent), title)
    imgui.separator()
    imgui.dummy(imgui.ImVec2(0, em(0.2)))


def popup(title: str, is_open: bool, theme: Theme = THEME) -> tuple[bool, bool]:
    """
    Begin a centered, auto-sized, closable window.

    Returns
    -------
    (draw_contents, still_open); call ``imgui.end()`` either way.
    """
    imgui.set_next_window_pos(
        imgui.get_main_viewport().get_center(), imgui.Cond_.appearing, pivot=imgui.ImVec2(0.5, 0.5)
    )
    imgui.push_style_var(imgui.StyleVar_.window_rounding, theme.rounding)
    imgui.push_style_var(imgui.StyleVar_.window_padding, imgui.ImVec2(em(1.0), em(0.8)))
    opened, is_open = imgui.begin(
        f"{title}###{title}",
        is_open,
        flags=imgui.WindowFlags_.no_saved_settings | imgui.WindowFlags_.always_auto_resize,
    )
    imgui.pop_style_var(2)
    return opened, is_open


def close_button(theme: Theme = THEME) -> bool:
    imgui.dummy(imgui.ImVec2(0, em(0.3)))
    return imgui.button("Close", imgui.ImVec2(em(6), 0))


LABEL_BUTTON_ALPHA = 0.5
SELECTED_BUTTON = (0.26, 0.59, 1.00, 1.0)
SELECTED_BUTTON_HOVER = (0.34, 0.66, 1.00, 1.0)


@contextmanager
def button_colors(button, hovered=None, active=None):
    """Push button colors for the block; popped on exit even if it raises."""
    n = 1
    imgui.push_style_color(imgui.Col_.button, to_vec4(button))
    if hovered is not None:
        imgui.push_style_color(imgui.Col_.button_hovered, to_vec4(hovered))
        n += 1
    if active is not None:
        imgui.push_style_color(imgui.Col_.button_active, to_vec4(active))
        n += 1
    try:
        yield
    finally:
        imgui.pop_style_color(n)


def label_button(rgb):
    """Button colored like a class label."""
    r, g, b = rgb[:3]
    return button_colors((r, g, b, LABEL_BUTTON_ALPHA))


def danger_button(theme: Theme = THEME):
    """Button colored for a destructive action."""
    return button_colors(theme.danger, theme.danger_hover)


@contextmanager
def toggle_button(active: bool):
    """Style one button as the chosen option while ``active``; a no-op otherwise."""
    if not active:
        yield
        return
    with button_colors(SELECTED_BUTTON, SELECTED_BUTTON_HOVER):
        yield


def label_color(rgb, alpha: float = 1.0) -> imgui.ImVec4:
    """A class color tuple as an ImVec4."""
    r, g, b = rgb[:3]
    return imgui.ImVec4(r, g, b, alpha)


def set_tooltip(text: str):
    """Tooltip on the item just drawn, shown for a disabled item too."""
    if imgui.is_item_hovered(imgui.HoveredFlags_.allow_when_disabled):
        imgui.set_tooltip(text)
