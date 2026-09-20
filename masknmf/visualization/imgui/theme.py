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


@contextmanager
def settings_table(table_id: str, captions):
    """
    A two-column settings table: dim captions in a fixed column as wide as the longest of
    ``captions``, controls in the stretch column. Yields False when the table is clipped; start
    each row with :func:`settings_row`.
    """
    caption_w = max(imgui.calc_text_size(c).x for c in captions) + em(0.8)
    flags = imgui.TableFlags_.sizing_stretch_prop | imgui.TableFlags_.no_pad_outer_x
    if not imgui.begin_table(table_id, 2, flags):
        yield False
        return
    try:
        imgui.table_setup_column("caption", imgui.TableColumnFlags_.width_fixed, caption_w)
        imgui.table_setup_column("control", imgui.TableColumnFlags_.width_stretch)
        yield True
    finally:
        imgui.end_table()


def settings_row(caption: str):
    """Start a row of a :func:`settings_table`: the dim caption on its widgets' frame baseline, the cursor in the control cell."""
    imgui.table_next_row()
    imgui.table_next_column()
    imgui.align_text_to_frame_padding()
    imgui.text_disabled(caption)
    imgui.table_next_column()


def right_aligned_text(text: str):
    """Dim ``text`` flush with the right edge of the current cell or window, on the next line when this one is full."""
    room = imgui.get_content_region_avail().x - imgui.calc_text_size(text).x
    if room < 0:
        imgui.new_line()
        room = imgui.get_content_region_avail().x - imgui.calc_text_size(text).x
    if room > 0:
        imgui.set_cursor_pos_x(imgui.get_cursor_pos_x() + room)
    imgui.align_text_to_frame_padding()
    imgui.text_disabled(text)


@contextmanager
def button_colors(fill, hover, text=None, on: bool = True):
    """Fill and hover colors, and optionally the text color, for the buttons drawn inside; a no-op unless ``on``."""
    if on:
        imgui.push_style_color(imgui.Col_.button, to_vec4(fill))
        imgui.push_style_color(imgui.Col_.button_hovered, to_vec4(hover))
        imgui.push_style_color(imgui.Col_.button_active, to_vec4(hover))
        if text is not None:
            imgui.push_style_color(imgui.Col_.text, to_vec4(text))
    try:
        yield
    finally:
        if on:
            imgui.pop_style_color(4 if text is not None else 3)


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
