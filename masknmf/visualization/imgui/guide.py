"""
Drawing primitives the guide pages and the launcher's Overview share: the palette, boxes and arrows on a draw
list, section headings and clickable cards.
"""

from imgui_bundle import imgui

ACCENT = imgui.ImVec4(0.40, 0.68, 1.00, 1.0)
KEY = imgui.ImVec4(1.00, 0.80, 0.20, 1.0)
DIM = imgui.ImVec4(0.62, 0.62, 0.65, 1.0)
TEXT = imgui.ImVec4(0.92, 0.92, 0.94, 1.0)
CARD = imgui.ImVec4(0.17, 0.18, 0.21, 1.0)
CARD_HOVER = imgui.ImVec4(0.21, 0.22, 0.26, 1.0)
EDGE = imgui.ImVec4(0.35, 0.35, 0.37, 1.0)
PLOT = imgui.ImVec4(0.06, 0.06, 0.08, 1.0)


def u32(color: imgui.ImVec4, alpha: float = 1.0) -> int:
    """
    Pack a color for a draw list.

    Parameters
    ----------
    color : imgui.ImVec4
        The color; its own alpha is ignored.
    alpha : float
        The alpha packed with it.

    Returns
    -------
    int
        The ImU32 color.
    """
    return imgui.color_convert_float4_to_u32(imgui.ImVec4(color.x, color.y, color.z, alpha))


def box(dl, x: float, y: float, w: float, h: float, label: str, color: imgui.ImVec4, fill_alpha: float = 0.0) -> None:
    """
    A rounded box with its label centered.

    Parameters
    ----------
    dl : imgui.ImDrawList
        Where to draw.
    x, y, w, h : float
        The box, in screen pixels.
    label : str
        Centered in the box.
    color : imgui.ImVec4
        The label and border color; ``fill_alpha`` tints the card fill with it.
    """
    a, b = imgui.ImVec2(x, y), imgui.ImVec2(x + w, y + h)
    dl.add_rect_filled(a, b, u32(CARD), 4.0)
    if fill_alpha:
        dl.add_rect_filled(a, b, u32(color, fill_alpha), 4.0)
    dl.add_rect(a, b, u32(color, 0.6), 4.0)
    size = imgui.calc_text_size(label)
    dl.add_text(imgui.ImVec2(x + (w - size.x) / 2, y + (h - size.y) / 2), u32(color), label)


def arrow(dl, x0: float, x1: float, y: float) -> None:
    """
    A dim arrow pointing right.

    Parameters
    ----------
    dl : imgui.ImDrawList
        Where to draw.
    x0, x1 : float
        The tail and the tip, in screen pixels.
    y : float
        The line the arrow sits on.
    """
    col = u32(DIM)
    dl.add_line(imgui.ImVec2(x0, y), imgui.ImVec2(x1 - 5, y), col, 1.5)
    dl.add_triangle_filled(imgui.ImVec2(x1, y), imgui.ImVec2(x1 - 7, y - 4), imgui.ImVec2(x1 - 7, y + 4), col)


def arrow_down(dl, x: float, y0: float, y1: float) -> None:
    """
    A dim arrow pointing down.

    Parameters
    ----------
    dl : imgui.ImDrawList
        Where to draw.
    x : float
        The line the arrow sits on.
    y0, y1 : float
        The tail and the tip, in screen pixels.
    """
    col = u32(DIM)
    dl.add_line(imgui.ImVec2(x, y0), imgui.ImVec2(x, y1 - 5), col, 1.5)
    dl.add_triangle_filled(imgui.ImVec2(x, y1), imgui.ImVec2(x - 4, y1 - 7), imgui.ImVec2(x + 4, y1 - 7), col)


def heading(icon: str, text: str) -> None:
    """
    An accent section heading, spaced from what is above it.

    Parameters
    ----------
    icon : str
        A Font Awesome icon drawn before the text.
    text : str
        The heading.
    """
    em = imgui.get_font_size()
    imgui.dummy(imgui.ImVec2(0, 0.7 * em))
    imgui.text_colored(ACCENT, f"{icon}  {text}")
    imgui.dummy(imgui.ImVec2(0, 0.1 * em))


def card_button(dl, name: str, size: imgui.ImVec2, selected: bool = False) -> tuple[bool, imgui.ImVec2]:
    """
    A clickable card at the cursor: its fill, hover and border; the caller draws what is inside.

    Parameters
    ----------
    dl : imgui.ImDrawList
        Where to draw.
    name : str
        The card's imgui id.
    size : imgui.ImVec2
        The card's size.
    selected : bool
        Whether the border is the accent.

    Returns
    -------
    clicked : bool
        Whether the card was clicked.
    top_left : imgui.ImVec2
        The card's top left corner, in screen pixels.
    """
    clicked = imgui.invisible_button(f"##card_{name}", size)
    hovered = imgui.is_item_hovered()
    a, b = imgui.get_item_rect_min(), imgui.get_item_rect_max()
    dl.add_rect_filled(a, b, u32(CARD_HOVER if hovered else CARD), 5.0)
    dl.add_rect(a, b, u32(ACCENT if selected else EDGE), 5.0, 1.5 if selected else 1.0)
    if hovered:
        imgui.set_mouse_cursor(imgui.MouseCursor_.hand)
    return clicked, a
