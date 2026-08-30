from contextlib import contextmanager
from typing import Optional

from imgui_bundle import imgui

ImVec4 = imgui.ImVec4

# status text
OK = ImVec4(0.35, 0.90, 0.35, 1.0)
WARN = ImVec4(1.00, 0.75, 0.25, 1.0)
ERROR = ImVec4(1.00, 0.30, 0.30, 1.0)
HINT = ImVec4(1.00, 0.85, 0.40, 1.0)
CODE = ImVec4(0.55, 0.75, 1.00, 1.0)

# canvas overlays
HIGHLIGHT = ImVec4(1.00, 0.90, 0.20, 0.9)
CONTOUR = ImVec4(0.20, 1.00, 0.40, 0.9)
TEXT_ON_DARK = ImVec4(1.0, 1.0, 1.0, 1.0)
TEXT_ON_LIGHT = ImVec4(0.0, 0.0, 0.0, 1.0)
LUMA_LIGHT = 140  # 0-255 luma above which TEXT_ON_LIGHT is used

# buttons
DANGER = ImVec4(0.75, 0.15, 0.15, 0.8)
DANGER_HOVERED = ImVec4(0.90, 0.20, 0.20, 1.0)
LABEL_BUTTON_ALPHA = 0.5


def label_color(rgb, alpha: float = 1.0) -> ImVec4:
    """Class colour tuple from LabelSet.color as an ImVec4."""
    return ImVec4(*rgb, alpha)


def u32(color: ImVec4) -> int:
    """Packed colour for draw-list calls."""
    return imgui.color_convert_float4_to_u32(color)


def text_on(luma: float) -> int:
    """Packed overlay text colour that reads against a pixel of the given luma."""
    return u32(TEXT_ON_LIGHT if luma > LUMA_LIGHT else TEXT_ON_DARK)


@contextmanager
def button_colors(button: ImVec4, hovered: Optional[ImVec4] = None, active: Optional[ImVec4] = None):
    """Push button colours for the block; popped on exit even if it raises."""
    n = 1
    imgui.push_style_color(imgui.Col_.button, button)
    if hovered is not None:
        imgui.push_style_color(imgui.Col_.button_hovered, hovered)
        n += 1
    if active is not None:
        imgui.push_style_color(imgui.Col_.button_active, active)
        n += 1
    try:
        yield
    finally:
        imgui.pop_style_color(n)


def label_button(rgb):
    return button_colors(label_color(rgb, LABEL_BUTTON_ALPHA))


def danger_button():
    return button_colors(DANGER, DANGER_HOVERED)
