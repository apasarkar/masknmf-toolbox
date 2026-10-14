from fastplotlib import ui
from imgui_bundle import imgui, imgui_toggle

from masknmf.visualization.imgui.theme import THEME, to_vec4


class CheckboxWindow(ui.ImguiWindow):
    """
    Imgui window with a single checkbox; read/write state via ``.value``.
    Place it with ``figure.add_imgui_window(window, location=..., size=..., title=...)``.
    """

    def __init__(self, label, value=False):
        super().__init__()
        self._label = label
        self.value = value

    def update(self):
        _, self.value = imgui.checkbox(self._label, self.value)


class SourceRightClickMenu(ui.StandardRightClickMenu):
    """
    The standard right-click menu with a choice of what the right-clicked subplot shows on top.
    ``choices(name)`` gives (the names to offer, the current one) for subplot ``name`` and
    ``on_pick(name, choice)`` switches it. Set it with ``figure.set_imgui_right_click(menu)``.
    """

    def __init__(self, choices, on_pick):
        super().__init__()
        self._choices = choices
        self._on_pick = on_pick

    def update(self):
        names, current = self._choices(self.subplot.name)
        for name in names:
            if imgui.menu_item(name, "", name == current)[0] and name != current:
                self._on_pick(self.subplot.name, name)
        imgui.separator()
        super().update()


def switch_width(left: str, right: str) -> float:
    """What :func:`draw_switch` takes."""
    return (
        imgui.calc_text_size(left).x
        + imgui.calc_text_size(right).x
        + imgui.get_frame_height() * imgui_toggle.ToggleConfig().width_ratio
        + 2 * imgui.get_style().item_inner_spacing.x
    )


def draw_switch(key: str, value: bool, left: str, right: str, live: bool = True) -> tuple[bool, bool]:
    """
    A two-way switch, ``left`` (False) or ``right`` (True): the side in use is lit and the knob takes the accent
    while ``live``, all grey otherwise; a click on either word picks that side. Returns (changed, value).
    """
    inner = imgui.get_style().item_inner_spacing.x
    dim, lit = imgui.get_style().color_(imgui.Col_.text_disabled), imgui.get_style().color_(imgui.Col_.text)
    frame = THEME.accent if live else (0.28, 0.28, 0.31)
    hover = (0.55, 0.78, 1.0) if live else (0.38, 0.38, 0.42)
    imgui.align_text_to_frame_padding()
    imgui.text_colored(lit if live and not value else dim, left)
    changed = imgui.is_item_clicked() and value
    value = value and not changed
    imgui.same_line(0, inner)
    for color, rgb in (
        (imgui.Col_.frame_bg, frame),
        (imgui.Col_.button, frame),
        (imgui.Col_.frame_bg_hovered, hover),
        (imgui.Col_.button_hovered, hover),
        (imgui.Col_.text, lit if live else dim),
    ):
        imgui.push_style_color(color, to_vec4(rgb))
    flipped, value = imgui_toggle.toggle(f"##switch_{key}", value, imgui_toggle.ToggleFlags_.animated)
    imgui.pop_style_color(5)
    imgui.same_line(0, inner)
    imgui.text_colored(lit if live and value else dim, right)
    if imgui.is_item_clicked() and not value:
        changed, value = True, True
    return changed or flipped, value
