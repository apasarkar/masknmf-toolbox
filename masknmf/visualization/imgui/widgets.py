from fastplotlib import ui
from imgui_bundle import imgui


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
