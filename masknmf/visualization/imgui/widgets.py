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
