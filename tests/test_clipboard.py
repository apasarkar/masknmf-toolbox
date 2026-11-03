"""Copy and paste in an imgui window go through the system clipboard, here a dict standing in for it."""

import pyperclip
import pytest
import wgpu
from imgui_bundle import imgui
from rendercanvas.offscreen import RenderCanvas
from wgpu.utils.imgui import ImguiRenderer

from masknmf.visualization.imgui.clipboard import connect_clipboard


class TextField:
    """One text field, given keyboard focus on the first frame."""

    def __init__(self):
        self.text = ""
        self.frames = 0

    def __call__(self) -> None:
        imgui.begin("field")
        if self.frames == 0:
            imgui.set_keyboard_focus_here()
        _, self.text = imgui.input_text("##field", self.text)
        imgui.end()
        self.frames += 1


@pytest.fixture
def system_clipboard(monkeypatch):
    system = {"text": "copied in a terminal"}
    monkeypatch.setattr(pyperclip, "paste", lambda: system["text"])
    monkeypatch.setattr(pyperclip, "copy", lambda text: system.update(text=text))
    monkeypatch.setattr(pyperclip, "determine_clipboard", lambda: (pyperclip.copy, pyperclip.paste))
    return system


@pytest.fixture
def context():
    context = imgui.create_context()
    imgui.set_current_context(context)
    yield context
    imgui.destroy_context(context)


def draw(canvas: RenderCanvas, frames: int) -> None:
    """Draw frames, each after dispatching submitted events, as a canvas's loop does and an offscreen draw does not."""
    for _ in range(frames):
        canvas._process_events()
        canvas.draw()


def test_ctrl_v_on_a_canvas_pastes_the_system_clipboard(system_clipboard):
    canvas = RenderCanvas(size=(400, 200), pixel_ratio=1)
    renderer = ImguiRenderer(wgpu.utils.get_default_device(), canvas)
    connect_clipboard(renderer.imgui_context)
    field = TextField()
    renderer.set_gui(field)
    canvas.request_draw(renderer.render)
    draw(canvas=canvas, frames=2)
    for event_type, key in (("key_down", "Control"), ("key_down", "v"), ("key_up", "v"), ("key_up", "Control")):
        canvas.submit_event({"event_type": event_type, "key": key, "modifiers": ("Control",)})
        draw(canvas=canvas, frames=2)
    assert field.text == "copied in a terminal"


def test_imgui_copies_onto_the_system_clipboard(system_clipboard, context):
    connect_clipboard(context)
    imgui.set_clipboard_text("copied in imgui")
    assert system_clipboard["text"] == "copied in imgui"
    assert imgui.get_clipboard_text() == "copied in imgui"


def test_without_a_clipboard_tool_it_warns_and_pastes_nothing(monkeypatch, context):
    monkeypatch.setattr(pyperclip, "determine_clipboard", pyperclip.init_no_clipboard)
    with pytest.warns(UserWarning, match="No system clipboard"):
        connect_clipboard(context)
    imgui.set_clipboard_text("dropped")
    assert imgui.get_clipboard_text() == ""
