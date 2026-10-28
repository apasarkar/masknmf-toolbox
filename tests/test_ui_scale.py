"""The UI scale: where it starts, what it saves, and how an imgui context, its edge windows and its fonts follow it."""

import json

import pytest
from imgui_bundle import imgui
from rendercanvas.offscreen import RenderCanvas

from masknmf import launcher
from masknmf.visualization.imgui import scale


@pytest.fixture(autouse=True)
def settings(tmp_path, monkeypatch):
    """A settings file of the test's own and a desktop at 2x."""
    monkeypatch.setattr(scale, "DIR_CONFIG", tmp_path)
    monkeypatch.setattr(scale, "FILEPATH_SETTINGS", tmp_path / "settings.json")
    monkeypatch.setattr(scale, "SIZE", scale.Size())
    monkeypatch.setattr(scale, "desktop_scale", lambda: 2.0)
    monkeypatch.delenv("MASKNMF_UI_SCALE", raising=False)
    return tmp_path / "settings.json"


def test_auto_is_the_desktop_scale_over_the_pixel_ratio():
    assert scale.auto_scale(1.0) == 2.0
    assert scale.auto_scale(2.0) == 1.0
    # a canvas drawn finer than the desktop is never shrunk
    assert scale.auto_scale(3.0) == 1.0


def test_initial_size_takes_the_env_over_the_saved_and_clamps(settings, monkeypatch):
    settings.write_text(json.dumps({"ui_scale": 1.5, "text_size": 9}))
    assert scale.initial_size() == scale.Size(ui_scale=1.5, text_size=scale.RANGE_TEXT[1])
    monkeypatch.setenv("MASKNMF_UI_SCALE", "3")
    assert scale.initial_size().ui_scale == 3.0


@pytest.mark.parametrize("content", ["not json", "[1, 2]", '{"ui_scale": "big", "text_size": -1}'])
def test_unreadable_settings_fall_back_to_auto(settings, content):
    settings.write_text(content)
    assert scale.initial_size() == scale.Size()


def test_setting_saves_beside_other_settings(settings):
    settings.write_text(json.dumps({"other": 1}))
    scale.set_ui_scale(1.25)
    scale.set_text_size(1.1)
    assert json.loads(settings.read_text()) == {"other": 1, "ui_scale": 1.25, "text_size": 1.1}
    scale.set_ui_scale(None)
    assert json.loads(settings.read_text())["ui_scale"] is None
    assert scale.SIZE.ui_scale is None


class Loop:
    """Holds what the launcher hands to its loop until the test runs it."""

    def __init__(self):
        self.callbacks = []

    def call_soon(self, callback, *args):
        self.callbacks.append((callback, args))

    def run_pending(self):
        callbacks, self.callbacks = self.callbacks, []
        for callback, args in callbacks:
            callback(*args)


def open_launcher(after=None) -> launcher.Launcher:
    """The launcher on an offscreen canvas at pixel ratio 1, as on a desktop that does not tell GLFW its scale."""
    window = launcher.Launcher()
    if after is not None:
        gui = window.draw_window
        window.draw_window = lambda: (gui(), after())
    window.attach(canvas=RenderCanvas(size=(1800, 1400), pixel_ratio=1), loop=Loop())
    return window


def draw(window: launcher.Launcher, frames: int = 3) -> None:
    for _ in range(frames):
        window.canvas.draw()
        window.loop.run_pending()


def test_style_follows_the_scale_without_drift():
    window = open_launcher()
    draw(window)
    assert window.scaler.applied == 2.0
    assert imgui.get_style().window_padding.x == 16.0
    # stepped up and back: the sizes come back exactly, being the default style's times the scale
    for target in (2.2, 2.42, 2.2, 2.0):
        scale.set_ui_scale(target)
        draw(window, frames=1)
    assert imgui.get_style().window_padding.x == 16.0
    assert imgui.get_style().font_scale_main == 2.0


def test_text_size_scales_the_fonts_only():
    window = open_launcher()
    scale.set_text_size(1.5)
    draw(window)
    assert imgui.get_style().font_scale_main == 3.0
    assert imgui.get_style().window_padding.x == 16.0


def test_minimum_size_grows_with_the_scale():
    window = open_launcher()
    draw(window)
    window.canvas.set_logical_size(100, 100)
    draw(window)
    assert window.canvas.get_logical_size() == tuple(2 * v for v in launcher.SIZE_MIN)


def test_push_font_em_is_not_scaled_twice():
    seen = {}

    def measure():
        em = imgui.get_font_size()
        scale.push_font_em(1.5)
        seen["ratio"] = imgui.get_font_size() / em
        imgui.pop_font()

    draw(open_launcher(after=measure))
    assert seen["ratio"] == pytest.approx(1.5, abs=0.05)
