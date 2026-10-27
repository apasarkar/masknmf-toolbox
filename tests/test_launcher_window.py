"""
The launcher hosted in a rendercanvas, as run_launcher opens it, on an offscreen canvas: it draws,
keeps its minimum size, and closes on Quit and Run.
"""

import os
import sys
import time

import numpy as np
import pytest
from rendercanvas.offscreen import RenderCanvas

from masknmf import launcher
from masknmf.visualization.imgui import scale


class Loop:
    """Holds what the launcher hands to its loop until the test runs it, as a running loop would between frames."""

    def __init__(self):
        self.callbacks = []

    def call_soon(self, callback, *args):
        self.callbacks.append((callback, args))

    def run_pending(self):
        callbacks, self.callbacks = self.callbacks, []
        for callback, args in callbacks:
            callback(*args)


@pytest.fixture
def window(monkeypatch):
    # a scale of 1 whatever the desktop running the tests is scaled to
    monkeypatch.setattr(scale, "SIZE", scale.Size(ui_scale=1.0))
    window = launcher.Launcher()
    window.attach(canvas=RenderCanvas(size=launcher.SIZE_WINDOW, pixel_ratio=1), loop=Loop())
    return window


def draw(window: launcher.Launcher, frames: int = 3) -> np.ndarray:
    """Draw a few frames, running what the launcher handed to its loop after each, and return the last frame."""
    for _ in range(frames):
        image = np.asarray(window.canvas.draw())
        window.loop.run_pending()
    return image


def test_window_draws_at_its_size(window):
    image = draw(window=window)
    assert image.shape[:2] == (launcher.SIZE_WINDOW[1], launcher.SIZE_WINDOW[0])
    assert image[..., :3].std() > 0


def test_window_is_kept_at_least_its_minimum_size(window):
    window.canvas.set_logical_size(100, 100)
    draw(window=window)
    width, height = window.canvas.get_logical_size()
    assert (width, height) == launcher.SIZE_MIN


def test_resizing_above_the_minimum_is_left_alone(window):
    size = (launcher.SIZE_MIN[0] + 50, launcher.SIZE_MIN[1] + 80)
    window.canvas.set_logical_size(*size)
    draw(window=window)
    assert window.canvas.get_logical_size() == size


def test_quit_closes_the_window_after_the_frame_without_arguments(window):
    draw(window=window)
    window.quit()
    assert not window.canvas.get_closed()
    window.loop.run_pending()
    assert window.canvas.get_closed()
    assert window.argv is None


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="linux dialogs run a helper like zenity")
def test_a_pick_lands_though_the_dialog_helper_never_exits(window, tmp_path, monkeypatch):
    # a zenity hung on a dropped network mount prints the pick and never finishes exiting
    filepath_results = tmp_path / "results.hdf5"
    filepath_zenity = tmp_path / "zenity"
    filepath_zenity.write_text(f"#!/bin/sh\necho '{filepath_results}'\nexec sleep 30\n")
    filepath_zenity.chmod(0o755)
    monkeypatch.setenv("PATH", f"{tmp_path}{os.pathsep}{os.environ['PATH']}")
    window.store.path = tmp_path / "recent.json"
    window.page = "open"

    window.open_picker(target=window.paths_open, key="source", filetypes=launcher.FILETYPES_RESULTS)
    picker = window.picker
    for _ in range(100):
        draw(window=window, frames=1)
        if window.picker is None:
            break
        time.sleep(0.02)
    picker.kill()
    assert window.paths_open["source"] == str(filepath_results)
