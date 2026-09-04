"""
Picking a path without depending on a native file dialog.

A dialog opens where the process runs, so a viewer driven from a notebook on
another machine has nowhere to draw one. Every path the user has to give goes
through :func:`draw_path_prompt`, which is a typed field first and offers the
native dialog only as a shortcut.
"""

import os
import shutil
import socket
import sys
from typing import Optional

from imgui_bundle import imgui

from masknmf.visualization.imgui.theme import em, popup, set_tooltip

__all__ = ["NATIVE_DIALOGS", "PathPrompt", "draw_path_prompt", "native_dialogs_available"]


def native_dialogs_available() -> bool:
    """
    Whether a file dialog can appear on the machine this process runs on.

    Windows and macOS always have one. Linux needs a display and one of the
    dialog helpers the portable-file-dialogs backend shells out to.
    """
    if sys.platform in ("win32", "darwin"):
        return True
    if not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
        return False
    return any(
        shutil.which(name) for name in ("zenity", "matedialog", "qarma", "kdialog")
    )


NATIVE_DIALOGS = native_dialogs_available()


class PathPrompt:
    """
    State for one path popup: whether it is open, the path, and what it said.

    Args:
        title (str): popup title
        path (str): initial path
        action (str): label of the button that accepts the path
        hint (str): line under the title saying what the path is for
    """

    def __init__(self, title: str, path: str = "", action: str = "open", hint: str = ""):
        self.title = title
        self.path = path
        self.action = action
        self.hint = hint
        self.open = False
        self.status = ""

    def start(self, path: Optional[str] = None):
        """Open the popup, optionally on a different path."""
        if path is not None:
            self.path = str(path)
        self.status = ""
        self.open = True


def draw_path_prompt(prompt: PathPrompt) -> tuple:
    """
    Draw one path popup.

    Returns (submitted_path, browse_clicked): the path when the action button or
    enter was pressed, else None, and whether the browse shortcut was pressed.
    The caller closes the popup by setting ``prompt.open``, so a failed action
    can leave it up with a message in ``prompt.status``.
    """
    if not prompt.open:
        return None, False
    opened, prompt.open = popup(prompt.title, prompt.open)
    submitted, browse = None, False
    if opened:
        imgui.text_disabled(f"{prompt.hint or 'read by this process'}, on {socket.gethostname()}")
        imgui.set_next_item_width(em(28))
        entered, prompt.path = imgui.input_text(
            f"##path-{prompt.title}", prompt.path, imgui.InputTextFlags_.enter_returns_true
        )
        if imgui.button(prompt.action, imgui.ImVec2(em(6), 0)) or entered:
            submitted = prompt.path
        imgui.same_line(0, em(0.5))
        if not NATIVE_DIALOGS:
            imgui.begin_disabled()
        if imgui.button("browse", imgui.ImVec2(em(6), 0)):
            browse = True
        if not NATIVE_DIALOGS:
            imgui.end_disabled()
        set_tooltip(
            "pick a path in a file dialog"
            if NATIVE_DIALOGS
            else "no file dialog on this machine; type the path instead"
        )
        imgui.same_line(0, em(0.5))
        if imgui.button("close", imgui.ImVec2(em(6), 0)):
            prompt.open = False
        if prompt.status:
            imgui.text_disabled(prompt.status)
    imgui.end()
    return submitted, browse
