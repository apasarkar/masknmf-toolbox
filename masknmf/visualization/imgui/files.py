"""
Picking a path without depending on a native file dialog.

A dialog opens where the process runs, so a viewer driven from a notebook on another machine has nowhere
to draw one, and a linux box without zenity or kdialog on PATH cannot draw one at all. Every path a viewer
asks for goes through :func:`draw_path_prompt`: a typed field first, with the native dialog as a browse
shortcut that is disabled when nothing can draw it. ``mbo_utilities.gui._files`` mirrors this module so
the two GUIs behave the same way on the same machine.
"""

import os
import shutil
import socket
import sys
from typing import Optional, Sequence

from imgui_bundle import imgui, portable_file_dialogs as pfd

from masknmf.visualization.imgui.theme import em, popup, tooltip


def native_dialogs_available() -> bool:
    """Whether a file dialog can appear where this process runs: always on windows and macos, on linux with a display and one of the helpers portable-file-dialogs shells out to."""
    if sys.platform in ("win32", "darwin"):
        return True
    if not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
        return False
    return any(shutil.which(name) for name in ("zenity", "matedialog", "qarma", "kdialog"))


NATIVE_DIALOGS = native_dialogs_available()


class PathPrompt:
    """
    State for one path popup: whether it is open, the path, and what it said.

    ``kind`` is what browse opens: "open" (a file; several joined with ";" when ``multiple``), "save" or
    "folder"; ``filters`` are portable-file-dialogs name / pattern pairs.
    """

    def __init__(
        self,
        title: str,
        path: str = "",
        action: str = "open",
        hint: str = "",
        kind: str = "open",
        filters: Sequence[str] = ("All files", "*"),
        multiple: bool = False,
    ):
        self.title = title
        self.path = path
        self.action = action
        self.hint = hint
        self.kind = kind
        self.filters = list(filters)
        self.multiple = multiple
        self.open = False
        self.status = ""
        self.dialog = None

    def start(self, path: Optional[str] = None):
        """Open the popup, optionally on a different path."""
        if path is not None:
            self.path = str(path)
        self.status = ""
        self.open = True


def draw_path_prompt(prompt: PathPrompt) -> Optional[str]:
    """
    Draw one path popup; the path when its action button or enter was pressed, else None.

    The caller closes the popup by setting ``prompt.open``, so a failed action can leave it up with a
    message in ``prompt.status``. A pick in the native dialog lands in the field, for the action to take.
    """
    if prompt.dialog is not None and prompt.dialog.ready(0):
        result = prompt.dialog.result()
        prompt.dialog = None
        if result:
            prompt.path = ";".join(result) if isinstance(result, list) else result
    if not prompt.open:
        return None
    opened, prompt.open = popup(prompt.title, prompt.open)
    submitted = None
    if opened:
        imgui.text_disabled(f"{prompt.hint or 'read by this process'}, on {socket.gethostname()}")
        imgui.set_next_item_width(em(28))
        entered, prompt.path = imgui.input_text(
            f"##path-{prompt.title}", prompt.path, imgui.InputTextFlags_.enter_returns_true
        )
        if imgui.is_window_appearing():
            imgui.set_keyboard_focus_here(-1)
        if imgui.button(prompt.action, imgui.ImVec2(em(6), 0)) or entered:
            submitted = prompt.path.strip().strip('"') or None
        imgui.same_line(0, em(0.5))
        imgui.begin_disabled(not NATIVE_DIALOGS or prompt.dialog is not None)
        if imgui.button("browse", imgui.ImVec2(em(6), 0)):
            start = prompt.path.strip().strip('"') or os.getcwd()
            if prompt.kind == "folder":
                prompt.dialog = pfd.select_folder(prompt.title, start if os.path.isdir(start) else os.path.dirname(start))
            elif prompt.kind == "save":
                prompt.dialog = pfd.save_file(prompt.title, start, prompt.filters)
            else:
                prompt.dialog = pfd.open_file(
                    prompt.title, start, prompt.filters, pfd.opt.multiselect if prompt.multiple else pfd.opt.none
                )
        imgui.end_disabled()
        tooltip(
            "pick it in a file dialog on the machine running python"
            if NATIVE_DIALOGS
            else "no file dialog on this machine; type the path instead"
        )
        imgui.same_line(0, em(0.5))
        if imgui.button("close", imgui.ImVec2(em(6), 0)):
            prompt.open = False
        if prompt.status:
            imgui.push_text_wrap_pos(em(30))
            imgui.text_disabled(prompt.status)
            imgui.pop_text_wrap_pos()
    imgui.end()
    return submitted
