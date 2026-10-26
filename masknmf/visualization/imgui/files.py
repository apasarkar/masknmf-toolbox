"""
Picking a path without depending on a native file dialog.

A dialog opens where the process runs, so a viewer driven from a notebook on another machine has nowhere
to draw one, and a linux box without zenity or kdialog on PATH cannot draw one at all. Every path a viewer
asks for goes through :func:`draw_path_prompt`: a typed field first, with the native dialog as a browse
shortcut that is disabled when nothing can draw it. ``mbo_utilities.gui._files`` mirrors this module so
the two GUIs behave the same way on the same machine.
"""

import os
import select
import shutil
import socket
import subprocess
import sys
from typing import Optional, Sequence

from imgui_bundle import imgui, portable_file_dialogs as pfd

from masknmf.visualization.imgui.theme import em, popup, tooltip

# the helpers portable-file-dialogs shells out to on linux, in the order it prefers them
HELPERS_LINUX = ("zenity", "matedialog", "qarma", "kdialog")


def native_dialogs_available() -> bool:
    """Whether a file dialog can appear where this process runs: always on windows and macos, on linux with a display and one of the helpers portable-file-dialogs shells out to."""
    if sys.platform in ("win32", "darwin"):
        return True
    if not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
        return False
    return any(shutil.which(name) for name in HELPERS_LINUX)


NATIVE_DIALOGS = native_dialogs_available()


class Picker:
    """
    A native dialog for one file or one folder, polled like a portable-file-dialogs dialog: ready, result, kill.

    On linux portable-file-dialogs runs zenity or kdialog and hands over the pick only once that process has
    exited. A zenity with a thread hung on an unresponsive network mount (a dropped sshfs, nfs) prints the pick
    and never finishes exiting, so the pick never arrives and the dialog's destructor blocks the process at exit.
    Here the helper's output is read as it comes, the pick lands at its newline, and kill does not wait on the
    helper. Windows and macos keep portable-file-dialogs.

    Args:
        title (str): The dialog's title
        start (str): The folder it opens in
        filters (Sequence[str] | None): portable-file-dialogs name / pattern pairs; None picks a folder
    """

    def __init__(self, title: str, start: str, filters: Optional[Sequence[str]] = None):
        self.dialog = None
        self.process = None
        self.output = b""
        self.closed = False
        if sys.platform in ("win32", "darwin"):
            if filters is None:
                self.dialog = pfd.select_folder(title, start)
            else:
                self.dialog = pfd.open_file(title, start, list(filters), pfd.opt.none)
            return
        helper = next(name for name in HELPERS_LINUX if shutil.which(name))
        pairs = list(zip(filters[::2], filters[1::2])) if filters is not None else []
        if helper == "kdialog":
            command = [helper, "--getexistingdirectory" if filters is None else "--getopenfilename", start]
            if filters is not None:
                command.append(" | ".join(f"{name}({patterns})" for name, patterns in pairs))
            command += ["--title", title]
        else:
            # zenity opens a file dialog in the folder's parent unless the folder ends in a separator
            if filters is not None and not start.endswith("/") and os.path.isdir(start):
                start += "/"
            command = [helper, "--file-selection", f"--filename={start}", "--title", title]
            command += [f"--file-filter={name}|{patterns}" for name, patterns in pairs]
            if filters is None:
                command.append("--directory")
        self.process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)

    def ready(self) -> bool:
        """Whether the dialog closed: the pick printed, or the helper's output ended without one."""
        if self.dialog is not None:
            return self.dialog.ready(0)
        fd = self.process.stdout.fileno()
        while not self.closed and len(select.select([fd], [], [], 0)[0]) > 0:
            chunk = os.read(fd, 65536)
            self.output += chunk
            self.closed = len(chunk) == 0 or b"\n" in self.output
        return self.closed

    def result(self):
        """The pick, "" when the dialog was cancelled; a list from portable-file-dialogs' open_file."""
        if self.dialog is not None:
            return self.dialog.result()
        return os.fsdecode(self.output.split(b"\n")[0])

    def kill(self) -> None:
        """Close the dialog without waiting on a helper that will not exit."""
        if self.dialog is not None:
            self.dialog.kill()
        elif self.process.poll() is None:
            self.process.kill()


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
