"""
Picking a path without depending on a native file dialog.

A dialog opens where the process runs, so a viewer driven from a notebook on another machine has nowhere
to draw one, and a linux box without a helper or a desktop portal cannot draw one at all. Every path a viewer
asks for goes through :func:`draw_path_prompt`: a typed field first, with the native dialog as a browse
shortcut that is disabled when nothing can draw it. ``mbo_utilities.gui._files`` mirrors this module so
the two GUIs behave the same way on the same machine.

What draws the dialog, by platform:

- windows, macos: portable-file-dialogs (the system's own dialog).
- linux with zenity, matedialog, qarma or kdialog on PATH (Ubuntu / GNOME, KDE): that helper.
- linux without one (Omarchy, other Hyprland or sway setups): xdg-desktop-portal, see ``portal_dialog``.
"""

import json
import os
import select
import shutil
import socket
import subprocess
import sys
from typing import Optional, Sequence

from imgui_bundle import imgui, portable_file_dialogs as pfd

from masknmf.visualization.imgui import portal_dialog
from masknmf.visualization.imgui.theme import em, popup, tooltip

# the helpers portable-file-dialogs shells out to on linux, in the order it prefers them
HELPERS_LINUX = ("zenity", "matedialog", "qarma", "kdialog")
PORTAL = "portal"
# zenity's separator between several picks: a character no path holds in practice
SEPARATOR_PICKS = "\x1f"


def linux_backend() -> Optional[str]:
    """What draws a dialog on this linux machine: a helper on PATH, else the desktop portal, else None."""
    if not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
        return None
    helper = next((name for name in HELPERS_LINUX if shutil.which(name)), None)
    if helper is not None:
        return helper
    return PORTAL if portal_dialog.portal_available() else None


def native_dialogs_available() -> bool:
    """Whether a file dialog can appear where this process runs: always on windows and macos, on linux with a display and a helper or the desktop portal."""
    return sys.platform in ("win32", "darwin") or linux_backend() is not None


NATIVE_DIALOGS = native_dialogs_available()


def helper_command(backend: str, kind: str, title: str, start: str, filters: Sequence[str], multiple: bool) -> list[str]:
    """The command line that opens one dialog with ``backend``; arguments as :class:`Picker`."""
    pairs = list(zip(filters[::2], filters[1::2]))
    if backend == PORTAL:
        command = [sys.executable, "-I", portal_dialog.__file__, kind, title, start]
        return command + (["--multiple"] if multiple else []) + list(filters)
    if backend == "kdialog":
        mode = {"folder": "--getexistingdirectory", "save": "--getsavefilename", "open": "--getopenfilename"}[kind]
        command = [backend, mode, start]
        if kind != "folder" and len(pairs) > 0:
            command.append(" | ".join(f"{name}({patterns})" for name, patterns in pairs))
        if multiple:
            command += ["--multiple", "--separate-output"]
        return command + ["--title", title]
    # zenity opens a file dialog in the folder's parent unless the folder ends in a separator
    if kind != "folder" and not start.endswith("/") and os.path.isdir(start):
        start += "/"
    command = [backend, "--file-selection", f"--filename={start}", "--title", title]
    if kind != "folder":
        command += [f"--file-filter={name}|{patterns}" for name, patterns in pairs]
    if kind == "folder":
        command.append("--directory")
    if kind == "save":
        command += ["--save", "--confirm-overwrite"]
    if multiple:
        command += ["--multiple", f"--separator={SEPARATOR_PICKS}"]
    return command


class Picker:
    """
    A native dialog for a file, several files, a file to save or a folder, polled like a portable-file-dialogs
    dialog: ready, result, kill.

    On linux portable-file-dialogs runs zenity or kdialog and hands over the pick only once that process has
    exited. A zenity with a thread hung on an unresponsive network mount (a dropped sshfs, nfs) prints the pick
    and never finishes exiting, so the pick never arrives and the dialog's destructor blocks the process at exit.
    Here the helper's output is read as it comes, the pick lands at its newline, and kill does not wait on the
    helper. Without a helper on PATH the desktop portal draws the dialog (``portal_dialog``, run as a helper too).
    Windows and macos keep portable-file-dialogs.

    Args:
        title (str): The dialog's title
        start (str): The folder it opens in, or a file in it
        filters (Sequence[str] | None): portable-file-dialogs name / pattern pairs
        kind (str | None): "open", "save" or "folder"; None is "folder" without filters, else "open"
        multiple (bool): Whether an open dialog takes several files; result is then a list
    """

    def __init__(
        self,
        title: str,
        start: str,
        filters: Optional[Sequence[str]] = None,
        kind: Optional[str] = None,
        multiple: bool = False,
    ):
        self.dialog = None
        self.process = None
        self.output = b""
        self.closed = False
        self.kind = kind if kind is not None else ("folder" if filters is None else "open")
        self.multiple = multiple and self.kind == "open"
        filters = [] if filters is None else list(filters)
        if sys.platform in ("win32", "darwin"):
            if self.kind == "folder":
                self.dialog = pfd.select_folder(title, start if os.path.isdir(start) else os.path.dirname(start))
            elif self.kind == "save":
                self.dialog = pfd.save_file(title, start, filters)
            else:
                self.dialog = pfd.open_file(title, start, filters, pfd.opt.multiselect if self.multiple else pfd.opt.none)
            return
        self.backend = linux_backend()
        if self.backend is None:
            raise RuntimeError("no file dialog on this machine: no zenity or kdialog on PATH and no desktop portal")
        command = helper_command(self.backend, self.kind, title, start, filters, self.multiple)
        self.process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)

    def ready(self) -> bool:
        """Whether the dialog closed: the pick printed, or the helper's output ended without one."""
        if self.dialog is not None:
            return self.dialog.ready(0)
        fd = self.process.stdout.fileno()
        # kdialog prints several picks one per line, so only the end of its output closes those
        lines = self.backend == "kdialog" and self.multiple
        while not self.closed and len(select.select([fd], [], [], 0)[0]) > 0:
            chunk = os.read(fd, 65536)
            self.output += chunk
            self.closed = len(chunk) == 0 or (b"\n" in self.output and not lines)
        return self.closed

    def result(self):
        """The pick, "" when the dialog was cancelled; a list of picks when ``multiple``."""
        if self.dialog is not None:
            return self.dialog.result()
        text = os.fsdecode(self.output)
        if self.backend == PORTAL:
            picks = json.loads(text) if text.strip() else []
        elif self.backend == "kdialog":
            picks = [line for line in text.split("\n") if line]
        else:
            picks = [pick for pick in text.split("\n")[0].split(SEPARATOR_PICKS) if pick]
        if self.multiple:
            return picks
        return picks[0] if len(picks) > 0 else ""

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
    if prompt.dialog is not None and prompt.dialog.ready():
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
            prompt.dialog = Picker(prompt.title, start, prompt.filters, kind=prompt.kind, multiple=prompt.multiple)
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
