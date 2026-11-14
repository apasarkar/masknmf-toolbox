"""
A file dialog from xdg-desktop-portal, the desktop's own (GTK, KDE, GNOME), for a linux box without zenity or kdialog:
Omarchy and other Hyprland or sway setups, flatpak and snap sandboxes.

Run as a script, it opens one dialog and prints the picks as a JSON list on one line, empty when cancelled:

    python portal_dialog.py open|save|folder TITLE START [--multiple] [NAME PATTERNS]...

The portal answers on the D-Bus connection that asked and closes the dialog when that connection goes, so a one-shot
``gdbus call`` or ``busctl call`` cannot hold one open. This talks to it through libgio with ctypes instead, which
every machine running the portal has, and needs no D-Bus package. It imports the standard library only, so
``files.Picker`` runs it without importing masknmf.

https://flatpak.github.io/xdg-desktop-portal/docs/doc-org.freedesktop.portal.FileChooser.html
"""

import ctypes
import ctypes.util
import json
import os
import sys
from pathlib import Path
from typing import Sequence
from urllib.parse import unquote, urlparse

PORTAL = "org.freedesktop.portal.Desktop"
PORTAL_PATH = "/org/freedesktop/portal/desktop"
SERVICE_FILE = Path("dbus-1") / "services" / f"{PORTAL}.service"
G_BUS_TYPE_SESSION = 2
RESPONSE_PICKED = 0

SIGNAL_CALLBACK = ctypes.CFUNCTYPE(
    None,
    ctypes.c_void_p,
    ctypes.c_char_p,
    ctypes.c_char_p,
    ctypes.c_char_p,
    ctypes.c_char_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
)


class GError(ctypes.Structure):
    _fields_ = [("domain", ctypes.c_uint32), ("code", ctypes.c_int), ("message", ctypes.c_char_p)]


def portal_available() -> bool:
    """Whether a session bus, libgio and the portal's service file are there; nothing is started to find out."""
    if not os.environ.get("DBUS_SESSION_BUS_ADDRESS") or ctypes.util.find_library("gio-2.0") is None:
        return False
    dirs_data = [os.environ.get("XDG_DATA_HOME") or str(Path.home() / ".local" / "share")]
    dirs_data += (os.environ.get("XDG_DATA_DIRS") or "/usr/local/share:/usr/share").split(":")
    return any((Path(d) / SERVICE_FILE).is_file() for d in dirs_data if d)


def quoted(text: str) -> str:
    """text as a string in GVariant's text format."""
    return "'" + text.replace("\\", "\\\\").replace("'", "\\'").replace("\n", "\\n") + "'"


def path_bytes(path: str) -> str:
    """path as the zero-terminated byte array the portal takes for a folder, in GVariant's text format."""
    return "[byte " + ", ".join(f"0x{b:02x}" for b in os.fsencode(path) + b"\0") + "]"


def request_text(kind: str, title: str, start: str, filters: Sequence[str], multiple: bool, token: str) -> str:
    """
    The arguments of OpenFile or SaveFile, ``(parent_window, title, options)``, in GVariant's text format.

    Parameters
    ----------
    kind : str
        "open", "save" or "folder".
    title : str
        The dialog's title.
    start : str
        The folder the dialog opens in, or a file in it; a file's name is the suggested name of a save.
    filters : Sequence[str]
        Name / pattern pairs, patterns separated by spaces: ``("Results", "*.hdf5 *.h5")``.
    multiple : bool
        Whether an open dialog takes several files.
    token : str
        Names the request, so its answer can be subscribed to before it is made.
    """
    folder = start if os.path.isdir(start) else os.path.dirname(start)
    options = {"handle_token": f"<{quoted(token)}>"}
    if os.path.isdir(folder):
        options["current_folder"] = f"<{path_bytes(folder)}>"
    if kind == "folder":
        options["directory"] = "<true>"
    if kind == "open" and multiple:
        options["multiple"] = "<true>"
    if kind == "save" and not os.path.isdir(start) and os.path.basename(start):
        options["current_name"] = f"<{quoted(os.path.basename(start))}>"
    pairs = list(zip(filters[::2], filters[1::2]))
    if kind != "folder" and len(pairs) > 0:
        # a filter is (name, [(0, glob), ...])
        entries = [
            f"({quoted(name)}, [{', '.join(f'(uint32 0, {quoted(glob)})' for glob in patterns.split())}])"
            for name, patterns in pairs
        ]
        options["filters"] = f"<[{', '.join(entries)}]>"
    body = ", ".join(f"{quoted(key)}: {value}" for key, value in options.items())
    return f"('', {quoted(title)}, {{{body}}})"


def load_gio() -> ctypes.CDLL:
    """libgio with the signatures used here."""
    gio = ctypes.CDLL(ctypes.util.find_library("gio-2.0"))
    pointer = ctypes.c_void_p
    signatures = {
        "g_bus_get_sync": (pointer, [ctypes.c_int, pointer, pointer]),
        "g_dbus_connection_get_unique_name": (ctypes.c_char_p, [pointer]),
        "g_dbus_connection_signal_subscribe": (
            ctypes.c_uint,
            [pointer, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int, SIGNAL_CALLBACK, pointer, pointer],
        ),
        "g_dbus_connection_call_sync": (
            pointer,
            [pointer, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p, pointer, pointer, ctypes.c_int, ctypes.c_int, pointer, pointer],
        ),
        "g_variant_type_new": (pointer, [ctypes.c_char_p]),
        "g_variant_parse": (pointer, [pointer, ctypes.c_char_p, pointer, pointer, pointer]),
        "g_variant_get_child_value": (pointer, [pointer, ctypes.c_size_t]),
        "g_variant_get_uint32": (ctypes.c_uint32, [pointer]),
        "g_variant_lookup_value": (pointer, [pointer, ctypes.c_char_p, pointer]),
        "g_variant_get_strv": (ctypes.POINTER(ctypes.c_char_p), [pointer, ctypes.POINTER(ctypes.c_size_t)]),
        "g_main_loop_new": (pointer, [pointer, ctypes.c_int]),
        "g_main_loop_run": (None, [pointer]),
        "g_main_loop_quit": (None, [pointer]),
    }
    for name, (restype, argtypes) in signatures.items():
        function = getattr(gio, name)
        function.restype, function.argtypes = restype, argtypes
    return gio


def raise_gerror(error: ctypes.POINTER(GError), doing: str) -> None:
    if error:
        raise RuntimeError(f"{doing}: {error.contents.message.decode(errors='replace')}")


class Request:
    """One portal request: collects the picks of its Response signal and stops the main loop."""

    def __init__(self, gio: ctypes.CDLL, loop: int):
        self.gio = gio
        self.loop = loop
        self.picks: list[str] = []
        self.callback = SIGNAL_CALLBACK(self.on_response)  # held here: ctypes frees a callback nothing refers to

    def on_response(self, connection, sender, path, interface, signal, parameters, data) -> None:
        gio = self.gio
        response = gio.g_variant_get_uint32(gio.g_variant_get_child_value(parameters, 0))
        uris = gio.g_variant_lookup_value(gio.g_variant_get_child_value(parameters, 1), b"uris", None)
        if response == RESPONSE_PICKED and uris:
            count = ctypes.c_size_t()
            values = gio.g_variant_get_strv(uris, ctypes.byref(count))
            self.picks = [unquote(urlparse(values[i].decode()).path) for i in range(count.value)]
        gio.g_main_loop_quit(self.loop)


def pick(kind: str, title: str, start: str, filters: Sequence[str] = (), multiple: bool = False) -> list[str]:
    """Open the portal's dialog and wait for it; the picked paths, empty when cancelled. Arguments as request_text."""
    gio = load_gio()
    error = ctypes.POINTER(GError)()
    connection = gio.g_bus_get_sync(G_BUS_TYPE_SESSION, None, ctypes.byref(error))
    raise_gerror(error, "connecting to the session bus")
    request = Request(gio, gio.g_main_loop_new(None, 0))
    # the portal answers at a path made of the caller's bus name and the token: subscribed to before the call,
    # so an answer cannot slip in between
    token = f"masknmf_{os.getpid()}"
    sender = gio.g_dbus_connection_get_unique_name(connection).decode().lstrip(":").replace(".", "_")
    handle = f"{PORTAL_PATH}/request/{sender}/{token}"
    gio.g_dbus_connection_signal_subscribe(
        connection, PORTAL.encode(), b"org.freedesktop.portal.Request", b"Response", handle.encode(), None, 0, request.callback, None, None
    )
    text = request_text(kind, title, start, filters, multiple, token)
    parameters = gio.g_variant_parse(gio.g_variant_type_new(b"(ssa{sv})"), text.encode(), None, None, ctypes.byref(error))
    raise_gerror(error, "building the request")
    method = b"SaveFile" if kind == "save" else b"OpenFile"
    gio.g_dbus_connection_call_sync(
        connection, PORTAL.encode(), PORTAL_PATH.encode(), b"org.freedesktop.portal.FileChooser", method, parameters, None, 0, -1, None, ctypes.byref(error)
    )
    raise_gerror(error, "asking the portal for a file dialog")
    gio.g_main_loop_run(request.loop)
    return request.picks


if __name__ == "__main__":
    arguments = [a for a in sys.argv[1:] if a != "--multiple"]
    picks = pick(arguments[0], arguments[1], arguments[2], arguments[3:], multiple="--multiple" in sys.argv[1:])
    print(json.dumps(picks), flush=True)
