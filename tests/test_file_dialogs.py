"""The native file dialog behind browse: which helper draws it, its command line, and how its picks are read."""

import ctypes
import ctypes.util
import sys
import time

import pytest

from masknmf.visualization.imgui import files, portal_dialog

FILTERS = ("Results", "*.hdf5 *.h5", "All files", "*")


def wait(picker: files.Picker) -> None:
    for _ in range(200):
        if picker.ready():
            return
        time.sleep(0.02)
    picker.kill()
    raise AssertionError("the helper never printed a pick")


def printing(text: str) -> list[str]:
    """A helper that prints text and exits, as a closed dialog does."""
    return [sys.executable, "-c", f"import sys; sys.stdout.write({text!r})"]


def test_the_portal_draws_the_dialog_when_no_helper_is_on_path(monkeypatch):
    monkeypatch.setenv("WAYLAND_DISPLAY", "wayland-1")
    monkeypatch.setattr(files.shutil, "which", lambda name: None)
    monkeypatch.setattr(portal_dialog, "portal_available", lambda: True)
    assert files.linux_backend() == files.PORTAL
    monkeypatch.setattr(portal_dialog, "portal_available", lambda: False)
    assert files.linux_backend() is None
    monkeypatch.setattr(files.shutil, "which", lambda name: "/usr/bin/zenity" if name == "zenity" else None)
    assert files.linux_backend() == "zenity"


def test_no_display_means_no_dialog(monkeypatch):
    monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
    monkeypatch.delenv("DISPLAY", raising=False)
    assert files.linux_backend() is None


def test_helper_commands_cover_open_save_and_folder(tmp_path):
    start = str(tmp_path)
    zenity = files.helper_command("zenity", "open", "t", start, FILTERS, True)
    assert zenity[:3] == ["zenity", "--file-selection", f"--filename={start}/"]
    assert "--file-filter=Results|*.hdf5 *.h5" in zenity and "--multiple" in zenity
    assert "--save" in files.helper_command("zenity", "save", "t", start + "/a.npz", FILTERS, False)
    assert "--directory" in files.helper_command("zenity", "folder", "t", start, [], False)
    kdialog = files.helper_command("kdialog", "save", "t", start, FILTERS, False)
    assert kdialog[:4] == ["kdialog", "--getsavefilename", start, "Results(*.hdf5 *.h5) | All files(*)"]
    portal = files.helper_command(files.PORTAL, "open", "t", start, FILTERS, True)
    assert portal[2:] == [portal_dialog.__file__, "open", "t", start, "--multiple", *FILTERS]


@pytest.mark.skipif(sys.platform in ("win32", "darwin"), reason="windows and macos use portable-file-dialogs")
@pytest.mark.parametrize(
    ("backend", "multiple", "printed", "expected"),
    [
        (files.PORTAL, True, '["/a b.h5", "/c.h5"]\n', ["/a b.h5", "/c.h5"]),
        (files.PORTAL, False, '["/a b.h5"]\n', "/a b.h5"),
        (files.PORTAL, False, "[]\n", ""),
        ("zenity", True, "/a b.h5\x1f/c.h5\n", ["/a b.h5", "/c.h5"]),
        ("zenity", False, "/a b.h5\n", "/a b.h5"),
        ("zenity", False, "", ""),
        ("kdialog", True, "/a b.h5\n/c.h5\n", ["/a b.h5", "/c.h5"]),
    ],
)
def test_picker_reads_each_helpers_picks(monkeypatch, backend, multiple, printed, expected):
    monkeypatch.setattr(files, "linux_backend", lambda: backend)
    monkeypatch.setattr(files, "helper_command", lambda *args: printing(printed))
    picker = files.Picker("t", "/", FILTERS, multiple=multiple)
    wait(picker)
    assert picker.result() == expected


@pytest.mark.skipif(ctypes.util.find_library("gio-2.0") is None, reason="needs libgio")
@pytest.mark.parametrize("kind", ["open", "save", "folder"])
def test_portal_request_is_a_valid_gvariant(tmp_path, kind):
    """A name with a quote and a backslash still parses as the (ssa{sv}) the portal takes."""
    gio = portal_dialog.load_gio()
    text = portal_dialog.request_text(kind, "it's a \\ title", str(tmp_path / "it's.npz"), FILTERS, True, "token")
    error = ctypes.POINTER(portal_dialog.GError)()
    parsed = gio.g_variant_parse(gio.g_variant_type_new(b"(ssa{sv})"), text.encode(), None, None, ctypes.byref(error))
    assert parsed and not error


@pytest.mark.skipif(ctypes.util.find_library("gio-2.0") is None, reason="needs libgio")
def test_portal_answer_becomes_paths():
    gio = portal_dialog.load_gio()
    request = portal_dialog.Request(gio, gio.g_main_loop_new(None, 0))
    answer = gio.g_variant_parse(None, b"(uint32 0, {'uris': <['file:///data/a%20b.h5', 'file:///c.h5']>})", None, None, None)
    request.on_response(None, None, None, None, None, answer, None)
    assert request.picks == ["/data/a b.h5", "/c.h5"]
    cancelled = gio.g_variant_parse(None, b"(uint32 1, @a{sv} {})", None, None, None)
    request.on_response(None, None, None, None, None, cancelled, None)
