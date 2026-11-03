"""
Copy and paste in the imgui windows through the system clipboard.

No rendercanvas backend (glfw, qt, jupyter, anywidget) hands imgui a clipboard. imgui's own then raises on Ctrl+V
until something was copied in the same window (imgui_bundle 1.92.801 builds a string from its null text), which
leaves the frame unfinished and every later frame failing. pyperclip reaches the operating system's clipboard
directly (wl-clipboard or xclip on Linux, pbcopy on macOS, the win32 API on Windows), whatever the canvas. In a
notebook that is the clipboard of the machine running the kernel, which is the browser's only when both are the same
machine.
"""

import warnings

import pyperclip
from imgui_bundle import imgui


def paste_text(context: imgui.internal.Context) -> str:
    """Returns the system clipboard's text; imgui calls this on Ctrl+V."""
    return pyperclip.paste()


def copy_text(context: imgui.internal.Context, text: str) -> None:
    """Puts the text imgui copied on the system clipboard."""
    pyperclip.copy(text)


def paste_nothing(context: imgui.internal.Context) -> str:
    """Returns no text, for a machine without a system clipboard."""
    return ""


def copy_nothing(context: imgui.internal.Context, text: str) -> None:
    """Drops the text imgui copied, for a machine without a system clipboard."""


def connect_clipboard(context: imgui.internal.Context) -> None:
    """
    Routes one imgui context's copy and paste through the system clipboard.

    Without a copy/paste tool (wl-clipboard or xclip on Linux) it warns and turns copy and paste off, so Ctrl+V pastes
    nothing instead of raising.
    """
    copy, _ = pyperclip.determine_clipboard()
    if not copy:
        warnings.warn("No system clipboard found (install wl-clipboard or xclip); copy and paste are off in this window")
        context.platform_io.platform_get_clipboard_text_fn = paste_nothing
        context.platform_io.platform_set_clipboard_text_fn = copy_nothing
        return
    context.platform_io.platform_get_clipboard_text_fn = paste_text
    context.platform_io.platform_set_clipboard_text_fn = copy_text
