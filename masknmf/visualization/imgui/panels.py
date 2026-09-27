"""Small imgui panels shared by the viewers."""

from typing import Mapping

from imgui_bundle import imgui

from masknmf.visualization.imgui.keybinds import Bind
from masknmf.visualization.imgui.theme import THEME, close_button, em, help_mark, popup, to_vec4


def draw_keybinds_popup(bindings: Mapping[str, Bind], is_open: bool, title: str = "Keybinds") -> bool:
    """Key reference window listing a keybinds table's rows. Returns the new open state."""
    if not is_open:
        return False
    opened, is_open = popup(title, is_open)
    if opened:
        flags = imgui.TableFlags_.row_bg | imgui.TableFlags_.borders_inner_h
        if imgui.begin_table("##keybinds-table", 2, flags):
            imgui.table_setup_column("key", imgui.TableColumnFlags_.width_fixed, em(10))
            imgui.table_setup_column("action", imgui.TableColumnFlags_.width_fixed, em(36))
            for bind in bindings.values():
                imgui.table_next_row()
                imgui.table_next_column()
                imgui.text_colored(to_vec4(THEME.warn), bind.label)
                imgui.table_next_column()
                # a fixed wrap position: a stretch column is not sized on the frame an auto-resize window first draws it
                imgui.push_text_wrap_pos(imgui.get_cursor_pos_x() + em(35.5))
                imgui.text(bind.action)
                imgui.pop_text_wrap_pos()
            imgui.end_table()
        if close_button():
            is_open = False
    imgui.end()
    return is_open


def draw_keybinds_button(is_open: bool) -> bool:
    """The keybinds button, k's twin; it toggles the popup the caller draws. Returns the popup's new open state."""
    if imgui.button("keybinds"):
        is_open = not is_open
    if imgui.is_item_hovered():
        imgui.set_tooltip("k")
    return is_open


def draw_help_buttons(help_open: bool, keys_open: bool) -> tuple[bool, bool]:
    """The help and keybinds buttons, h's and k's twins, on the current line. Returns their new open states."""
    if imgui.button("help"):
        help_open = not help_open
    if imgui.is_item_hovered():
        imgui.set_tooltip("h")
    imgui.same_line(0, em(0.4))
    return help_open, draw_keybinds_button(keys_open)


def draw_path_popup(
    title: str, is_open: bool, path: str, hint: str, action: str, browse=None, note: str = ""
) -> tuple[bool, str, bool]:
    """
    A window with a path field, a "browse" button when ``browse`` is given (it starts a native dialog
    whose pick the caller writes back into ``path``) and an ``action`` button; Enter in the field counts
    as the action. ``note`` shows under the field. Returns (still open, path, action pressed).
    """
    if not is_open:
        return False, path, False
    opened, is_open = popup(title, is_open)
    confirmed = False
    if opened:
        imgui.set_next_item_width(em(26))
        entered, path = imgui.input_text_with_hint(
            "##path", hint, path, imgui.InputTextFlags_.enter_returns_true
        )
        if browse is not None:
            imgui.same_line(0, em(0.4))
            if imgui.button("browse"):
                browse()
            help_mark("native file dialog on the machine running python; on a remote kernel type the path")
        if note:
            imgui.push_text_wrap_pos(em(32))
            imgui.text_disabled(note)
            imgui.pop_text_wrap_pos()
        imgui.dummy(imgui.ImVec2(0, em(0.3)))
        if imgui.button(action, imgui.ImVec2(em(6), 0)) or entered:
            confirmed = bool(path.strip())
        imgui.same_line(0, em(0.6))
        if imgui.button("Close", imgui.ImVec2(em(6), 0)):
            is_open = False
    imgui.end()
    return is_open, path, confirmed
