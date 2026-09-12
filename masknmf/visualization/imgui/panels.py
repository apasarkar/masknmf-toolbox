"""Small imgui panels shared by the viewers."""

from typing import Sequence

from imgui_bundle import imgui

from masknmf.visualization.imgui.theme import THEME, close_button, em, popup, to_vec4


def draw_keybinds_popup(bindings: Sequence[tuple], is_open: bool, title: str = "Keybinds") -> bool:
    """Key reference window built from ``(key, action)`` pairs. Returns the new open state."""
    if not is_open:
        return False
    opened, is_open = popup(title, is_open)
    if opened:
        flags = imgui.TableFlags_.row_bg | imgui.TableFlags_.borders_inner_h
        if imgui.begin_table("##keybinds-table", 2, flags):
            imgui.table_setup_column("key", imgui.TableColumnFlags_.width_fixed, em(10))
            imgui.table_setup_column("action")
            for key, action in bindings:
                imgui.table_next_row()
                imgui.table_next_column()
                imgui.text_colored(to_vec4(THEME.warn), key)
                imgui.table_next_column()
                imgui.text(action)
            imgui.end_table()
        if close_button():
            is_open = False
    imgui.end()
    return is_open
