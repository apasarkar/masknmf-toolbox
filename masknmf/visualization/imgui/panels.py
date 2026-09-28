"""Small imgui panels shared by the viewers."""

from typing import Mapping

from imgui_bundle import imgui

from masknmf.visualization.imgui.keybinds import Bind
from masknmf.visualization.imgui.theme import THEME, close_button, em, popup, to_vec4


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


def hint_button(name: str, hint: str) -> bool:
    """A button reading ``name`` with ``hint`` dimmed after it, e.g. its key in parentheses."""
    style = imgui.get_style()
    imgui.push_style_var(imgui.StyleVar_.frame_rounding, THEME.rounding)
    clicked = imgui.button(f"##{name} {hint}", imgui.ImVec2(hint_button_width(name, hint), 0))
    imgui.pop_style_var()
    corner = imgui.get_item_rect_min()
    x, y = corner.x + style.frame_padding.x, corner.y + style.frame_padding.y
    draw = imgui.get_window_draw_list()
    draw.add_text(imgui.ImVec2(x, y), imgui.get_color_u32(imgui.Col_.text), name)
    draw.add_text(
        imgui.ImVec2(x + imgui.calc_text_size(f"{name} ").x, y), imgui.get_color_u32(imgui.Col_.text_disabled), hint
    )
    return clicked


def hint_button_width(name: str, hint: str) -> float:
    return imgui.calc_text_size(f"{name} {hint}").x + 2 * imgui.get_style().frame_padding.x


def draw_keybinds_button(is_open: bool, right: float | None = None) -> bool:
    """
    The keybinds button, k's twin; it toggles the popup the caller draws. With ``right`` it joins the current
    line with its right edge at that window x. Returns the popup's new open state.
    """
    if right is not None:
        imgui.same_line(right - hint_button_width("Keybinds", "(k)"))
    if hint_button("Keybinds", "(k)"):
        is_open = not is_open
    return is_open


def help_buttons_width(guide: str) -> float:
    """The width :func:`draw_help_buttons` takes for ``guide``."""
    return hint_button_width(guide, "(h)") + em(0.4) + hint_button_width("Keybinds", "(k)")


def draw_help_buttons(help_open: bool, keys_open: bool, guide: str) -> tuple[bool, bool]:
    """The ``guide`` and keybinds buttons, h's and k's twins, from the cursor. Returns their new open states."""
    if hint_button(guide, "(h)"):
        help_open = not help_open
    imgui.same_line(0, em(0.4))
    return help_open, draw_keybinds_button(keys_open)
