"""Label and keybind panels shared by the ROI viewers."""

from typing import Optional, Sequence

from imgui_bundle import imgui

from masknmf.visualization.imgui.labels import UNLABELED, UNLABEL_ALL
from masknmf.visualization.imgui.theme import (
    THEME,
    danger_button,
    em,
    label_button,
    label_color,
    to_vec4,
)


def draw_progress(label_set, id_suffix: str = "") -> bool:
    """Labeled n/total; True when "next unlabeled" was clicked."""
    done, total = label_set.progress()
    imgui.text_colored(
        to_vec4(THEME.ok if done == total else THEME.warn), f"labeled {done}/{total}"
    )
    if done >= total:
        return False
    imgui.same_line(0, em(0.8))
    clicked = imgui.button(f"next unlabeled##{id_suffix}")
    imgui.same_line(0, em(0.3))
    imgui.text_disabled("(u)")
    return clicked


def draw_label_buttons(label_set, id_suffix: str = "") -> Optional[int]:
    """
    One colored button per class, plus unlabel and unlabel all.

    Returns the clicked label index, ``UNLABELED`` to clear the current item,
    ``UNLABEL_ALL`` to clear every item, or None.
    """
    picked = None
    for i, name in enumerate(label_set.names):
        if i:
            imgui.same_line(0, em(0.6))
        with label_button(label_set.color(i)):
            if imgui.button(f"{name} ({label_set.count(i)})##label{i}{id_suffix}"):
                picked = i
        if i < 9:
            imgui.same_line(0, em(0.25))
            imgui.text_disabled(f"({i + 1})")
    if not label_set.names:
        imgui.text_disabled("no labels yet: add one")
        return None
    imgui.same_line(0, em(0.8))
    if imgui.button(f"unlabel##{id_suffix}"):
        picked = UNLABELED
    imgui.same_line(0, em(0.25))
    imgui.text_disabled("(0)")
    imgui.same_line(0, em(0.6))
    with danger_button():
        if imgui.button(f"unlabel all##{id_suffix}"):
            picked = UNLABEL_ALL
    return picked


def draw_label_buttons(label_set, id_suffix: str = "") -> Optional[int]:
    """
    The unlabel actions across the top, then one button per class in columns
    filled evenly down the space the card has, at least two wide.

    Returns the clicked label index, ``UNLABELED`` to clear the current item,
    ``UNLABEL_ALL`` to clear every item, or None.
    """
    if not label_set.names:
        imgui.text_disabled("no labels yet: add one")
        return None
    picked = _draw_unlabel_row(id_suffix)
    n = len(label_set.names)
    avail = imgui.get_content_region_avail()
    rows = max(int(avail.y // imgui.get_frame_height_with_spacing()), 1)
    ncols = max(2, -(-n // rows))
    per_col = -(-n // ncols)
    gap, hint = em(0.8), em(2.0)
    col_w = max((avail.x - gap * (ncols - 1)) / ncols, em(5))
    # a touch narrower than the column: the buttons carried more empty space
    # than the names needed, and the hint sits outside them
    size = imgui.ImVec2(max(col_w - hint - em(0.7), em(3.0)), 0)
    count_w = max(
        imgui.calc_text_size(_count_text(label_set, i)).x for i in range(n)
    ) + em(0.5)
    for start in range(0, n, per_col):
        if start:
            imgui.same_line(0, gap)
        imgui.begin_group()
        for i in range(start, min(start + per_col, n)):
            with label_button(label_set.color(i)):
                if imgui.button(f"##lab{i}{id_suffix}", size):
                    picked = i
            _draw_class_text(label_set, i, count_w)
            if i < 9:
                imgui.same_line(0, 4)
                imgui.text_disabled(f"({i + 1})")
        imgui.end_group()
    return picked


def draw_label_editor(label_set, new_label: str, id_suffix: str = "") -> tuple:
    """Add and remove class controls. Returns (new_label_text, changed)."""
    changed = False
    imgui.set_next_item_width(em(7))
    entered, new_label = imgui.input_text_with_hint(
        f"##new-label{id_suffix}", "new label", new_label,
        imgui.InputTextFlags_.enter_returns_true,
    )
    imgui.same_line(0, em(0.3))
    if (imgui.button(f"add##{id_suffix}") or entered) and new_label.strip():
        changed = label_set.add(new_label.strip())
        new_label = ""
    imgui.same_line(0, em(0.3))
    if imgui.button(f"del##{id_suffix}"):
        imgui.open_popup(f"##del-labels{id_suffix}")
    if imgui.begin_popup(f"##del-labels{id_suffix}"):
        if not label_set.names:
            imgui.text_disabled("no labels")
        remove = None
        for i, name in enumerate(label_set.names):
            if imgui.small_button(f"x##del{i}{id_suffix}"):
                remove = i
            imgui.same_line(0, em(0.5))
            imgui.text_colored(
                label_color(label_set.color(i)), f"{name} ({label_set.count(i)})"
            )
        if remove is not None:
            changed = label_set.remove(remove)
        imgui.end_popup()
    return new_label, changed


def draw_keybinds_popup(bindings: Sequence[tuple], is_open: bool, title: str = "Keybinds") -> bool:
    """Key reference window. Returns the new open state."""
    if not is_open:
        return False
    imgui.set_next_window_pos(
        imgui.get_main_viewport().get_center(), imgui.Cond_.appearing,
        pivot=imgui.ImVec2(0.5, 0.5),
    )
    opened, is_open = imgui.begin(
        f"{title}###keybinds", is_open,
        flags=imgui.WindowFlags_.no_saved_settings | imgui.WindowFlags_.always_auto_resize,
    )
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
    imgui.end()
    return is_open
