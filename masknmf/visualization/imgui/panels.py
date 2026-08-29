from typing import Optional, Sequence

import numpy as np
from imgui_bundle import imgui

from masknmf.visualization.imgui.labels import UNLABEL_ALL, UNLABELED

DONE_COLOR = imgui.ImVec4(0.35, 0.9, 0.35, 1.0)
TODO_COLOR = imgui.ImVec4(1.0, 0.75, 0.25, 1.0)
HINT_COLOR = imgui.ImVec4(1.0, 0.85, 0.4, 1.0)


def draw_progress(labels: np.ndarray, session_sizes: Optional[Sequence[int]] = None) -> bool:
    """Labeled n/total with a per-session split. Returns True if "next unlabeled" was clicked."""
    labeled = labels >= 0
    done, total = int(labeled.sum()), len(labeled)
    imgui.text_colored(DONE_COLOR if done == total else TODO_COLOR, f"labeled {done}/{total}")
    if session_sizes is not None and len(session_sizes) > 1:
        start = 0
        for k, n in enumerate(session_sizes):
            hit = int(labeled[start : start + n].sum())
            imgui.same_line(0, 10)
            imgui.text_colored(DONE_COLOR if hit == n else TODO_COLOR, f"s{k}: {hit}/{n}")
            start += n
    clicked = False
    if done < total:
        imgui.same_line(0, 12)
        clicked = imgui.button("next unlabeled")
        imgui.same_line(0, 4)
        imgui.text_disabled("(u)")
    return clicked


def draw_keybinds_popup(bindings: Sequence[tuple], is_open: bool, title: str = "Keybinds") -> bool:
    """Modal-ish key reference. Returns the new open state."""
    if not is_open:
        return False
    em = imgui.get_font_size()
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
            imgui.table_setup_column("key", imgui.TableColumnFlags_.width_fixed, 10 * em)
            imgui.table_setup_column("action")
            for key, action in bindings:
                imgui.table_next_row()
                imgui.table_next_column()
                imgui.text_colored(HINT_COLOR, key)
                imgui.table_next_column()
                imgui.text(action)
            imgui.end_table()
    imgui.end()
    return is_open


def draw_label_buttons(label_set, id_suffix: str = "") -> Optional[int]:
    """
    One coloured button per class, plus unlabel and unlabel all.

    Returns the clicked label index, ``UNLABELED`` (-1) to clear the current
    item, ``UNLABEL_ALL`` (-2) to clear every item, or None.
    """
    picked = None
    for i, name in enumerate(label_set.names):
        imgui.same_line(0, 10)
        imgui.push_style_color(imgui.Col_.button, imgui.ImVec4(*label_set.color(i), 0.5))
        if imgui.button(f"{name} ({label_set.count(i)})##label{i}{id_suffix}"):
            picked = i
        imgui.pop_style_color()
        if i < 9:
            imgui.same_line(0, 4)
            imgui.text_disabled(f"({i + 1})")
    if label_set.names:
        imgui.same_line(0, 10)
        if imgui.button(f"unlabel##{id_suffix}"):
            picked = UNLABELED
        imgui.same_line(0, 4)
        imgui.text_disabled("(0)")
        imgui.same_line(0, 10)
        imgui.push_style_color(imgui.Col_.button, imgui.ImVec4(0.75, 0.15, 0.15, 0.8))
        imgui.push_style_color(
            imgui.Col_.button_hovered, imgui.ImVec4(0.90, 0.20, 0.20, 1.0)
        )
        if imgui.button(f"unlabel all##{id_suffix}"):
            picked = UNLABEL_ALL
        imgui.pop_style_color(2)
    return picked


def draw_label_editor(label_set, new_label: str, id_suffix: str = "") -> tuple:
    """Add/remove class controls. Returns (new_label_text, changed)."""
    changed = False
    imgui.set_next_item_width(120)
    entered, new_label = imgui.input_text_with_hint(
        f"##new-label{id_suffix}", "new label", new_label,
        imgui.InputTextFlags_.enter_returns_true,
    )
    imgui.same_line(0, 5)
    if (imgui.button(f"add##{id_suffix}") or entered) and new_label.strip():
        changed = label_set.add(new_label.strip())
        new_label = ""
    imgui.same_line(0, 5)
    if imgui.button(f"del##{id_suffix}"):
        imgui.open_popup(f"##del-labels{id_suffix}")
    if imgui.begin_popup(f"##del-labels{id_suffix}"):
        if not label_set.names:
            imgui.text_disabled("no labels")
        remove = None
        for i, name in enumerate(label_set.names):
            if imgui.small_button(f"x##del{i}{id_suffix}"):
                remove = i
            imgui.same_line(0, 8)
            imgui.text_colored(
                imgui.ImVec4(*label_set.color(i), 1.0), f"{name} ({label_set.count(i)})"
            )
        if remove is not None:
            changed = label_set.remove(remove)
        imgui.end_popup()
    return new_label, changed
