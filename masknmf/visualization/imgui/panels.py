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
    set_tooltip,
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


def _count_text(label_set, i: int) -> str:
    """
    The count column of a class button.

    "n=3", not "(3)": the (1-9) that follows the button is the keybind, and a
    bare count beside it read as one too.
    """
    return f"n={label_set.count(i)}"


def _draw_class_text(label_set, i: int, count_w: float) -> None:
    """
    Paint one class button's count and name as two left-aligned columns.
    """
    lo, hi = imgui.get_item_rect_min(), imgui.get_item_rect_max()
    pad = imgui.get_style().frame_padding.x
    y = lo.y + (hi.y - lo.y - imgui.get_text_line_height()) * 0.5
    draw = imgui.get_window_draw_list()
    color = imgui.get_color_u32(imgui.Col_.text)

    # a long name is clipped to its own button instead of bleeding into the
    # column beside it
    draw.push_clip_rect(imgui.ImVec2(lo.x, lo.y), imgui.ImVec2(hi.x - 1, hi.y), True)

    draw.add_text(imgui.ImVec2(lo.x + pad, y), color, _count_text(label_set, i))
    draw.add_text(imgui.ImVec2(lo.x + pad + count_w, y), color, label_set.names[i])
    draw.pop_clip_rect()


def _draw_unlabel_row(size, gap: float, id_suffix: str) -> Optional[int]:
    """
    The two clear actions, on the same grid as the class buttons under them so
    the card reads as one set of columns rather than a toolbar over a grid.
    """
    picked = None
    if imgui.button(f"unlabel##{id_suffix}", size):
        picked = UNLABELED
    set_tooltip("clear the selected ROI's label")
    imgui.same_line(0, 4)
    imgui.text_disabled("(0)")
    imgui.same_line(0, gap)
    with danger_button():
        if imgui.button(f"unlabel all##{id_suffix}", size):
            picked = UNLABEL_ALL
    set_tooltip("clear every label")
    imgui.dummy(imgui.ImVec2(0, em(0.25)))
    return picked


def draw_label_buttons(label_set, id_suffix: str = "") -> Optional[int]:
    """
    The two clear actions, then one button per class, in columns filled evenly
    down the space the card has and at least two wide.

    Returns the clicked label index, ``UNLABELED`` to clear the current item,
    ``UNLABEL_ALL`` to clear every item, or None.
    """
    if not label_set.names:
        imgui.text_disabled("no labels yet: add one")
        return None
    n = len(label_set.names)
    avail = imgui.get_content_region_avail()
    # one row goes to the clear actions, the rest to the classes
    rows = max(int(avail.y // imgui.get_frame_height_with_spacing()) - 1, 1)
    ncols = max(2, -(-n // rows))
    per_col = -(-n // ncols)
    gap, hint = em(0.8), em(2.0)
    col_w = max((avail.x - gap * (ncols - 1)) / ncols, em(5))
    # a touch narrower than the column: the buttons carried more empty space
    # than the names needed, and the keybind hint sits outside them
    size = imgui.ImVec2(max(col_w - hint - em(0.7), em(3.0)), 0)
    picked = _draw_unlabel_row(size, gap, id_suffix)
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
        f"##new-label{id_suffix}",
        "new label",
        new_label,
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


def draw_keybinds_popup(
    bindings: Sequence[tuple], is_open: bool, title: str = "Keybinds"
) -> bool:
    """Key reference window. Returns the new open state."""
    if not is_open:
        return False
    imgui.set_next_window_pos(
        imgui.get_main_viewport().get_center(),
        imgui.Cond_.appearing,
        pivot=imgui.ImVec2(0.5, 0.5),
    )
    opened, is_open = imgui.begin(
        f"{title}###keybinds",
        is_open,
        flags=imgui.WindowFlags_.no_saved_settings
        | imgui.WindowFlags_.always_auto_resize,
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
