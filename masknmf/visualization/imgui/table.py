"""Sortable, filterable ROI table shared by the ROI viewers."""

from typing import Callable, NamedTuple, Optional, Sequence

import numpy as np
from imgui_bundle import imgui

from masknmf.visualization.imgui.labels import UNLABELED
from masknmf.visualization.imgui.theme import label_color

FILTER_ALL = -2


class RowAction(NamedTuple):
    """
    One icon button in the table's trailing actions column.

    Args:
        icon (str): glyph drawn on the button; the name belongs in ``tooltip``
        tooltip (str): hover text
        on_click (Callable[[int], None]): called with the item index on press
        disabled (Optional[Callable[[int], Optional[str]]]): why the action cannot
            run for that item; greys the button and replaces the tooltip. None
            means available.
    """

    icon: str
    tooltip: str
    on_click: Callable[[int], None]
    disabled: Optional[Callable[[int], Optional[str]]] = None


class RoiOrder:
    """
    Filter and stable sort over per-item columns; yields the visible item order.

    Three filters compose: the class label, an integer range over one column
    (``range_column``), and an exact value of one categorical column
    (``category_column``). ``pos`` is the cursor into ``order``.
    """

    def __init__(self, columns: dict, labels: np.ndarray, n_items: int):
        self.columns = columns
        self.labels = labels
        self.n_items = n_items
        self.filter_label = FILTER_ALL
        self.range_column: Optional[str] = None
        self.range_span = (0, 0)
        self.range_limits = (0, 0)
        self.category_column: Optional[str] = None
        self.category: Optional[int] = None
        self.sort_column = 0
        self.ascending = True
        self.order = np.arange(n_items)
        self.pos = 0

    def set_range_column(self, name: str):
        """Filter on ``name``, with the limits reset to its full span."""
        self.range_column = name
        values = self.columns[name]
        self.range_span = (0, int(np.max(values, initial=0)))
        self.range_limits = self.range_span

    def refresh_range(self):
        """Widen untouched range limits to the column's new span."""
        if self.range_column is not None and self.range_limits == self.range_span:
            self.set_range_column(self.range_column)

    @property
    def current(self) -> Optional[int]:
        if len(self.order) == 0:
            return None
        return int(self.order[self.pos])

    def rebuild(self):
        """Reapply the filters and the sort, keeping the cursor on its item."""
        current = self.current
        mask = np.ones(self.n_items, dtype=bool)
        if self.range_column is not None:
            values = self.columns[self.range_column]
            mask &= (values >= self.range_limits[0]) & (values <= self.range_limits[1])
        if self.category_column is not None and self.category is not None:
            mask &= self.columns[self.category_column] == self.category
        if self.filter_label >= UNLABELED:
            mask &= self.labels == self.filter_label
        idx = np.flatnonzero(mask)
        if self.sort_column:
            # keys has no entry for column 0, the id, which is the natural order
            keys = [self.labels, *self.columns.values()]
            idx = idx[np.argsort(keys[self.sort_column - 1][idx], kind="stable")]
        if not self.ascending:
            idx = idx[::-1]
        self.order = idx
        hits = np.flatnonzero(self.order == current) if current is not None else ()
        self.pos = int(hits[0]) if len(hits) else int(min(self.pos, max(len(idx) - 1, 0)))

    def step(self, delta: int) -> bool:
        if not len(self.order):
            return False
        self.pos = int(np.clip(self.pos + delta, 0, len(self.order) - 1))
        return True

    def goto(self, item: int) -> bool:
        hits = np.flatnonzero(self.order == item)
        if not len(hits):
            return False
        self.pos = int(hits[0])
        return True

    def hidden_by(self, item: int) -> list:
        """Names of the filters that keep ``item`` out of the current view."""
        out = []
        if self.filter_label >= UNLABELED and int(self.labels[item]) != self.filter_label:
            out.append("label")
        if self.category_column is not None and self.category is not None:
            if int(self.columns[self.category_column][item]) != self.category:
                out.append(self.category_column)
        if self.range_column is not None:
            value = self.columns[self.range_column][item]
            if not self.range_limits[0] <= value <= self.range_limits[1]:
                out.append(self.range_column)
        return out

    def clear_filter(self, name: str):
        if name == "label":
            self.filter_label = FILTER_ALL
        elif name == self.category_column:
            self.category = None
        elif name == self.range_column:
            self.set_range_column(self.range_column)

    def reveal(self, item: int) -> list:
        """
        Put ``item`` under the cursor, dropping whatever filters hide it.

        Returns the filters cleared, so a caller that reveals an item the user had
        filtered away can say which ones it undid.
        """
        cleared = self.hidden_by(item)
        for name in cleared:
            self.clear_filter(name)
        if cleared:
            self.rebuild()
        self.goto(item)
        return cleared

    def next_unlabeled(self) -> bool:
        """First unlabeled item after the cursor, wrapping."""
        hits = np.flatnonzero(self.labels[self.order] < 0)
        if not len(hits):
            return False
        after = hits[hits > self.pos]
        self.pos = int(after[0] if len(after) else hits[0])
        return True


def draw_roi_table(
    order: RoiOrder,
    label_set,
    column_names: Sequence[str],
    formatters: dict,
    scroll_to_current: bool,
    table_id: str = "rois",
    on_select: Optional[Callable[[int], None]] = None,
    actions: Sequence[RowAction] = (),
    is_grouped: Optional[Callable[[int], bool]] = None,
    on_ctrl_select: Optional[Callable[[int], None]] = None,
    on_shift_select: Optional[Callable[[int], None]] = None,
) -> bool:
    """
    Sortable, clipped ROI table. Returns the new ``scroll_to_current`` flag.

    ``column_names[0]`` is the id column and "label" draws in its class color;
    every other name is rendered by ``formatters[name](item)``. ``actions`` adds
    a trailing, unsortable column of icon buttons that sit over the row's
    selectable, so clicking one does not change the selection. ``is_grouped``
    highlights rows beyond the cursor; ctrl and shift clicks route to
    ``on_ctrl_select`` / ``on_shift_select`` when given, else to ``on_select``.
    """
    flags = (
        imgui.TableFlags_.sortable
        | imgui.TableFlags_.row_bg
        | imgui.TableFlags_.resizable
        | imgui.TableFlags_.scroll_y
    )
    avail = imgui.get_content_region_avail()
    n_columns = len(column_names) + bool(actions)
    if not imgui.begin_table(table_id, n_columns, flags, imgui.ImVec2(0, avail.y)):
        return scroll_to_current
    imgui.table_setup_scroll_freeze(0, 1)
    imgui.table_setup_column(column_names[0], imgui.TableColumnFlags_.default_sort)
    for name in column_names[1:]:
        imgui.table_setup_column(name)
    if actions:
        imgui.table_setup_column(
            "##actions",
            imgui.TableColumnFlags_.no_sort | imgui.TableColumnFlags_.width_fixed,
            len(actions) * imgui.get_font_size() * 2.0,
        )
    imgui.table_headers_row()

    specs = imgui.table_get_sort_specs()
    if specs is not None and specs.specs_dirty:
        if specs.specs_count > 0:
            order.sort_column = int(specs.specs.column_index)
            order.ascending = specs.specs.sort_direction == imgui.SortDirection.ascending
        specs.specs_dirty = False
        order.rebuild()

    clipper = imgui.ListClipper()
    clipper.begin(len(order.order))
    if scroll_to_current:
        clipper.include_item_by_index(order.pos)
    while clipper.step():
        for row in range(clipper.display_start, clipper.display_end):
            item = int(order.order[row])
            imgui.table_next_row()
            imgui.table_next_column()
            sel_flags = imgui.SelectableFlags_.span_all_columns
            if actions:
                sel_flags |= imgui.SelectableFlags_.allow_overlap
            highlighted = row == order.pos or (is_grouped is not None and is_grouped(item))
            clicked, _ = imgui.selectable(f"{item}##row{row}", highlighted, sel_flags)
            if clicked:
                io = imgui.get_io()
                if io.key_ctrl and on_ctrl_select is not None:
                    on_ctrl_select(item)
                elif io.key_shift and on_shift_select is not None:
                    on_shift_select(item)
                else:
                    order.pos = row
                    if on_select is not None:
                        on_select(item)
            if row == order.pos and scroll_to_current:
                imgui.set_scroll_here_y(0.5)
                scroll_to_current = False
            for name in column_names[1:]:
                imgui.table_next_column()
                if name == "label":
                    label = int(label_set.labels[item])
                    if label >= 0:
                        imgui.text_colored(
                            label_color(label_set.color(label)), label_set.names[label]
                        )
                    else:
                        imgui.text("-")
                else:
                    imgui.text(formatters[name](item))
            if actions:
                imgui.table_next_column()
                _draw_row_actions(actions, item, row)
    imgui.end_table()
    return scroll_to_current


def _draw_row_actions(actions: Sequence[RowAction], item: int, row: int):
    """Icon buttons for one row; ``row`` only keeps the imgui ids unique."""
    for k, action in enumerate(actions):
        if k:
            imgui.same_line(0, 2)
        reason = action.disabled(item) if action.disabled is not None else None
        if reason is not None:
            imgui.begin_disabled()
        if imgui.small_button(f"{action.icon}##act{k}_{row}"):
            action.on_click(item)
        if reason is not None:
            imgui.end_disabled()
        if imgui.is_item_hovered(imgui.HoveredFlags_.allow_when_disabled):
            imgui.set_tooltip(reason or action.tooltip)


def draw_label_filter(order: RoiOrder, label_set, id_suffix: str = "", width: float = -1) -> bool:
    """Label filter combo. Does not rebuild; True when the selection changed."""
    names = ("all", "unlabeled", *label_set.names)
    imgui.set_next_item_width(width)
    changed, sel = imgui.combo(f"##filter{id_suffix}", order.filter_label + 2, list(names))
    if changed:
        order.filter_label = sel - 2
    return changed


def draw_category_filter(
    order: RoiOrder, names: Sequence[str], id_suffix: str = "", width: float = -1
) -> bool:
    """
    Combo over the values of ``order.category_column``, with "all" first.

    ``names[i]`` labels category value ``i``. Does not rebuild; True when the
    selection changed.
    """
    if order.category_column is None:
        return False
    current = 0 if order.category is None else order.category + 1
    imgui.set_next_item_width(width)
    changed, sel = imgui.combo(
        f"##category{id_suffix}", min(current, len(names)), ["all", *names]
    )
    if changed:
        order.category = None if sel == 0 else sel - 1
    return changed


def draw_range_filter(order: RoiOrder, id_suffix: str = "", width: float = -1) -> bool:
    """
    Range slider for ``order.range_column``, or nothing when no column is set.

    Does not rebuild; True when the limits changed.
    """
    if order.range_column is None:
        return False
    values = order.columns[order.range_column]
    imgui.set_next_item_width(width)
    changed, lo, hi = imgui.drag_int_range2(
        f"##range{id_suffix}",
        order.range_limits[0], order.range_limits[1], 1, 0,
        int(np.max(values, initial=0)),
        f"{order.range_column} >= %d", f"{order.range_column} <= %d",
    )
    if changed:
        order.range_limits = (lo, hi)
    return changed
