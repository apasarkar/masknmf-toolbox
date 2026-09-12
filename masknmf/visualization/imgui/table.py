"""Sortable, filterable ROI table shared by the ROI viewers."""

from typing import Callable, Optional, Sequence

import numpy as np
from imgui_bundle import imgui


class RoiOrder:
    """
    Filter and stable sort over per-item columns; yields the visible item order.

    ``columns`` maps a name to one value per item. An integer range over
    ``range_column`` filters; ``pos`` is the cursor into ``order``.
    """

    def __init__(self, columns: dict, n_items: int):
        self.columns = columns
        self.n_items = n_items
        self.range_column: Optional[str] = None
        self.range_span = (0, 0)
        self.range_limits = (0, 0)
        self.sort_column = 0
        self.ascending = True
        self.order = np.arange(n_items)
        self.pos = 0

    def set_range_column(self, name: str):
        """Filter on ``name``, with the limits reset to its full span."""
        self.range_column = name
        self.range_span = (0, int(np.max(self.columns[name], initial=0)))
        self.range_limits = self.range_span

    @property
    def current(self) -> Optional[int]:
        if len(self.order) == 0:
            return None
        return int(self.order[self.pos])

    def rebuild(self):
        """Reapply the filter and the sort, keeping the cursor on its item."""
        current = self.current
        mask = np.ones(self.n_items, dtype=bool)
        if self.range_column is not None:
            values = self.columns[self.range_column]
            mask &= (values >= self.range_limits[0]) & (values <= self.range_limits[1])
        idx = np.flatnonzero(mask)
        # keys has no entry for column 0, the id, which is the natural order
        keys = list(self.columns.values())
        if 0 < self.sort_column <= len(keys):
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

    def hidden(self, item: int) -> bool:
        """Whether the range filter keeps ``item`` out of the current view."""
        if self.range_column is None:
            return False
        value = self.columns[self.range_column][item]
        return not self.range_limits[0] <= value <= self.range_limits[1]

    def reveal(self, item: int) -> bool:
        """Put ``item`` under the cursor, widening the range filter if it hides it. True when it did."""
        cleared = self.hidden(item)
        if cleared:
            self.set_range_column(self.range_column)
            self.rebuild()
        self.goto(item)
        return cleared


def draw_roi_table(
    order: RoiOrder,
    column_names: Sequence[str],
    formatters: dict,
    scroll_to_current: bool,
    table_id: str = "rois",
    on_select: Optional[Callable[[int], None]] = None,
    is_grouped: Optional[Callable[[int], bool]] = None,
    on_ctrl_select: Optional[Callable[[int], None]] = None,
    on_shift_select: Optional[Callable[[int], None]] = None,
    row_color: Optional[Callable[[int], Optional[tuple]]] = None,
) -> bool:
    """
    Sortable, clipped ROI table. Returns the new ``scroll_to_current`` flag.

    ``column_names[0]`` is the id column; every other name is rendered by
    ``formatters[name](item)`` and is sortable when it is a key of
    ``order.columns``. ``is_grouped`` highlights rows beyond the cursor; ctrl and
    shift clicks route to ``on_ctrl_select`` / ``on_shift_select`` when given,
    else to ``on_select``. ``row_color`` tints the id cell (rgb in 0-1).
    """
    flags = (
        imgui.TableFlags_.sortable
        | imgui.TableFlags_.row_bg
        | imgui.TableFlags_.resizable
        | imgui.TableFlags_.scroll_y
    )
    avail = imgui.get_content_region_avail()
    if not imgui.begin_table(table_id, len(column_names), flags, imgui.ImVec2(0, avail.y)):
        return scroll_to_current
    imgui.table_setup_scroll_freeze(0, 1)
    imgui.table_setup_column(column_names[0], imgui.TableColumnFlags_.default_sort)
    for name in column_names[1:]:
        sortable = name in order.columns
        imgui.table_setup_column(name, 0 if sortable else imgui.TableColumnFlags_.no_sort)
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
            highlighted = row == order.pos or (is_grouped is not None and is_grouped(item))
            rgb = row_color(item) if row_color is not None else None
            if rgb is not None:
                imgui.push_style_color(imgui.Col_.text, imgui.ImVec4(*rgb[:3], 1.0))
            clicked, _ = imgui.selectable(
                f"{item}##row{row}", highlighted, imgui.SelectableFlags_.span_all_columns
            )
            if rgb is not None:
                imgui.pop_style_color()
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
                imgui.text(formatters[name](item))
    imgui.end_table()
    return scroll_to_current


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
