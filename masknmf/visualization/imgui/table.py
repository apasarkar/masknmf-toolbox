from typing import Callable, Optional, Sequence

import numpy as np
from imgui_bundle import imgui

from masknmf.visualization.imgui.labels import UNLABELED

FILTER_ALL = -2


class RoiOrder:
    """Filter + stable sort over per-ROI columns; yields the visible index order."""

    def __init__(self, columns: dict, labels: np.ndarray, n_items: int):
        self.columns = columns
        self.labels = labels
        self.n_items = n_items
        self.filter_label = FILTER_ALL
        self.range_column: Optional[str] = None
        self.range_limits = (0, 0)
        self.sort_column = 0
        self.ascending = True
        self.order = np.arange(n_items)
        self.pos = 0

    def set_range_column(self, name: str):
        self.range_column = name
        values = self.columns[name]
        self.range_limits = (0, int(np.max(values, initial=0)))

    @property
    def current(self) -> Optional[int]:
        if len(self.order) == 0:
            return None
        return int(self.order[self.pos])

    def rebuild(self):
        current = self.current
        mask = np.ones(self.n_items, dtype=bool)
        if self.range_column is not None:
            values = self.columns[self.range_column]
            mask &= (values >= self.range_limits[0]) & (values <= self.range_limits[1])
        if self.filter_label >= UNLABELED:
            mask &= self.labels == self.filter_label
        idx = np.flatnonzero(mask)
        if self.sort_column:
            keys = [self.labels, *self.columns.values()]
            idx = idx[np.argsort(keys[self.sort_column][idx], kind="stable")]
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

    def next_unlabeled(self) -> bool:
        """First unlabeled item after the cursor, wrapping."""
        hits = np.flatnonzero(self.labels[self.order] < 0)
        if not len(hits):
            return False
        after = hits[hits > self.pos]
        self.pos = int(after[0] if len(after) else hits[0])
        return True

    def step_group(self, direction: int) -> bool:
        """First item in view of the next/previous label class."""
        if self.current is None:
            return False
        labels = self.labels[self.order]
        values = np.unique(labels)
        if len(values) < 2:
            return False
        i = int(np.flatnonzero(values == int(self.labels[self.current]))[0])
        target = values[(i + direction) % len(values)]
        self.pos = int(np.flatnonzero(labels == target)[0])
        return True


def draw_roi_table(
    order: RoiOrder,
    label_set,
    column_names: Sequence[str],
    formatters: dict,
    scroll_to_current: bool,
    table_id: str = "rois",
    on_select: Optional[Callable[[int], None]] = None,
) -> bool:
    """
    Sortable, clipped ROI table. Returns the new scroll_to_current flag.

    column_names[0] is the id column; "label" is drawn in its class colour.
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
        imgui.table_setup_column(name)
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
            clicked, _ = imgui.selectable(
                f"{item}##row{row}", row == order.pos,
                imgui.SelectableFlags_.span_all_columns,
            )
            if clicked:
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
                            imgui.ImVec4(*label_set.color(label), 1.0),
                            label_set.names[label],
                        )
                    else:
                        imgui.text("-")
                else:
                    imgui.text(formatters[name](item))
    imgui.end_table()
    return scroll_to_current


def draw_filter_row(order: RoiOrder, label_set, id_suffix: str = "") -> bool:
    """Label filter combo + optional range slider. Returns True if the view changed."""
    changed_any = False
    names = ("all", "unlabeled", *label_set.names)
    imgui.set_next_item_width(-1)
    changed, sel = imgui.combo(f"##filter{id_suffix}", order.filter_label + 2, list(names))
    if changed:
        order.filter_label = sel - 2
        changed_any = True
    if order.range_column is not None:
        values = order.columns[order.range_column]
        imgui.set_next_item_width(-1)
        changed, lo, hi = imgui.drag_int_range2(
            f"##range{id_suffix}",
            order.range_limits[0], order.range_limits[1], 1, 0,
            int(np.max(values, initial=0)),
            f"{order.range_column} >= %d", f"{order.range_column} <= %d",
        )
        if changed:
            order.range_limits = (lo, hi)
            changed_any = True
    imgui.text(f"{len(order.order)}/{order.n_items} in view")
    if changed_any:
        order.rebuild()
    return changed_any
