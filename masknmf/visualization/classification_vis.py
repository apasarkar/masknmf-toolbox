from typing import *
import os
import numpy as np
import fastplotlib as fpl
from imgui_bundle import imgui

try:
    from imgui_data_loader import ButtonSpec, FileDialog, FileDialogConfig, FileType, PickKind
except ImportError:
    FileDialog = None

_LABEL_COLORS = (
    (0.12, 0.47, 0.71), (1.00, 0.50, 0.05), (0.17, 0.63, 0.17),
    (0.84, 0.15, 0.16), (0.58, 0.40, 0.74), (0.55, 0.34, 0.29),
    (0.89, 0.47, 0.76), (0.50, 0.50, 0.50), (0.74, 0.74, 0.13),
    (0.09, 0.75, 0.81),
)
_LABEL_KEYS = (
    imgui.Key._1, imgui.Key._2, imgui.Key._3, imgui.Key._4, imgui.Key._5,
    imgui.Key._6, imgui.Key._7, imgui.Key._8, imgui.Key._9,
)
_COLUMNS = ("id", "label", "area", "peak", "y", "x")


class ClassificationVis:
    """
    Label ROI images (e.g. from ROICat) one neuron at a time.
    """

    def __init__(
        self,
        roi_images: np.ndarray,
        label_names: Sequence[str] = (),
        class_labels: Optional[np.ndarray] = None,
        save_path: Optional[str] = None,
    ):
        """
        roi_images: (num_neurons, Y, X) spatial footprint images.
        save_path: .npz file; label names and per-neuron labels are written on every
        change and restored from it on launch unless overridden by arguments.
        """
        roi_images = np.asarray(roi_images, dtype=np.float32)
        if roi_images.ndim != 3:
            raise ValueError(f"roi_images must be (num_neurons, Y, X), got {roi_images.shape}")
        self._roi_images = roi_images
        num_neurons = roi_images.shape[0]

        self._save_path = str(save_path) if save_path is not None else None
        self._error: Optional[str] = None
        self._new_label = ""
        if self._save_path is not None and os.path.exists(self._save_path):
            saved = np.load(self._save_path)
            if not label_names:
                label_names = [str(n) for n in saved["label_names"]]
            if class_labels is None:
                class_labels = saved["class_labels"]

        self._label_names = tuple(label_names)
        if class_labels is None:
            class_labels = np.full((num_neurons,), -1, dtype=np.int64)
        else:
            class_labels = np.asarray(class_labels).astype(np.int64)
            if class_labels.shape[0] != num_neurons:
                raise ValueError(
                    f"class_labels has {class_labels.shape[0]} entries, expected {num_neurons}"
                )
        self._class_labels = class_labels

        if FileDialog is not None:
            self._label_dialog = FileDialog(FileDialogConfig(close_on_select=False))
            self._label_open_spec = ButtonSpec(
                "load labels",
                PickKind.OPEN_FILE,
                filetypes=[
                    FileType("Label sets", "*.json *.txt *.npz *.h5 *.hdf5"),
                    FileType("All Files", "*"),
                ],
            )
        else:
            self._label_dialog = None

        flat = roi_images.reshape(num_neurons, -1)
        self._peak = flat.max(axis=1)
        self._area = np.count_nonzero(flat, axis=1).astype(np.int64)
        totals = flat.sum(axis=1)
        totals[totals == 0] = 1
        ys = np.arange(roi_images.shape[1], dtype=np.float32)
        xs = np.arange(roi_images.shape[2], dtype=np.float32)
        self._cy = (roi_images.sum(axis=2) @ ys) / totals
        self._cx = (roi_images.sum(axis=1) @ xs) / totals

        self._mip = roi_images.max(axis=0)
        self._mip_max = float(self._mip.max()) or 1.0
        self._show_mip = True
        self._mip_alpha = 0.5
        self._mip_saturation = 0.0

        self._filter_label = -2  # -2 all, -1 unlabeled, >=0 label index
        self._area_range = (0, int(self._area.max(initial=0)))
        self._sort_column = 0
        self._sort_ascending = True
        self._order = np.arange(num_neurons)
        self._pos = 0
        self._advance_on_label = True
        self._scroll_to_current = True

        self._figure = fpl.Figure(size=(1200, 900))
        subplot = self._figure[0, 0]
        self._bg = subplot.add_image(self._mip, cmap="gray", name="mip")
        self._fg = subplot.add_image(
            np.zeros((*roi_images.shape[1:], 4), dtype=np.float32), name="roi"
        )
        self._apply_mip()
        self._show_current()

        self._figure.add_imgui_window(
            self._draw_panel, location="top", size=120, title="Classification"
        )
        self._figure.add_imgui_window(
            self._draw_table, location="right", size=360, title="Neurons"
        )

    @property
    def current(self) -> Optional[int]:
        """Neuron id at the current position in the sorted/filtered view"""
        if len(self._order) == 0:
            return None
        return int(self._order[self._pos])

    def step(self, delta: int):
        """Move through the current view by delta neurons"""
        if len(self._order):
            self._pos = int(np.clip(self._pos + delta, 0, len(self._order) - 1))
            self._show_current()

    def goto(self, neuron: int):
        """Jump to a neuron if it is in the current view"""
        hits = np.flatnonzero(self._order == neuron)
        if len(hits):
            self._pos = int(hits[0])
            self._show_current()

    def label(self, neuron_ids: Sequence[int], label_index: int):
        """Assign a class label to the given neurons; -1 clears"""
        self._class_labels[list(neuron_ids)] = label_index
        self._autosave()
        if self._filter_label != -2:
            self._rebuild_order()
        self._show_current()

    def label_current(self, label_index: int):
        """Label the current neuron and advance to the next one in view"""
        if self.current is None:
            return
        size_before = len(self._order)
        self.label([self.current], label_index)
        # if the filter dropped the labeled neuron, pos already sits on its successor
        if self._advance_on_label and len(self._order) == size_before:
            self.step(1)

    def add_label(self, name: str):
        """Add a new class name to the label set"""
        if name and name not in self._label_names:
            self._label_names = (*self._label_names, name)
            self._autosave()

    def load_label_names(self, path: str):
        """
        Merge a label-name set from a file: .json (list of names), .npz (a previous
        save), .h5/.hdf5 (DemixingResults/label_names), or plain text
        (comma/newline separated)
        """
        path = str(path)
        if path.endswith(".json"):
            import json

            with open(path) as f:
                names = json.load(f)
        elif path.endswith(".npz"):
            names = [str(n) for n in np.load(path)["label_names"]]
        elif path.endswith((".h5", ".hdf5")):
            import h5py

            with h5py.File(path, "r") as f:
                names = [n.decode() for n in f["DemixingResults/label_names"][()]]
        else:
            with open(path) as f:
                names = [t.strip() for t in f.read().replace(",", "\n").splitlines()]
        for name in names:
            self.add_label(str(name).strip())

    def save(self, path: Optional[str] = None):
        """Write label names and per-neuron labels to an npz file"""
        path = path if path is not None else self._save_path
        np.savez(path, label_names=np.array(self._label_names), class_labels=self._class_labels)

    def _autosave(self):
        if self._save_path is None:
            return
        try:
            self.save()
            self._error = None
        except OSError as e:
            self._error = f"save failed: {e}"

    def _rebuild_order(self):
        current = self.current
        mask = (self._area >= self._area_range[0]) & (self._area <= self._area_range[1])
        if self._filter_label >= -1:
            mask &= self._class_labels == self._filter_label
        idx = np.flatnonzero(mask)
        if self._sort_column:
            keys = (self._class_labels, self._area, self._peak, self._cy, self._cx)
            idx = idx[np.argsort(keys[self._sort_column - 1][idx], kind="stable")]
        if not self._sort_ascending:
            idx = idx[::-1]
        self._order = idx
        hits = np.flatnonzero(self._order == current) if current is not None else ()
        if len(hits):
            self._pos = int(hits[0])
        else:
            self._pos = int(min(self._pos, max(len(self._order) - 1, 0)))
        self._show_current()

    def _label_color(self, label_index: int) -> tuple:
        return _LABEL_COLORS[label_index % len(_LABEL_COLORS)]

    def _show_current(self):
        neuron = self.current
        rgba = np.zeros((*self._roi_images.shape[1:], 4), dtype=np.float32)
        title = "no neurons in view"
        if neuron is not None:
            label = int(self._class_labels[neuron])
            color = self._label_color(label) if label >= 0 else (1.0, 1.0, 1.0)
            rgba[..., :3] = color
            rgba[..., 3] = self._roi_images[neuron] / (self._peak[neuron] or 1.0)
            name = self._label_names[label] if label >= 0 else "unlabeled"
            title = f"neuron {neuron}  [{name}]  ({self._pos + 1}/{len(self._order)})"
        self._fg.data = rgba
        self._figure[0, 0].title = title
        self._scroll_to_current = True

    def _apply_mip(self):
        self._bg.visible = self._show_mip
        self._bg.alpha = self._mip_alpha
        self._bg.vmin = 0.0
        self._bg.vmax = self._mip_max * (1.0 - 0.99 * self._mip_saturation)

    def _handle_keys(self):
        if imgui.get_io().want_text_input:
            return
        if imgui.is_key_pressed(imgui.Key.left_arrow):
            self.step(-1)
        if imgui.is_key_pressed(imgui.Key.right_arrow) or imgui.is_key_pressed(imgui.Key.space):
            self.step(1)
        if imgui.is_key_pressed(imgui.Key._0, False):
            self.label_current(-1)
        for i, key in enumerate(_LABEL_KEYS[: len(self._label_names)]):
            if imgui.is_key_pressed(key, False):
                self.label_current(i)

    def _draw_panel(self):
        self._handle_keys()

        if imgui.button("prev"):
            self.step(-1)
        imgui.same_line(0, 5)
        if imgui.button("next"):
            self.step(1)
        imgui.same_line(0, 10)
        imgui.set_next_item_width(200)
        changed, pos = imgui.slider_int("##pos", self._pos, 0, max(len(self._order) - 1, 0))
        if changed and len(self._order):
            self._pos = pos
            self._show_current()
        imgui.same_line(0, 30)
        _, self._advance_on_label = imgui.checkbox("advance on label", self._advance_on_label)
        imgui.same_line(0, 30)
        imgui.text(f"autosave: {self._save_path if self._save_path else 'off'}")
        if self._error is not None:
            imgui.same_line(0, 30)
            imgui.text(self._error)

        changed_mip, self._show_mip = imgui.checkbox("MIP background", self._show_mip)
        imgui.same_line(0, 20)
        imgui.set_next_item_width(150)
        changed_alpha, self._mip_alpha = imgui.slider_float("opacity", self._mip_alpha, 0.0, 1.0)
        imgui.same_line(0, 20)
        imgui.set_next_item_width(150)
        changed_sat, self._mip_saturation = imgui.slider_float(
            "saturation", self._mip_saturation, 0.0, 1.0
        )
        if changed_mip or changed_alpha or changed_sat:
            self._apply_mip()

        if self._label_dialog is not None:
            if imgui.button("load labels"):
                self._label_dialog.pick(self._label_open_spec)
            self._label_dialog.poll()
            result = self._label_dialog.take_result()
            if result:
                try:
                    self.load_label_names(result.path)
                    self._error = None
                except (OSError, KeyError, ValueError) as e:
                    self._error = f"label load failed: {e}"
            imgui.same_line(0, 10)
        imgui.set_next_item_width(120)
        entered, self._new_label = imgui.input_text_with_hint(
            "##new-label",
            "new label",
            self._new_label,
            imgui.InputTextFlags_.enter_returns_true,
        )
        imgui.same_line(0, 5)
        if (imgui.button("add") or entered) and self._new_label.strip():
            self.add_label(self._new_label.strip())
            self._new_label = ""

        for i, name in enumerate(self._label_names):
            imgui.same_line(0, 10)
            count = int((self._class_labels == i).sum())
            key_hint = f"{i + 1}:" if i < len(_LABEL_KEYS) else ""
            imgui.push_style_color(imgui.Col_.button, imgui.ImVec4(*self._label_color(i), 0.5))
            if imgui.button(f"{key_hint}{name} ({count})"):
                self.label_current(i)
            imgui.pop_style_color()
        if self._label_names:
            imgui.same_line(0, 10)
            if imgui.button("0:unlabel"):
                self.label_current(-1)

    def _draw_table(self):
        names = ("all", "unlabeled", *self._label_names)
        imgui.set_next_item_width(-1)
        changed, sel = imgui.combo("##filter", self._filter_label + 2, list(names))
        if changed:
            self._filter_label = sel - 2
            self._rebuild_order()
        imgui.set_next_item_width(-1)
        changed, lo, hi = imgui.drag_int_range2(
            "##area",
            self._area_range[0],
            self._area_range[1],
            1,
            0,
            int(self._area.max(initial=0)),
            "area >= %d",
            "area <= %d",
        )
        if changed:
            self._area_range = (lo, hi)
            self._rebuild_order()
        imgui.text(f"{len(self._order)}/{len(self._roi_images)} in view")

        flags = (
            imgui.TableFlags_.sortable
            | imgui.TableFlags_.row_bg
            | imgui.TableFlags_.resizable
            | imgui.TableFlags_.scroll_y
        )
        avail = imgui.get_content_region_avail()
        if not imgui.begin_table("neurons", len(_COLUMNS), flags, imgui.ImVec2(0, avail.y)):
            return
        imgui.table_setup_scroll_freeze(0, 1)
        imgui.table_setup_column(_COLUMNS[0], imgui.TableColumnFlags_.default_sort)
        for name in _COLUMNS[1:]:
            imgui.table_setup_column(name)
        imgui.table_headers_row()

        specs = imgui.table_get_sort_specs()
        if specs is not None and specs.specs_dirty:
            if specs.specs_count > 0:
                self._sort_column = int(specs.specs.column_index)
                self._sort_ascending = (
                    specs.specs.sort_direction == imgui.SortDirection.ascending
                )
            specs.specs_dirty = False
            self._rebuild_order()

        clipper = imgui.ListClipper()
        clipper.begin(len(self._order))
        if self._scroll_to_current:
            clipper.include_item_by_index(self._pos)
        while clipper.step():
            for row in range(clipper.display_start, clipper.display_end):
                neuron = int(self._order[row])
                label = int(self._class_labels[neuron])
                imgui.table_next_row()
                imgui.table_next_column()
                clicked, _ = imgui.selectable(
                    f"{neuron}##row{row}",
                    row == self._pos,
                    imgui.SelectableFlags_.span_all_columns,
                )
                if clicked:
                    self._pos = row
                    self._show_current()
                if row == self._pos and self._scroll_to_current:
                    imgui.set_scroll_here_y(0.5)
                    self._scroll_to_current = False
                imgui.table_next_column()
                if label >= 0:
                    imgui.text_colored(
                        imgui.ImVec4(*self._label_color(label), 1.0), self._label_names[label]
                    )
                else:
                    imgui.text("-")
                imgui.table_next_column()
                imgui.text(f"{self._area[neuron]}")
                imgui.table_next_column()
                imgui.text(f"{self._peak[neuron]:.3g}")
                imgui.table_next_column()
                imgui.text(f"{self._cy[neuron]:.0f}")
                imgui.table_next_column()
                imgui.text(f"{self._cx[neuron]:.0f}")
        imgui.end_table()

    @property
    def class_labels(self) -> np.ndarray:
        """(num_neurons,) label indices into label_names, -1 = unlabeled"""
        return self._class_labels

    @property
    def label_names(self) -> tuple[str, ...]:
        return self._label_names

    @property
    def roi_images(self) -> np.ndarray:
        return self._roi_images

    @property
    def figure(self) -> fpl.Figure:
        return self._figure

    def show(self):
        return self._figure.show()


def main(argv=None):
    import argparse

    parser = argparse.ArgumentParser(description="Label ROI images one neuron at a time")
    parser.add_argument("path", help=".npy file with a (num_neurons, Y, X) array")
    parser.add_argument(
        "--labels",
        default=None,
        help="comma-separated class names, e.g. soma,dendrite,junk",
    )
    parser.add_argument(
        "--save",
        default=None,
        help="labels npz path (default: <path>.labels.npz)",
    )
    args = parser.parse_args(argv)

    roi_images = np.load(args.path)
    label_names = args.labels.split(",") if args.labels else ()
    save_path = args.save if args.save else f"{args.path}.labels.npz"
    vis = ClassificationVis(roi_images, label_names=label_names, save_path=save_path)
    vis.show()
    fpl.loop.run()


if __name__ == "__main__":
    main()
