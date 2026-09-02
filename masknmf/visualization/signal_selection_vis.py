import os
import time
from collections.abc import Sequence
from functools import partial
from itertools import permutations

import numpy as np
import fastplotlib as fpl
from imgui_bundle import imgui, portable_file_dialogs as pfd
from fastplotlib.widgets.nd_widget._index import ReferenceIndex
from fastplotlib.graphics.selectors._polygon import point_in_polygon

from masknmf.compression import PMDArray
from masknmf.demixing.demixing_results import DemixingResults
from masknmf.visualization.imgui import component_at_pixel, resolve_time_reference
from masknmf.visualization.imgui.theme import em, to_vec4
from masknmf.visualization.classification_vis import _LABEL_COLORS, _LABEL_KEYS

_ROI_COLORS = (
    (1.00, 0.50, 0.05),
    (0.17, 0.63, 0.17),
    (0.84, 0.15, 0.16),
    (0.58, 0.40, 0.74),
    (0.89, 0.47, 0.76),
    (0.74, 0.74, 0.13),
    (0.09, 0.75, 0.81),
    (0.12, 0.47, 0.71),
)
_NPZ_FILTERS = ["NumPy archive", "*.npz", "All files", "*"]
_PREVIEW_DELAY = 0.4  # seconds after the last ROI edit before the preview recomputes
_COMPONENT_COLOR = (
    1.0,
    0.25,
    0.25,
)  # matches the contour highlight of a selected signal
_TRACE_BAND = 0.3  # fraction of the figure height the trace panels share
DEFAULT_LABEL_NAMES = ("cell", "not cell")


def _as_numpy(img) -> np.ndarray:
    return img.cpu().numpy() if hasattr(img, "cpu") else np.asarray(img)


class SignalSelectionVis:
    """
    Viewer for manually selecting signals after motion correction + PMD compression
    (and optionally demixing).
    """

    def __init__(
        self,
        results: DemixingResults | PMDArray,
        frame_timings: np.ndarray | None = None,
        ref_range: dict | None = None,
        summary_images: dict | None = None,
        show_contours: bool = True,
        label_names: Sequence[str] = DEFAULT_LABEL_NAMES,
        device: str = "cpu",
    ):
        """
        Args:
            results (DemixingResults | PMDArray): demixing results (enables the residual
                movie and existing ROI contours) or just a PMD array
            frame_timings (Optional[np.ndarray]): per-frame timestamps
            ref_range (Optional[dict]): reference range for the time axis
            summary_images (Optional[dict]): {name: (fov dim1, fov dim2) array} projections
                to display; defaults to the PMD mean image
            show_contours (bool): initial state of the contour overlay of existing demixed signals
            label_names (Sequence[str]): class names the drawn ROIs can be labeled with
            device (str): 'cpu' or 'cuda'
        """

        self._results = results
        results.to(device)

        if isinstance(results, DemixingResults):
            self._pmd_array = results.pmd_array
            self._residual_array = results.residual_array
            self._ac_array = results.ac_array
            self._contours = results.ac_array.contours
        else:
            self._pmd_array = results
            self._residual_array = None
            self._ac_array = None
            self._contours = None

        self._shape = self._pmd_array.shape
        ref_range, self._frame_timings = resolve_time_reference(
            self._shape[0], frame_timings, ref_range
        )
        time_transform = {"time": self._frame_timings}

        if summary_images is None:
            summary_images = {"mean img": self._pmd_array.mean_img}
        summary_images = {name: _as_numpy(img) for name, img in summary_images.items()}

        movies = {"pmd movie": self._pmd_array}
        self._trace_movies = {"pmd": self._pmd_array}
        if self._residual_array is not None:
            movies["residual movie"] = self._residual_array
            self._trace_movies["residual"] = self._residual_array

        self._source_names = [*summary_images, *movies]
        self._source_idx = 0

        step = _TRACE_BAND / len(self._trace_movies)
        extents = {
            panel: (0, 1, i * step, (i + 1) * step)
            for i, panel in enumerate(self._trace_movies)
        }
        extents["fov"] = (0, 1, _TRACE_BAND, 1)

        self._ndw = fpl.NDWidget(ref_range, extents=extents, size=(1000, 1200))
        self._fov_subplot = self._ndw.figure["fov"]

        self._nd_images = {}
        for name, img in summary_images.items():
            self._nd_images[name] = self._ndw["fov"].add_nd_image(
                img,
                ["m", "n"],
                ["m", "n"],
                compute_histogram=name == self._source_names[0],
                name=name,
            )
        for name, movie in movies.items():
            self._nd_images[name] = self._ndw["fov"].add_nd_image(
                movie,
                ["time", "m", "n"],
                ["m", "n"],
                slider_dim_transforms=time_transform.copy(),
                compute_histogram=False,
                name=name,
            )
        for name, nd in self._nd_images.items():
            if name != self._source_names[0]:
                nd.graphic.visible = False
                nd.pause = True
        self._fov_subplot.title = self._source_names[0]

        self._show_contours = False
        self._contour_selector = None
        if self._contours is not None:
            self._contour_selector = fpl.ImageHighlightSelector(
                selection_options={"pixels": self._contours},
                options_color="w",
                options_alpha=0.5,
            )
            if show_contours:
                self._set_contours(True)

        self._trace_graphics = {}
        for panel in self._trace_movies:
            self._trace_graphics[panel] = self._ndw[panel].add_nd_timeseries(
                None,
                ("l", "time", "d"),
                ("l", "time", "d"),
                slider_dim_transforms=time_transform.copy(),
                max_display_datapoints=5000,
                x_range_mode="auto",
                display_window=None,
                name=panel,
            )
            self._ndw.figure[panel].title = f"{panel} roi average"

        # the trace panels share the time axis, but each keeps its own y scale
        for panel, other in permutations(self._trace_movies, 2):
            self._ndw.figure[panel].controller.add_camera(
                self._ndw.figure[other].camera, include_state={"x", "width"}
            )

        self._rois = {}  # PolygonSelector -> {"color": rgb, "label": class index}
        self._active_roi = None
        self._active_component = None
        self._preview_stale = False
        self._last_roi_event = 0.0
        self._status = ""
        self._file_dialog = None
        self._new_label = ""
        self._label_colors = []
        self._set_label_names(label_names)

        self._ndw.figure.renderer.add_event_handler(self._click_select, "double_click")
        self._ndw.figure.add_imgui_window(
            self._draw_panel, location="top", size=96, title="Signal Selection"
        )
        for subplot in self._ndw.figure:
            subplot.tooltip.enabled = False
            subplot.toolbar = False

    @property
    def results(self) -> DemixingResults | PMDArray:
        return self._results

    @property
    def reference_index(self) -> ReferenceIndex:
        return self._ndw.indices

    @property
    def widget(self) -> fpl.NDWidget:
        return self._ndw

    @property
    def rois(self) -> list:
        """The drawn ROIs as a list of PolygonSelector graphics"""
        return list(self._rois)

    @property
    def label_names(self) -> tuple:
        return self._label_names

    def _masks_and_labels(self) -> tuple[np.ndarray, np.ndarray]:
        shape = tuple(self._shape[1:3])
        masks, labels = [], []
        for selector, roi in self._rois.items():
            indices = selector.get_selected_indices(self._base_graphic)
            if indices.shape[0] == 0:
                continue
            mask = np.zeros(shape, dtype=np.float32)
            mask[indices[:, 1], indices[:, 0]] = 1.0
            masks.append(mask)
            labels.append(roi["label"])
        if not masks:
            return np.zeros((*shape, 0), dtype=np.float32), np.zeros(0, dtype=np.int64)
        return np.stack(masks, axis=-1), np.array(labels, dtype=np.int64)

    @property
    def roi_masks(self) -> np.ndarray:
        """The drawn ROIs as a binary mask stack of shape (fov dim1, fov dim2, num_rois)"""
        return self._masks_and_labels()[0]

    @property
    def roi_labels(self) -> np.ndarray:
        """Class label index per drawn ROI (-1 = unlabeled), aligned with roi_masks"""
        return self._masks_and_labels()[1]

    def _append_to_signals(self, masks: np.ndarray) -> np.ndarray:
        if self._ac_array is None:
            raise ValueError("combined footprints need demixing results")
        return np.concatenate([self._ac_array.export_a(), masks], axis=-1)

    def combined_footprints(self) -> np.ndarray:
        """
        The existing demixed spatial footprints with the drawn ROI masks appended,
        shape (fov dim1, fov dim2, num_signals + num_rois). Passing this to
        SignalDemixer.initialize_signals(is_custom=True) re-demixes with the drawn
        ROIs added to the existing signals.
        """
        return self._append_to_signals(self.roi_masks)

    def export_rois(self, path: str) -> str:
        """
        Save the drawn ROIs to an .npz file: 'spatial_footprints' is the
        (fov dim1, fov dim2, num_rois) mask stack for a custom demixing initialization,
        'class_labels' / 'label_names' carry the labels, and, when demixing results are
        loaded, 'spatial_footprints_combined' appends the drawn ROIs to the existing
        signals so SignalDemixer.initialize_signals(is_custom=True) re-demixes with
        the drawn ROIs added to the results.
        """
        masks, labels = self._masks_and_labels()
        if masks.shape[-1] == 0:
            raise ValueError("no rois have been drawn")
        path = str(path)
        if not path.endswith(".npz"):
            path += ".npz"
        data = dict(
            spatial_footprints=masks,
            class_labels=labels,
            label_names=np.array(self._label_names),
        )
        if self._ac_array is not None:
            data["spatial_footprints_combined"] = self._append_to_signals(masks)
        np.savez_compressed(path, **data)
        return path

    @property
    def _base_graphic(self):
        return self._nd_images[self._source_names[0]].graphic

    def _set_source(self, index: int):
        if index == self._source_idx:
            return
        old_nd = self._nd_images[self._source_names[self._source_idx]]
        new_nd = self._nd_images[self._source_names[index]]
        old_nd.compute_histogram = False
        old_nd.graphic.visible = False
        old_nd.pause = True
        new_nd.pause = False
        new_nd.graphic.visible = True
        new_nd.compute_histogram = True
        self._source_idx = index
        self._fov_subplot.title = self._source_names[index]
        self._ndw.indices = new_nd.indices  # nudge the unpaused graphic into view

    def _set_contours(self, show: bool):
        self._show_contours = show
        for nd in self._nd_images.values():
            if show:
                self._contour_selector.add_graphic(nd.graphic)
            else:
                self._contour_selector.remove_graphic(nd.graphic)

    def _start_roi(self):
        color = _ROI_COLORS[len(self._rois) % len(_ROI_COLORS)]
        selector = self._base_graphic.add_polygon_selector(
            fill_color=color,
            edge_color=color,
            vertex_color=color,
            edge_thickness=2,
            vertex_size=8,
        )
        selector.add_event_handler(partial(self._roi_changed, selector), "selection")
        self._rois[selector] = {"color": color, "label": -1}
        self._active_roi = selector
        self._clear_component()

    def _roi_changed(self, selector, ev):
        self._active_roi = selector
        self._clear_component()
        self._preview_stale = True
        self._last_roi_event = time.perf_counter()

    def _drawing(self) -> bool:
        return (
            self._active_roi is not None
            and self._active_roi._move_info.mode == "create"
        )

    def _roi_at(self, col: int, row: int):
        for selector in reversed(self._rois):
            polygon = selector.selection[:, :2]
            if polygon.shape[0] >= 3 and point_in_polygon((col, row), polygon):
                return selector
        return None

    def _click_select(self, ev):
        """Double click selects what is under the cursor: a drawn ROI first, else an
        existing demixed signal; the background clears the selection"""
        if self._drawing() or imgui.get_io().want_capture_mouse:
            return
        pos = self._fov_subplot.map_screen_to_world(ev)
        if pos is None:
            return
        col, row = round(float(pos[0])), round(float(pos[1]))
        if not (0 <= row < self._shape[1] and 0 <= col < self._shape[2]):
            return
        selector = self._roi_at(col, row)
        if selector is not None:
            self._select_roi(selector)
            return
        if self._ac_array is not None:
            component = component_at_pixel(
                self._ac_array.a, self._ac_array.centers, self._shape[1:], (col, row)
            )
            if component is not None:
                self._select_component(component)
                return
        self._clear_selection()

    def _select_roi(self, selector):
        self._clear_component()
        self._active_roi = selector
        self._preview_stale = True
        self._last_roi_event = 0.0

    def _select_component(self, component: int):
        self._active_roi = None
        self._preview_stale = False
        self._active_component = int(component)
        if self._contour_selector is not None:
            self._contour_selector.selection = self._active_component
        traces = {
            panel: getattr(self._results, f"{panel}_roi_averages")[component]
            .cpu()
            .numpy()
            for panel in self._trace_movies
        }
        self._plot_traces(
            traces, _COMPONENT_COLOR, f" (signal {self._active_component})"
        )

    def _clear_component(self):
        if self._active_component is not None:
            self._active_component = None
            if self._contour_selector is not None:
                self._contour_selector.selection = None

    def _clear_selection(self):
        self._active_roi = None
        self._clear_component()
        self._clear_traces()

    def _set_label_names(
        self, names: Sequence[str], colors: Sequence[tuple] | None = None
    ):
        """Replace the label set, keeping (or extending from the palette) one color per label."""
        names = tuple(names)
        colors = list(colors if colors is not None else self._label_colors)[
            : len(names)
        ]
        colors += [
            _LABEL_COLORS[i % len(_LABEL_COLORS)]
            for i in range(len(colors), len(names))
        ]
        self._label_names, self._label_colors = names, colors

    def add_label(self, name: str):
        """Add a new class name to the label set"""
        if name and name not in self._label_names:
            self._set_label_names((*self._label_names, name))

    def label_selected(self, label_index: int):
        """Give the selected drawn ROI a class label; -1 clears it"""
        if self._active_roi is not None:
            self._rois[self._active_roi]["label"] = int(label_index)

    def _delete_roi(self, selector):
        if selector._move_info.mode is not None:
            selector._end_move_mode()
        self._fov_subplot.delete_graphic(selector)
        del self._rois[selector]
        if self._active_roi is selector:
            self._active_roi = next(reversed(self._rois), None)
            if self._active_roi is None:
                self._clear_traces()
            else:
                self._preview_stale = True
                self._last_roi_event = 0.0

    def _clear_rois(self):
        for selector in list(self._rois):
            self._delete_roi(selector)

    def _roi_average(self, movie, indices: np.ndarray) -> np.ndarray:
        cols, rows = indices[:, 0], indices[:, 1]
        row_slice = slice(int(rows.min()), int(rows.max()) + 1)
        col_slice = slice(int(cols.min()), int(cols.max()) + 1)
        crop = np.asarray(movie[:, row_slice, col_slice])
        return crop[:, rows - row_slice.start, cols - col_slice.start].mean(axis=1)

    def _plot_traces(self, traces: dict, color, suffix: str = ""):
        for panel, trace in traces.items():
            graphic = self._trace_graphics[panel]
            graphic.data = fpl.utils.heatmap_to_positions(
                trace[None, :], self._frame_timings
            )
            graphic.graphic.colors = np.array([color])
            low, high = float(np.amin(trace)), float(np.amax(trace))
            subplot = self._ndw.figure[panel]
            subplot.y_range = (low, high) if high > low else (low - 1.0, low + 1.0)
            subplot.title = f"{panel} roi average{suffix}"

    def _update_preview(self):
        self._preview_stale = False
        selector = self._active_roi
        if selector is None:
            return
        indices = selector.get_selected_indices(self._base_graphic)
        if indices.shape[0] == 0:
            return
        traces = {
            panel: self._roi_average(movie, indices)
            for panel, movie in self._trace_movies.items()
        }
        roi_num = list(self._rois).index(selector)
        self._plot_traces(
            traces,
            self._rois[selector]["color"],
            f" (roi {roi_num}, {indices.shape[0]} px)",
        )

    def _clear_traces(self):
        zeros = np.zeros(self._shape[0], dtype=np.float32)
        self._plot_traces(
            {panel: zeros for panel in self._trace_movies}, (0.5, 0.5, 0.5)
        )

    def _browse_export(self):
        if self._file_dialog is None:
            start = os.path.join(os.getcwd(), "rois.npz")
            self._file_dialog = pfd.save_file("Export ROIs", start, _NPZ_FILTERS)

    def _poll_file_dialog(self):
        if self._file_dialog is None or not self._file_dialog.ready(0):
            return
        result = self._file_dialog.result()
        self._file_dialog = None
        if not result:
            return
        try:
            path = self.export_rois(result)
            self._status = f"exported {len(self._rois)} roi(s) to {path}"
        except (OSError, ValueError) as e:
            self._status = f"export failed: {e}"

    def _handle_keys(self):
        if imgui.get_io().want_text_input:
            return
        if imgui.is_key_pressed(imgui.Key._0, False):
            self.label_selected(-1)
        if (
            imgui.is_key_pressed(imgui.Key.delete, False)
            and self._active_roi is not None
        ):
            self._delete_roi(self._active_roi)
        for i, key in enumerate(_LABEL_KEYS[: len(self._label_names)]):
            if imgui.is_key_pressed(key, False):
                self.label_selected(i)

    def _selection_status(self) -> str:
        if self._active_component is not None:
            return f"signal {self._active_component} selected"
        if self._active_roi in self._rois:
            label = self._rois[self._active_roi]["label"]
            name = (
                self._label_names[label]
                if 0 <= label < len(self._label_names)
                else "unlabeled"
            )
            return f"roi {list(self._rois).index(self._active_roi)} selected ({name})"
        return "double-click a mask or roi to see its trace"

    def _draw_label_row(self):
        imgui.text_disabled("labels")
        imgui.same_line(0, em(0.5))
        active = self._rois.get(self._active_roi)
        for i, name in enumerate(self._label_names):
            r, g, b = self._label_colors[i]
            alpha = 1.0 if active is not None and active["label"] == i else 0.55
            imgui.push_style_color(imgui.Col_.button, to_vec4((r, g, b, alpha)))
            if imgui.button(f"{name}##label{i}"):
                self.label_selected(i)
            imgui.pop_style_color()
            if imgui.is_item_hovered():
                imgui.set_tooltip(f"label the selected roi ({i + 1}; 0 clears)")
            imgui.same_line(0, em(0.4))
        imgui.set_next_item_width(em(6))
        entered, self._new_label = imgui.input_text_with_hint(
            "##new-label",
            "new label",
            self._new_label,
            imgui.InputTextFlags_.enter_returns_true,
        )
        imgui.same_line(0, em(0.4))
        if (imgui.button("add") or entered) and self._new_label.strip():
            self.add_label(self._new_label.strip())
            self._new_label = ""
        imgui.same_line(0, em(1))
        imgui.text_disabled(self._selection_status())

    def _draw_panel(self):
        self._poll_file_dialog()
        self._handle_keys()
        drawing = self._drawing()

        imgui.set_next_item_width(em(10))
        changed, idx = imgui.combo("source", self._source_idx, self._source_names)
        if changed:
            self._set_source(idx)
        imgui.same_line(0, em(1))
        if self._contour_selector is not None:
            changed, show = imgui.checkbox("contours", self._show_contours)
            if changed:
                self._set_contours(show)
            imgui.same_line(0, em(1))

        imgui.begin_disabled(drawing)
        if imgui.button("draw roi"):
            self._start_roi()
        imgui.end_disabled()
        imgui.same_line(0, em(0.5))
        imgui.begin_disabled(self._active_roi is None)
        if imgui.button("delete roi"):
            self._delete_roi(self._active_roi)
        imgui.end_disabled()
        imgui.same_line(0, em(0.5))
        imgui.begin_disabled(not self._rois)
        if imgui.button("clear rois"):
            self._clear_rois()
        imgui.same_line(0, em(0.5))
        if imgui.button("export rois"):
            self._browse_export()
        imgui.end_disabled()
        imgui.same_line(0, em(1))
        if drawing:
            imgui.text_disabled("click to add points; click the first point to close")
        else:
            imgui.text_disabled(f"{len(self._rois)} roi(s)  {self._status}")

        self._draw_label_row()

        if (
            self._preview_stale
            and not drawing
            and (time.perf_counter() - self._last_roi_event > _PREVIEW_DELAY)
        ):
            self._update_preview()

    def show(self):
        return self._ndw.show()

    def close(self):
        self._ndw.close()


def main(argv=None):
    import argparse
    import h5py

    parser = argparse.ArgumentParser(
        description="Select signals by drawing ROIs over a summary image, PMD movie, and residual movie"
    )
    parser.add_argument("path", help="masknmf demixing_results or PMD .hdf5 file")
    parser.add_argument("--device", default="cpu", help="'cpu' or 'cuda'")
    parser.add_argument(
        "--labels",
        default=None,
        help="comma-separated class names for labeling drawn ROIs, e.g. cell,dendrite,junk",
    )
    args = parser.parse_args(argv)

    with h5py.File(args.path, "r") as f:
        cls = DemixingResults if "DemixingResults" in f else PMDArray
    vis = SignalSelectionVis(
        cls.from_hdf5(args.path, device=args.device),
        label_names=args.labels.split(",") if args.labels else DEFAULT_LABEL_NAMES,
        device=args.device,
    )
    vis.show()
    fpl.loop.run()


if __name__ == "__main__":
    main()
