from typing import *
import os
import time
import numpy as np
import fastplotlib as fpl
from imgui_bundle import imgui, portable_file_dialogs as pfd
from functools import partial
from fastplotlib.widgets.nd_widget._index import ReferenceIndex
from masknmf.compression import PMDArray
from masknmf.demixing.demixing_results import DemixingResults
from masknmf.utils import display
from masknmf.visualization.imgui import resolve_time_reference, is_notebook_canvas
from masknmf.visualization.imgui.theme import em

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


def _as_numpy(img) -> np.ndarray:
    if hasattr(img, "cpu"):
        return img.cpu().numpy()
    return np.asarray(img)


class SignalSelectionVis:
    """
    Viewer for manually selecting signals after motion correction + PMD compression
    (and optionally demixing).

    A single FOV panel switches (via the source selector) between summary images,
    the PMD movie, and the residual movie, with existing ROI contours overlaid when
    demixing results are provided. Polygon ROIs drawn on the panel preview their
    ROI-average traces and can be exported as spatial footprints for demixing.
    """

    def __init__(
        self,
        results: DemixingResults | PMDArray,
        frame_timings: Optional[np.ndarray] = None,
        ref_range: Optional[dict] = None,
        summary_images: Optional[dict] = None,
        show_contours: bool = True,
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
            device (str): 'cpu' or 'cuda'
        """

        self._device = device
        self._results = results
        # will device ever be none here?
        results.to(device)

        if isinstance(results, DemixingResults):
            self._pmd_array = results.pmd_array
            self._residual_array = results.residual_array
            self._contours = results.ac_array.contours
        else:
            self._pmd_array = results
            self._residual_array = None
            self._contours = None

        self._shape = self._pmd_array.shape
        num_frames = self._shape[0]

        ref_range, frame_timings = resolve_time_reference(
            num_frames, frame_timings, ref_range
        )
        movie_index_mapping = {"time": frame_timings}

        if summary_images is None:
            summary_images = {"mean img": self._pmd_array.mean_img}
        summary_images = {name: _as_numpy(img) for name, img in summary_images.items()}

        self._source_names = list(summary_images) + ["pmd movie"]
        if self._residual_array is not None:
            self._source_names.append("residual movie")
        self._source_idx = 0

        self._ndw_fov = fpl.NDWidget(
            ref_range,
            shape=(1, 1),
            names=["fov"],
            size=(1000, 1000),
        )
        self._reference_index = self._ndw_fov.indices
        self._fov_subplot = self._ndw_fov.figure["fov"]

        self._nd_images = {}
        for name, img in summary_images.items():
            self._nd_images[name] = self._ndw_fov["fov"].add_nd_image(
                img,
                ["m", "n"],
                ["m", "n"],
                compute_histogram=name == self._source_names[0],
                name=name,
            )

        self._nd_images["pmd movie"] = self._ndw_fov["fov"].add_nd_image(
            self._pmd_array,
            ["time", "m", "n"],
            ["m", "n"],
            slider_dim_transforms=movie_index_mapping.copy(),
            compute_histogram=False,
            name="pmd movie",
        )

        if self._residual_array is not None:
            self._nd_images["residual movie"] = self._ndw_fov["fov"].add_nd_image(
                self._residual_array,
                ["time", "m", "n"],
                ["m", "n"],
                slider_dim_transforms=movie_index_mapping.copy(),
                compute_histogram=False,
                name="residual movie",
            )

        for name, nd in self._nd_images.items():
            if name != self._source_names[0]:
                nd.graphic.visible = False
                nd.pause = True
        self._fov_subplot.title = self._source_names[0]

        self._show_contours = show_contours and self._contours is not None
        if self._contours is not None:
            self._contour_selector = fpl.ImageHighlightSelector(
                selection_options={"pixels": self._contours},
                options_color="w",
                options_alpha=0.5,
            )
            if self._show_contours:
                for nd in self._nd_images.values():
                    self._contour_selector.add_graphic(nd.graphic)
        else:
            self._contour_selector = None

        self._trace_panels = ["pmd roi average"]
        self._trace_movies = {"pmd roi average": self._pmd_array}
        if self._residual_array is not None:
            self._trace_panels.append("residual roi average")
            self._trace_movies["residual roi average"] = self._residual_array

        step = 1.0 / len(self._trace_panels)
        self._trace_extents = {
            panel: (0, 1, k * step, (k + 1) * step)
            for k, panel in enumerate(self._trace_panels)
        }

        self._ndw_traces = fpl.NDWidget(
            ref_ranges=self.reference_index.ref_ranges,
            ref_index=self.reference_index,
            extents=self._trace_extents,
            names=[*self._trace_panels],
            controller_ids=[tuple(self._trace_panels)],
            size=(1000, 500),
        )

        self._trace_graphics = {}
        for panel in self._trace_panels:
            self._trace_graphics[panel] = self._ndw_traces[panel].add_nd_timeseries(
                None,
                ("l", "time", "d"),
                ("l", "time", "d"),
                slider_dim_transforms=movie_index_mapping.copy(),
                max_display_datapoints=5000,
                x_range_mode="auto",
                display_window=None,
                name=panel,
            )

        self._rois = {}  # PolygonSelector -> its display color
        self._active_roi = None
        self._preview_stale = False
        self._last_roi_event = 0.0
        self._status = ""
        self._file_dialog = None

        self._ndw_fov.figure.add_imgui_window(
            self._draw_panel, location="top", size=64, title="Signal Selection"
        )

        for subplot in self._ndw_fov.figure:
            subplot.tooltip.enabled = False
        for subplot in self._ndw_traces.figure:
            subplot.tooltip.enabled = False

    @property
    def results(self) -> DemixingResults | PMDArray:
        return self._results

    @property
    def pmd_array(self) -> PMDArray:
        return self._pmd_array

    @property
    def residual_array(self):
        return self._residual_array

    @property
    def device(self) -> str:
        return self._device

    @property
    def reference_index(self) -> ReferenceIndex:
        return self._reference_index

    @property
    def fov_widget(self) -> fpl.NDWidget:
        return self._ndw_fov

    @property
    def trace_widget(self) -> fpl.NDWidget:
        return self._ndw_traces

    @property
    def rois(self) -> list:
        """The drawn ROIs as a list of PolygonSelector graphics"""
        return list(self._rois)

    @property
    def roi_masks(self) -> np.ndarray:
        """The drawn ROIs as a binary mask stack of shape (fov dim1, fov dim2, num_rois)"""
        height, width = self._shape[1], self._shape[2]
        masks = []
        for sel in self._rois:
            indices = sel.get_selected_indices(self._base_graphic)
            if indices.shape[0] == 0:
                continue
            mask = np.zeros((height, width), dtype=np.float32)
            mask[indices[:, 1], indices[:, 0]] = 1.0
            masks.append(mask)
        if not masks:
            return np.zeros((height, width, 0), dtype=np.float32)
        return np.stack(masks, axis=-1)

    def export_rois(self, path: str) -> str:
        """
        Save the drawn ROIs to an .npz file with key 'spatial_footprints', shaped
        (fov dim1, fov dim2, num_rois); this can be passed directly to
        SignalDemixer.initialize_signals for a custom initialization.
        """
        masks = self.roi_masks
        if masks.shape[-1] == 0:
            raise ValueError("no rois have been drawn")
        path = str(path)
        if not path.endswith(".npz"):
            path += ".npz"
        np.savez(path, spatial_footprints=masks)
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
        if "time" in new_nd.dims:
            self._ndw_fov.indices = {"time": self.reference_index["time"]}

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
        self._rois[selector] = color
        self._active_roi = selector

    def _roi_changed(self, selector, ev):
        self._active_roi = selector
        self._preview_stale = True
        self._last_roi_event = time.perf_counter()

    def _drawing(self) -> bool:
        sel = self._active_roi
        return sel is not None and sel._move_info.mode == "create"

    def _delete_roi(self, selector):
        if selector._move_info.mode is not None:
            selector._end_move_mode()
        self._fov_subplot.delete_graphic(selector)
        del self._rois[selector]
        if self._active_roi is selector:
            self._active_roi = next(reversed(self._rois)) if self._rois else None
            if self._active_roi is not None:
                self._preview_stale = True
                self._last_roi_event = 0.0
            else:
                self._clear_preview()

    def _clear_rois(self):
        for selector in list(self._rois):
            self._delete_roi(selector)

    def _roi_average(self, movie, indices: np.ndarray) -> np.ndarray:
        cols, rows = indices[:, 0], indices[:, 1]
        row_slice = slice(int(rows.min()), int(rows.max()) + 1)
        col_slice = slice(int(cols.min()), int(cols.max()) + 1)
        crop = np.asarray(movie[:, row_slice, col_slice])
        return crop[:, rows - row_slice.start, cols - col_slice.start].mean(axis=1)

    def _update_preview(self):
        self._preview_stale = False
        selector = self._active_roi
        if selector is None:
            return
        indices = selector.get_selected_indices(self._base_graphic)
        if indices.shape[0] == 0:
            return
        color = self._rois[selector]
        x_data = np.arange(self._shape[0])
        for panel in self._trace_panels:
            trace = self._roi_average(self._trace_movies[panel], indices)
            graphic = self._trace_graphics[panel]
            graphic.data = fpl.utils.heatmap_to_positions(trace[None, :], x_data)
            graphic.graphic.colors = np.array([color])
            self._ndw_traces.figure[panel].y_range = (
                float(np.amin(trace)),
                float(np.amax(trace)),
            )
            self._ndw_traces.figure[panel].title = f"{panel} ({indices.shape[0]} px)"

    def _clear_preview(self):
        x_data = np.arange(self._shape[0])
        zeros = fpl.utils.heatmap_to_positions(np.zeros((1, self._shape[0])), x_data)
        for panel in self._trace_panels:
            self._trace_graphics[panel].data = zeros
            self._ndw_traces.figure[panel].title = panel

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
            self._status = f"exported {self.roi_masks.shape[-1]} roi(s) to {path}"
        except (OSError, ValueError) as e:
            self._status = f"export failed: {e}"

    def _draw_panel(self):
        self._poll_file_dialog()
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

        if (
            self._preview_stale
            and not drawing
            and (time.perf_counter() - self._last_roi_event > _PREVIEW_DELAY)
        ):
            self._update_preview()

    def show(self):
        # parse based on canvas type
        if is_notebook_canvas(self.fov_widget.figure):
            from ipywidgets import VBox

            return VBox([self.fov_widget.show(), self.trace_widget.show()])
        else:
            return self.fov_widget.show(), self.trace_widget.show()
