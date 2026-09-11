import os
import threading
import time
from typing import *
import numpy as np
import fastplotlib as fpl
from imgui_bundle import imgui, portable_file_dialogs as pfd
from fastplotlib import ui
from fastplotlib.graphics.selectors._polygon import point_in_polygon
import pygfx
import torch
from collections import OrderedDict
import masknmf.arrays
from masknmf.utils import display
from functools import partial
from fastplotlib.widgets.nd_widget._index import ReferenceIndex
from masknmf.visualization.imgui import (
    TracePlot,
    resolve_time_reference,
    component_at_pixel,
)
from masknmf.visualization.rois import FootprintSet
from masknmf.demixing import add_signals, replace_results
from masknmf.pipelines.configs.demixing_configs import NMFConfig

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
_PREVIEW_DELAY = (
    0.4  # seconds after the last roi edit before the preview trace recomputes
)
# compressed/background/residual, in that order, so the 3 base lines read apart in the legend
_BASE_LINE_COLORS = ((0.85, 0.85, 0.85), (0.95, 0.55, 0.15), (0.35, 0.65, 0.95))


class SingleSessionDemixingVis:
    """
    View and curate demixing results. Can be used whether demixing has been ran (pass in DemixingResults) or not (PMDArray).
    existing masknmf.DemixingResults (or a bare PMDArray before demixing has run), and draw
    and export ROIs for a custom SignalDemixer.initialize_signals(is_custom=True) pass.
    Footprints show as feathered masks and/or contours over the summary image.

    With ``source_path`` set, "add to results" runs the drawn ROIs through the demixer's NMF pass
    (``nmf_config``, the pipeline defaults when None) and rewrites that file with them as ordinary signals.

    TODO:
    -----
    - Could really support registration arrays?

    """

    def __init__(
        self,
        demixing_results: masknmf.DemixingResults
        | masknmf.PMDArray
        | List[masknmf.DemixingResults],
        frame_timings: Optional[np.ndarray | List[np.ndarray]] = None,
        ref_range: Optional[dict] = None,
        summary_img: np.ndarray | masknmf.ArrayLike | None = None,
        summary_img_name: str | None = None,
        show_contours: bool = False,
        show_masks: bool = True,
        mask_opacity: float = 0.5,
        device="cpu",
        source_path: str | os.PathLike | None = None,
        nmf_config: NMFConfig | None = None,
    ):
        self._source_path = None if source_path is None else str(source_path)
        self._nmf_config = NMFConfig(min_brightness=None) if nmf_config is None else nmf_config
        self._min_brightness_cache = self._nmf_config.min_brightness or 1.0
        self._worker = None
        self._pending = None
        if device == "cpu":
            display(
                "Using CPU; it will be much slower. Use CUDA for much faster rendering"
            )
        self._demixing_results = demixing_results
        self._device = device

        self._demixing_results.to(self.device)
        self._has_ac = isinstance(demixing_results, masknmf.DemixingResults)
        self._shape = self.demixing_results.shape

        ref_range, frame_timings = resolve_time_reference(
            self._shape[0], frame_timings, ref_range
        )

        self._video_panels = (
            "compressed+denoised",
            "signals",
            "background",
            "residual",
            "colorful_signals",
            "summary img",
        )

        self._bind_arrays()

        self._video_extents = {
            self._video_panels[0]: (0, 0.333, 0.0, 0.5),
            self._video_panels[1]: (0.33, 0.666, 0.0, 0.5),
            self._video_panels[2]: (0.666, 1, 0.0, 0.5),
            self._video_panels[3]: (0.0, 0.333, 0.5, 1.0),
            self._video_panels[4]: (0.333, 0.666, 0.5, 1.0),
            self._video_panels[5]: (0.666, 1, 0.5, 1.0),
        }

        self._ndw_fov = fpl.NDWidget(
            ref_range,
            extents=self._video_extents,
            names=[*self._video_panels],
            controller_ids=[
                tuple(self._video_panels),
            ],
            size=(1200, 1200),
        )

        self._reference_index = self._ndw_fov.indices
        self._panel_graphics = OrderedDict()

        movie_dims = ["time", "m", "n"]
        movie_spatial_dims = ["m", "n"]
        movie_index_mapping = {"time": frame_timings}
        self._pmd_graphic = self._ndw_fov[self._video_panels[0]].add_nd_image(
            self._pmd_array,
            movie_dims,
            movie_spatial_dims,
            slider_dim_transforms=movie_index_mapping.copy(),
            name=self._video_panels[0],
        )
        self._panel_graphics[self._video_panels[0]] = self._pmd_graphic

        if self._ac_array is not None:
            self._ac_graphic = self._ndw_fov[self._video_panels[1]].add_nd_image(
                self._ac_array,
                movie_dims,
                movie_spatial_dims,
                slider_dim_transforms=movie_index_mapping.copy(),
                name=self._video_panels[1],
            )

            self._background_graphic = self._ndw_fov[
                self._video_panels[2]
            ].add_nd_image(
                self._fluctuating_background_array,
                movie_dims,
                movie_spatial_dims,
                slider_dim_transforms=movie_index_mapping.copy(),
                name=self._video_panels[2],
            )

            self._residual_graphic = self._ndw_fov[self._video_panels[3]].add_nd_image(
                self._residual_array,
                movie_dims,
                movie_spatial_dims,
                slider_dim_transforms=movie_index_mapping.copy(),
                name=self._video_panels[3],
            )

            movie_dims_rgb = ["time", "m", "n", "c"]
            movie_spatial_dims_rgb = ["m", "n", "c"]
            self._colorful_signal_graphic = self._ndw_fov[
                self._video_panels[4]
            ].add_nd_image(
                self._colorful_ac_array,
                movie_dims_rgb,
                movie_spatial_dims_rgb,
                slider_dim_transforms=movie_index_mapping.copy(),
                rgb_dim="c",
                name=self._video_panels[4],
            )
            self._panel_graphics[self._video_panels[1]] = self._ac_graphic
            self._panel_graphics[self._video_panels[2]] = self._background_graphic
            self._panel_graphics[self._video_panels[3]] = self._residual_graphic
        else:
            self._ac_graphic = None
            self._background_graphic = None
            self._residual_graphic = None
            self._colorful_signal_graphic = None

        self._own_summary = summary_img is None and self._has_ac
        if summary_img is not None:
            dimension_data = ["m", "n"] if summary_img.ndim == 2 else ["time", "m", "n"]
            self._summary_image = self._ndw_fov[self._video_panels[5]].add_nd_image(
                summary_img,
                dimension_data,
                ["m", "n"],
                name=self._video_panels[5],
            )
            self._ndw_fov.figure[self._video_panels[5]].title = (
                summary_img_name if summary_img_name is not None else "Summary Image"
            )
        elif self._has_ac:
            self._summary_image = self._ndw_fov[self._video_panels[5]].add_nd_image(
                self.demixing_results.global_residual_correlation_image.cpu().numpy(),
                ["m", "n"],
                ["m", "n"],
                name=self._video_panels[5],
            )
            self._ndw_fov.figure[self._video_panels[5]].title = (
                summary_img_name
                if summary_img_name is not None
                else "Residual Correlation Image"
            )
        else:
            self._summary_image = self._ndw_fov[self._video_panels[5]].add_nd_image(
                self._pmd_array.mean_img.cpu().numpy(),
                ["m", "n"],
                ["m", "n"],
                name=self._video_panels[5],
            )
            self._ndw_fov.figure[self._video_panels[5]].title = (
                summary_img_name if summary_img_name is not None else "Mean Image"
            )

        self._panel_graphics[self._video_panels[5]] = self._summary_image
        self._fov_subplot = self._ndw_fov.figure[self._video_panels[5]]

        self._active_component = None
        self._show_masks = show_masks
        self._mask_opacity = mask_opacity
        self._footprints = None
        self._mask_overlays = {}
        if self._has_ac:
            blank = np.zeros((*self._shape[1:3], 4), np.uint8)
            for name in self._video_panels:
                overlay = self._ndw_fov.figure[name].add_image(
                    blank, name="masks", alpha_mode="blend", offset=(0, 0, 0.5)
                )
                # literal RGBA bytes: auto-ranging the all-zero start saturates to white
                overlay.vmin, overlay.vmax = 0, 255
                for tile in overlay.world_object.children:
                    tile.material.pick_write = False
                self._mask_overlays[name] = overlay
            self._make_footprints()

        self._set_gray_cmaps()

        self._traces = TracePlot(("traces",), self._shape[0], frame_timings)
        self._traces.dock(self._ndw_fov.figure, size=360, title="traces")
        self._traces.link(self.reference_index)
        self._traces.on_pick = self._select_signal
        self._base_lines = ("compressed", "background", "residual")
        self._selected_signals = None

        self._pick_selector = None
        self._image_selector = None
        self._show_contours = show_contours
        if self._ac_array is not None:
            self._make_selectors()

        self._rois = OrderedDict()  # PolygonSelector -> {"color": rgb}
        self._active_roi = None
        self._preview_stale = False
        self._last_roi_event = 0.0
        self._status = ""
        self._file_dialog = None

        self._bind_click_handlers()

        for subplot in self._ndw_fov.figure:
            subplot.tooltip.enabled = False
            subplot.toolbar = False

        # "right", not "bottom": NDWidget already docks its own play/pause/slider toolbar
        # at "bottom", and a figure only keeps one window per edge (a second add_imgui_window
        # at the same edge replaces it rather than stacking).
        self._ndw_fov.figure.add_imgui_window(
            self._draw_roi_panel, location="right", size=240, title="ROI Tools"
        )
        if self._has_ac and len(self._footprints):
            self._select_component(0)

    def _make_footprints(self):
        self._footprints = FootprintSet.from_sparse(self._ac_array.a, tuple(self._shape[1:3]))
        self._refresh_masks()

    def _refresh_masks(self):
        if not self._mask_overlays:
            return
        rgba = (
            self._footprints.rgba(
                tuple(self._shape[1:3]), self._mask_opacity, self._active_component
            )
            if self._show_masks
            else None
        )
        for overlay in self._mask_overlays.values():
            overlay.visible = self._show_masks
            if rgba is not None:
                overlay.data = rgba

    def _bind_arrays(self):
        if self._has_ac:
            self._pmd_array = self.demixing_results.pmd_array
            self._fluctuating_background_array = (
                self.demixing_results.fluctuating_background_array
            )
            self._residual_array = self.demixing_results.residual_array
            self._colorful_ac_array = self.demixing_results.colorful_ac_array
            self._ac_array = self.demixing_results.ac_array
        else:
            self._pmd_array = self.demixing_results
            self._fluctuating_background_array = None
            self._residual_array = None
            self._colorful_ac_array = None
            self._ac_array = None

    def _bind_click_handlers(self):
        """Re-attach double-click handlers: NDGraphic.data= replaces the graphic instance."""
        for graphic in self._panel_graphics.values():
            graphic.graphic.add_event_handler(
                partial(self._click_update), "double_click"
            )

    def _set_gray_cmaps(self):
        """NDGraphic.data= replaces the graphic instance, dropping its cmap too."""
        for g in self._panel_graphics.values():
            g.graphic.cmap = "gray"

    def _video_graphics(self):
        """The NDImage wrapper for every video panel, in ``_video_panels`` order."""
        return (
            self._pmd_graphic,
            self._ac_graphic,
            self._background_graphic,
            self._residual_graphic,
            self._colorful_signal_graphic,
            self._summary_image,
        )

    def _make_selectors(self):
        """(Re)build the footprint selectors over the current signals."""
        show = self._show_contours
        if self._image_selector is not None:
            self._set_contours(False)
        # footprint of the signal picked in the trace dock, drawn over the signals movie
        self._pick_selector = fpl.ImageHighlightSelector(
            color="w",
            selection_options={"pixels": self._ac_array.contours},
            options_alpha=0.0,
            alpha=0.6,
        )
        self._pick_selector.add_graphic(self._ac_graphic.graphic)
        # all known footprints, toggled from the roi panel; also drives a picked existing component
        self._image_selector = fpl.ImageHighlightSelector(
            lut="tab10",
            lut_wrap="repeat",
            selection_options={"pixels": self._ac_array.contours},
            options_color="w",
            options_alpha=0.1,
            alpha=0.7,
        )
        self._show_contours = False
        self._set_contours(show)

    def _load_results(self, results: masknmf.DemixingResults):
        """Swap in re-demixed results: every movie panel, the selectors and the summary image follow."""
        results.to(self.device)
        self._demixing_results = results
        self._bind_arrays()
        self._clear_rois()
        self._clear_component()
        self._selected_signals = None
        self._clear_traces()
        self._pmd_graphic.data = self._pmd_array
        self._ac_graphic.data = self._ac_array
        self._background_graphic.data = self._fluctuating_background_array
        self._residual_graphic.data = self._residual_array
        self._colorful_signal_graphic.data = self._colorful_ac_array
        if self._own_summary:
            self._summary_image.data = (
                results.global_residual_correlation_image.cpu().numpy()
            )
        self._bind_click_handlers()
        self._set_gray_cmaps()
        self._make_selectors()
        self._make_footprints()

    def add_to_results(self):
        """
        Run the drawn ROIs through the demixer's NMF pass, appended to the existing signals, and rewrite the
        results file with the outcome. Runs on a thread; the viewer reloads when it finishes.
        """
        if self._ac_array is None or self._source_path is None:
            raise ValueError("adding rois needs demixing results loaded from a file")
        masks = self.roi_masks
        if masks.shape[-1] == 0:
            raise ValueError("no rois have been drawn")
        if self._worker is not None:
            raise RuntimeError("a demixing pass is already running")
        self._status = f"demixing {masks.shape[-1]} roi(s)..."
        self._worker = threading.Thread(
            target=self._demix_rois, args=(masks,), daemon=True
        )
        self._worker.start()

    def _demix_rois(self, masks: np.ndarray):
        try:
            results = add_signals(
                self.demixing_results, masks, self._nmf_config, device=self.device
            )
            replace_results(self._source_path, results)
            self._pending = results
        except Exception as e:
            self._pending = e

    def _poll_worker(self):
        if self._worker is None or self._worker.is_alive():
            return
        self._worker = None
        pending, self._pending = self._pending, None
        if isinstance(pending, Exception):
            self._status = f"add to results failed: {pending}"
            return
        before = self._ac_array.a.shape[1]
        try:
            self._load_results(pending)
        except Exception as e:
            self._status = f"reload after demix failed: {e}"
            return
        self._status = (
            f"{pending.a.shape[1]} signals (was {before}) written to "
            f"{os.path.basename(self._source_path)}"
        )

    def _select_signal(self, panel: str, index: int):
        """Highlight the footprint of the demixed signal whose line was double-clicked."""
        index -= len(self._base_lines)
        if self._selected_signals is None or not 0 <= index < len(
            self._selected_signals
        ):
            self._pick_selector.selection = []
            return
        self._pick_selector.selection = [int(self._selected_signals[index])]

    def _click_update(self, ev: pygfx.PointerEvent):
        """Priority: a drawn roi, then an existing component, else clear the traces."""
        if self._drawing() or imgui.get_io().want_capture_mouse:
            return
        col, row = ev.pick_info["index"]

        roi = self._roi_at(col, row)
        if roi is not None:
            self._select_roi(roi)
            return

        if self._ac_array is not None:
            component = component_at_pixel(
                self._ac_array.a, self._ac_array.centers, self._shape[1:], (col, row)
            )
            if component is not None:
                self._select_component(component)
                return

        self._clear_component()
        self._active_roi = None
        self._selected_signals = None
        if self._pick_selector is not None:
            self._pick_selector.selection = []
        self._clear_traces()

    @property
    def _base_graphic(self):
        return self._summary_image.graphic

    def _select_roi(self, selector):
        self._clear_component()
        self._active_roi = selector
        self._preview_stale = True
        self._last_roi_event = 0.0

    def _select_component(self, component: int):
        self._active_roi = None
        self._preview_stale = False
        self._active_component = int(component)
        if self._image_selector is not None:
            self._image_selector.selection = [self._active_component]
        self._refresh_masks()
        fields = (
            "pmd_roi_averages",
            "fluctuating_background_roi_averages",
            "residual_roi_averages",
        )
        lines = list(
            zip(
                self._base_lines,
                (
                    getattr(self.demixing_results, f)[component].cpu().numpy()
                    for f in fields
                ),
                _BASE_LINE_COLORS,
            )
        )
        self._traces.set("traces", lines)

    def _clear_component(self):
        if self._active_component is not None:
            self._active_component = None
            if self._image_selector is not None:
                self._image_selector.selection = []
            self._refresh_masks()

    def _set_contours(self, show: bool):
        self._show_contours = show
        if self._image_selector is None:
            return
        for g in self._video_graphics():
            if g is None:
                continue
            graphic = g.graphic
            attached = graphic in self._image_selector.graphics
            if show and not attached:
                self._image_selector.add_graphic(graphic)
            elif not show and attached:
                self._image_selector.remove_graphic(graphic)

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
        self._rois[selector] = {"color": color}
        self._active_roi = selector
        self._clear_component()

    def _roi_changed(self, selector, ev):
        self._active_roi = selector
        self._clear_component()
        self._preview_stale = True
        self._last_roi_event = time.perf_counter()

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

    def _update_preview(self):
        self._preview_stale = False
        selector = self._active_roi
        if selector is None:
            return
        indices = selector.get_selected_indices(self._base_graphic)
        if indices.shape[0] == 0:
            return
        lines = [
            (
                "compressed",
                self._roi_average(self._pmd_array, indices),
                _BASE_LINE_COLORS[0],
            )
        ]
        if self._ac_array is not None:
            lines.append(
                (
                    "background",
                    self._roi_average(self._fluctuating_background_array, indices),
                    _BASE_LINE_COLORS[1],
                )
            )
            lines.append(
                (
                    "residual",
                    self._roi_average(self._residual_array, indices),
                    _BASE_LINE_COLORS[2],
                )
            )
        self._traces.set("traces", lines)

    def _clear_traces(self):
        self._traces.clear()

    @property
    def roi_masks(self) -> np.ndarray:
        """The drawn ROIs as a binary mask stack of shape (fov dim1, fov dim2, num_rois)"""
        shape = tuple(self._shape[1:3])
        masks = []
        for selector in self._rois:
            indices = selector.get_selected_indices(self._base_graphic)
            if indices.shape[0] == 0:
                continue
            mask = np.zeros(shape, dtype=np.float32)
            mask[indices[:, 1], indices[:, 0]] = 1.0
            masks.append(mask)
        if not masks:
            return np.zeros((*shape, 0), dtype=np.float32)
        return np.stack(masks, axis=-1)

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
        (fov dim1, fov dim2, num_rois) mask stack for a custom demixing initialization and,
        when demixing results are loaded, 'spatial_footprints_combined' appends the drawn
        ROIs to the existing signals so SignalDemixer.initialize_signals(is_custom=True)
        re-demixes with the drawn ROIs added to the results.
        """
        masks = self.roi_masks
        if masks.shape[-1] == 0:
            raise ValueError("no rois have been drawn")
        path = str(path)
        if not path.endswith(".npz"):
            path += ".npz"
        data = dict(spatial_footprints=masks)
        if self._ac_array is not None:
            data["spatial_footprints_combined"] = self._append_to_signals(masks)
        np.savez_compressed(path, **data)
        return path

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
        if (
            imgui.is_key_pressed(imgui.Key.delete, False)
            and self._active_roi is not None
        ):
            self._delete_roi(self._active_roi)

    def _selection_status(self) -> str:
        if self._active_component is not None:
            return f"signal {self._active_component} selected"
        if self._active_roi in self._rois:
            return f"roi {list(self._rois).index(self._active_roi)} selected"
        return "double-click a mask or roi to see its trace"

    def _draw_roi_panel(self):
        """A narrow, vertically-stacked sidebar (docked at "right") so it stays out of the
        NDWidget playback toolbar's way at "bottom"."""
        self._poll_file_dialog()
        self._poll_worker()
        self._handle_keys()
        drawing = self._drawing()

        existing = len(self._footprints) if self._footprints is not None else 0
        imgui.text_disabled(
            f"{existing + len(self._rois)} roi(s) total"
            f" ({existing} existing, {len(self._rois)} drawn)"
        )
        imgui.separator()

        if self._image_selector is not None:
            changed, show = imgui.checkbox("masks", self._show_masks)
            if changed:
                self._show_masks = show
                self._refresh_masks()
            imgui.same_line()
            changed, show = imgui.checkbox("contours", self._show_contours)
            if changed:
                self._set_contours(show)
            imgui.set_next_item_width(-1)
            changed, self._mask_opacity = imgui.slider_float(
                "##mask-opacity", self._mask_opacity, 0.05, 1.0, "opacity %.2f"
            )
            if changed and self._show_masks:
                self._refresh_masks()

        imgui.begin_disabled(drawing)
        if imgui.button("Add ROI", imgui.ImVec2(-1, 0)):
            self._start_roi()
        imgui.end_disabled()

        imgui.begin_disabled(self._active_roi is None)
        if imgui.button("Delete ROI", imgui.ImVec2(-1, 0)):
            self._delete_roi(self._active_roi)
        imgui.end_disabled()

        imgui.begin_disabled(not self._rois)
        if imgui.button("export rois", imgui.ImVec2(-1, 0)):
            self._browse_export()
        imgui.end_disabled()

        imgui.begin_disabled(
            not self._rois
            or self._ac_array is None
            or self._source_path is None
            or self._worker is not None
        )
        if imgui.button("Demix", imgui.ImVec2(-1, 0)):
            self.add_to_results()
        imgui.end_disabled()
        if imgui.is_item_hovered(imgui.HoveredFlags_.allow_when_disabled):
            imgui.set_tooltip(
                "demix the drawn rois with the existing signals and rewrite the results file"
                if self._source_path is not None
                else "open the results with source_path to enable"
            )

        changed, filter_dim = imgui.checkbox(
            "filter dim rois", self._nmf_config.min_brightness is not None
        )
        if changed:
            if filter_dim:
                self._nmf_config.min_brightness = self._min_brightness_cache
            else:
                self._min_brightness_cache = (
                    self._nmf_config.min_brightness or self._min_brightness_cache
                )
                self._nmf_config.min_brightness = None
        if imgui.is_item_hovered():
            imgui.set_tooltip(
                "delete signals that never get bright enough during the nmf pass. "
                "turn off if a hand-drawn roi keeps disappearing from add to results."
            )

        imgui.push_text_wrap_pos(0)
        if drawing:
            imgui.text_disabled("click to add points; click the first point to close")
        else:
            imgui.text_disabled(f"{len(self._rois)} roi(s)  {self._status}")
        imgui.pop_text_wrap_pos()

        imgui.separator()
        imgui.push_text_wrap_pos(0)
        imgui.text_disabled(self._selection_status())
        imgui.pop_text_wrap_pos()

        if (
            self._preview_stale
            and not drawing
            and (time.perf_counter() - self._last_roi_event > _PREVIEW_DELAY)
        ):
            self._update_preview()

    @property
    def device(self) -> str:
        return self._device

    @property
    def demixing_results(self) -> masknmf.DemixingResults | masknmf.PMDArray:
        return self._demixing_results

    @property
    def fov_widget(self) -> fpl.NDWidget:
        return self._ndw_fov

    @property
    def traces(self) -> TracePlot:
        return self._traces

    @property
    def reference_index(self) -> ReferenceIndex:
        return self._reference_index

    def show(self):
        return self.fov_widget.show()

    def close(self):
        self._ndw_fov.close()


def visualize_superpixels_peaks(init_results: masknmf.InitializationResults):
    superpixel_map = init_results.nmf_seed_map
    pure_superpixel_map = init_results.pure_nmf_seed_map
    correlation_image = init_results.correlation_img

    superpixel_img = np.stack([correlation_image.copy()] * 3, axis=-1)
    superpixel_img[superpixel_map > 0] = [4, 0, 0]

    pure_superpixel_img = np.stack([correlation_image.copy()] * 3, axis=-1)
    pure_superpixel_img[pure_superpixel_map > 0] = [4, 0, 0]

    corr_rgb = np.stack([correlation_image] * 3, axis=-1)

    image_panels = ("corr image", "nmf seed map", "pure nmf seed map")

    extents = {
        image_panels[0]: (0, 0.333, 0.0, 1),
        image_panels[1]: (0.33, 0.666, 0.0, 1),
        image_panels[2]: (0.666, 1, 0.0, 1),
    }

    ndw_corr = fpl.NDWidget(
        extents=extents,
        names=[*image_panels],
        controller_ids=[
            tuple(image_panels),
        ],
        size=(1200, 1200),
    )

    corr_img_graphic = ndw_corr[image_panels[0]].add_nd_image(
        corr_rgb, ["m", "n", "c"], ["m", "n", "c"], rgb_dim="c", name=image_panels[0]
    )
    nmf_seed_graphic = ndw_corr[image_panels[1]].add_nd_image(
        superpixel_img,
        ["m", "n", "c"],
        ["m", "n", "c"],
        rgb_dim="c",
        name=image_panels[1],
    )
    pure_seed_graphic = ndw_corr[image_panels[2]].add_nd_image(
        pure_superpixel_img,
        ["m", "n", "c"],
        ["m", "n", "c"],
        rgb_dim="c",
        name=image_panels[2],
    )

    return ndw_corr.show()
