import os
import threading
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
    RoiOrder,
    TracePlot,
    component_at_pixel,
    draw_keybinds_popup,
    draw_range_filter,
    draw_roi_table,
    em,
    resolve_time_reference,
)
from masknmf.visualization.rois import FootprintSet
from masknmf.demixing import update_signals, replace_results
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
_SIGNAL_COLUMNS = ("id", "area", "peak", "del")
_KEYBINDS = (
    ("up / down", "previous / next signal in the table (shift: by 10)"),
    ("ctrl + click", "toggle a signal in the group, in the image or the table"),
    ("shift + click", "add a signal to the group; in the table, every row up to it"),
    ("esc", "empty the group"),
    ("f", "center the view on the selection and keep following it"),
    ("delete", "remove the selected roi, or mark the selected signal for deletion"),
    ("k", "show these keybinds"),
)
# compressed/background/residual, in that order, so the 3 base lines read apart in the legend
_BASE_LINE_COLORS = ((0.85, 0.85, 0.85), (0.95, 0.55, 0.15), (0.35, 0.65, 0.95))


class SingleSessionDemixingVis:
    """
    View and curate demixing results. Can be used whether demixing has been ran (pass in DemixingResults) or not (PMDArray).
    existing masknmf.DemixingResults (or a bare PMDArray before demixing has run), and draw
    and export ROIs for a custom SignalDemixer.initialize_signals(is_custom=True) pass.
    Footprints show as feathered masks and/or contours over the summary image.

    With ``source_path`` set, "Demix" runs the drawn ROIs and the signals marked with "Delete" through the
    demixer's NMF pass (``nmf_config``, the pipeline defaults when None) and rewrites that file: drawn ROIs
    become ordinary signals, marked signals are gone.

    The "Signals" tab lists every demixed signal; ctrl / shift select a group whose traces share the plot.

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
        mb = self._nmf_config.min_brightness
        self._min_brightness_cache = 1.0 if mb is None else mb
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
        self._marked = set()  # signal indices "Delete" has marked; removed on the next "Demix"
        self._group: list = []  # signals selected together; their traces share the plot
        self._order = None  # RoiOrder over the signals, built with the footprints
        self._follow = False
        self._scroll_to_current = False
        self._keybinds_open = False
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
        self._selected_signals = None  # the signal behind each plotted line, when lines are signals

        self._image_selector = None
        self._show_contours = show_contours
        if self._ac_array is not None:
            self._make_selectors()

        self._rois = OrderedDict()  # PolygonSelector -> {"color": rgb}
        self._active_roi = None
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
            self._draw_side_panel, location="right", size=300, title="Tools"
        )
        if self._has_ac and len(self._footprints):
            self._select_component(0)

    def _make_footprints(self):
        self._footprints = FootprintSet.from_sparse(self._ac_array.a, tuple(self._shape[1:3]))
        peaks = self.demixing_results.c.max(dim=0).values.cpu().numpy()
        self._order = RoiOrder({"area": self._footprints.areas, "peak": peaks}, len(self._footprints))
        self._order.set_range_column("area")
        self._refresh_masks()

    def _refresh_masks(self):
        if not self._mask_overlays:
            return
        rgba = (
            self._footprints.rgba(
                tuple(self._shape[1:3]),
                self._mask_opacity,
                self._active_component,
                self._marked,
                self._group,
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
        # all known footprints, toggled from the roi panel; also drives the selected and grouped components
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
        self._marked.clear()
        self._group.clear()
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

    def demix(self):
        """
        Run the drawn ROIs (appended) and the marked signals (removed) through the demixer's NMF pass and
        rewrite the results file with the outcome. Runs on a thread; the viewer reloads when it finishes.
        """
        if self._ac_array is None or self._source_path is None:
            raise ValueError("editing signals needs demixing results loaded from a file")
        masks = self.roi_masks
        drop = sorted(self._marked)
        if masks.shape[-1] == 0 and not drop:
            raise ValueError("no rois drawn and no signals marked for deletion")
        if self._worker is not None:
            raise RuntimeError("a demixing pass is already running")
        self._status = f"demixing: +{masks.shape[-1]} roi(s), -{len(drop)} signal(s)..."
        self._worker = threading.Thread(
            target=self._demix, args=(masks, drop), daemon=True
        )
        self._worker.start()

    def _demix(self, masks: np.ndarray, drop: list):
        try:
            results = update_signals(
                self.demixing_results,
                masks,
                drop,
                self._nmf_config,
                device=self.device,
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
            self._status = f"demix failed: {pending}"
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
        """Select the signal whose line was double-clicked in the trace dock."""
        if self._selected_signals is not None and 0 <= index < len(self._selected_signals):
            self._select_component(self._selected_signals[index])

    def _click_update(self, ev: pygfx.PointerEvent):
        """
        Priority: a drawn roi, then an existing component, else clear the selection.
        ctrl / shift on a component grow the group instead of replacing the selection.
        """
        if self._drawing() or imgui.get_io().want_capture_mouse:
            return
        col, row = ev.pick_info["index"]
        mods = set(getattr(ev, "modifiers", ()) or ())

        roi = self._roi_at(col, row)
        if roi is not None:
            self._select_roi(roi)
            return

        if self._ac_array is not None:
            component = component_at_pixel(
                self._ac_array.a, self._ac_array.centers, self._shape[1:], (col, row)
            )
            if component is not None:
                if mods & {"Control", "Ctrl"}:
                    self.group_toggle(component)
                elif "Shift" in mods:
                    self.group_add(component)
                else:
                    self.group_clear()
                    self._select_component(component)
                return

        self.group_clear()
        self._clear_component()
        self._active_roi = None
        self._selected_signals = None
        self._clear_traces()

    @property
    def _base_graphic(self):
        return self._summary_image.graphic

    def _select_roi(self, selector):
        self.group_clear()
        self._clear_component()
        self._active_roi = selector
        self._clear_traces()

    def _select_component(self, component: int):
        self._active_roi = None
        self._active_component = int(component)
        if self._order is not None:
            if self._order.reveal(self._active_component):
                self._status = "area filter widened to show the selection"
            self._scroll_to_current = True
        if self._follow:
            self._center_on(self._active_component)
        self._sync_highlight()
        self._update_traces()

    def _clear_component(self):
        if self._active_component is not None:
            self._active_component = None
            self._sync_highlight()

    def _highlighted(self) -> list:
        picks = list(self._group)
        if self._active_component is not None and self._active_component not in picks:
            picks.append(self._active_component)
        return picks

    def _sync_highlight(self):
        """The contour selector and the mask overlay both show the group plus the selection."""
        if self._image_selector is not None:
            self._image_selector.selection = self._highlighted()
        self._refresh_masks()

    def _update_traces(self):
        """
        One signal: its compressed / background / residual roi averages. A group: every
        member's compressed roi average, colored like its mask.
        """
        results = self.demixing_results
        if len(self._group) > 1:
            self._selected_signals = list(self._group)
            averages = results.pmd_roi_averages
            lines = [
                (f"signal {k}", averages[k].cpu().numpy(), self._footprints.color(k))
                for k in self._selected_signals
            ]
        elif self._active_component is not None:
            k = self._active_component
            self._selected_signals = None
            fields = (
                "pmd_roi_averages",
                "fluctuating_background_roi_averages",
                "residual_roi_averages",
            )
            lines = list(
                zip(
                    self._base_lines,
                    (getattr(results, f)[k].cpu().numpy() for f in fields),
                    _BASE_LINE_COLORS,
                )
            )
        else:
            self._selected_signals = None
            self._clear_traces()
            return
        self._traces.set("traces", lines)

    def _seed_group(self):
        """A first ctrl or shift pick keeps the current selection in the group."""
        if not self._group and self._active_component is not None:
            self._group.append(self._active_component)

    def group_add(self, component: int):
        self._seed_group()
        if component not in self._group:
            self._group.append(int(component))
        self._select_component(component)

    def group_toggle(self, component: int):
        self._seed_group()
        if component in self._group:
            self._group.remove(component)
        else:
            self._group.append(int(component))
        self._select_component(component)

    def group_extend_to(self, component: int):
        """Add every table row between the cursor and ``component`` to the group."""
        if self._order is None:
            return
        self._seed_group()
        order = list(self._order.order)
        current = self._order.current
        if component not in order or current not in order:
            self.group_add(component)
            return
        start, stop = order.index(current), order.index(component)
        for k in order[min(start, stop) : max(start, stop) + 1]:
            if k not in self._group:
                self._group.append(int(k))
        self._select_component(component)

    def group_clear(self):
        if self._group:
            self._group.clear()
            self._sync_highlight()
            self._update_traces()

    def _center_on(self, component: int):
        """Frame every video panel on one footprint with some context around it."""
        ypix, xpix, _lam = self._footprints.footprints[component]
        if not len(ypix):
            return
        y0, y1 = float(ypix.min()), float(ypix.max())
        x0, x1 = float(xpix.min()), float(xpix.max())
        cy, cx = (y0 + y1) / 2, (x0 + x1) / 2
        half = max(max(y1 - y0, x1 - x0, 1.0) * 2.0, 40.0)
        for name in self._video_panels:
            self._ndw_fov.figure[name].camera.show_rect(cx - half, cx + half, cy - half, cy + half)

    def _toggle_follow(self):
        self._follow = not self._follow
        if self._follow and self._active_component is not None:
            self._center_on(self._active_component)

    def _step(self, delta: int):
        """Move the table cursor and select what it lands on."""
        if self._order is not None and self._order.step(delta):
            self.group_clear()
            self._select_component(self._order.current)

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

    def _delete_roi(self, selector):
        if selector._move_info.mode is not None:
            selector._end_move_mode()
        self._fov_subplot.delete_graphic(selector)
        del self._rois[selector]
        if self._active_roi is selector:
            self._active_roi = next(reversed(self._rois), None)

    def _clear_rois(self):
        for selector in list(self._rois):
            self._delete_roi(selector)

    def _toggle_marked(self, component: int):
        """Mark a signal for deletion on the next demix, or unmark it."""
        self._marked.symmetric_difference_update({int(component)})
        self._refresh_masks()

    def _delete_selected(self):
        """The Delete action: drop an active drawn roi, else toggle the active signal's mark."""
        if self._active_roi is not None:
            self._delete_roi(self._active_roi)
        elif self._active_component is not None:
            self._toggle_marked(self._active_component)

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
        io = imgui.get_io()
        if io.want_text_input:
            return
        if imgui.is_key_pressed(imgui.Key.delete, False):
            self._delete_selected()
        if imgui.is_key_pressed(imgui.Key.escape, False):
            self.group_clear()
        stride = 10 if io.key_shift else 1
        if imgui.is_key_pressed(imgui.Key.down_arrow, True):
            self._step(stride)
        if imgui.is_key_pressed(imgui.Key.up_arrow, True):
            self._step(-stride)
        if imgui.is_key_pressed(imgui.Key.f, False):
            self._toggle_follow()
        if imgui.is_key_pressed(imgui.Key.k, False):
            self._keybinds_open = not self._keybinds_open

    def _selection_status(self) -> str:
        if len(self._group) > 1:
            return f"{len(self._group)} signals grouped: {sorted(self._group)}"
        if self._active_component is not None:
            marked = " (marked for deletion)" if self._active_component in self._marked else ""
            return f"signal {self._active_component} selected{marked}"
        if self._active_roi in self._rois:
            return f"roi {list(self._rois).index(self._active_roi)} selected"
        return "double-click a mask or roi to see its trace"

    def _draw_side_panel(self):
        """Docked at "right" (the NDWidget owns "bottom"): the roi tools and the signal table as tabs."""
        self._poll_file_dialog()
        self._poll_worker()
        self._handle_keys()
        if imgui.begin_tab_bar("##side"):
            if imgui.begin_tab_item("ROI Tools")[0]:
                self._draw_roi_tools()
                imgui.end_tab_item()
            if imgui.begin_tab_item("Signals")[0]:
                self._draw_signal_tab()
                imgui.end_tab_item()
            imgui.end_tab_bar()
        self._keybinds_open = draw_keybinds_popup(_KEYBINDS, self._keybinds_open)

    def _table_select(self, component: int):
        self.group_clear()
        self._select_component(component)

    def _format_cell(self, name: str, component: int) -> str:
        if name == "del":
            return "x" if component in self._marked else ""
        value = self._order.columns[name][component]
        return f"{int(value)}" if name == "area" else f"{float(value):.3g}"

    def _draw_signal_tab(self):
        if self._order is None:
            imgui.text_disabled("no demixed signals")
            return
        if draw_range_filter(self._order, "_signals"):
            self._order.rebuild()
        imgui.text_disabled(f"{len(self._order.order)}/{self._order.n_items} in view")
        changed, self._follow = imgui.checkbox("center on selection", self._follow)
        if changed and self._follow and self._active_component is not None:
            self._center_on(self._active_component)
        imgui.same_line(0, em(0.3))
        imgui.text_disabled("(f)")
        imgui.same_line(0, em(0.8))
        if imgui.small_button("keys"):
            self._keybinds_open = not self._keybinds_open
        footer = imgui.get_frame_height_with_spacing() * 2.5
        if imgui.begin_child("##signal_table", imgui.ImVec2(0, -footer)):
            formatters = {name: partial(self._format_cell, name) for name in _SIGNAL_COLUMNS[1:]}
            self._scroll_to_current = draw_roi_table(
                self._order,
                _SIGNAL_COLUMNS,
                formatters,
                self._scroll_to_current,
                table_id="signals",
                on_select=self._table_select,
                is_grouped=self._group.__contains__,
                on_ctrl_select=self.group_toggle,
                on_shift_select=self.group_extend_to,
                row_color=self._footprints.color,
            )
        imgui.end_child()
        imgui.separator()
        if len(self._group) > 1:
            if imgui.small_button("ungroup"):
                self.group_clear()
            imgui.same_line(0, em(0.3))
            imgui.text_disabled("(esc)")
            imgui.same_line(0, em(0.8))
        imgui.push_text_wrap_pos(0)
        imgui.text_disabled(self._selection_status())
        imgui.pop_text_wrap_pos()

    def _draw_roi_tools(self):
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

        imgui.begin_disabled(self._active_roi is None and self._active_component is None)
        label = (
            "Unmark signal"
            if self._active_component is not None and self._active_component in self._marked
            else "Delete"
        )
        if imgui.button(label, imgui.ImVec2(-1, 0)):
            self._delete_selected()
        imgui.end_disabled()
        if imgui.is_item_hovered(imgui.HoveredFlags_.allow_when_disabled):
            imgui.set_tooltip(
                "remove the selected drawn roi, or mark the selected signal for deletion "
                "on the next demix (press again to unmark)"
            )

        imgui.begin_disabled(not self._rois)
        if imgui.button("export rois", imgui.ImVec2(-1, 0)):
            self._browse_export()
        imgui.end_disabled()

        imgui.begin_disabled(
            (not self._rois and not self._marked)
            or self._ac_array is None
            or self._source_path is None
            or self._worker is not None
        )
        if imgui.button("Demix", imgui.ImVec2(-1, 0)):
            self.demix()
        imgui.end_disabled()
        if imgui.is_item_hovered(imgui.HoveredFlags_.allow_when_disabled):
            imgui.set_tooltip(
                "re-demix: add the drawn rois, remove the marked signals, rewrite the results file"
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
                self._min_brightness_cache = self._nmf_config.min_brightness
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
            imgui.text_disabled(
                f"{len(self._rois)} roi(s), {len(self._marked)} marked  {self._status}"
            )
        imgui.pop_text_wrap_pos()

        imgui.separator()
        imgui.push_text_wrap_pos(0)
        imgui.text_disabled(self._selection_status())
        imgui.pop_text_wrap_pos()

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
