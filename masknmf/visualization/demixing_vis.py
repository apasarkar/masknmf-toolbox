import os
import threading
from pathlib import Path
from dataclasses import replace
from typing import *
import numpy as np
import fastplotlib as fpl
from imgui_bundle import imgui, icons_fontawesome_6 as fa, portable_file_dialogs as pfd
from fastplotlib import ui
from fastplotlib.graphics.selectors._polygon import point_in_polygon
import pygfx
import h5py
import torch
from collections import OrderedDict
import masknmf.arrays
from masknmf.arrays import TiffArray
from masknmf.utils import display
from functools import partial
from masknmf.visualization.imgui import (
    THEME,
    RoiOrder,
    TracePlot,
    component_at_pixel,
    draw_keybinds_popup,
    draw_path_popup,
    draw_range_filter,
    draw_roi_table,
    em,
    resolve_time_reference,
    section,
    to_vec4,
    grid,
    help_mark,
    right_aligned_text,
    button_colors,
)
from masknmf.visualization.rois import MARKED_COLOR, FootprintSet
from masknmf.demixing import CellStats, update_signals, write_curated
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
_STATS_FILTERS = ["Cell stats", "*.npy *.npz *.csv *.tsv *.txt", "All files", "*"]
_CLICK_SLOP = (
    4  # px the pointer may travel between press and release and still be a click
)
_UNDO_DEPTH = 50  # ctrl+z snapshots kept
# every grid's captions, so the caption column is one width across the sections and the tabs
_CAPTIONS = (
    "masks", "contours", "color by", "traces",
    "rois", "on disk", "polygon", "side", "run", "options",
    "filter", "in view", "selection", "merge", "view", "stats",
)
# signals selected together, in order of mutual contrast on the dark plot; no red, a mask marked for
# deletion is red
_GROUP_COLORS = (
    (1.00, 0.55, 0.10),
    (0.25, 0.85, 0.35),
    (0.95, 0.35, 0.90),
    (0.35, 0.80, 1.00),
    (1.00, 0.95, 0.35),
    (0.65, 0.50, 1.00),
    (1.00, 1.00, 1.00),
)
# TODO: should abstract keybinds out of the curation widget, where most keybinds live
# They share a lot of common functionality with demixing vis
_KEYBINDS = (
    ("up / down", "previous / next signal in the table (shift: by 10)"),
    (
        "click",
        "on an empty pixel: add its 5x5 pixel average to the plot as if grouped; on a drawn roi: plot its average alone",
    ),
    (
        "ctrl + click",
        "toggle a signal, drawn roi or pixel average in the group, in the image or the table",
    ),
    (
        "shift + click",
        "add a signal or drawn roi to the group; in the table, every row up to it",
    ),
    ("esc", "cancel a new roi, stop a poly-select (the selection stays), else deselect everything and drop the pixel averages"),
    ("ctrl + a", "group every signal the table shows"),
    ("ctrl + z", "undo the last mark, drawn roi, pixel average or deselect"),
    ("f", "center the view on the selection and keep following it"),
    (
        "p",
        "toggle quick pixel trace: a click on an empty pixel adds its 5x5 average to the plot",
    ),
    (
        "delete",
        "remove the selected roi, drop the active pixel average, or mark the selected signals for deletion (unmark when all are)",
    ),
    ("shift / alt + scroll", "in the trace plot, zoom x only / y only"),
    ("k", "show these keybinds"),
)
# compressed/signal/background/residual, in that order, so the 4 base lines read apart in the legend
_BASE_LINE_COLORS = (
    (0.85, 0.85, 0.85),
    (0.30, 0.85, 0.40),
    (0.95, 0.55, 0.15),
    (0.35, 0.65, 0.95),
)


class SingleSessionDemixingVis:
    """
    View and curate demixing results. Can be used whether demixing has been ran (pass in DemixingResults) or not (PMDArray).
    existing masknmf.DemixingResults (or a bare PMDArray before demixing has run), and draw
    and export ROIs for a custom SignalDemixer.initialize_signals(is_custom=True) pass.
    Footprints show as feathered masks and/or contours over the summary image.

    With ``results_path`` set, "Demix" runs the drawn ROIs and the signals marked with "Delete" through the
    demixer's NMF pass (``nmf_config``, the pipeline defaults when None) and writes the outcome to a new
    ``<timestamp>.curated.hdf5`` beside the file, never over it: drawn ROIs become ordinary signals,
    marked signals are gone, the new file's description says what was done, and the viewer moves on to it so
    further passes chain.

    The "Signals" tab lists every demixed signal; ctrl / shift select a group whose traces share the plot.
    Traces plot only with the Curation tab's "show selected traces" on (off by default): selecting then only
    highlights, however big the selection.
    Clicking the selected mask, its trace or its table row again deselects it; a pan or drag on a panel
    leaves the selection alone. Esc deselects everything, ctrl+a groups every signal the table shows, and
    ctrl+z undoes the last mark, drawn roi, pixel average or deselect (a Demix empties the undo stack; roi
    vertex drags are not undone). The selection's contour always shows; the Overlay "contours" checkbox
    adds every other footprint's.
    With "pixel traces" on (Curation tab checkbox or the p key, off by default), clicking an empty pixel adds the compressed movie's 5x5
    average there to the plot as if it were a grouped signal, and lists it at the top of the Signals table,
    marked. Pixel averages are diagnostic only: Demix and export ignore them, Delete drops them.
    A drawn roi gets the same kind of trace once it is closed ("roi n", groupable, in the table too) and,
    unlike a pixel average, is kept: Demix seeds the NMF pass with it and export writes it.
    The filter above the table takes any column, del included (0 or 1), and "delete in view" marks every
    signal it shows, as Delete does one at a time. The Curation tab's "color by" colors the masks and the
    table's ids by a column's rank instead of by signal id.
    poly-select (the Curation tab's polygon button) draws a polygon on any panel that selects every signal in
    view whose center falls inside it (or outside, per the toggle): they form the group, highlighted in the
    panels and the table, and the selection follows the polygon as it is drawn and later dragged (its traces,
    when shown, plot once it settles). Clicking the button again or esc leaves the mode and keeps the selection, so
    Delete (or the Signals tab's delete button) marks it like any other selection. Nothing is removed until the
    next Demix. line-select is a placeholder, not implemented yet.

    ``raw`` (a movie, or a .tif path) shows the raw movie in place of the signals panel and ``shifts`` (an
    array, or a motion correction hdf5 path) adds the registration shifts as a panel above the traces
    (piecewise rigid: the largest block shift per frame). With ``results_path`` set, a lone .tif beside the
    results, and the registration shifts from the results file itself or from a motion_correction.hdf5 beside
    it, are picked up when their frames match the results; given ones must match.

    The Signals table always carries the results' own stats (:meth:`CellStats.from_results`: mean, std, snr
    and skew of each demixed trace; fit, resid and bkgd from the roi averages the results hold), hidden until
    the Signals tab's "columns" button shows them; click a header to order the signals by a stat, then step
    through the top or bottom of the order. ``cell_stats`` (a :class:`CellStats`, or a .npy / .npz / .csv /
    .tsv it reads, one row per signal) joins more columns, shown, replacing same-named ones. ``cell_order``
    (signal ids in a custom order, or a
    .npy / text file of them) adds an "order" column of ranks and opens the table in that order; signals it
    leaves out sort last. :meth:`load_cell_order` and :meth:`add_cell_stats` (or the Signals tab's "load stats"
    button: a .txt of ids is an order, anything else stats) do the same with the results open. "load stats"
    and "Export" open a window with a typed path, so they work on a remote kernel; "browse" there is the
    native dialog for a local one. Marked signals always come first. A Demix pass recomputes the results'
    stats for the new signals and drops given ones.

    TODO:
    -----
    - Could really support registration arrays?

    """

    def __init__(
        self,
        demixing_results: masknmf.DemixingResults
        | masknmf.CompressionArray
        | List[masknmf.DemixingResults],
        frame_timings: Optional[np.ndarray | List[np.ndarray]] = None,
        ref_range: Optional[dict] = None,
        summary_img: np.ndarray | masknmf.ArrayLike | None = None,
        summary_img_name: str | None = None,
        show_contours: bool = False,
        show_masks: bool = True,
        mask_opacity: float = 0.5,
        device="cpu",
        results_path: str | os.PathLike | None = None,
        nmf_config: NMFConfig | None = None,
        raw: masknmf.ArrayLike | np.ndarray | str | os.PathLike | None = None,
        shifts: np.ndarray | torch.Tensor | str | os.PathLike | None = None,
        cell_stats: CellStats | str | os.PathLike | None = None,
        cell_order: Sequence[int] | np.ndarray | str | os.PathLike | None = None,
    ):
        self._results_path = None if results_path is None else str(results_path)
        base = NMFConfig() if nmf_config is None else nmf_config
        self._min_brightness_cache = (
            1.0 if base.min_brightness is None else base.min_brightness
        )
        # "filter dim rois" starts off: a hand-drawn roi that never gets bright would vanish from the pass
        self._nmf_config = replace(base, min_brightness=None)
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

        folder = None if self._results_path is None else Path(self._results_path).parent
        num_signals = demixing_results.spatial_demixed.shape[1] if self._has_ac else 0
        # the results' own stats, hidden; given stats join them, shown, replacing same-named columns
        self._cell_stats = CellStats.from_results(demixing_results) if self._has_ac else None
        self._shown_stats = set()
        if isinstance(cell_stats, (str, os.PathLike)):
            cell_stats = CellStats.read(cell_stats)
        if cell_stats is not None:
            if cell_stats.values.shape[0] != num_signals:
                raise ValueError(f"{cell_stats.values.shape[0]} cell stat rows for {num_signals} signals")
            if set(cell_stats.names) & {"id", "area", "peak", "del"}:
                raise ValueError(f"cell stat names clash with the table's own columns: {cell_stats.names}")
            self._cell_stats = cell_stats if self._cell_stats is None else self._cell_stats.join(cell_stats)
            self._shown_stats = set(cell_stats.names)
        if self._cell_stats is not None:
            display(f"cell stats: {', '.join(self._cell_stats.names)}; the Signals tab's columns button shows them")

        # raw movie and shifts: data or a path, or found beside the results; a found mismatch is skipped, a given one raises
        found_raw = found_shifts = False
        if raw is None and folder is not None:
            tifs = sorted(p for ext in ("*.tif", "*.tiff") for p in folder.glob(ext))
            if len(tifs) == 1:
                raw, found_raw = tifs[0], True
        raw_src = raw if isinstance(raw, (str, os.PathLike)) else None
        if raw_src is not None:
            try:
                raw = TiffArray(str(raw_src), memmap=True)
            except (ValueError, TypeError):
                raw = TiffArray(str(raw_src))
        if raw is not None and tuple(raw.shape) != tuple(self._shape):
            if not found_raw:
                raise ValueError(
                    f"raw movie has shape {tuple(raw.shape)}, the results have {tuple(self._shape)}"
                )
            display(
                f"skipping {raw_src}: shape {tuple(raw.shape)} does not match the results"
            )
            raw = None
        if shifts is None and folder is not None:
            # the results file itself when the pipeline wrote every stage to it, else the old separate file
            for candidate in (Path(self._results_path), folder / "motion_correction.hdf5"):
                if candidate.is_file():
                    with h5py.File(candidate, "r") as f:
                        found_shifts = "PiecewiseRigidRegistrationArray" in f or "RigidRegistrationArray" in f
                    if found_shifts:
                        shifts = candidate
                        break
        shifts_src = shifts if isinstance(shifts, (str, os.PathLike)) else None
        if shifts_src is not None:
            with h5py.File(shifts_src, "r") as f:
                groups = [
                    g
                    for g in (
                        "PiecewiseRigidRegistrationArray",
                        "RigidRegistrationArray",
                    )
                    if g in f
                ]
                if not groups:
                    raise ValueError(f"{shifts_src} holds no registration array")
                shifts = f[groups[0]]["shifts"][()]
        if shifts is not None:
            if isinstance(shifts, torch.Tensor):
                shifts = shifts.cpu().numpy()
            shifts = np.asarray(shifts, np.float32)
            if shifts.shape[0] != self._shape[0]:
                if not found_shifts:
                    raise ValueError(
                        f"{shifts.shape[0]} shifts for {self._shape[0]} frames"
                    )
                display(
                    f"skipping {shifts_src}: {shifts.shape[0]} shifts for {self._shape[0]} frames"
                )
                shifts = None
        # say what was picked up, and how to add what was not, so the panels are discoverable
        if raw is not None:
            display(f"raw panel: {raw_src if raw_src is not None else 'movie given'}")
        else:
            display(
                "no raw movie: raw= (a movie or a .tif path) adds a raw panel; a lone .tif beside the results is picked up"
            )
        if shifts is not None:
            display(
                f"shift traces: {shifts_src if shifts_src is not None else 'shifts given'}"
            )
        else:
            display(
                "no motion shifts: shifts= (an array or a motion correction hdf5) adds shift traces; "
                "shifts in the results file or a motion_correction.hdf5 beside it are picked up"
            )
        self._raw = raw
        self._shifts = shifts
        self._shift_lines = None
        if shifts is not None:
            # piecewise rigid shifts are (frames, height blocks, width blocks, 2): show the largest block shift
            summary = np.abs(shifts).max(axis=(1, 2)) if shifts.ndim == 4 else shifts
            prefix = "max block shift" if shifts.ndim == 4 else "shift"
            self._shift_lines = [
                (f"{prefix} height", summary[:, 0], (1.0, 0.6, 0.2)),
                (f"{prefix} width", summary[:, 1], (0.4, 0.7, 1.0)),
            ]

        ref_range, frame_timings = resolve_time_reference(
            self._shape[0], frame_timings, ref_range
        )

        # TODO: make these a variable at the top of the file when the names for these visualizations are set
        # raw takes the place of the signals panel, so the grid stays three by two
        self._video_panels = (
            ("raw", "compressed+denoised")
            if self._raw is not None
            else ("compressed+denoised", "signals")
        ) + (
            "background",
            "residual",
            "colorful_signals",
            "summary img",
        )

        self._bind_arrays()

        self._video_extents = {
            name: ((i % 3) / 3, (i % 3 + 1) / 3, (i // 3) / 2, (i // 3) / 2 + 0.5)
            for i, name in enumerate(self._video_panels)
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
        movie_display_dims = ["m", "n"]
        movie_index_mapping = {"time": frame_timings}
        self._pmd_graphic = self._ndw_fov["compressed+denoised"].add_nd_image(
            self._pmd_array,
            movie_dims,
            movie_display_dims,
            slider_maps=movie_index_mapping.copy(),
            name="compressed+denoised",
        )
        self._panel_graphics["compressed+denoised"] = self._pmd_graphic

        self._raw_graphic = None
        if self._raw is not None:
            self._raw_graphic = self._ndw_fov["raw"].add_nd_image(
                self._raw,
                movie_dims,
                movie_display_dims,
                slider_maps=movie_index_mapping.copy(),
                name="raw",
            )
            self._panel_graphics["raw"] = self._raw_graphic

        self._ac_graphic = None
        if self._ac_array is not None:
            if "signals" in self._video_panels:
                self._ac_graphic = self._ndw_fov["signals"].add_nd_image(
                    self._ac_array,
                    movie_dims,
                    movie_display_dims,
                    slider_maps=movie_index_mapping.copy(),
                    name="signals",
                )
                self._panel_graphics["signals"] = self._ac_graphic

            self._background_graphic = self._ndw_fov["background"].add_nd_image(
                self._fluctuating_background_array,
                movie_dims,
                movie_display_dims,
                slider_maps=movie_index_mapping.copy(),
                name="background",
            )

            self._residual_graphic = self._ndw_fov["residual"].add_nd_image(
                self._residual_array,
                movie_dims,
                movie_display_dims,
                slider_maps=movie_index_mapping.copy(),
                name="residual",
            )

            movie_dims_rgb = ["time", "m", "n", "c"]
            movie_display_dims_rgb = ["m", "n", "c"]
            self._colorful_signal_graphic = self._ndw_fov[
                "colorful_signals"
            ].add_nd_image(
                self._colorful_ac_array,
                movie_dims_rgb,
                movie_display_dims_rgb,
                slider_maps=movie_index_mapping.copy(),
                rgb_dim="c",
                name="colorful_signals",
            )
            self._panel_graphics["background"] = self._background_graphic
            self._panel_graphics["residual"] = self._residual_graphic
            self._panel_graphics["colorful_signals"] = self._colorful_signal_graphic
        else:
            self._background_graphic = None
            self._residual_graphic = None
            self._colorful_signal_graphic = None

        self._own_summary = summary_img is None and self._has_ac
        if summary_img is not None:
            dimension_data = ["m", "n"] if summary_img.ndim == 2 else ["time", "m", "n"]
            self._summary_image = self._ndw_fov["summary img"].add_nd_image(
                summary_img,
                dimension_data,
                ["m", "n"],
                name="summary img",
            )
            self._ndw_fov.figure["summary img"].title = (
                summary_img_name if summary_img_name is not None else "Summary Image"
            )
        elif self._has_ac:
            self._summary_image = self._ndw_fov["summary img"].add_nd_image(
                self.demixing_results.global_residual_correlation_image.cpu().numpy(),
                ["m", "n"],
                ["m", "n"],
                name="summary img",
            )
            self._ndw_fov.figure["summary img"].title = (
                summary_img_name
                if summary_img_name is not None
                else "Residual Correlation Image"
            )
        else:
            self._summary_image = self._ndw_fov["summary img"].add_nd_image(
                self._pmd_array.mean_image.cpu().numpy(),
                ["m", "n"],
                ["m", "n"],
                name="summary img",
            )
            self._ndw_fov.figure["summary img"].title = (
                summary_img_name if summary_img_name is not None else "Mean Image"
            )

        self._panel_graphics["summary img"] = self._summary_image
        self._fov_subplot = self._ndw_fov.figure["summary img"]

        self._active_component = None
        self._marked = (
            set()
        )  # signal indices "Delete" has marked; removed on the next "Demix"
        self._poly_hits = []  # the signals the poly-select polygon holds
        self._show_traces = False  # plot the selection's traces; off, selecting only highlights
        self._undo = []  # curation snapshots for ctrl+z, newest last
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
        if cell_order is not None:
            self.load_cell_order(cell_order)

        self._set_gray_cmaps()

        # no autofit: the zoom set on one signal's traces is kept while selecting others
        self._traces = TracePlot(
            ("shift (px)", "traces") if self._shift_lines else ("traces",),
            self._shape[0],
            frame_timings,
            autofit=False,
        )
        self._traces.dock(
            self._ndw_fov.figure, size=480 if self._shift_lines else 360, title="traces"
        )
        if self._shift_lines:
            self._traces.set("shift (px)", self._shift_lines)
        self._traces.link(self.reference_index)
        self._traces.on_pick = self._select_signal
        self._base_lines = ("compressed", "signal", "background", "residual")
        self._selected_signals = (
            None  # the signal behind each plotted line, when lines are signals
        )
        # diagnostic only: (row, col) -> (compressed 5x5 average, pixel count), newest first; they join the group
        self._pixel_traces = False
        self._pixels = OrderedDict()
        self._active_pixel = None

        self._image_selector = None
        self._show_contours = show_contours
        self._contour_opacity = 0.9
        if self._ac_array is not None:
            self._make_selectors()

        # PolygonSelector -> color, subplot, compressed average over it (None until closed), pixel count, dirty
        self._rois = OrderedDict()
        self._active_roi = None
        self._status = ""
        self._file_dialog = None
        self._order_dialog = None
        self._export_popup = False
        self._export_path = os.path.join(os.getcwd(), "rois.npz")
        self._stats_popup = False
        self._stats_path = ""
        self._press = None  # screen position of the last pointer press on a video panel
        self._armed = None  # "roi" / "poly": the next press on any video panel starts that polygon there
        # the poly-select polygon, its subplot, and the vertices / side / filter its hits were last computed for
        self._poly = None
        self._poly_panel = None
        self._poly_key = None
        self._poly_outside = False

        self._bind_click_handlers()

        for subplot in self._ndw_fov.figure:
            subplot.tooltip.enabled = False
            subplot.toolbar = False

        self._ndw_fov.figure.add_imgui_window(
            self._draw_side_panel, location="right", size=300, title="Tools"
        )
        if self._has_ac and len(self._footprints):
            self._select_component(0)

    def _make_footprints(self):
        self._footprints = FootprintSet.from_sparse(
            self._ac_array.spatial_demixed, tuple(self._shape[1:3])
        )
        peaks = self.demixing_results.temporal_demixed.max(dim=0).values.cpu().numpy()
        columns = {"area": self._footprints.areas, "peak": peaks}
        if self._cell_stats is not None:
            columns.update(zip(self._cell_stats.names, self._cell_stats.values.T))
        columns["del"] = np.zeros(len(self._footprints), np.int8)
        self._order = RoiOrder(columns, len(self._footprints), pinned="del")
        self._order.set_range_column("area")
        self._order.rebuild()
        self._color_by = "signal id"
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
                {
                    k: rgb
                    for k, rgb in self._group_colors().items()
                    if isinstance(k, int)
                },
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
            self._pmd_array = self.demixing_results.compression_array
            self._fluctuating_background_array = (
                self.demixing_results.fluctuating_background_array
            )
            self._residual_array = self.demixing_results.residual_array
            self._colorful_ac_array = self.demixing_results.colorful_signals_array
            self._ac_array = self.demixing_results.signals_array
        else:
            self._pmd_array = self.demixing_results
            self._fluctuating_background_array = None
            self._residual_array = None
            self._colorful_ac_array = None
            self._ac_array = None

    def _bind_click_handlers(self):
        """Re-attach the click handlers to every video panel: NDGraphic.data= replaces the graphic instance."""
        for name, graphic in self._panel_graphics.items():
            graphic.graphic.add_event_handler(
                partial(self._pointer_down, name), "pointer_down"
            )
            graphic.graphic.add_event_handler(self._click_update, "click")

    def _pointer_down(self, name: str, ev: pygfx.PointerEvent):
        self._press = (ev.x, ev.y)
        if self._armed == "roi":
            self._begin_roi(name)
        elif self._armed == "poly":
            self._begin_poly(name)

    def _set_gray_cmaps(self):
        """NDGraphic.data= replaces the graphic instance, dropping its cmap too."""
        for g in self._panel_graphics.values():
            if g is not self._colorful_signal_graphic:
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
            self._raw_graphic,
        )

    def _make_selectors(self):
        """(Re)build the footprint selectors over the current signals."""
        show = self._show_contours
        # the selected and grouped components' contours; the roi panel's "contours" adds every footprint's
        self._image_selector = fpl.ImageHighlightSelector(
            lut="tab10",
            lut_wrap="repeat",
            selection_options={"pixels": self._ac_array.contours},
            options_color="w",
            options_alpha=self._contour_opacity,
            alpha=0.7,
        )
        self._set_contours(show)

    def _load_results(self, results: masknmf.DemixingResults):
        """Swap in re-demixed results: every movie panel, the selectors and the summary image follow."""
        results.to(self.device)
        self._demixing_results = results
        self._cell_stats = CellStats.from_results(results)
        self._shown_stats.clear()
        self._bind_arrays()
        self._clear_rois()
        self._drop_poly()
        self._marked.clear()
        self._group.clear()
        self._clear_component()
        self._selected_signals = None
        self._clear_traces()
        self._pixels.clear()
        self._pmd_graphic.data = self._pmd_array
        if self._ac_graphic is not None:
            self._ac_graphic.data = self._ac_array
        self._background_graphic.data = self._fluctuating_background_array
        self._residual_graphic.data = self._residual_array
        self._colorful_signal_graphic.data = self._colorful_ac_array
        if self._raw_graphic is not None:
            # same movie, new graphic instance: keeps the re-bound click handlers from doubling up
            self._raw_graphic.data = self._raw
        if self._own_summary:
            self._summary_image.data = (
                results.global_residual_correlation_image.cpu().numpy()
            )
        self._bind_click_handlers()
        self._set_gray_cmaps()
        self._make_selectors()
        self._make_footprints()
        self._undo.clear()

    def demix(self):
        """
        Run the drawn ROIs (appended) and the marked signals (removed) through the demixer's NMF pass and
        write the outcome to a new curated file beside the results. Runs on a thread; the viewer reloads
        from the new file when it finishes.
        """
        if self._ac_array is None or self._results_path is None:
            raise ValueError(
                "editing signals needs demixing results loaded from a file"
            )
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
            path = write_curated(self._results_path, results, drop, masks.shape[-1])
            self._pending = (results, path)
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
        results, path = pending
        before = self._ac_array.spatial_demixed.shape[1]
        try:
            self._load_results(results)
        except Exception as e:
            self._status = f"reload after demix failed: {e}"
            return
        parent, self._results_path = self._results_path, path
        self._status = (
            f"{results.spatial_demixed.shape[1]} signals (was {before}) written to {os.path.basename(path)}; "
            f"{os.path.basename(parent)} kept"
        )

    def _select_signal(self, panel: str, index: int):
        """Select the signal whose line was double-clicked in the trace dock."""
        if self._selected_signals is not None and 0 <= index < len(
            self._selected_signals
        ):
            picked = self._selected_signals[index]
            if isinstance(picked, tuple):
                self._active_pixel = picked
            elif picked in self._rois:
                self._active_roi = picked
            else:
                self._select_component(picked)
        elif self._selected_signals is None and self._active_component is not None and not self._group:
            # the single signal's own lines: a second double-click deselects it
            self._snapshot()
            self._clear_component()
            self._clear_traces()

    def _click_update(self, ev: pygfx.PointerEvent):
        """
        Priority: a drawn roi, then an existing component, else clear the selection.
        ctrl / shift on a component grow the group instead of replacing the selection.
        """
        if self._drawing() or imgui.get_io().want_capture_mouse:
            return
        # pygfx reports a click after any press and release on one graphic, a pan drag included
        if self._press is not None and (
            abs(ev.x - self._press[0]) + abs(ev.y - self._press[1]) > _CLICK_SLOP
        ):
            return
        col, row = ev.pick_info["index"]
        mods = set(getattr(ev, "modifiers", ()) or ())

        roi = self._roi_at(col, row)
        if roi is not None:
            if mods & {"Control", "Ctrl"}:
                self.group_toggle(roi)
            elif "Shift" in mods:
                self.group_add(roi)
            else:
                self._select_roi(roi)
            return

        if self._ac_array is not None:
            component = component_at_pixel(
                self._ac_array.spatial_demixed, self._ac_array.centers, self._shape[1:], (col, row)
            )
            if component is not None:
                if mods & {"Control", "Ctrl"}:
                    self.group_toggle(component)
                elif "Shift" in mods:
                    self.group_add(component)
                elif component == self._active_component and not self._group:
                    self._snapshot()
                    self._clear_component()
                    self._clear_traces()
                else:
                    self.group_clear()
                    self._select_component(component)
                return

        if (
            self._pixel_traces
            and 0 <= row < self._shape[1]
            and 0 <= col < self._shape[2]
        ):
            pixel = (int(row), int(col))
            if mods & {"Control", "Ctrl"} and pixel in self._group:
                self._group.remove(pixel)
            else:
                if pixel not in self._pixels:
                    self._snapshot()
                    rows, cols = np.mgrid[
                        max(pixel[0] - 2, 0) : min(pixel[0] + 3, self._shape[1]),
                        max(pixel[1] - 2, 0) : min(pixel[1] + 3, self._shape[2]),
                    ]
                    self._pixels[pixel] = (
                        self._pmd_average(rows.ravel(), cols.ravel()),
                        rows.size,
                    )
                    self._pixels.move_to_end(pixel, last=False)
                self._seed_group()
                if pixel not in self._group:
                    self._group.append(pixel)
            self._active_pixel = pixel
            self._active_roi = None
            self._sync_highlight()
            self._update_traces()
            return

        if self._group or self._active_component is not None or self._active_roi is not None:
            self._snapshot()
        self.group_clear()
        self._clear_component()
        self._active_roi = None
        self._selected_signals = None
        self._clear_traces()

    def _select_roi(self, selector):
        """A drawn roi alone: its average is the plot, once it has one."""
        self.group_clear()
        self._clear_component()
        self._select_component(selector)

    def _select_component(self, component):
        if isinstance(component, tuple) or component in self._rois:
            # pixel averages and drawn rois are only ever plotted as group members
            if component not in self._group:
                self._group.append(component)
            self._active_pixel = component if isinstance(component, tuple) else None
            self._active_roi = None if isinstance(component, tuple) else component
            self._sync_highlight()
            self._update_traces()
            return
        self._active_roi = None
        self._active_pixel = None
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

    def _group_colors(self) -> dict:
        """One contrasting color per grouped signal, shared by its trace, mask and table row."""
        if len(self._group) < 2:
            return {}
        return {
            k: _GROUP_COLORS[i % len(_GROUP_COLORS)] for i, k in enumerate(self._group)
        }

    def _highlighted(self) -> list:
        picks = [k for k in self._group if isinstance(k, int)]
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
        One signal: its compressed / signal / background / residual roi averages. A group, or any pixel
        average or drawn roi: every member's compressed average, colored like its mask or table row.
        Nothing unless "show selected traces" is on.
        """
        if not self._show_traces:
            self._selected_signals = None
            self._traces.set("traces", [])
            return
        results = self.demixing_results
        if len(self._group) > 1 or any(not isinstance(k, int) for k in self._group):
            self._selected_signals = []
            lines = []
            for i, k in enumerate(self._group):
                rgb = _GROUP_COLORS[i % len(_GROUP_COLORS)]
                if isinstance(k, tuple):
                    lines.append(
                        (f"pixel avg ({k[0]}, {k[1]})", self._pixels[k][0], rgb)
                    )
                elif k in self._rois:
                    if self._rois[k]["trace"] is None:
                        continue
                    lines.append(
                        (
                            f"roi {list(self._rois).index(k)}",
                            self._rois[k]["trace"],
                            rgb,
                        )
                    )
                else:
                    lines.append(
                        (f"signal {k}", results.pmd_roi_averages[k].cpu().numpy(), rgb)
                    )
                self._selected_signals.append(k)
        elif self._active_component is not None:
            k = self._active_component
            self._selected_signals = None
            ypix, xpix, _lam = self._footprints.footprints[k]
            support = torch.as_tensor(
                ypix.astype(np.int64) * self._shape[2] + xpix, device=results.spatial_demixed.device
            )
            # the signal movie averaged over the footprint's support, like the stored roi averages
            signal = torch.sparse.mm(
                torch.index_select(results.spatial_demixed, 0, support), results.temporal_demixed.T
            ).mean(dim=0)
            traces = (
                results.pmd_roi_averages[k],
                signal,
                results.fluctuating_background_roi_averages[k],
                results.residual_roi_averages[k],
            )
            lines = [
                (label, trace.cpu().numpy(), rgb)
                for label, trace, rgb in zip(
                    self._base_lines, traces, _BASE_LINE_COLORS
                )
            ]
        else:
            self._selected_signals = None
            self._clear_traces()
            return
        self._traces.set("traces", lines)

    def _seed_group(self):
        """A first ctrl or shift pick keeps the current selection in the group."""
        if not self._group and self._active_component is not None:
            self._group.append(self._active_component)

    def group_add(self, component):
        self._seed_group()
        if component not in self._group:
            self._group.append(
                int(component)
                if isinstance(component, (int, np.integer))
                else component
            )
        self._select_component(component)

    def group_toggle(self, component):
        self._seed_group()
        if component in self._group:
            self._group.remove(component)
            if not isinstance(component, (int, np.integer)):
                self._active_pixel = component if isinstance(component, tuple) else None
                self._active_roi = None if isinstance(component, tuple) else component
                self._sync_highlight()
                self._update_traces()
                return
            # the cursor moves to another member, so the removed signal loses its highlight too
            signals = [k for k in self._group if isinstance(k, int)]
            self._active_component = signals[-1] if signals else None
            if self._active_component is not None and self._order is not None:
                self._order.goto(self._active_component)
            self._sync_highlight()
            self._update_traces()
            return
        else:
            self._group.append(
                int(component)
                if isinstance(component, (int, np.integer))
                else component
            )
        self._select_component(component)

    def group_extend_to(self, component):
        """Add every table row between the cursor and ``component`` to the group; a pixel average or drawn roi just joins."""
        if self._order is None or not isinstance(component, (int, np.integer)):
            self.group_add(component)
            return
        self._seed_group()
        order = [int(k) for k in self._order.order]
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

    def deselect(self):
        """Drop the selection, the group and the pixel averages: esc, or the Signals tab's deselect button."""
        if self._group or self._pixels or self._active_component is not None or self._active_roi is not None:
            self._snapshot()
        self._pixels.clear()
        self.group_clear()
        self._clear_component()
        self._active_roi = None
        self._selected_signals = None
        self._clear_traces()

    def _center_on(self, component: int):
        """
        Pan every video panel to one footprint. Zoomed out to the whole fov, this also zooms in on
        it with some context; once zoomed in, the zoom is the user's and only the center moves.
        """
        ypix, xpix, _lam = self._footprints.footprints[component]
        if not len(ypix):
            return
        y0, y1 = float(ypix.min()), float(ypix.max())
        x0, x1 = float(xpix.min()), float(xpix.max())
        cy, cx = (y0 + y1) / 2, (x0 + x1) / 2
        camera = self._fov_subplot.camera
        fov_height, fov_width = self._shape[1:3]
        if camera.width >= fov_width or camera.height >= fov_height:
            width = height = max(max(y1 - y0, x1 - x0, 1.0) * 4.0, 80.0)
        else:
            width, height = camera.width, camera.height
        for name in self._video_panels:
            self._ndw_fov.figure[name].camera.show_rect(
                cx - width / 2, cx + width / 2, cy - height / 2, cy + height / 2
            )

    def _set_pixel_traces(self, on: bool):
        """Turning pixel traces off drops every pixel average, from the plot and the table."""
        self._pixel_traces = on
        if on or not self._pixels:
            return
        self._group[:] = [k for k in self._group if not isinstance(k, tuple)]
        self._pixels.clear()
        self._active_pixel = None
        self._sync_highlight()
        self._update_traces()

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
        """The selection's contour is always drawn; ``show`` adds every other footprint's at the contour opacity."""
        self._show_contours = show
        if self._image_selector is None:
            return
        self._image_selector.options_alpha = self._contour_opacity if show else 0.0
        for g in self._video_graphics():
            if g is not None and g.graphic not in self._image_selector.graphics:
                self._image_selector.add_graphic(g.graphic)

    def _drawing(self) -> bool:
        return self._armed is not None or any(
            s is not None and s._move_info.mode == "create"
            for s in (self._active_roi, self._poly)
        )

    def _roi_at(self, col: int, row: int):
        for selector in reversed(self._rois):
            polygon = selector.selection[:, :2]
            if polygon.shape[0] >= 3 and point_in_polygon((col, row), polygon):
                return selector
        return None

    def _start_roi(self):
        """Arm a new roi: it is created on whichever video panel gets the next press."""
        self._armed = "roi"
        self._clear_component()

    def _begin_roi(self, name: str):
        """
        Create the armed roi on panel ``name``. The press that got here also places its first
        vertex: pygfx bubbles it up to the renderer last, where the selector's fresh handlers wait.
        """
        self._armed = None
        self._snapshot()
        color = _ROI_COLORS[len(self._rois) % len(_ROI_COLORS)]
        selector = self._panel_graphics[name].graphic.add_polygon_selector(
            fill_color=color,
            edge_color=color,
            vertex_color=color,
            edge_thickness=2,
            vertex_size=8,
        )
        selector.add_event_handler(partial(self._roi_changed, selector), "selection")
        self._rois[selector] = {
            "color": color,
            "panel": name,
            "subplot": self._ndw_fov.figure[name],
            "trace": None,
            "area": 0,
            "dirty": True,
        }
        self._active_roi = selector

    def _roi_changed(self, selector, ev):
        self._active_roi = selector
        self._rois[selector]["dirty"] = True
        self._clear_component()

    def _poll_rois(self):
        """Average the compressed movie over each drawn roi once its vertices settle; a new roi joins the plot."""
        for selector, roi in self._rois.items():
            if not roi["dirty"] or selector._move_info.mode is not None:
                continue
            roi["dirty"] = False
            indices = selector.get_selected_indices(selector.parent)
            if indices.shape[0] == 0:
                continue
            first = roi["trace"] is None
            roi["trace"] = self._pmd_average(indices[:, 1], indices[:, 0])
            roi["area"] = int(indices.shape[0])
            if first:
                self._seed_group()
                self._select_component(selector)
            elif selector in self._group:
                self._update_traces()

    def _pmd_average(self, rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
        """The compressed movie averaged over the pixels (rows, cols), as the panel shows it, without building frames."""
        pmd = self._pmd_array
        idx = torch.as_tensor(
            np.asarray(rows, np.int64) * self._shape[2] + np.asarray(cols, np.int64),
            device=pmd.temporal_compressed.device,
        )
        u = torch.index_select(pmd.spatial_compressed, 0, idx).to_dense()
        if pmd.rescale:
            u = u * pmd.noise_variance_image.flatten()[idx, None]
        trace = u.mean(dim=0) @ pmd.temporal_compressed
        if pmd.rescale:
            trace = trace + pmd.mean_image.flatten()[idx].mean()
            if (
                pmd.include_trend
                and pmd.spatial_trend_basis is not None
                and pmd.temporal_trend_basis is not None
            ):
                trace = (
                    trace
                    + pmd.spatial_trend_basis[idx].mean(dim=0)
                    @ pmd.temporal_trend_basis
                )
        return trace.cpu().numpy()

    def _delete_roi(self, selector, record: bool = True):
        if record:
            self._snapshot()
        if selector._move_info.mode is not None:
            selector._end_move_mode()
        self._rois.pop(selector)["subplot"].delete_graphic(selector)
        if self._active_roi is selector:
            self._active_roi = next(reversed(self._rois), None)
        if selector in self._group:
            self._group.remove(selector)
            self._sync_highlight()
            self._update_traces()

    def _clear_rois(self):
        for selector in list(self._rois):
            self._delete_roi(selector)

    def _start_poly(self):
        """The poly-select button: arm a polygon for the next press on a video panel, or leave the mode (the selection stays)."""
        if self._armed == "poly":
            self._armed = None
        elif self._poly is not None:
            self._drop_poly()
        else:
            self._armed = "poly"

    def _begin_poly(self, name: str):
        """Create the armed poly-select polygon on panel ``name``; the press places its first vertex, as for a roi."""
        self._armed = None
        self._snapshot()
        self._drop_poly()
        self._poly = self._panel_graphics[name].graphic.add_polygon_selector(
            fill_color=(0.0, 0.0, 0.0, 0.0),
            edge_color=THEME.accent[:3],
            vertex_color=THEME.accent[:3],
            edge_thickness=2,
            vertex_size=8,
        )
        self._poly_panel = self._ndw_fov.figure[name]

    def _poll_poly(self):
        """Select the signals in view the poly-select polygon holds, following it as it is drawn or dragged."""
        if self._poly is None:
            return
        polygon = self._poly.selection[:, :2]
        moving = self._poly._move_info.mode is not None
        if polygon.shape[0] < 3:
            if not moving:
                self._drop_poly()
            return
        key = (polygon.tobytes(), self._poly_outside, self._order.range_limits, moving)
        if key == self._poly_key:
            return
        self._poly_key = key
        view = self._order.order
        centers = self._ac_array.centers.cpu().numpy()[view]
        inside = np.fromiter(
            (point_in_polygon((col, row), polygon) for row, col in centers),
            bool,
            len(view),
        )
        hits = [int(k) for k in (view[~inside] if self._poly_outside else view[inside])]
        if hits != self._poly_hits:
            self._poly_hits = hits
            self._group[:] = hits
            self._active_component = hits[-1] if hits else None
            if self._active_component is not None:
                self._order.goto(self._active_component)
            self._sync_highlight()
            where = "outside" if self._poly_outside else "inside"
            self._status = f"poly-select: {len(hits)} signal(s) {where} the polygon"
        # no traces while the polygon is being drawn or dragged; they plot once it settles
        if moving:
            self._selected_signals = None
            self._clear_traces()
        else:
            self._update_traces()

    def _drop_poly(self):
        if self._poly is None:
            return
        if self._poly._move_info.mode is not None:
            self._poly._end_move_mode()
        self._poly_panel.delete_graphic(self._poly)
        self._poly = None
        self._poly_key = None
        self._poly_hits = []
        self._update_traces()

    def _mark(self, signals, on: bool, record: bool = True):
        """Mark ``signals`` for deletion on the next demix, or unmark them; the masks and the table follow."""
        signals = {int(k) for k in signals}
        if record and signals:
            self._snapshot()
        if on:
            self._marked |= signals
        else:
            self._marked -= signals
        self._order.columns["del"][list(signals)] = on
        self._order.rebuild()
        self._refresh_masks()

    def _snapshot(self):
        """Push the curation state for ctrl+z: the marks, drawn rois, pixel averages and selection."""
        self._undo.append(
            {
                "marked": set(self._marked),
                "rois": [
                    (sel, roi["panel"], roi["color"], np.array(sel.selection), roi["trace"], roi["area"])
                    for sel, roi in self._rois.items()
                ],
                "pixels": OrderedDict(self._pixels),
                "group": list(self._group),
                "active": self._active_component,
                "active_roi": self._active_roi,
                "active_pixel": self._active_pixel,
            }
        )
        del self._undo[:-_UNDO_DEPTH]

    def undo(self):
        """Ctrl+z: back to the last snapshot; a deleted roi is redrawn, a live poly-select polygon is dropped."""
        if not self._undo:
            return
        state = self._undo.pop()
        self._armed = None
        self._drop_poly()
        kept = {sel for sel, *_ in state["rois"]}
        for sel in list(self._rois):
            if sel not in kept:
                self._delete_roi(sel, record=False)
        swap = {}
        for sel, panel, color, vertices, trace, area in state["rois"]:
            if sel in self._rois:
                continue
            new = self._panel_graphics[panel].graphic.add_polygon_selector(
                fill_color=color, edge_color=color, vertex_color=color, edge_thickness=2, vertex_size=8
            )
            new.selection = vertices
            new._end_move_mode()
            new.add_event_handler(partial(self._roi_changed, new), "selection")
            self._rois[new] = {
                "color": color,
                "panel": panel,
                "subplot": self._ndw_fov.figure[panel],
                "trace": trace,
                "area": area,
                "dirty": trace is None,
            }
            swap[sel] = new
        rois = OrderedDict((swap.get(sel, sel), self._rois[swap.get(sel, sel)]) for sel, *_ in state["rois"])
        self._rois.clear()
        self._rois.update(rois)
        self._marked.clear()
        self._marked.update(state["marked"])
        if self._order is not None:
            self._order.columns["del"][:] = 0
            self._order.columns["del"][list(self._marked)] = 1
            self._order.rebuild()
        self._pixels.clear()
        self._pixels.update(state["pixels"])
        self._group[:] = [swap.get(k, k) for k in state["group"]]
        self._active_component = state["active"]
        self._active_roi = swap.get(state["active_roi"], state["active_roi"])
        self._active_pixel = state["active_pixel"]
        if self._active_component is not None and self._order is not None:
            self._order.reveal(self._active_component)
            self._scroll_to_current = True
        self._sync_highlight()
        self._update_traces()
        self._status = f"undone, {len(self._undo)} more"

    def _delete_selected(self):
        """The Delete action: drop an active drawn roi, else the active pixel average, else mark the selected signals."""
        if self._active_roi is not None:
            self._delete_roi(self._active_roi)
        elif self._active_pixel is not None:
            self._snapshot()
            self._pixels.pop(self._active_pixel, None)
            if self._active_pixel in self._group:
                self._group.remove(self._active_pixel)
            self._active_pixel = None
            self._sync_highlight()
            self._update_traces()
        elif self._active_component is not None or any(isinstance(k, int) for k in self._group):
            signals = [k for k in self._group if isinstance(k, int)] or [self._active_component]
            self._mark(signals, not all(k in self._marked for k in signals))

    def _clear_traces(self):
        self._active_pixel = None
        self._traces.set("traces", [])

    @property
    def roi_masks(self) -> np.ndarray:
        """The drawn ROIs as a binary mask stack of shape (fov dim1, fov dim2, num_rois)"""
        shape = tuple(self._shape[1:3])
        masks = []
        for selector in self._rois:
            indices = selector.get_selected_indices(selector.parent)
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
        return np.concatenate([self._ac_array.export_spatial_demixed(), masks], axis=-1)

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
        if result:
            self._export_path = result

    def _browse_stats(self):
        if self._order_dialog is None:
            self._order_dialog = pfd.open_file("Load cell stats", os.getcwd(), _STATS_FILTERS)

    def _poll_order_dialog(self):
        if self._order_dialog is None or not self._order_dialog.ready(0):
            return
        result = self._order_dialog.result()
        self._order_dialog = None
        if result:
            self._stats_path = result[0]

    def _handle_keys(self):
        io = imgui.get_io()
        if io.want_text_input:
            return
        if imgui.is_key_pressed(imgui.Key.delete, False):
            self._delete_selected()
        if imgui.is_key_pressed(imgui.Key.escape, False):
            if self._armed is not None:
                self._armed = None
            elif self._poly is not None:
                self._drop_poly()
            else:
                self.deselect()
        if io.key_ctrl and imgui.is_key_pressed(imgui.Key.a, False) and self._order is not None:
            self._group[:] = [int(k) for k in self._order.order]
            self._sync_highlight()
            self._update_traces()
        if io.key_ctrl and imgui.is_key_pressed(imgui.Key.z, False):
            self.undo()
        stride = 10 if io.key_shift else 1
        if imgui.is_key_pressed(imgui.Key.down_arrow, True):
            self._step(stride)
        if imgui.is_key_pressed(imgui.Key.up_arrow, True):
            self._step(-stride)
        if imgui.is_key_pressed(imgui.Key.f, False):
            self._toggle_follow()
        if imgui.is_key_pressed(imgui.Key.p, False):
            self._set_pixel_traces(not self._pixel_traces)
        if imgui.is_key_pressed(imgui.Key.k, False):
            self._keybinds_open = not self._keybinds_open

    def _selection_status(self) -> str:
        pixels = [k for k in self._group if isinstance(k, tuple)]
        rois = [list(self._rois).index(k) for k in self._group if k in self._rois]
        if len(self._group) > 1 or pixels or rois:
            signals = sorted(k for k in self._group if isinstance(k, int))
            shown = f"{len(signals)} signals" if len(signals) > 12 else f"signals {signals}"
            note = "; pixel avgs are marked, delete them when done" if pixels else ""
            return f"{len(self._group)} grouped: {shown}, rois {rois}, pixel avgs {pixels}{note}"
        if self._active_component is not None:
            marked = (
                " (marked for deletion)"
                if self._active_component in self._marked
                else ""
            )
            return f"signal {self._active_component} selected{marked}"
        if self._active_roi in self._rois:
            return f"roi {list(self._rois).index(self._active_roi)} selected"
        return "double-click a mask or roi to see its trace"

    def _draw_side_panel(self):
        """Docked at "right" (the NDWidget owns "bottom"): the roi tools and the signal table as tabs."""
        self._poll_file_dialog()
        self._poll_order_dialog()
        self._poll_worker()
        self._poll_rois()
        self._poll_poly()
        self._handle_keys()
        if imgui.begin_tab_bar("##side"):
            if imgui.begin_tab_item("Curation")[0]:
                self._draw_roi_tools()
                imgui.end_tab_item()
            if imgui.begin_tab_item("Signals")[0]:
                self._draw_signal_tab()
                imgui.end_tab_item()
            imgui.end_tab_bar()
        self._keybinds_open = draw_keybinds_popup(_KEYBINDS, self._keybinds_open)
        self._export_popup, self._export_path, go = draw_path_popup(
            "Export ROIs", self._export_popup, self._export_path, "rois.npz", "export", self._browse_export, self._status
        )
        if go:
            try:
                path = self.export_rois(self._export_path)
                self._status = f"exported {len(self._rois)} roi(s) to {path}"
                self._export_popup = False
            except (OSError, ValueError) as e:
                self._status = f"export failed: {e}"
        self._stats_popup, self._stats_path, go = draw_path_popup(
            "Load cell stats", self._stats_popup, self._stats_path,
            ".npy / .npz / .csv / .tsv of stats, or a .txt of signal ids in order", "load", self._browse_stats, self._status,
        )
        if go:
            try:
                if self._stats_path.lower().endswith(".txt"):
                    self.load_cell_order(self._stats_path)
                else:
                    self.add_cell_stats(self._stats_path)
                self._status = f"cell stats loaded from {os.path.basename(self._stats_path)}"
                self._stats_popup = False
            except (OSError, ValueError, TypeError) as e:
                self._status = f"cell stats failed: {e}"

    def _table_select(self, component):
        if component == self._active_component and not self._group:
            self._snapshot()
            self._clear_component()
            self._clear_traces()
            return
        self.group_clear()
        if isinstance(component, tuple):
            self._clear_component()
        self._select_component(component)

    def _format_cell(self, name: str, item) -> str:
        if isinstance(item, tuple):
            trace, area = self._pixels[item]
            return {"area": f"{area}", "peak": f"{float(trace.max()):.3g}", "del": "x"}.get(name, "")
        if item in self._rois:
            roi = self._rois[item]
            peak = "" if roi["trace"] is None else f"{float(roi['trace'].max()):.3g}"
            return {"area": f"{roi['area']}", "peak": peak, "del": ""}.get(name, "")
        if name == "del":
            return "x" if item in self._marked else ""
        value = self._order.columns[name][item]
        if np.isnan(value):
            return ""
        return f"{int(value)}" if name == "area" else f"{float(value):.3g}"

    def _draw_signal_tab(self):
        if self._order is None and not self._pixels and not self._rois:
            imgui.text_disabled("no demixed signals")
            return
        # a bare pmd array has no signals, but its pixel averages and drawn rois still get the table
        order = (
            self._order
            if self._order is not None
            else RoiOrder({"area": np.zeros(0), "peak": np.zeros(0), "del": np.zeros(0)}, 0)
        )
        g = grid(_CAPTIONS)
        names = () if self._cell_stats is None else self._cell_stats.names
        if order.range_column is not None:
            g.row("filter")
            if draw_range_filter(order, "_signals"):
                order.rebuild()
        g.row("in view")
        on = not all(int(k) in self._marked for k in order.order)
        imgui.begin_disabled(not len(order.order))
        with button_colors(THEME.danger, THEME.danger_hover):
            if imgui.button(f"{'delete' if on else 'unmark'} in view", imgui.ImVec2(g.w, 0)):
                self._mark(order.order, on)
        imgui.end_disabled()
        help_mark("mark every signal the filter shows for deletion on the next demix, or unmark them")
        g.cell(1)
        right_aligned_text(f"{len(order.order)} / {order.n_items}")
        if self._order is not None:
            g.row("selection")
            signals = [k for k in self._group if isinstance(k, int)]
            if not signals and self._active_component is not None:
                signals = [self._active_component]
            on = not signals or not all(k in self._marked for k in signals)
            imgui.begin_disabled(not signals or self._worker is not None)
            with button_colors(THEME.danger, THEME.danger_hover):
                if imgui.button(f"{'delete' if on else 'unmark'} {len(signals)}", imgui.ImVec2(g.w, 0)):
                    self._mark(signals, on)
            imgui.end_disabled()
            help_mark("mark the selected signals for deletion on the next demix, or unmark them (delete)")
            g.cell(1)
            selected = bool(self._group or self._pixels) or self._active_component is not None or self._active_roi is not None
            imgui.begin_disabled(not selected)
            if imgui.button("deselect", imgui.ImVec2(g.w, 0)):
                self.deselect()
            imgui.end_disabled()
            help_mark("drop the selection, the group and the pixel averages (esc)")
            g.row("merge")
            imgui.begin_disabled(True)
            imgui.button("merge", imgui.ImVec2(g.w, 0))
            imgui.end_disabled()
            help_mark("not yet implemented")
        g.row("view")
        changed, self._follow = imgui.checkbox("center on selection", self._follow)
        if changed and self._follow and self._active_component is not None:
            self._center_on(self._active_component)
        help_mark("pan every panel to the selected signal, and keep following it (f)")
        g.cell(1)
        if imgui.button("keys", imgui.ImVec2(g.w, 0)):
            self._keybinds_open = not self._keybinds_open
        help_mark("the keybinds (k)")
        if self._order is not None:
            g.row("stats")
            if imgui.button("load stats", imgui.ImVec2(g.w, 0)):
                self._stats_popup = True
            help_mark(
                "one row per signal, in signal id order, as sortable table columns:\n"
                "- .npy: a (signals,) or (signals, stats) array, or a structured array of stats\n"
                "- .npz: one (signals,) array per stat, named by key\n"
                "- .csv / .tsv: a header row of names, then one row per signal\n"
                "- .txt: signal ids in a custom order, becomes the 'order' column"
            )
            if names:
                g.cell(1)
                if imgui.button("columns", imgui.ImVec2(g.w, 0)):
                    imgui.open_popup("##columns")
                help_mark("which stat columns the table shows")
                if imgui.begin_popup("##columns"):
                    for name in names:
                        changed, on = imgui.checkbox(name, name in self._shown_stats)
                        if changed and on:
                            self._shown_stats.add(name)
                        elif changed:
                            self._shown_stats.discard(name)
                            if self._order is not None and self._order.sort_by == name:
                                self._order.sort_by = None
                                self._order.rebuild()
                    imgui.end_popup()
        footer = imgui.get_frame_height_with_spacing() * 2.5
        if imgui.begin_child("##signal_table", imgui.ImVec2(0, -footer)):
            columns = ("id", "area", "peak", *[n for n in names if n in self._shown_stats], "del")
            formatters = {name: partial(self._format_cell, name) for name in columns[1:]}
            colors = self._group_colors()
            self._scroll_to_current = draw_roi_table(
                order,
                columns,
                formatters,
                self._scroll_to_current,
                table_id="signals",
                cursor=self._active_component is not None,
                on_select=self._table_select,
                is_grouped=self._group.__contains__,
                on_ctrl_select=self.group_toggle,
                on_shift_select=self.group_extend_to,
                row_color=lambda k: colors.get(
                    k,
                    MARKED_COLOR
                    if isinstance(k, tuple)
                    else self._rois[k]["color"]
                    if k in self._rois
                    else self._footprints.color(k),
                ),
                prefix_rows=[(k, f"px {k[0]},{k[1]}") for k in self._pixels]
                + [(sel, f"roi {i}") for i, sel in enumerate(self._rois)],
            )
        imgui.end_child()
        imgui.separator()
        imgui.push_text_wrap_pos(0)
        imgui.text_disabled(self._selection_status())
        imgui.pop_text_wrap_pos()

    def _draw_roi_tools(self):
        drawing = self._drawing()
        existing = len(self._footprints) if self._footprints is not None else 0
        g = grid(_CAPTIONS)

        section("OVERLAY")
        if self._image_selector is not None:
            changed, show = imgui.checkbox("masks", self._show_masks)
            if changed:
                self._show_masks = show
                self._refresh_masks()
            g.cell(0)
            imgui.set_next_item_width(g.span)
            changed, self._mask_opacity = imgui.slider_float(
                "##mask-opacity", self._mask_opacity, 0.05, 1.0, "opacity %.2f"
            )
            if changed and self._show_masks:
                self._refresh_masks()
            changed, show = imgui.checkbox("contours", self._show_contours)
            if changed:
                self._set_contours(show)
            g.cell(0)
            imgui.set_next_item_width(g.span)
            changed, self._contour_opacity = imgui.slider_float(
                "##contour-opacity", self._contour_opacity, 0.05, 1.0, "opacity %.2f"
            )
            if changed and self._show_contours:
                self._image_selector.options_alpha = self._contour_opacity
            help_mark("every footprint's contour at this opacity; the selection's contour always shows")
        if self._order is not None:
            g.row("color by")
            names = ["signal id", *[n for n in self._order.columns if n != "del"]]
            imgui.set_next_item_width(g.span)
            changed, index = imgui.combo("##color_by", names.index(self._color_by) if self._color_by in names else 0, names)
            if changed:
                self._color_by = names[index]
                self._footprints.recolor(None if index == 0 else self._order.columns[self._color_by])
                self._refresh_masks()
            help_mark("color the masks and the table's ids by a column's rank instead of by signal id")
        g.row("traces")
        changed, on = imgui.checkbox("quick pixel trace", self._pixel_traces)
        if changed:
            self._set_pixel_traces(on)
        help_mark(
            "click an empty pixel to add the compressed movie's 5x5 average there to the plot, grouped "
            "with whatever is shown, and to the top of the signals table, marked. demix and export "
            "ignore it; delete drops it (p)"
        )
        g.cell(1)
        changed, self._show_traces = imgui.checkbox("show selected traces", self._show_traces)
        if changed:
            self._update_traces()
        help_mark(
            "plot whatever is selected: a signal's four averages, or one compressed average per grouped "
            "signal, pixel average and drawn roi. off, selecting only highlights, however big the selection"
        )

        section("ROIS")
        g.row("rois")
        imgui.begin_disabled(drawing)
        if imgui.button("Add ROI", imgui.ImVec2(g.w, 0)):
            self._start_roi()
        imgui.end_disabled()
        help_mark("draw a polygon roi on any panel; its average joins the plot and Demix seeds the nmf pass with it")
        g.cell(1)
        signals = [k for k in self._group if isinstance(k, int)]
        if not signals and self._active_component is not None:
            signals = [self._active_component]
        nothing = self._active_roi is None and self._active_pixel is None and not signals
        unmark = (
            self._active_roi is None
            and self._active_pixel is None
            and bool(signals)
            and all(k in self._marked for k in signals)
        )
        imgui.begin_disabled(nothing)
        with button_colors(THEME.danger, THEME.danger_hover, on=not unmark):
            if imgui.button("Unmark" if unmark else "Delete", imgui.ImVec2(g.w, 0)):
                self._delete_selected()
        imgui.end_disabled()
        help_mark(
            "remove the selected drawn roi, drop the active pixel average, or mark the selected "
            "signals for deletion on the next demix; again to unmark (delete)"
        )
        g.row("on disk")
        imgui.begin_disabled(not self._rois)
        if imgui.button("Export", imgui.ImVec2(g.w, 0)):
            self._export_popup = True
        imgui.end_disabled()
        help_mark("write the drawn rois to a .npz")
        g.cell(1)
        right_aligned_text(f"{existing} existing, {len(self._rois)} drawn")

        section("SELECT")
        selecting = self._armed == "poly" or self._poly is not None
        g.row("polygon")
        imgui.begin_disabled(self._order is None or (drawing and not selecting))
        with button_colors(THEME.accent, THEME.accent, (0.05, 0.05, 0.05), on=selecting):
            if imgui.button(f"{fa.ICON_FA_DRAW_POLYGON} {'stop' if selecting else 'poly-select'}", imgui.ImVec2(g.w, 0)):
                self._start_poly()
        imgui.end_disabled()
        help_mark(
            "click again or esc to leave poly-select; the selection stays"
            if selecting
            else "draw a polygon on any panel to select every signal in view whose center is inside (or outside) "
            "it; the selection follows the polygon as it is drawn and dragged, and Delete marks it"
        )
        g.cell(1)
        imgui.begin_disabled(True)
        imgui.button(f"{fa.ICON_FA_PEN_RULER} line-select", imgui.ImVec2(g.w, 0))
        imgui.end_disabled()
        help_mark("not yet implemented")
        g.row("side")
        if imgui.radio_button("inside", not self._poly_outside):
            self._poly_outside = False
        g.cell(1)
        if imgui.radio_button("outside", self._poly_outside):
            self._poly_outside = True
        help_mark("which side of the polygon poly-select takes")

        section("DEMIX")
        g.row("run")
        imgui.begin_disabled(
            (not self._rois and not self._marked)
            or self._ac_array is None
            or self._results_path is None
            or self._worker is not None
        )
        if imgui.button("Demix", imgui.ImVec2(g.w, 0)):
            self.demix()
        imgui.end_disabled()
        help_mark(
            "re-demix: add the drawn rois, remove the marked signals, write a new curated results file "
            "beside the original (kept)"
            if self._results_path is not None
            else "open the results with results_path to enable"
        )
        g.cell(1)
        right_aligned_text(f"{len(self._rois)} roi(s), {len(self._marked)} marked")
        g.row("options")
        changed, filter_dim = imgui.checkbox(
            "filter dim rois", self._nmf_config.min_brightness is not None
        )
        if changed:
            if filter_dim:
                self._nmf_config.min_brightness = self._min_brightness_cache
            else:
                self._min_brightness_cache = self._nmf_config.min_brightness
                self._nmf_config.min_brightness = None
        help_mark(
            "delete signals that never get bright enough during the nmf pass. "
            "turn off if a hand-drawn roi keeps disappearing from add to results."
        )

        imgui.spacing()
        imgui.push_text_wrap_pos(0)
        if self._armed is not None:
            imgui.text_disabled("click on any panel to start the polygon (esc or the button cancels)")
        elif drawing:
            imgui.text_disabled("click to add points; click the first point to close")
        else:
            imgui.text_disabled(self._status)
        imgui.pop_text_wrap_pos()
        imgui.separator()
        imgui.push_text_wrap_pos(0)
        imgui.text_disabled(self._selection_status())
        imgui.pop_text_wrap_pos()

    @property
    def device(self) -> str:
        return self._device

    @property
    def demixing_results(self) -> masknmf.DemixingResults | masknmf.CompressionArray:
        return self._demixing_results

    @property
    def fov_widget(self) -> fpl.NDWidget:
        return self._ndw_fov

    @property
    def traces(self) -> TracePlot:
        return self._traces

    @property
    def raw(self) -> masknmf.ArrayLike | np.ndarray | None:
        """The raw movie behind the "raw" panel, None without one."""
        return self._raw

    @property
    def shifts(self) -> np.ndarray | None:
        """The registration shifts behind the "shift (px)" panel, None without one."""
        return self._shifts

    @property
    def cell_stats(self) -> CellStats | None:
        """The per-signal stats behind the extra Signals-table columns, None without any."""
        return self._cell_stats

    def add_cell_stats(self, stats: CellStats | str | os.PathLike):
        """Join stat columns onto the Signals table, shown and sorted by the first; same-named columns are replaced."""
        if isinstance(stats, (str, os.PathLike)):
            stats = CellStats.read(stats)
        if self._order is None:
            raise ValueError("cell stats need demixed signals")
        if stats.values.shape[0] != self._order.n_items:
            raise ValueError(f"{stats.values.shape[0]} cell stat rows for {self._order.n_items} signals")
        if set(stats.names) & {"id", "area", "peak", "del"}:
            raise ValueError(f"cell stat names clash with the table's own columns: {stats.names}")
        new = stats
        self._cell_stats = new if self._cell_stats is None else self._cell_stats.join(new)
        self._shown_stats |= set(new.names)
        columns = {"area": self._order.columns["area"], "peak": self._order.columns["peak"]}
        columns.update(zip(stats.names, stats.values.T))
        columns["del"] = self._order.columns["del"]
        self._order.columns = columns
        self._order.sort_by, self._order.ascending = new.names[0], True
        self._order.rebuild()
        display(f"cell stats: {', '.join(new.names)} added; sorted by {new.names[0]}")

    def load_cell_order(self, order, name: str = "order"):
        """Add a column of ranks from signal ids in a custom order (a sequence, or a .npy / text file of ids)."""
        if isinstance(order, (str, os.PathLike)):
            order = np.load(order) if str(order).endswith(".npy") else np.loadtxt(order, dtype=np.int64, ndmin=1)
        self.add_cell_stats(CellStats.from_order(order, 0 if self._order is None else self._order.n_items, name))

    @property
    def reference_index(self) -> fpl.ReferenceIndices:
        return self._reference_index

    def show(self):
        return self.fov_widget.show()

    def close(self):
        self._ndw_fov.close()


def visualize_superpixels_peaks(init_results: masknmf.InitializationResults):
    superpixel_map = init_results.nmf_seed_map
    pure_superpixel_map = init_results.pure_nmf_seed_map
    correlation_image = init_results.corr_image

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
