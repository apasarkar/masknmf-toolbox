import os
import threading
from pathlib import Path
from dataclasses import replace
import numpy as np
import fastplotlib as fpl
from imgui_bundle import imgui, imgui_toggle, icons_fontawesome_6 as fa
from fastplotlib import ui
from fastplotlib.graphics.selectors._polygon import point_in_polygon
import pygfx
from fastplotlib.widgets.nd_widget._async import run_sync
import h5py
import torch
from collections.abc import Sequence
from collections import OrderedDict
import masknmf.arrays
from masknmf.arrays import SwitchableArray, TiffArray
from masknmf.utils import display
from functools import partial
from masknmf.visualization.imgui import (
    THEME,
    GROUP_COLORS,
    PathPrompt,
    RoiOrder,
    SourceRightClickMenu,
    TracePlot,
    CLICK_SLOP,
    component_at_pixel,
    draw_keybinds_popup,
    draw_panels_popup,
    PANELS_LABEL,
    PANELS_TIP,
    draw_path_prompt,
    draw_help_buttons,
    help_buttons_width,
    draw_range_filter,
    draw_roi_table,
    em,
    popup,
    opaque_popups,
    resolve_time_reference,
    section,
    to_vec4,
    grid,
    help_mark,
    right_aligned_text,
    button_colors,
    tooltip,
)
from masknmf.visualization.imgui.curation_help import draw_curation_help
from masknmf.visualization.imgui.keybinds import DEMIXING, pressed
from masknmf.visualization.imgui.options import OPTIONS_LABEL, draw_options_popup
from masknmf.visualization.rois import MARKED_COLOR, SELECTED_ALPHA, FootprintSet
from masknmf.visualization.summary_widget import SummaryImageViewer
from masknmf.demixing import CellStats, update_signals, write_curated
from masknmf.diagnostics import pmd_autocovariance_diagnostics
from masknmf.pipelines.configs.demixing_configs import NMFConfig
from masknmf.demixing._base_results import BaseResults

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
_HDF5_FILTERS = ["masknmf demixing results", "*.hdf5 *.h5", "All files", "*"]
_UNDO_DEPTH = 50  # ctrl+z snapshots kept
# every grid's captions, so the caption column is one width across the sections and the tabs
# what the panels open on, first come: raw | compressed+denoised | signals when a raw movie is there
_DEFAULT_ORDER = ("raw", "compressed+denoised", "signals", "background", "registered", "residual")
_CAPTIONS = (
    "masks", "contours", "sel masks", "sel contours", "color by", "traces",
    "filter", "range", "applied", "run", "options",
)
# signals selected together, in order of mutual contrast on the dark plot; no red, a mask marked for
# deletion is red
# compressed/signal/background/residual, in that order, so the 4 base lines read apart in the legend
_BASE_LINE_COLORS = (
    (0.85, 0.85, 0.85),
    (0.30, 0.85, 0.40),
    (0.95, 0.55, 0.15),
    (0.35, 0.65, 0.95),
)
# pink, apart from the four base lines and the green and navy trace backgrounds
_RAW_LINE_COLOR = (0.95, 0.45, 0.70)
# behind the traces, naming what the lines are: near-black with nothing shown, navy for the sources under a
# double-clicked pixel, forest green for a selection
_TRACE_MODES = {
    "normal": (0.02, 0.02, 0.03, 1.0),
    "source": (0.06, 0.09, 0.16, 1.0),
    "selection": (0.05, 0.14, 0.08, 1.0),
}


class SingleSessionDemixingVis:
    """
    View and curate demixing results. Takes whatever stage a run got to: masknmf.DemixingResults, a bare
    CompressionArray before demixing has run, or a registration array (with its raw input movie) before
    compression. Draw and export ROIs for a custom SignalDemixer.initialize_signals(is_custom=True) pass.
    Footprints show as feathered masks and/or contours over the summary image.

    With ``results_path`` set, "Demix" runs the drawn ROIs and the signals marked with "Delete" through the
    demixer's NMF pass (``nmf_config``, the pipeline defaults when None) and writes the outcome to a new
    ``<stem>.<timestamp>.curated.hdf5`` beside the file (``results.hdf5 -> results.<timestamp>.curated.hdf5``),
    never over it: drawn ROIs become ordinary signals, marked signals are gone, the new file's description says
    what was done, and the viewer moves on to it so further passes chain.

    The "Signals" tab lists every demixed signal; ctrl / shift select a group whose traces share the plot.
    Traces plot only with the Curation tab's "show selected traces" on (off by default): selecting then only
    highlights, however big the selection.
    Clicking the selected mask or its table row again deselects it; a pan or drag on a panel leaves the
    selection alone, and a double-click on a trace or shift panel only refits its axes. Esc deselects everything, ctrl+a groups every signal the table shows, and
    ctrl+z undoes the last mark, drawn roi, pixel average or deselect (a Demix empties the undo stack; roi
    vertex drags are not undone). Left / right step the movie a frame (shift: 10); m and c toggle the masks and
    contours, r arms a new roi. Every key is listed by the keybinds button at the top of both tabs (k) and the
    help button beside it (h) opens the curation help page. The Overlay section shows masks and contours in two pairs, each a checkbox
    and an opacity: "masks" / "contours" over every footprint, "sel masks" / "sel contours" over the selection
    and its group, which take their signal's mask color so a contour matches its trace.
    With "pixel traces" on (Curation tab checkbox or the p key, off by default), clicking an empty pixel adds the compressed movie's 5x5
    average there to the plot as if it were a grouped signal, and lists it at the top of the Signals table,
    marked. Pixel averages are diagnostic only: Demix and export ignore them, Delete drops them.
    A drawn roi gets the same kind of trace once it is closed ("roi n", groupable, in the table too) and,
    unlike a pixel average, is kept: Demix seeds the NMF pass with it and export writes it.
    The Signals tab's filter takes any column, del included (0 or 1): a slider with two lines over the
    column's span, everything in view at the full span. Its "apply" checkbox makes the signals on the filter
    switch's side of the range (outside by default) the selection, as ctrl+a does for the table, and keeps
    it following the lines as they move; editing the selection by hand switches it off. Delete then marks the
    selection like any other and remembers the filter it came from, listed under the range, and Demix writes
    which filter removed which signals into the curated file's description. The Curation tab's "color by"
    colors the masks and the table's ids by a column's rank instead of by signal id.
    draw (the Curation tab's polygon button, a) puts one region on any panel: a polygon that selects every signal
    in view whose center falls inside it (or outside, per its own switch): they form the group, highlighted in the
    panels and the table, and the selection follows the polygon as it is drawn and later dragged (its traces,
    when shown, plot once it settles). Add ROI (r) keeps the region as a drawn roi. The button again, esc, or a
    click or table pick that edits the selection by hand drops it and keeps the selection, so Delete (or the
    Curation tab's delete button) marks it like any other selection. Nothing is removed until the next Demix.

    ``raw`` (a movie, or a .tif path) adds the raw movie, ``registered`` (the results' registration replayed
    on it, for compression or demixing results) the registered one, and ``shifts`` (an array, or a motion
    correction hdf5 path) adds the registration shifts as a panel above the traces (piecewise rigid: the
    largest block shift per frame); results that hold their own ``raw_array``, ``registered_array`` or
    ``shifts`` show those when none are given.
    With ``results_path`` set, a lone .tif beside the results, and the
    registration shifts from the results file itself or from a motion_correction.hdf5 beside it, are picked up
    when their frames match the results; given ones must match. Up to three panels, one per movie the results
    hold, each switchable to any of them: the Panels button at the top of the Tools panel opens the array x
    panel matrix, and the top of a panel's right-click menu offers the same choice for that panel. The movies
    are raw, registered, compressed+denoised, the residual, the fitted background (when the demixer fit one)
    and the demixed signals under their masks, as available. They open on raw | compressed+denoised | signals, else
    compressed+denoised | signals | background, else whatever there is (raw | registered for a registration
    array, whose input movie is its raw and whose shifts are the shift traces). Switching keeps the zoom and
    the drawn rois. A registration array has no compressed movie to average, so pixel traces and drawn rois
    are off for it.

    The Signals table always carries the results' own stats (:meth:`CellStats.from_results`: mean, std, snr
    and skew of each demixed trace; fit, resid and bkgd from the roi averages the results hold), hidden until
    a right-click on a table header shows them; click a header to order the signals by a stat, then step
    through the top or bottom of the order. ``cell_stats`` (a :class:`CellStats`, or a .npy / .npz / .csv /
    .tsv it reads, one row per signal) joins more columns, shown, replacing same-named ones. ``cell_order``
    (signal ids in a custom order, or a
    .npy / text file of them) adds an "order" column of ranks and opens the table in that order; signals it
    leaves out sort last. :meth:`load_cell_order` and :meth:`add_cell_stats` (or File > load cell stats: a
    .txt of ids is an order, anything else stats) do the same with the results open; File > load results.hdf5
    swaps in another results file of the same movie. Every path the viewer asks for ("Export" too) is a
    window with a typed field, so it works on a remote kernel; "browse" there is the native dialog for a
    local one. Marked signals always come first. A Demix pass recomputes the results'
    stats for the new signals and drops given ones.

    :meth:`compute_lag1_acf` (``masknmf view --compression``) adds the lag-1 autocorrelation images of the movie
    the compression saw, of the compressed movie and of their residual to the Static images window.
    """

    def __init__(
        self,
        demixing_results: BaseResults | masknmf.DemixingResults
        | masknmf.CompressionArray
        | masknmf.BaseRegistrationArray,
        frame_timings: np.ndarray | list[np.ndarray] | None = None,
        ref_range: dict | None = None,
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
        registered: masknmf.ArrayLike | None = None,
        cell_stats: CellStats | str | os.PathLike | None = None,
        cell_order: Sequence[int] | np.ndarray | str | os.PathLike | None = None,
        frame_batch_size: int = 300,
    ):
        self._results_path = None if results_path is None else str(results_path)
        if nmf_config is None:
            ## At a baseline, we don't want to modify user-input ROIs
            base = NMFConfig(maxiter=40,
                             update_frequency=45)
        else:
            base = nmf_config
        self._min_brightness_cache = (
            0.0 if base.min_brightness is None else base.min_brightness
        )
        self.frame_batch_size=frame_batch_size ## Useful for any interactive analysis
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

        self._is_masknmf_result = isinstance(demixing_results, masknmf.DemixingResults)
        self._has_ac = hasattr(demixing_results, "signals_array")
        self._is_registration = isinstance(demixing_results, masknmf.BaseRegistrationArray)
        # a registration array computes on its strategy's device and hands frames over from output_device
        if not self._is_registration:
            self._demixing_results.to(self.device)
        self._shape = self.demixing_results.shape

        folder = None if self._results_path is None else Path(self._results_path).parent
        num_signals = demixing_results.spatial_demixed.shape[1] if self._has_ac else 0
        # the results' own stats, hidden; given stats join them, shown, replacing same-named columns
        self._cell_stats = CellStats.from_results(demixing_results) if self._is_masknmf_result else None
        self._hidden_stats = set() if self._cell_stats is None else set(self._cell_stats.names)
        if isinstance(cell_stats, (str, os.PathLike)):
            cell_stats = CellStats.read(cell_stats)
        if cell_stats is not None:
            if cell_stats.values.shape[0] != num_signals:
                raise ValueError(f"{cell_stats.values.shape[0]} cell stat rows for {num_signals} signals")
            if set(cell_stats.names) & {"id", "area", "peak", "del"}:
                raise ValueError(f"cell stat names clash with the table's own columns: {cell_stats.names}")
            self._cell_stats = cell_stats if self._cell_stats is None else self._cell_stats.join(cell_stats)
            self._hidden_stats -= set(cell_stats.names)
        if self._cell_stats is not None:
            display(f"cell stats: {', '.join(self._cell_stats.names)}; right-click a Signals table header to show them")

        # raw movie and shifts: data or a path, or found beside the results; a found mismatch is skipped, a given one raises
        found_raw = found_shifts = False
        if raw is None and self._is_registration:
            raw = demixing_results.input_movie
        if raw is None and isinstance(demixing_results, BaseResults):
            raw = demixing_results.raw_array
        if registered is None and isinstance(demixing_results, BaseResults):
            registered = demixing_results.registered_array
        if shifts is None and (self._is_registration or isinstance(demixing_results, BaseResults)):
            shifts = demixing_results.shifts
        if raw is None and folder is not None:
            tifs = sorted(p for ext in ("*.tif", "*.tiff") for p in folder.glob(ext))
            if len(tifs) == 1:
                raw, found_raw = tifs[0], True
        raw_src = raw if isinstance(raw, (str, os.PathLike)) else None
        if raw_src is not None:
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
                        found_shifts = "PiecewiseRigidRegistrationArray" in f or "RigidRegistrationArray" in f or "GradientRegistrationArray" in f
                    if found_shifts:
                        shifts = candidate
                        break
        shifts_src = shifts if isinstance(shifts, (str, os.PathLike)) else None
        # the registration template, the still a registration stage leaves: beside the shifts in the file, or on the array
        template = None
        if shifts_src is not None:
            with h5py.File(shifts_src, "r") as f:
                groups = [
                    g
                    for g in (
                        "PiecewiseRigidRegistrationArray",
                        "RigidRegistrationArray",
                        "GradientRegistrationArray",
                    )
                    if g in f
                ]
                if not groups:
                    raise ValueError(f"{shifts_src} holds no registration array")
                shifts = f[groups[0]]["shifts"][()]
                strategy = getattr(masknmf, groups[0])._strategy_cls.__name__
                if strategy in f and "template" in f[strategy]:
                    template = f[strategy]["template"][()]
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
                template = None
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
        if registered is not None and self._is_registration:
            raise ValueError("registered= goes with compression or demixing results; a registration array is the registered movie")
        if registered is not None and tuple(registered.shape) != tuple(self._shape):
            raise ValueError(f"registered movie has shape {tuple(registered.shape)}, the results have {tuple(self._shape)}")
        self._registered = registered
        if self._is_registration:
            template = demixing_results.strategy.template
        elif isinstance(registered, masknmf.BaseRegistrationArray):
            template = registered.strategy.template
        self._template = (
            None
            if template is None
            else np.asarray(template.cpu() if isinstance(template, torch.Tensor) else template, np.float32)
        )
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

        self._bind_arrays()
        self._summary_img = summary_img
        self._summary_name = "summary image" if summary_img_name is None else summary_img_name
        self._lag1 = {}
        self._stills = self._static_images()
        # one panel per movie, up to three, each switchable to any of them; they open in _DEFAULT_ORDER
        movies = self._movies()
        self._panels = OrderedDict()
        for i, name in enumerate([name for name in _DEFAULT_ORDER if name in movies][:3], start=1):
            self._panels[str(i)] = SwitchableArray(movies, self._shape)
            self._panels[str(i)].current = name
        n = len(self._panels)
        self._video_extents = {name: (i / n, (i + 1) / n, 0.0, 1.0) for i, name in enumerate(self._panels)}

        self._ndw_fov = fpl.NDWidget(
            ref_range,
            extents=self._video_extents,
            names=[*self._panels],
            controller_ids=[
                tuple(self._panels),
            ],
            size=(1200, 800),
        )

        self._reference_index = self._ndw_fov.indices
        self._panel_graphics = OrderedDict()
        for name, array in self._panels.items():
            self._panel_graphics[name] = self._ndw_fov[name].add_nd_image(
                array, ["time", "m", "n"], ["m", "n"], slider_maps={"time": frame_timings}, name=name
            )
            self._ndw_fov.figure[name].title = array.current
        self._fov_subplot = self._ndw_fov.figure[next(iter(self._panels))]
        self._ndw_fov.figure.set_imgui_right_click(SourceRightClickMenu(self._panel_choices, self._set_source))

        self._active_component = None
        self._marked = (
            set()
        )  # signal indices "Delete" has marked; removed on the next "Demix"
        self._show_traces = True  # plot the selection's traces; off, selecting only highlights
        self._show_raw_trace = True  # with a single signal, its trace re-estimated from the raw movie, when the results hold one
        self._roi_radius = 1  # a double-click splits the square this far around the pixel into its sources
        self._undo = []  # curation snapshots for ctrl+z, newest last
        self._group: list = []  # signals selected together; their traces share the plot
        self._order = None  # RoiOrder over the signals, built with the footprints
        self._follow = False
        self._scroll_to_current = False
        self._keybinds_open = False
        self._help_open = False
        self._options_open = False
        self._show_masks = show_masks
        self._mask_opacity = mask_opacity
        self._show_selected_masks = True
        self._selected_mask_opacity = SELECTED_ALPHA
        self._footprints = None
        self._mask_overlays = {}
        if self._has_ac:
            blank = np.zeros((*self._shape[1:3], 4), np.uint8)
            for name in self._panels:
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
            (*(("shift (px)",) if self._shift_lines else ()), *(("traces",) if self._pmd_array is not None or self._has_ac else ())),
            self._shape[0],
            frame_timings,
            autofit=False,
        )
        self._traces.dock(self._ndw_fov.figure, size=440 if self._shift_lines else 320)
        self._set_trace_mode("normal")
        if self._shift_lines:
            self._traces.set("shift (px)", self._shift_lines)
        self._traces.link(self.reference_index)
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
        self._show_selected_contours = True
        self._selected_contour_opacity = 0.7
        if self._ac_array is not None:
            self._make_selectors()

        # PolygonSelector -> color, subplot, compressed average over it (None until closed), pixel count, dirty
        self._rois = OrderedDict()
        self._active_roi = None
        self._status = ""
        self._panels_open = False
        self._summary = SummaryImageViewer(self._ndw_fov.figure, title="Static images")
        # typed-path windows, so they work on a remote kernel; browse there is the native dialog
        self._export_prompt = PathPrompt(
            "Export ROIs", os.path.join(os.getcwd(), "rois.npz"), "export", "a .npz of the drawn rois", "save", _NPZ_FILTERS
        )
        self._stats_prompt = PathPrompt(
            "Load cell stats", "", "load", ".npy / .npz / .csv / .tsv of stats, or a .txt of signal ids in order", "open", _STATS_FILTERS
        )
        self._results_prompt = PathPrompt(
            "Load results", self._results_path or "", "load", "a results.hdf5 of the same movie", "open", _HDF5_FILTERS
        )
        # the Signals tab's filter: select keeps the group on the filter switch's side of the range, and a Delete
        # of that selection records the filter in _filters for the curated file
        self._filter_select = False
        self._filter_side = set()
        self._filters = []
        self._press = None  # screen position of the last pointer press on a video panel
        self._press_drawn = False  # that press drew the region, so the click it makes is not a pick
        self._same_spot = False
        # the one drawn region (draw / a): a polygon that selects the signals it holds until + keeps it as a roi
        # or esc drops it; armed = the next press on any video panel starts it there
        self._armed = False
        self._region = None
        self._region_panel = None
        self._region_key = None  # the vertices / side / filter its hits were last computed for
        self._region_hits = None  # the signals it selected, None until it first did
        self._region_outside = False
        self._filter_outside = True

        self._bind_click_handlers()

        for subplot in self._ndw_fov.figure:
            subplot.tooltip.enabled = False
            subplot.toolbar = False

        # 31.5 em fits the Panels, Static images, guide and keybinds buttons on one row, with a few px to spare
        self._ndw_fov.figure.add_imgui_window(
            self._draw_side_panel,
            location="right",
            size=round(31.5 * self._ndw_fov.figure.default_imgui_font.legacy_size),
            title="Tools",
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
        visible = self._show_masks or self._show_selected_masks
        rgba = (
            self._footprints.rgba(
                tuple(self._shape[1:3]),
                self._mask_opacity if self._show_masks else 0.0,
                self._active_component if self._show_selected_masks else None,
                self._marked,
                {
                    k: rgb
                    for k, rgb in self._group_colors().items()
                    if isinstance(k, int)
                }
                if self._show_selected_masks
                else {},
                self._selected_mask_opacity,
            )
            if visible
            else None
        )
        for overlay in self._mask_overlays.values():
            overlay.visible = visible
            if rgba is not None:
                overlay.data = rgba

    def _bind_arrays(self):
        if self._has_ac:
            self._pmd_array = self.demixing_results.compression_array
            self._fluctuating_background_array = (
                self.demixing_results.fluctuating_background_array
            )
            self._residual_array = self.demixing_results.residual_array
            self._ac_array = self.demixing_results.signals_array
            # an all-zero background term means the demixer never fit one
            if self._is_masknmf_result:
                self._has_background = bool(torch.count_nonzero(self.demixing_results.factorized_background_term1))
            else:
                self._has_background = self.demixing_results.fluctuating_background_array is not None ## Ok to show nothing here for now
        else:
            self._pmd_array = None if self._is_registration else self.demixing_results
            self._fluctuating_background_array = None
            self._residual_array = None
            self._ac_array = None
            self._has_background = False

    def _movies(self) -> OrderedDict:
        """The movies the results hold, name to (frames, height, width) array, in pipeline order: what a panel can show."""
        movies = OrderedDict()
        if self._raw is not None:
            movies["raw"] = self._raw
        if self._is_registration:
            movies["registered"] = self.demixing_results
        elif self._registered is not None:
            movies["registered"] = self._registered
        if self._pmd_array is not None:
            movies["compressed+denoised"] = self._pmd_array
        if self._residual_array is not None:
            movies["residual"] = self._residual_array
        if self._has_background:
            movies["background"] = self._fluctuating_background_array
        if self._ac_array is not None:
            movies["signals"] = self._ac_array
        return movies

    def _static_images(self) -> dict:
        """The stills the results hold, name to 2-D image, for the Static images window."""
        stills = {}
        if self._template is not None:
            stills["registration template"] = self._template
        if self._summary_img is not None:
            stills[self._summary_name] = self._summary_img
        if self._has_ac and self.demixing_results.global_residual_correlation_image is not None:
            stills["residual correlation image"] = self.demixing_results.global_residual_correlation_image.cpu().numpy()
        if self._pmd_array is not None:
            stills["mean image"] = self._pmd_array.mean_image.cpu().numpy()
            stills["noise variance image"] = self._pmd_array.noise_variance_image.cpu().numpy()
        stills.update(self._lag1)
        return stills

    def _panel_choices(self, panel: str) -> tuple[list, str]:
        """What ``panel`` can show and what it shows, for its right-click menu."""
        return list(self._panels[panel].sources), self._panels[panel].current

    def _set_source(self, panel: str, name: str):
        """Show movie ``name`` in ``panel``: it re-slices and refits its color limits; zoom and rois stay."""
        self._panels[panel].current = name
        self._ndw_fov.figure[panel].title = name
        # re-slice now, not on the scheduled fetch, so the color limits refit to the new frame; then what the
        # spatial_func setter does after a change of what the slicer sees
        graphic = self._panel_graphics[panel]
        run_sync(graphic._set_indices_())
        graphic.slicer._recompute_histogram()
        graphic._reset_histogram()

    def _bind_click_handlers(self):
        """Re-attach the click handlers to every video panel: NDGraphic.data= replaces the graphic instance."""
        for name, graphic in self._panel_graphics.items():
            graphic.graphic.add_event_handler(
                partial(self._pointer_down, name), "pointer_down"
            )
            graphic.graphic.add_event_handler(self._click_update, "click")
            graphic.graphic.add_event_handler(self._source_click, "double_click")

    def _source_click(self, ev: pygfx.PointerEvent):
        """
        emulate click_update from https://github.com/apasarkar/masknmf-toolbox/blob/f5c22fa01d6a1c87ab120592c0ec7cd5a1a01567/masknmf/visualization/demixing_vis.py#L328
        split the compressed average over the square around a double-clicked pixel into its sources.
        """

        if self._ac_array is None or not self._show_traces or self._press_drawn or not self._same_spot:
            return
        num_frames, height, width = self._shape
        col, row = ev.pick_info["index"]

        col_start, col_stop = (
            max(0, col - self._roi_radius),
            min(width, col + self._roi_radius + 1),
        )
        row_start, row_stop = (
            max(0, row - self._roi_radius),
            min(height, row + self._roi_radius + 1),
        )

        pmd_trace = np.mean(
            self._pmd_array[:, row_start:row_stop, col_start:col_stop], axis=(1, 2)
        )
        residual_trace = np.mean(
            self._residual_array[:, row_start:row_stop, col_start:col_stop], axis=(1, 2)
        )
        background_trace = np.mean(
            self._fluctuating_background_array[
                :, row_start:row_stop, col_start:col_stop
            ],
            axis=(1, 2),
        )

        separated_ac_signals, unique_signals = extract_per_trace_roi_averages(
            self._ac_array, slice(row_start, row_stop), slice(col_start, col_stop)
        )
        self._selected_signals = None
        lines = [("compressed", pmd_trace, _BASE_LINE_COLORS[0])]
        if separated_ac_signals is not None:
            # each source in its mask's color, so the split reads against the panels and the table
            lines += [
                (f"signal {k}", trace, self._footprints.color(int(k)))
                for k, trace in zip(unique_signals, separated_ac_signals)
            ]
        lines.append(("background", background_trace, _BASE_LINE_COLORS[2]))
        lines.append(("residual", residual_trace, _BASE_LINE_COLORS[3]))
        self._traces.set("traces", lines)
        self._set_trace_mode("source")
        self._status = f"sources over the {row_stop - row_start}x{col_stop - col_start} square at ({row}, {col})"

    def _pointer_down(self, name: str, ev: pygfx.PointerEvent):
        # pygfx reports any two quick presses on one panel as a double-click, however far apart
        self._same_spot = (
            self._press is not None
            and abs(ev.x - self._press[0]) + abs(ev.y - self._press[1]) <= CLICK_SLOP
        )
        self._press = (ev.x, ev.y)
        self._press_drawn = self._drawing()
        if self._armed:
            self._begin_region(name)

    def _set_gray_cmaps(self):
        for g in self._panel_graphics.values():
            g.graphic.cmap = "gray"

    def _video_graphics(self):
        """The NDImage wrapper for every video panel, in panel order."""
        return tuple(self._panel_graphics.values())

    def _make_selectors(self):
        """(Re)build the footprint selectors over the current signals."""
        show = self._show_contours
        # the selected and grouped components' contours; the roi panel's "contours" adds every footprint's
        self._image_selector = fpl.ImageHighlightSelector(
            lut_wrap="repeat",
            selection_options={"pixels": self._ac_array.contours},
            options_color="w",
            options_alpha=self._contour_opacity,
            alpha=self._selected_contour_opacity if self._show_selected_contours else 0.0,
        )
        self._set_contours(show)

    def _load_results(self, results: masknmf.DemixingResults):
        """Swap in re-demixed results: every movie panel, the selectors and the summary image follow."""
        results.to(self.device)
        self._demixing_results = results
        self._cell_stats = CellStats.from_results(results)
        self._hidden_stats = set(self._cell_stats.names)
        self._bind_arrays()
        self._clear_rois()
        self._armed = False
        self._drop_region()
        self._marked.clear()
        self._filter_select = False
        self._filter_side = set()
        self._filters.clear()
        self._group.clear()
        self._clear_component()
        self._selected_signals = None
        self._clear_traces()
        self._pixels.clear()
        # fresh wrappers, so every panel gets a new graphic instance: keeps the re-bound click handlers from doubling up
        movies = self._movies()
        for name, panel in list(self._panels.items()):
            keep = panel.current
            self._panels[name] = SwitchableArray(movies, self._shape)
            if keep in movies:
                self._panels[name].current = keep
            self._panel_graphics[name].data = self._panels[name]
            self._ndw_fov.figure[name].title = self._panels[name].current
        self._bind_click_handlers()
        self._stills = self._static_images()
        if self._summary.is_open:
            self._summary.set_images(self._stills)
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
        # each recorded filter minus any of its signals unmarked since
        filters = [dict(kept, signals=kept["signals"] & set(drop)) for kept in self._filters]
        filters = [kept for kept in filters if kept["signals"]]
        self._status = f"demixing: +{masks.shape[-1]} roi(s), -{len(drop)} signal(s)..."
        self._worker = threading.Thread(
            target=self._demix, args=(masks, drop, filters), daemon=True
        )
        self._worker.start()

    def _demix(self, masks: np.ndarray, drop: list, filters: list):
        try:
            results = update_signals(
                self.demixing_results,
                masks,
                drop,
                self._nmf_config,
                device=self.device,
                frame_batch_size=self.frame_batch_size
            )
            path = write_curated(self._results_path, results, drop, masks.shape[-1], filters)
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

    def _click_update(self, ev: pygfx.PointerEvent):
        """
        Priority: a drawn roi, then an existing component, else clear the selection.
        ctrl / shift on a component grow the group instead of replacing the selection.
        """
        if self._press_drawn or imgui.get_io().want_capture_mouse:
            return
        # pygfx reports a click after any press and release on one graphic, a pan drag included
        if self._press is not None and (
            abs(ev.x - self._press[0]) + abs(ev.y - self._press[1]) > CLICK_SLOP
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
            k: GROUP_COLORS[i % len(GROUP_COLORS)] for i, k in enumerate(self._group)
        }

    def _highlighted(self) -> list:
        picks = [k for k in self._group if isinstance(k, int)]
        if self._active_component is not None and self._active_component not in picks:
            picks.append(self._active_component)
        return picks

    def _sync_highlight(self):
        """The contour selector and the mask overlay both show the group plus the selection, in the same colors."""
        if self._image_selector is not None:
            picks = self._highlighted()
            grouped = self._group_colors()
            if picks:
                # the contour of a highlighted signal takes its mask's color, so it matches its trace too
                self._image_selector.lut = np.array(
                    [
                        (
                            *(
                                MARKED_COLOR
                                if k in self._marked
                                else grouped.get(k, self._footprints.color(k))
                            ),
                            1.0,
                        )
                        for k in picks
                    ],
                    np.float32,
                )
            self._image_selector.selection = picks
        self._refresh_masks()

    def _update_traces(self):
        """
        One signal: its compressed / signal / background / residual roi averages, and when the results hold
        temporal_demixed_raw and "raw trace" is on, the signal line from those raw traces drawn under it. A group, or any pixel
        average or drawn roi: one line per member, colored like its mask or table row - a signal's
        demixed trace, a pixel average or drawn roi's compressed average.
        Nothing unless "show selected traces" is on.
        """
        if not self._show_traces:
            self._selected_signals = None
            self._traces.set("traces", [])
            self._set_trace_mode("normal")
            return
        results = self.demixing_results
        if len(self._group) > 1 or any(not isinstance(k, int) for k in self._group):
            self._selected_signals = []
            lines = []
            for i, k in enumerate(self._group):
                rgb = GROUP_COLORS[i % len(GROUP_COLORS)]
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
                    # lam to scale
                    _y, _x, lam = self._footprints.footprints[k]
                    trace = float(lam.mean()) * results.temporal_demixed[:, k]
                    lines.append((f"signal {k}", trace.cpu().numpy(), rgb))
                self._selected_signals.append(k)
        elif self._active_component is not None:
            k = self._active_component
            self._selected_signals = None
            ypix, xpix, _lam = self._footprints.footprints[k]
            support = torch.as_tensor(
                ypix.astype(np.int64) * self._shape[2] + xpix, device=results.spatial_demixed.device
            )
            # the signal movie averaged over the footprint's support, like the stored roi averages; averaging the
            # footprints first keeps a huge roi from building a (pixels, frames) matrix
            weights = torch.sparse.sum(torch.index_select(results.spatial_demixed, 0, support), dim=0).to_dense() / len(support)
            signal = results.temporal_demixed @ weights
            traces = (
                results.compression_array_roi_averages[k],
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
            if self._show_raw_trace and results.temporal_demixed_raw is not None:
                raw = results.temporal_demixed_raw @ weights
                # before the signal line so the signal draws over it
                lines.insert(1, ("raw", raw.cpu().numpy(), _RAW_LINE_COLOR))
        else:
            self._selected_signals = None
            self._clear_traces()
            return
        self._traces.set("traces", lines)
        self._set_trace_mode("selection")

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
        """Drop the selection, the group and the pixel averages: esc."""
        if self._group or self._pixels or self._active_component is not None or self._active_roi is not None:
            self._snapshot()
        self._pixels.clear()
        self.group_clear()
        self._clear_component()
        self._active_roi = None
        self._selected_signals = None
        self._clear_traces()
        self._filter_select = False

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
        for name in self._panels:
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

    def _toggle_trace_follow(self):
        self._traces.follow = not self._traces.follow

    def _toggle_show_traces(self):
        self._show_traces = not self._show_traces
        self._update_traces()

    def _toggle_raw_trace(self):
        self._show_raw_trace = not self._show_raw_trace
        self._update_traces()

    def _step(self, delta: int):
        """Move the table cursor and select what it lands on."""
        if self._order is not None and self._order.step(delta):
            self.group_clear()
            self._select_component(self._order.current)

    def _set_contours(self, show: bool):
        """``show`` draws every other footprint's contour at the contour opacity; the selection's has its own pair."""
        self._show_contours = show
        if self._image_selector is None:
            return
        self._image_selector.options_alpha = self._contour_opacity if show else 0.0
        for g in self._video_graphics():
            if g is not None and g.graphic not in self._image_selector.graphics:
                self._image_selector.add_graphic(g.graphic)

    def _drawing(self) -> bool:
        """The region has the pointer: armed, or its vertices are being placed or dragged."""
        return self._armed or (self._region is not None and self._region._move_info.mode is not None)

    def _roi_at(self, col: int, row: int):
        for selector in reversed(self._rois):
            polygon = selector.selection[:, :2]
            if polygon.shape[0] >= 3 and point_in_polygon((col, row), polygon):
                return selector
        return None

    def _commit_region(self):
        """The Add ROI action (r): the settled region becomes a roi, redrawn in the next roi color, and the selection."""
        vertices = np.array(self._region.selection)
        name = self._region_panel
        self._snapshot()
        self.group_clear()
        self._clear_component()
        self._drop_region()
        color = _ROI_COLORS[len(self._rois) % len(_ROI_COLORS)]
        selector = self._panel_graphics[name].graphic.add_polygon_selector(
            fill_color=color,
            edge_color=color,
            vertex_color=color,
            edge_thickness=2,
            vertex_size=8,
        )
        selector.selection = vertices
        selector._end_move_mode()
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

    def _start_region(self):
        """The draw button (a): arm a region for the next press on a video panel, or drop the one there is (the selection stays)."""
        if self._armed:
            self._armed = False
        elif self._region is not None:
            self._drop_region()
        else:
            self._armed = True
            self._filter_select = False

    def _begin_region(self, name: str):
        """
        Create the armed region on panel ``name``. The press that got here also places its first
        vertex: pygfx bubbles it up to the renderer last, where the selector's fresh handlers wait.
        """
        self._armed = False
        self._snapshot()
        self._region = self._panel_graphics[name].graphic.add_polygon_selector(
            fill_color=(0.0, 0.0, 0.0, 0.0),
            edge_color=THEME.accent[:3],
            vertex_color=THEME.accent[:3],
            edge_thickness=2,
            vertex_size=8,
        )
        self._region_panel = name

    def _poll_region(self):
        """
        The region selects the signals in view it holds, following it as it is drawn or dragged. Left with
        under three vertices, or once the selection is edited by hand, it is dropped.
        """
        if self._region is None:
            return
        polygon = self._region.selection[:, :2]
        moving = self._region._move_info.mode is not None
        if polygon.shape[0] < 3:
            if not moving:
                self._drop_region()
            return
        signals = [k for k in self._group if isinstance(k, int)]
        if self._region_hits is not None and not moving and signals != self._region_hits:
            self._drop_region()
            return
        key = (polygon.tobytes(), self._region_outside, self._order.range_limits, moving)
        if key == self._region_key:
            return
        self._region_key = key
        view = self._order.order
        centers = self._ac_array.centers.cpu().numpy()[view]
        inside = np.fromiter(
            (point_in_polygon((col, row), polygon) for row, col in centers),
            bool,
            len(view),
        )
        hits = [int(k) for k in (view[~inside] if self._region_outside else view[inside])]
        if hits != self._region_hits:
            self._region_hits = hits
            self._group[:] = hits
            self._active_roi = None
            self._active_pixel = None
            self._active_component = hits[-1] if hits else None
            if self._active_component is not None:
                self._order.goto(self._active_component)
            self._sync_highlight()
            where = "outside" if self._region_outside else "inside"
            self._status = f"region: {len(hits)} signal(s) {where}; + keeps it as a roi (r), esc drops it"
        # no traces while the polygon is being drawn or dragged; they plot once it settles
        if moving:
            self._selected_signals = None
            self._clear_traces()
        else:
            self._update_traces()

    def _drop_region(self):
        if self._region is None:
            return
        if self._region._move_info.mode is not None:
            self._region._end_move_mode()
        self._ndw_fov.figure[self._region_panel].delete_graphic(self._region)
        self._region = None
        self._region_key = None
        self._region_hits = None
        self._update_traces()

    def _mark(self, signals, on: bool, record: bool = True):
        """Mark ``signals`` for deletion on the next demix, or unmark them; the masks and the table follow."""
        signals = {int(k) for k in signals}
        if record and signals:
            self._snapshot()
        if on:
            self._marked |= signals
            # a filter's selection: remember which filter deleted what, for the curated file
            if self._filter_select and signals & self._filter_side:
                self._filters.append(
                    {
                        "column": self._order.range_column,
                        "range": tuple(self._order.range_limits),
                        "outside": self._filter_outside,
                        "signals": signals & self._filter_side,
                    }
                )
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
                "filter_select": self._filter_select,
                "filter_side": set(self._filter_side),
                "region_outside": self._region_outside,
                "filter_outside": self._filter_outside,
                "filters": [dict(kept, signals=set(kept["signals"])) for kept in self._filters],
            }
        )
        del self._undo[:-_UNDO_DEPTH]

    def undo(self):
        """Ctrl+z: back to the last snapshot; a deleted roi is redrawn, a live region is dropped."""
        if not self._undo:
            return
        state = self._undo.pop()
        self._armed = False
        self._drop_region()
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
        self._filter_select = state["filter_select"]
        self._filter_side = set(state["filter_side"])
        self._region_outside = state["region_outside"]
        self._filter_outside = state["filter_outside"]
        self._filters = [dict(kept, signals=set(kept["signals"])) for kept in state["filters"]]
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
        """
        The Delete action: drop an active drawn roi, else the active pixel average, else mark the selected signals;
        marking a single signal selects the next row, so a review keeps its place in the table.
        """
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
            on = not all(k in self._marked for k in signals)
            # a marked signal joins the marked rows at the top of the table: the selection moves on to the row after it
            view = [int(k) for k in self._order.order]
            follows = view[view.index(signals[0]) + 1] if on and len(signals) == 1 and signals[0] in view[:-1] else None
            self._mark(signals, on)
            if follows is not None:
                self.group_clear()
                self._select_component(follows)

    def _clear_traces(self):
        self._active_pixel = None
        if self._pmd_array is not None:
            self._traces.set("traces", [])
        self._set_trace_mode("normal")

    def _set_trace_mode(self, mode: str):
        self._traces.background = _TRACE_MODES[mode]
        self._traces.title = f"Traces (mode: {mode})"

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

    def _handle_keys(self):
        io = imgui.get_io()
        if io.want_text_input:
            return
        if pressed(DEMIXING["delete"]) and self._worker is None:
            self._delete_selected()
        if pressed(DEMIXING["escape"]):
            if self._help_open or self._keybinds_open or self._options_open:
                self._help_open = self._keybinds_open = self._options_open = False
            elif self._armed:
                self._armed = False
            elif self._region is not None:
                self._drop_region()
            else:
                self.deselect()
        if pressed(DEMIXING["select_all"]) and self._order is not None:
            self._group[:] = [int(k) for k in self._order.order]
            self._sync_highlight()
            self._update_traces()
        if pressed(DEMIXING["undo"]):
            self.undo()
        stride = 10 if io.key_shift else 1
        if pressed(DEMIXING["down"]):
            self._step(stride)
        if pressed(DEMIXING["up"]):
            self._step(-stride)
        if pressed(DEMIXING["right"]):
            self._step_frame(stride)
        if pressed(DEMIXING["left"]):
            self._step_frame(-stride)
        if pressed(DEMIXING["masks"]) and self._image_selector is not None:
            self._show_masks = not self._show_masks
            self._refresh_masks()
        if pressed(DEMIXING["contours"]):
            self._set_contours(not self._show_contours)
        if pressed(DEMIXING["follow"]):
            self._toggle_follow()
        if pressed(DEMIXING["trace_follow"]) and self._traces.panels:
            self._toggle_trace_follow()
        if pressed(DEMIXING["pixel_trace"]) and self._pmd_array is not None:
            self._set_pixel_traces(not self._pixel_traces)
        if pressed(DEMIXING["roi"]) and self._order is not None:
            if self._region is not None and self._region._move_info.mode is None:
                self._commit_region()
            elif not self._drawing():
                self._start_region()
        if pressed(DEMIXING["poly"]) and self._order is not None:
            self._start_region()
        if pressed(DEMIXING["help"]):
            self._help_open = not self._help_open
        if pressed(DEMIXING["keybinds"]):
            self._keybinds_open = not self._keybinds_open

    def _step_frame(self, delta: int):
        """Move the time index by delta frames; every panel and the trace playhead follow."""
        step = self.reference_index.ref_ranges["time"].step
        self.reference_index.set({"time": self.reference_index["time"] + delta * step})

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
        return "click a mask or roi to see its trace; double-click any pixel to split it into its sources"

    def _draw_side_panel(self):
        """
        Docked at "right" (the NDWidget owns "bottom"): a File menu, the Panels and Static images buttons, then the
        roi tools and the signal table as tabs.
        """
        opaque_popups()
        self._poll_worker()
        self._poll_rois()
        self._poll_region()
        self._handle_keys()
        # a child carries the menu bar, so the docked window itself needs no flag
        imgui.begin_child(
            "##menu",
            imgui.ImVec2(0, 0),
            imgui.ChildFlags_.auto_resize_y | imgui.ChildFlags_.always_auto_resize,
            imgui.WindowFlags_.menu_bar,
        )
        if imgui.begin_menu_bar():
            if imgui.begin_menu("File"):
                if imgui.menu_item_simple(f"{fa.ICON_FA_FILE_IMPORT}  load results.hdf5", enabled=self._has_ac):
                    self._results_prompt.start(self._results_path or "")
                tooltip(
                    "swap in another results.hdf5 of the same movie: every panel, the signals and their stats follow"
                    if self._has_ac
                    else "needs a viewer opened on demixing results"
                )
                if imgui.menu_item_simple(f"{fa.ICON_FA_CHART_SIMPLE}  load cell stats", enabled=self._order is not None):
                    self._stats_prompt.start()
                tooltip(
                    "one row per signal, in signal id order, as sortable table columns:\n"
                    "- .npy: a (signals,) or (signals, stats) array, or a structured array of stats\n"
                    "- .npz: one (signals,) array per stat, named by key\n"
                    "- .csv / .tsv: a header row of names, then one row per signal\n"
                    "- .txt: signal ids in a custom order, becomes the 'order' column"
                )
                imgui.separator()
                if imgui.menu_item_simple(OPTIONS_LABEL):
                    self._options_open = True
                imgui.end_menu()
            imgui.end_menu_bar()
        imgui.end_child()
        if imgui.button(PANELS_LABEL):
            self._panels_open = True
        if imgui.is_item_hovered():
            imgui.set_tooltip(PANELS_TIP)
        imgui.same_line()
        stills = self._stills
        imgui.begin_disabled(not stills)
        if imgui.button(f"{fa.ICON_FA_IMAGE} Static images"):
            self._summary.set_images(stills)
            self._summary.open()
        imgui.end_disabled()
        if imgui.is_item_hovered(imgui.HoveredFlags_.allow_when_disabled):
            imgui.set_tooltip(
                "browse every still the results hold at full size: zoom, colormap, contrast, pixel values"
                if stills
                else "the results hold no still images"
            )
        # the guide and keybinds buttons right-aligned on the same row, or on a row of their own when it is too narrow
        buttons_w = help_buttons_width("Curation Guide")
        imgui.same_line()
        if imgui.get_content_region_avail().x < buttons_w + em(0.6):
            imgui.new_line()
        imgui.set_cursor_pos_x(imgui.get_cursor_pos_x() + imgui.get_content_region_avail().x - buttons_w)
        self._help_open, self._keybinds_open = draw_help_buttons(self._help_open, self._keybinds_open, "Curation Guide")
        # each tab's body is a child that scrolls on its own: the menu, the buttons and the tabs stay put
        if imgui.begin_tab_bar("##side"):
            if imgui.begin_tab_item("Curation")[0]:
                imgui.begin_child("##curation_tab")
                self._draw_roi_tools()
                imgui.end_child()
                imgui.end_tab_item()
            if imgui.begin_tab_item("Signals")[0]:
                imgui.begin_child("##signals_tab")
                self._draw_signal_tab()
                imgui.end_child()
                imgui.end_tab_item()
            imgui.end_tab_bar()
        self._keybinds_open = draw_keybinds_popup(DEMIXING, self._keybinds_open)
        self._help_open, self._keybinds_open = draw_curation_help(self._help_open, self._keybinds_open)
        self._options_open = draw_options_popup(self._ndw_fov.figure, self._options_open)
        path = draw_path_prompt(self._export_prompt)
        if path is not None:
            try:
                path = self.export_rois(path)
                self._status = f"exported {len(self._rois)} roi(s) to {path}"
                self._export_prompt.open = False
            except (OSError, ValueError) as e:
                self._export_prompt.status = f"export failed: {e}"
        path = draw_path_prompt(self._stats_prompt)
        if path is not None:
            try:
                if path.lower().endswith(".txt"):
                    self.load_cell_order(path)
                else:
                    self.add_cell_stats(path)
                self._status = f"cell stats loaded from {os.path.basename(path)}"
                self._stats_prompt.open = False
            except (OSError, ValueError, TypeError) as e:
                self._stats_prompt.status = f"cell stats failed: {e}"
        path = draw_path_prompt(self._results_prompt)
        if path is not None:
            try:
                results = masknmf.DemixingResults.from_hdf5(path, device=self.device)
                if tuple(results.shape) != tuple(self._shape):
                    raise ValueError(f"results of shape {tuple(results.shape)} for a {tuple(self._shape)} movie")
                self._load_results(results)
                self._results_path = path
                self._status = f"loaded {os.path.basename(path)}"
                self._results_prompt.open = False
            except (OSError, KeyError, ValueError, TypeError) as e:
                self._results_prompt.status = f"load failed: {e}"
        self._panels_open = draw_panels_popup(self._panels, self._panels_open, self._set_source)
        self._summary.draw()

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
        names = () if self._cell_stats is None else self._cell_stats.names
        if self._order is not None and self._order.range_column is not None:
            self._draw_filter()
        footer = imgui.get_frame_height_with_spacing() * 2.5
        if imgui.begin_child("##signal_table", imgui.ImVec2(0, -footer)):
            columns = ("id", "area", "peak", *names, "del")
            formatters = {name: partial(self._format_cell, name) for name in columns[1:]}
            colors = self._group_colors()
            self._scroll_to_current = draw_roi_table(
                order,
                columns,
                formatters,
                self._scroll_to_current,
                table_id="signals",
                hidden=self._hidden_stats,
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

    def _draw_filter(self):
        """The filter over the table: its column, "apply" with its side switch, the range, and the filters applied."""
        order = self._order
        g = grid(_CAPTIONS)
        right = imgui.get_cursor_pos_x() + imgui.get_content_region_avail().x
        slider_w = min(0.7 * imgui.get_window_width(), right - g.cell_x[0] - g.mark_w)
        g.row("filter")
        imgui.set_next_item_width(g.w)
        glow = self._filter_select
        if glow:
            for color, value in (
                (imgui.Col_.frame_bg, THEME.emphasis),
                (imgui.Col_.frame_bg_hovered, THEME.emphasis_hover),
                (imgui.Col_.button, THEME.emphasis),
                (imgui.Col_.button_hovered, THEME.emphasis_hover),
                (imgui.Col_.text, (0.05, 0.05, 0.05)),
            ):
                imgui.push_style_color(color, to_vec4(value))
        # popped right after the preview, so the dropdown's items keep the normal text color
        opened = imgui.begin_combo("##range_column", order.range_column)
        if glow:
            imgui.pop_style_color(5)
        picked = None
        if opened:
            for name in order.columns:
                if imgui.selectable(name, name == order.range_column)[0]:
                    picked = name
            imgui.end_combo()
        if picked is not None and picked != order.range_column:
            # a new column starts at its full span; the selection stays as it is until apply is on again
            order.set_range_column(picked)
            order.rebuild()
            self._filter_select = False
        tooltip("Filter column: the range below spans it (del is 0 or 1); picking one puts the range back at its full span")
        imgui.same_line(0, g.gap)
        inner = imgui.get_style().item_inner_spacing.x
        need = imgui.get_frame_height() + inner + imgui.calc_text_size("apply").x + g.gap + _side_switch_width()
        if imgui.get_content_region_avail().x < need:
            imgui.new_line()
            imgui.same_line(g.cell_x[0])
        # a selection edited by hand is no longer the filter's: the checkbox switches itself off
        if self._filter_select and {k for k in self._group if isinstance(k, int)} != self._filter_side:
            self._filter_select = False
        toggled, self._filter_select = imgui.checkbox("apply", self._filter_select)
        if toggled and self._filter_select:
            # the filter takes over from the region
            self._armed = False
            self._drop_region()
        tooltip(
            "Apply to selection: the signals on the switch's side of the range become the selection, as ctrl+a does "
            "for the table, and follow the lines as they move; Delete marks them and records the filter. Editing "
            "the selection by hand switches this off"
        )
        imgui.same_line(0, g.gap)
        flipped, self._filter_outside = _side_switch(
            "filter",
            self._filter_outside,
            self._filter_select,
            "The filter takes the signals inside or outside the range"
            + ("" if self._filter_select else "; lit while it drives the selection"),
        )
        imgui.same_line(0, g.gap)
        right_aligned_text(f"{len(order.order)} / {order.n_items}")
        g.row("range")
        moved = draw_range_filter(order, "_signals", slider_w)
        if moved:
            order.rebuild()
        tooltip("Range: drag a line to move it, double-click for the full span; the table shows what is inside")
        for i, kept in enumerate(self._filters):
            g.row("applied" if i == 0 else "")
            fmt = "%.0f" if np.asarray(order.columns[kept["column"]]).dtype.kind in "iub" else "%.3g"
            imgui.push_text_wrap_pos(0)
            imgui.text_disabled(
                f"{kept['column']} {'outside' if kept['outside'] else 'inside'} "
                f"{fmt % kept['range'][0]} - {fmt % kept['range'][1]}: {len(kept['signals'])} deleted"
            )
            imgui.pop_text_wrap_pos()
            tooltip("the filter a Delete came from, written into the curated file's description on demix; ctrl+z undoes the delete")
        if self._filter_select and (toggled or moved or flipped):
            values = np.asarray(order.columns[order.range_column], dtype=np.float64)
            inside = (values >= order.range_limits[0]) & (values <= order.range_limits[1])
            self._filter_side = set(
                np.flatnonzero(np.isfinite(values) & ~inside if self._filter_outside else inside).tolist()
            )
            self._group[:] = sorted(self._filter_side)
            self._sync_highlight()
            self._update_traces()
        imgui.dummy(imgui.ImVec2(0, em(0.2)))

    def _draw_roi_tools(self):
        g = grid(_CAPTIONS)
        # sliders: seven tenths of the panel, or what their row has left before the (?) mark
        right = imgui.get_cursor_pos_x() + imgui.get_content_region_avail().x
        slider_w = min(0.7 * imgui.get_window_width(), right - g.cell_x[0] - g.mark_w)

        section("OVERLAY")
        if self._image_selector is not None:
            changed, show = imgui.checkbox("masks", self._show_masks)
            if changed:
                self._show_masks = show
                self._refresh_masks()
            g.cell(0)
            imgui.set_next_item_width(slider_w)
            changed, self._mask_opacity = imgui.slider_float(
                "##mask-opacity", self._mask_opacity, 0.05, 1.0, "%.2f"
            )
            if changed and self._show_masks:
                self._refresh_masks()
            help_mark("every footprint's mask at this opacity")
            changed, show = imgui.checkbox("sel masks", self._show_selected_masks)
            if changed:
                self._show_selected_masks = show
                self._refresh_masks()
            g.cell(0)
            imgui.set_next_item_width(slider_w)
            changed, self._selected_mask_opacity = imgui.slider_float(
                "##selected-mask-opacity", self._selected_mask_opacity, 0.05, 1.0, "%.2f"
            )
            if changed and self._show_selected_masks:
                self._refresh_masks()
            help_mark("the selected and grouped masks, filled at this opacity with a white rim")
            changed, show = imgui.checkbox("contours", self._show_contours)
            if changed:
                self._set_contours(show)
            g.cell(0)
            imgui.set_next_item_width(slider_w)
            changed, self._contour_opacity = imgui.slider_float(
                "##contour-opacity", self._contour_opacity, 0.05, 1.0, "%.2f"
            )
            if changed and self._show_contours:
                self._image_selector.options_alpha = self._contour_opacity
            help_mark("every other footprint's contour at this opacity")
            changed, show = imgui.checkbox("sel contours", self._show_selected_contours)
            if changed:
                self._show_selected_contours = show
                self._image_selector.alpha = self._selected_contour_opacity if show else 0.0
            g.cell(0)
            imgui.set_next_item_width(slider_w)
            changed, self._selected_contour_opacity = imgui.slider_float(
                "##selected-contour-opacity", self._selected_contour_opacity, 0.05, 1.0, "%.2f"
            )
            if changed and self._show_selected_contours:
                self._image_selector.alpha = self._selected_contour_opacity
            help_mark("the selected and grouped contours, in their mask's color at this opacity")
        if self._order is not None:
            g.row("color by")
            names = ["signal id", *[n for n in self._order.columns if n != "del"]]
            imgui.set_next_item_width(g.w)
            changed, index = imgui.combo("##color_by", names.index(self._color_by) if self._color_by in names else 0, names)
            if changed:
                self._color_by = names[index]
                self._footprints.recolor(None if index == 0 else self._order.columns[self._color_by])
                self._refresh_masks()
            help_mark("color the masks and the table's ids by a column's rank instead of by signal id")
        if self._traces.panels:
            g.row("traces")
            buttons = [
                (
                    fa.ICON_FA_ARROWS_LEFT_RIGHT_TO_LINE,
                    self._traces.follow,
                    self._toggle_trace_follow,
                    "Center: keep the current frame in the middle of the traces as the movie plays or the slider "
                    "moves, the zoom kept; near either end of the recording the view stops at that end (t)",
                )
            ]
            if self._pmd_array is not None:
                buttons += [
                    (
                        fa.ICON_FA_EYE_DROPPER,
                        self._pixel_traces,
                        lambda: self._set_pixel_traces(not self._pixel_traces),
                        "Quick pixel trace: click an empty pixel to add the compressed movie's 5x5 average there to "
                        "the plot, grouped with whatever is shown, and to the top of the signals table, marked. "
                        "Demix and export ignore it; delete drops it (p)",
                    ),
                    (
                        fa.ICON_FA_CHART_LINE,
                        self._show_traces,
                        self._toggle_show_traces,
                        "Show selected traces: plot whatever is selected, a signal's four averages, or one line per "
                        "group member - a grouped signal's demixed trace, a pixel average's or drawn roi's "
                        "compressed average. Off, selecting only highlights, however big the selection",
                    ),
                ]
            if isinstance(self.demixing_results, masknmf.DemixingResults) and self.demixing_results.temporal_demixed_raw is not None:
                buttons.append(
                    (
                        fa.ICON_FA_WAVE_SQUARE,
                        self._show_raw_trace,
                        self._toggle_raw_trace,
                        "Raw trace: with one signal selected, also plot its signal line from the traces re-estimated "
                        "on the raw movie, with no compression or denoising, under the signal line",
                    )
                )
            # icon toggle buttons, lit while on
            for i, (icon, on, action, tip) in enumerate(buttons):
                if i:
                    imgui.same_line(0, g.gap)
                with button_colors(THEME.accent, THEME.accent, (0.05, 0.05, 0.05), on=on):
                    if imgui.button(f"{icon}##traces_{i}", imgui.ImVec2(em(3.2), 0)):
                        action()
                tooltip(tip)

        section("SELECTION")
        selecting = self._armed or self._region is not None
        settled = self._region is not None and self._region._move_info.mode is None
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
        # one centered row of equally spaced buttons, capped so a wide panel does not bloat them
        gap, avail = em(0.6), imgui.get_content_region_avail().x
        w = min((avail - 6 * gap) / 7, em(3.2))
        size = imgui.ImVec2(w, imgui.get_frame_height() * 1.2)
        imgui.dummy(imgui.ImVec2(0, em(0.4)))
        imgui.set_cursor_pos_x(imgui.get_cursor_pos_x() + (avail - 7 * w - 6 * gap) / 2)
        with button_colors(THEME.accent, THEME.accent, (0.05, 0.05, 0.05), on=self._follow):
            if imgui.button(f"{fa.ICON_FA_LOCATION_CROSSHAIRS}##center", size):
                self._toggle_follow()
        tooltip("Center: every panel on the selected signal, following it as the selection moves (f)")
        imgui.same_line(0, gap)
        imgui.begin_disabled(self._order is None)
        with button_colors(THEME.emphasis, THEME.emphasis_hover, (0.05, 0.05, 0.05), on=selecting):
            if imgui.button(f"{fa.ICON_FA_DRAW_POLYGON}##draw", size):
                self._start_region()
        imgui.end_disabled()
        tooltip(
            "Draw is on: click again or esc to drop the region, the selection stays (a)"
            if selecting
            else "Draw: a polygon on any panel selects every signal in view whose center is on the switch's side of "
            "it, live as it is drawn and dragged; Add ROI keeps it, a click that picks by hand drops it (a)"
        )
        imgui.same_line(0, gap)
        imgui.begin_disabled(True)
        imgui.button(f"{fa.ICON_FA_SIGNATURE}##freehand", size)
        imgui.end_disabled()
        tooltip("Freehand selection: the region drawn in one stroke, not yet implemented")
        imgui.same_line(0, gap)
        imgui.begin_disabled(not settled)
        if imgui.button(f"{fa.ICON_FA_PLUS}##add_roi", size):
            self._commit_region()
        imgui.end_disabled()
        tooltip(
            "Add ROI: keep the drawn region as a roi; its average joins the plot and Demix seeds the nmf pass with it (r)"
            + ("" if settled else "; draw one first")
        )
        imgui.same_line(0, gap)
        imgui.begin_disabled(nothing or self._worker is not None)
        with button_colors(THEME.danger, THEME.danger_hover, on=not unmark):
            if imgui.button(f"{fa.ICON_FA_TRASH}##delete", size):
                self._delete_selected()
        imgui.end_disabled()
        key = DEMIXING["delete"].label
        tooltip(
            f"Unmark: the {len(signals)} selected signal(s) stay in the next demix ({key})"
            if unmark
            else f"Mark for deletion: the {len(signals)} selected signal(s) are removed by the next demix and kept "
            f"until then; a selected drawn roi or pixel average is dropped right away ({key})"
        )
        imgui.same_line(0, gap)
        imgui.begin_disabled(True)
        imgui.button(f"{fa.ICON_FA_OBJECT_GROUP}##merge", size)
        imgui.end_disabled()
        tooltip("Merge: the grouped signals into one, not yet implemented")
        imgui.same_line(0, gap)
        imgui.begin_disabled(not self._rois)
        if imgui.button(f"{fa.ICON_FA_FILE_EXPORT}##export", size):
            self._export_prompt.start()
        imgui.end_disabled()
        tooltip(f"Export: the {len(self._rois)} drawn roi(s) to a .npz, a window with a typed path, browse for the native dialog")
        # the side switch under the row, centered: grey but flippable until the region drives the selection
        imgui.dummy(imgui.ImVec2(0, em(0.4)))
        label_w = imgui.calc_text_size("poly").x + em(0.6)
        row_w = label_w + _side_switch_width()
        imgui.set_cursor_pos_x(imgui.get_cursor_pos_x() + max((imgui.get_content_region_avail().x - row_w) / 2, 0))
        _, self._region_outside = _side_switch(
            "poly",
            self._region_outside,
            selecting,
            "The region takes the signals inside or outside it"
            + ("" if selecting else "; lit while it drives the selection"),
            label_w,
        )

        w = imgui.get_frame_height() * 1.6
        section("DEMIX")
        g.row("run")
        imgui.begin_disabled(
            (not self._rois and not self._marked)
            or self._ac_array is None
            or not self._is_masknmf_result
            or self._results_path is None
            or self._worker is not None
        )
        if imgui.button(f"{fa.ICON_FA_PLAY}##demix", imgui.ImVec2(w, 0)):
            self.demix()
        imgui.end_disabled()
        tooltip(
            "Demix: add the drawn rois, remove the marked signals, write a new curated results file "
            "beside the original (kept)"
            if self._results_path is not None
            else "Demix needs the results opened with results_path"
        )
        imgui.same_line(0, g.gap)
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
        if self._armed:
            imgui.text_disabled("click on any panel to start the region (esc or the button cancels)")
        elif self._region is not None and self._region._move_info.mode == "create":
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
    def demixing_results(self) -> masknmf.DemixingResults | masknmf.CompressionArray | masknmf.BaseRegistrationArray:
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
        self._hidden_stats -= set(new.names)
        columns = {"area": self._order.columns["area"], "peak": self._order.columns["peak"]}
        columns.update(zip(self._cell_stats.names, self._cell_stats.values.T))
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

    def compute_lag1_acf(self, batch_size: int = 200):
        """
        Add the lag-1 autocorrelation images to Static images: of the movie the compression saw (the registered
        movie, else the raw one), of the compressed movie and of their residual, the last two normalized like the
        first (:func:`masknmf.pmd_autocovariance_diagnostics`). One pass over the movie, ``batch_size`` frames at a time.
        """
        if self._pmd_array is None:
            raise ValueError("lag-1 acf images need a compression")
        movie = self._raw if self._registered is None else self._registered
        if movie is None:
            raise ValueError("lag-1 acf images need the movie the compression saw: raw=, and registered= when the results hold a registration")
        display("computing the lag-1 acf images")
        # in raw units with the pixelwise trend, as the movie is: the residual is then mean 0, which the diagnostic assumes
        compressed = masknmf.CompressionArray.from_flyweight(
            self._pmd_array.shape, self._pmd_array.flyweight, rescale=True, include_trend=True
        )
        raw, compressed, residual = pmd_autocovariance_diagnostics(movie, compressed, batch_size=batch_size, device=self.device)
        name = "raw" if self._registered is None else "registered"
        self._lag1 = {f"{name} lag-1 acf": raw, "compressed lag-1 acf": compressed, "residual lag-1 acf": residual}
        self._stills = self._static_images()
        if self._summary.is_open:
            self._summary.set_images(self._stills)

    @property
    def reference_index(self) -> fpl.ReferenceIndices:
        return self._reference_index

    def show(self):
        return self.fov_widget.show()

    def close(self):
        self._summary.cleanup()
        self._ndw_fov.close()


def _side_switch_width() -> float:
    """What :func:`_side_switch` takes past its label."""
    return (
        imgui.calc_text_size("inside").x
        + imgui.calc_text_size("outside").x
        + imgui.get_frame_height() * imgui_toggle.ToggleConfig().width_ratio
        + 2 * imgui.get_style().item_inner_spacing.x
    )


def _side_switch(key: str, outside: bool, live: bool, tip: str, label_w: float = 0.0) -> tuple[bool, bool]:
    """An inside / outside toggle, after ``key`` when ``label_w``: accent with the side in use lit while ``live``, grey otherwise."""
    inner = imgui.get_style().item_inner_spacing.x
    dim, lit = imgui.get_style().color_(imgui.Col_.text_disabled), imgui.get_style().color_(imgui.Col_.text)
    frame = THEME.accent if live else (0.28, 0.28, 0.31)
    hover = (0.55, 0.78, 1.0) if live else (0.38, 0.38, 0.42)
    imgui.align_text_to_frame_padding()
    if label_w:
        x = imgui.get_cursor_pos_x()
        imgui.text_colored(lit if live else dim, key)
        imgui.same_line(x + label_w)
    imgui.text_colored(lit if live and not outside else dim, "inside")
    imgui.same_line(0, inner)
    for color, value in (
        (imgui.Col_.frame_bg, frame),
        (imgui.Col_.button, frame),
        (imgui.Col_.frame_bg_hovered, hover),
        (imgui.Col_.button_hovered, hover),
        (imgui.Col_.text, lit if live else dim),
    ):
        imgui.push_style_color(color, to_vec4(value))
    changed, outside = imgui_toggle.toggle(f"##side_{key}", outside, imgui_toggle.ToggleFlags_.animated)
    imgui.pop_style_color(5)
    tooltip(tip)
    imgui.same_line(0, inner)
    imgui.text_colored(lit if live and outside else dim, "outside")
    return changed, outside


def extract_per_trace_roi_averages(signals_array: masknmf.SignalsArray, rowslice: slice, colslice: slice):
    """
    Split the region's demixed signal into its sources: each signal's trace weighted by its footprint's
    average over the region.

    Args:
        signals_array (masknmf.SignalsArray): The signal array that contains the factorized signals
        rowslice (slice): rows of the region
        colslice (slice): columns of the region

    Returns:
        (traces, signals): (signals, frames) weighted traces and the ids they belong to, or (None, None) when
        no footprint touches the region
    """
    device = signals_array.device
    num_frames, height, width = signals_array.shape
    a = signals_array.spatial_demixed.coalesce()  # Shape (num_pixels, num_signals)
    c = signals_array.temporal_demixed  # Shape (num_frames, num_signals)

    pixel_space = (
        torch.arange(height * width, device=device).reshape(height, width).long()
    )
    good_row_values = pixel_space[rowslice, colslice].flatten()
    num_pixels = good_row_values.shape[0]

    row, col = a.indices()
    values = a.values()

    valid_indices = torch.isin(row, good_row_values)
    if torch.count_nonzero(valid_indices) == 0:
        return None, None
    else:
        valid_columns = col[valid_indices]
        unique_signals = torch.unique(valid_columns)

        a_subset = torch.index_select(a, 1, unique_signals).coalesce()
        filtered_rows, filtered_col = a_subset.indices()
        filtered_values = a_subset.values()

        valid_indices = torch.isin(filtered_rows, good_row_values)
        filtered_rows = filtered_rows[valid_indices]
        filtered_col = filtered_col[valid_indices]
        filtered_values = filtered_values[valid_indices]

        reduce_tensor = torch.zeros(a_subset.shape[1], device=device)
        reduce_tensor.scatter_reduce_(0, filtered_col, filtered_values, reduce="sum")
        reduce_tensor = reduce_tensor / num_pixels

        weighted_signals = (
            reduce_tensor[None, :] * c[:, unique_signals]
        )  # Shape (num_frames, neural_signals)

        return weighted_signals.T.cpu().numpy(), unique_signals.cpu().numpy()


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
