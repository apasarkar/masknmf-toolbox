from typing import *
from collections import OrderedDict
from functools import partial

import cv2
import numpy as np
import fastplotlib as fpl
from fastplotlib.widgets.nd_widget._async import run_sync
from cmap import Colormap
from imgui_bundle import imgui, icons_fontawesome_6 as fa
import h5py
import torch

import masknmf.arrays
from masknmf.arrays import SwitchableArray
from masknmf.multisession import RoicatTrackingResults
from masknmf.visualization.imgui import (
    PANELS_LABEL,
    PANELS_TIP,
    THEME,
    GROUP_COLORS,
    RoiOrder,
    SourceRightClickMenu,
    TracePlot,
    button_colors,
    CLICK_SLOP,
    component_at_pixel,
    draw_keybinds_button,
    draw_keybinds_popup,
    draw_panels_popup,
    draw_roi_table,
    em,
    grid,
    help_mark,
    opaque_popups,
    section,
    tooltip,
)
from masknmf.visualization.imgui.keybinds import MULTISESSION, pressed
from masknmf.visualization.imgui.options import draw_options_menu, draw_options_popup
from masknmf.visualization.imgui.panels import KEYBINDS_LABEL, hint_button_width
from masknmf.visualization.summary_widget import SummaryImageViewer

_UNCLUSTERED_COLOR = (0.45, 0.45, 0.45)
_SCALAR_CMAP = "viridis"
# ROICaT's quality_metrics key per table column, one value per cluster label
_QUALITY = {"similarity": "cluster_intra_means", "silhouette": "cluster_silhouette"}
_CAPTIONS = ("contours", "sel contours", "color by", "unclustered", "traces")
_DEFAULT_PANELS = 3
# panel name -> DemixingResults attribute, and the hdf5 dataset that has to hold something for it to be offered
_MOVIES = OrderedDict(
    [
        ("compressed+denoised", ("compression_array", None)),
        ("all signals", ("signals_array", None)),
        ("background", ("fluctuating_background_array", "factorized_background_term1")),
        ("residual", ("residual_array", None)),
        ("multiunit", ("multiunit_background_array", "multiunit_basis_term1")),
    ]
)
# static image name -> DemixingResults hdf5 dataset, warped into the aligned space
_STILLS = OrderedDict(
    [
        ("mean image", "mean_image"),
        ("noise variance image", "noise_variance_image"),
        ("residual correlation image", "global_residual_correlation_image"),
    ]
)


def _unit(image: np.ndarray) -> np.ndarray:
    """``image`` scaled to 0-1 between its 0.5 and 99.5 percentiles, NaNs as 0."""
    image = np.nan_to_num(np.asarray(image, dtype=np.float32))
    lo, hi = np.percentile(image, (0.5, 99.5))
    return np.clip((image - lo) / (hi - lo), 0, 1) if hi > lo else np.zeros_like(image)


class _Session:
    """
    One session's results file: its DemixingResults, loaded the first time one of its movies is shown, and ROICaT's
    remap from its own pixels into the tracking's aligned space.
    """

    def __init__(self, path: str, device: str, remap: Optional[np.ndarray]):
        self.path = path
        self.device = device
        self._results = None
        self._maps = None
        if remap is not None:
            # NaN marks a pixel the registration has no source for; -1 sends it to the zero border
            remap = np.nan_to_num(np.asarray(remap, dtype=np.float32), nan=-1.0)
            self._maps = (np.ascontiguousarray(remap[..., 0]), np.ascontiguousarray(remap[..., 1]))

    @property
    def results(self) -> "masknmf.DemixingResults":
        if self._results is None:
            self._results = masknmf.DemixingResults.from_hdf5(self.path, device=self.device)
        return self._results

    def warp(self, image: np.ndarray) -> np.ndarray:
        image = np.asarray(image, dtype=np.float32)
        if self._maps is None:
            return image
        return cv2.remap(image, *self._maps, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)


class _WarpedMovie:
    """(frames, height, width, 3) gray rgb of one of a session's movies, warped into the aligned space frame by frame."""

    def __init__(self, session: _Session, attribute: str, shape: tuple):
        self._session = session
        self._attribute = attribute
        self.shape = (*shape[:3], 3)
        self.ndim = 4
        self.dtype = np.dtype(np.float32)

    def __getitem__(self, item):
        key = item if isinstance(item, tuple) else (item,)
        frames = getattr(self._session.results, self._attribute)[key[0]]
        frames = np.asarray(frames.cpu().numpy() if isinstance(frames, torch.Tensor) else frames, dtype=np.float32)
        single = frames.ndim == 2
        warped = np.stack([self._session.warp(frame) for frame in (frames[None] if single else frames)])
        out = np.repeat(warped[..., None], 3, axis=-1)
        if single:
            return out[0][key[1:]]
        return out[(slice(None), *key[1:])]


class MultiSessionDemixingVis:

    def __init__(self,
                 tracking_results: RoicatTrackingResults,
                 session_ids: np.ndarray | list | None = None,
                 clusters: np.ndarray | Callable | None = None,
                 session_names: list[str] | None = None,
                 reference_ranges: dict | None = None,
                 reference_range_timeaxis: str | None = None,
                 session_frame_timings: list[np.ndarray] | None = None,
                 figure_shape: tuple[int, int] | None = None,
                 device='cuda'):

        """
        Visualization class to view tracking results for
        1. You have run a tracking algorithm and have a tracking_results object. This gives you a clusters x sessions matrix, C.
            C[i, j] gives the local index of the neuron in session "j" that belongs to cluster "i" (or -1 if no neuron exists).
        2. (Optional) You have specified the clusters (i.e. rows of the clustering matrix) you care about.
        3. (Optional) You have a specific subset of tracked sessions you care about

        One window: a grid of panels, figure_shape (rows, cols; default one row of up to three), each showing one
        session, picked in the Sessions tab or paged through with [ and ] so any number of sessions fits; the
        selected cluster's traces above them, those of the sessions on screen (or every session) in one panel, each
        in its own color and shown or hidden in the Sessions tab, and a Tools panel on the right listing the clusters.
        Each panel shows any movie its session's results hold (the Panels button, or right-click a panel); the stills
        (MIPs, FOVs, mean images, and each FOV over panel 1's, panel 2's for panel 1's session) are in the Static
        images window, the panels' sessions side by side under one selector that picks the still, the row refitted
        whole to the window whenever it is resized. Every movie and still is warped into the tracking's aligned
        space, so they line up with each other and with the contours. A session's full results load the first time
        one of its movies other than the tracked signals is shown.
        Selecting a cluster, from the table or by double-clicking a footprint, highlights it in every session.

        reference_ranges and synchronization:
        In this viewer, time reference space specifies units of time relative to the start of each session. So t = 1 refers to 1 unit after the start of a session.
        With this convention, all temporal data from all sessions can be synchronized along a single time axis

        If a reference range is provided, the user needs to specify reference_range_timeaxis as the key in the reference range dictionary that corresponds to the time axis.
        """

        self._device = device
        self._tracking_results = tracking_results

        if session_ids is None:
            session_ids = np.arange(self.tracking_results.num_sessions).astype('int')
        self._validate_session_ids(session_ids)
        self._session_ids = np.array(session_ids)

        if session_names is None:
            session_names = [f'Session_{i}' for i in self.session_ids]
        if len(session_names) != self.num_sessions_displayed:
            raise ValueError(
                f"You provided {len(session_names)} session names there are {self.num_sessions_displayed} sessions being visualized")
        self._session_names = list(session_names)

        if figure_shape is None:
            self._figure_shape = (1, min(self.num_sessions_displayed, _DEFAULT_PANELS))
        else:
            self._figure_shape = tuple(figure_shape)
        # a fixed grid of panels, each showing one session: panel k starts on session k
        num_panels = min(self._figure_shape[0] * self._figure_shape[1], self.num_sessions_displayed)
        self._panel_names = [f"panel {k + 1}" for k in range(num_panels)]
        self._panel_session = list(range(num_panels))

        if isinstance(clusters, Callable):
            self._cluster_ids = self.tracking_results.select(clusters)
        elif clusters is not None:
            self._cluster_ids = np.asarray(clusters).astype('int')
        else:
            self._cluster_ids = np.arange(self.tracking_results.num_clusters).astype('int')
        self._cluster_ids = np.unique(self._cluster_ids)

        self._clustering_mat = self.tracking_results.presence[np.ix_(self.cluster_ids, self.session_ids)].astype(
            np.float32)

        remaps = self._remaps()
        self._sessions = []
        self._movie_names = []
        self._stills_raw = []
        self._ac_arrays = []
        self._colorful_ac_arrays = []
        for index, sess_id in enumerate(self.session_ids):
            fpath = self.tracking_results.session_files[sess_id]
            with h5py.File(fpath, "r") as f:
                group = f["DemixingResults"]
                c = torch.from_numpy(group["temporal_demixed"][:])
                curr_shape = tuple([int(i) for i in group["shape"][:]])
                self._movie_names.append([
                    name for name, (_, needs) in _MOVIES.items()
                    if needs is None or (needs in group and np.any(group[needs][:]))
                ])
                self._stills_raw.append({name: group[key][:] for name, key in _STILLS.items() if key in group})
            self._sessions.append(_Session(fpath, device, None if remaps is None else remaps[sess_id]))

            curr_a = masknmf.demixing.demixing_utils.scipy_sparse_to_torch(
                self.tracking_results.aligned_rois[sess_id]).coalesce()

            self._ac_arrays.append(masknmf.SignalsArray.from_tensors(curr_shape[1:],
                                                                     curr_a.to(self.device),
                                                                     c.to(self.device)))
            self._colorful_ac_arrays.append(masknmf.ColorfulSignalsArray.from_tensors(curr_shape[1:],
                                                                                      curr_a.to(self.device),
                                                                                      c.to(self.device)))
        if remaps is None:
            # without the remap a session's own movies would not line up with the aligned footprints
            self._movie_names = [[] for _ in self.session_ids]

        if session_frame_timings is not None:
            if len(session_frame_timings) != self.num_sessions_displayed:
                raise ValueError(
                    f"Provide exactly one frame timing array for each of the {self.num_sessions_displayed} session(s) being visualized. You provided {len(session_frame_timings)}.")
            for index, elt in enumerate(session_frame_timings):
                if elt.shape[0] != self.ac_arrays[index].shape[0]:
                    raise ValueError(
                        f"session_frame_timings for {self.session_ids[index]} has shape {elt.shape[0]}, but the video for that session has {self.ac_arrays[index].shape[0]} frames.")
        if reference_ranges is not None:
            if reference_range_timeaxis is None:
                raise ValueError(
                    "If you provide your own reference_ranges, you need to specify which key in reference range represents the time axis in ``reference_range_timeaxis``")
            if reference_range_timeaxis not in reference_ranges:
                raise ValueError(
                    f"reference_range_timeaxis key {reference_range_timeaxis} must be a key in reference_ranges")
            if session_frame_timings is None:
                raise ValueError(
                    "If you provide your own reference_ranges, you need to provide frame timings for each session via the session_frame_timings parameter")
        else:
            reference_range_timeaxis = "time" if reference_range_timeaxis is None else reference_range_timeaxis
            # every session's frames, not just those of the panels shown first: a panel can switch to any session
            reference_ranges = {reference_range_timeaxis: (0, max(array.shape[0] for array in self.ac_arrays), 1)}
        if session_frame_timings is None:
            session_frame_timings = [None for _ in range(self.num_sessions_displayed)]

        self._reference_ranges = reference_ranges
        self._session_frame_timings = session_frame_timings
        self._reference_range_timeaxis = reference_range_timeaxis

        self._build_members()
        self._quality = self._read_quality()

        coloring = np.random.uniform(low=30, high=255, size=self.cluster_ids.shape[0] * 3).reshape(
            self.cluster_ids.shape[0], 3).astype('float32')
        coloring /= np.amax(coloring, axis=1, keepdims=True)
        self._coloring = coloring.astype('float32')
        self._color_by = "cluster"
        self._show_unclustered = False
        self._panels = None
        self._fovs = self._fov_images()
        self._apply_consistent_coloring()

        self._panels = OrderedDict()
        for k, name in enumerate(self._panel_names):
            self._panels[name] = self._session_array(self._panel_session[k])

        self._ndw = fpl.NDWidget(self.reference_ranges,
                                 extents=self._extents(),
                                 names=self._panel_names,
                                 controller_ids=[tuple(self._panel_names)],
                                 size=(1500, 1000))

        # one selector per session, holding the graphics of whichever panels show that session
        self._show_contours = False
        self._press = None  # screen position of the last pointer press on a panel
        self._same_spot = False  # that press landed where the one before it did
        self._contour_opacity = 0.9
        self._show_selected_contours = True
        self._selected_contour_opacity = 0.7
        self._masks_shown = np.ones(self.num_sessions_displayed, dtype=bool)
        self._image_highlight_selectors = []
        for j in range(self.num_sessions_displayed):
            selector = fpl.ImageHighlightSelector(lut="tab10",
                                                  lut_wrap="repeat",
                                                  selection_options={"pixels": self.ac_arrays[j].contours},
                                                  options_color="w",
                                                  options_alpha=0.0,
                                                  alpha=self._selected_contour_opacity)
            selector.selection = None
            self._image_highlight_selectors.append(selector)

        self._nd_image_graphics = []
        self._signal_clims = {}
        self._movie_clims = {}
        for k, name in enumerate(self._panel_names):
            timings = self.session_frame_timings[self._panel_session[k]]
            graphic = self._ndw[name].add_nd_image(
                self._panels[name],
                (self.reference_range_timeaxis, "m", "n", "c"),
                ("m", "n", "c"),
                rgb_dim="c",
                compute_histogram=False,
                slider_maps=None if timings is None else {self.reference_range_timeaxis: timings},
                name=name,
            )
            self._nd_image_graphics.append(graphic)
            self._bind_panel(k)
        self._ndw.figure.set_imgui_right_click(SourceRightClickMenu(self._panel_choices, self._set_source))

        for subplot in self._ndw.figure:
            subplot.tooltip.enabled = False
            subplot.toolbar = False

        self._trace_x, self._session_x = self._trace_axes()
        # every session's traces share one panel, each session in its own color, shown or hidden from the Sessions tab
        n = self.num_sessions_displayed
        self._trace_shown = np.ones(n, dtype=bool)
        self._traces_all_sessions = False
        self._traces = TracePlot(
            ["traces"],
            len(self._trace_x),
            None if self.session_frame_timings[0] is None else self._trace_x,
            autofit=False,
        )
        self._traces.dock(self._ndw.figure, size=260)
        if self.reference_range_timeaxis in self.reference_index.ref_ranges:
            self._traces.link(self.reference_index, dim=self.reference_range_timeaxis)

        columns = {"sessions": self.clustering_mat.sum(axis=1), **self._quality}
        columns.update({
            name: np.where(self._first_member[:, j] >= 0, self._first_member[:, j], np.nan)
            for j, name in enumerate(self.session_names)
        })
        self._order = RoiOrder(columns, len(self.cluster_ids))
        self._order.rebuild()
        self._active = None
        self._follow = False
        self._scroll_to_current = False
        self._keybinds_open = False
        self._options_open = False
        self._panels_open = False
        self._summary = SummaryImageViewer(self._ndw.figure, title="Static images")
        self._stills = None

        # 31.5 em fits the Panels, Static images and keybinds buttons on one row, like the single-session Tools
        self._ndw.figure.add_imgui_window(
            self._draw_side_panel,
            location="right",
            size=round(31.5 * self._ndw.figure.default_imgui_font.legacy_size),
            title="Tools",
        )

    def _remaps(self) -> Optional[list[np.ndarray]]:
        """ROICaT's per-session (height, width, 2) x/y source coordinates into the aligned space, or None without run data."""
        aligner = self.tracking_results.run_data.get("aligner", {})
        remaps = aligner.get("remappingIdx_nonrigid")
        if remaps is None:
            remaps = aligner.get("remappingIdx_geo")
        return remaps

    def _session_array(self, j: int) -> SwitchableArray:
        """What a panel on session j can show: the tracked signals, then every movie its results hold; stills go to Static images."""
        shape = self.colorful_ac_arrays[j].shape
        sources = OrderedDict(signals=self.colorful_ac_arrays[j])
        for name in self._movie_names[j]:
            sources[name] = _WarpedMovie(self._sessions[j], _MOVIES[name][0], shape)
        return SwitchableArray(sources, (*shape[:3], 3))

    def _bind_panel(self, k: int):
        """Hook panel k's graphic to its session: that session's contours and double-click selection, then its title and color limits."""
        graphic = self._nd_image_graphics[k].graphic
        self._image_highlight_selectors[self._panel_session[k]].add_graphic(graphic)
        graphic.add_event_handler(self._pointer_down, "pointer_down")
        graphic.add_event_handler(partial(self.neuron_selection, k), "double_click")
        self._set_title(k)
        self._refresh_panel(k)

    def set_session(self, k: int, j: int):
        """Show displayed session ``j`` in panel ``k``, on the movie the panel showed when session j has it; the zoom stays."""
        if self._panel_session[k] == j:
            return
        name = self._panel_names[k]
        nd_image = self._nd_image_graphics[k]
        self._image_highlight_selectors[self._panel_session[k]].remove_graphic(nd_image.graphic)
        keep = self._panels[name].current
        array = self._session_array(j)
        if keep in array.sources:
            array.current = keep
        self._panels[name] = array
        self._panel_session[k] = j
        timings = self.session_frame_timings[j]
        nd_image.slicer.slider_maps = None if timings is None else {self.reference_range_timeaxis: timings}
        # a new graphic instance: the selector and the click handler go onto it, and the panels keep their zoom
        cameras = [subplot.camera.get_state() for subplot in self._ndw.figure]
        nd_image.data = array
        self._bind_panel(k)
        for subplot, state in zip(self._ndw.figure, cameras):
            subplot.camera.set_state(state)
        # a session moved into a panel shows its contours as Display sets them, whatever its masks box said before
        self._masks_shown[j] = True
        self._set_contours(self._show_contours)
        self._set_selected_contours(self._show_selected_contours)
        # the stills, the overlay's partner and the traces follow the sessions on screen
        self._refresh_stills()
        self._update_traces(fit=False)

    def _on_screen(self) -> list[int]:
        """The sessions the panels show, each once, in panel order."""
        return list(dict.fromkeys(self._panel_session))

    def _refresh_stills(self):
        """Drop the Static images set so it is rebuilt on open; rebuild it now when the window is open."""
        self._stills = None
        if self._summary.is_open:
            self._stills = self._static_images()
            self._summary.set_images(self._stills)

    def _shift_sessions(self, delta: int):
        """Move every panel ``delta`` sessions along, wrapping: on sessions 0, 1, 2, -1 shows the last, 0 and 1."""
        for k in range(len(self._panel_names)):
            self.set_session(k, (self._panel_session[k] + delta) % self.num_sessions_displayed)

    def _static_images(self) -> dict:
        """
        The stills for the Static images window, in the aligned space: each kind is one row of (session name, image),
        a column per panel in panel order; 2-D, or rgb for the MIP and the FOV overlay. A still that some panel's
        session does not hold is left out.
        """
        sessions = self._panel_session
        names = [self.session_names[j] for j in sessions]
        stills = {"MIP": [(name, self._mips[j]) for name, j in zip(names, sessions)]}
        if self._fovs is not None:
            stills["aligned FOV"] = [(name, self._fovs[j]) for name, j in zip(names, sessions)]
            if len(sessions) > 1:
                stills["FOV overlay"] = [
                    (f"{name} (green: {self.session_names[self._overlay_partner(j)]})", self._overlay(j))
                    for name, j in zip(names, sessions)
                ]
        stills["ROI projection"] = [
            (name, self.tracking_results.roi_projection(self.session_ids[j]).astype(np.float32))
            for name, j in zip(names, sessions)
        ]
        for still in _STILLS:
            if all(self._movie_names[j] and still in self._stills_raw[j] for j in sessions):
                stills[still] = [
                    (name, self._sessions[j].warp(self._stills_raw[j][still])) for name, j in zip(names, sessions)
                ]
        return stills

    def _extents(self) -> dict:
        """The panels on the figure_shape grid."""
        rows, cols = self.figure_shape
        extents = {}
        for k, name in enumerate(self._panel_names):
            row, col = divmod(k, cols)
            extents[name] = (col / cols, (col + 1) / cols, row / rows, (row + 1) / rows)
        return extents

    def _build_members(self):
        """
        _rows_by_session[j] maps each ROI of displayed session j to its displayed cluster row (-1: none);
        _first_member[row, j] is the cluster's first ROI there (-1: none) and _member_count[row, j] counts them.
        """
        row_of = np.full(self.tracking_results.num_clusters, -1, dtype=np.int64)
        row_of[self.cluster_ids] = np.arange(len(self.cluster_ids))
        self._row_of = row_of
        self._rows_by_session = []
        self._first_member = np.full((len(self.cluster_ids), self.num_sessions_displayed), -1, dtype=np.int64)
        self._member_count = np.zeros((len(self.cluster_ids), self.num_sessions_displayed), dtype=np.int64)
        for j, sess_id in enumerate(self.session_ids):
            labels = self.tracking_results.labels_by_session[sess_id]
            rows = np.where(labels >= 0, row_of[np.maximum(labels, 0)], -1)
            self._rows_by_session.append(rows)
            for local in np.flatnonzero(rows >= 0)[::-1]:
                self._first_member[rows[local], j] = local
            np.add.at(self._member_count[:, j], rows[rows >= 0], 1)

    def _read_quality(self) -> dict:
        """ROICaT's per-cluster quality metrics, one value per displayed cluster row; empty when the results lack them."""
        metrics = self.tracking_results.results.get("clusters", {}).get("quality_metrics")
        if not metrics or "cluster_labels_unique" not in metrics:
            return {}
        labels = np.asarray(metrics["cluster_labels_unique"]).astype(np.int64)
        quality = {}
        for name, key in _QUALITY.items():
            if metrics.get(key) is None:
                continue
            values = np.full(self.tracking_results.num_clusters, np.nan)
            keep = labels >= 0
            values[labels[keep]] = np.asarray(metrics[key], dtype=np.float64)[keep]
            quality[name] = values[self.cluster_ids]
        return quality

    def _fov_images(self) -> list[np.ndarray] | None:
        """Each displayed session's aligned FOV image scaled to 0-1, or None when the run data lacks them."""
        images = self.tracking_results.aligned_fov_images
        if images is None:
            return None
        return [_unit(images[sess_id]) for sess_id in self.session_ids]

    def _overlay_partner(self, j: int) -> int:
        """The session session j's FOV overlay compares against: panel 1's, or panel 2's for panel 1's own."""
        return self._panel_session[1] if j == self._panel_session[0] else self._panel_session[0]

    def _overlay(self, j: int) -> np.ndarray:
        """Session j's FOV in magenta over its partner's in green: aligned structure reads white."""
        own, other = self._fovs[j], self._fovs[self._overlay_partner(j)]
        return np.stack([own, other, own], axis=2)

    def _trace_axes(self) -> tuple[np.ndarray, list[np.ndarray]]:
        """The trace plot's x samples, the longest session's frames or timings, and each session's own."""
        lengths = [array.shape[0] for array in self.ac_arrays]
        if self.session_frame_timings[0] is None:
            own = [np.arange(n, dtype=np.float64) for n in lengths]
        else:
            own = [np.asarray(t, dtype=np.float64) for t in self.session_frame_timings]
        return own[int(np.argmax(lengths))], own

    def _set_title(self, k: int):
        name = self._panel_names[k]
        self._ndw.figure[name].title = f"{self.session_names[self._panel_session[k]]} - {self._panels[name].current}"

    def _panel_choices(self, panel: str) -> tuple[list, str]:
        """What ``panel`` can show and what it shows, for its right-click menu."""
        return list(self._panels[panel].sources), self._panels[panel].current

    def _set_source(self, panel: str, name: str):
        """Show ``name`` in ``panel``; the zoom and the highlighted cluster stay."""
        self._panels[panel].current = name
        k = self._panel_names.index(panel)
        self._set_title(k)
        self._refresh_panel(k)

    def _refresh_panel(self, k: int):
        """
        Re-slice panel k now, with the color limits of what it shows: the tracked signals' own, set the first time
        its session's signals are shown, and for a movie its percentiles on the first frame it was shown at.
        """
        graphic = self._nd_image_graphics[k]
        run_sync(graphic._set_indices_())
        j = self._panel_session[k]
        current = self._panels[self._panel_names[k]].current
        if current == "signals":
            if j not in self._signal_clims:
                graphic.graphic.reset_vmin_vmax()
                self._signal_clims[j] = (graphic.graphic.vmin, graphic.graphic.vmax)
            vmin, vmax = self._signal_clims[j]
        else:
            if (j, current) not in self._movie_clims:
                frame = np.asarray(graphic.graphic.data.value, dtype=np.float32)[..., 0]
                lo, hi = np.percentile(frame, (0.5, 99.5))
                self._movie_clims[(j, current)] = (float(lo), float(hi if hi > lo else lo + 1))
            vmin, vmax = self._movie_clims[(j, current)]
        graphic.graphic.vmin, graphic.graphic.vmax = vmin, vmax

    def _row_colors(self) -> np.ndarray:
        """(clusters displayed, 3) rgb per cluster row under the current color-by."""
        if self._color_by == "cluster":
            return self._coloring
        values = self.clustering_mat.sum(axis=1) if self._color_by == "sessions" else self._quality[self._color_by]
        values = np.asarray(values, dtype=np.float64)
        finite = values[np.isfinite(values)]
        lo, hi = (finite.min(), finite.max()) if finite.size else (0.0, 1.0)
        scaled = np.clip((values - lo) / (hi - lo), 0, 1) if hi > lo else np.full_like(values, 0.5)
        colors = np.asarray(Colormap(_SCALAR_CMAP)(np.nan_to_num(scaled, nan=0.0)))[:, :3].astype(np.float32)
        colors[~np.isfinite(values)] = _UNCLUSTERED_COLOR
        return colors

    def _apply_consistent_coloring(self):
        """Color every session's ROIs by their cluster row, mask the rest (gray when shown), and remake the MIPs."""
        self._current_colors = self._row_colors()
        self._mips = []
        for j in range(self.num_sessions_displayed):
            rows = self._rows_by_session[j]
            tracked = rows >= 0
            colorful = self.colorful_ac_arrays[j]
            self.ac_arrays[j].mask = torch.from_numpy(tracked)
            colors = np.tile(np.asarray(_UNCLUSTERED_COLOR, np.float32), (len(rows), 1))
            colors[tracked] = self._current_colors[rows[tracked]]
            colorful.colors = torch.from_numpy(colors).float()
            colorful.mask = torch.from_numpy(tracked)
            mip = colorful.compute_mip().cpu().numpy()
            if self._show_unclustered:
                # compute_mip scales each pixel to its brightest channel, which turns gray white: the unclustered
                # get a pass of their own, dimmed, under the tracked
                colorful.mask = torch.from_numpy(~tracked)
                under = colorful.compute_mip().cpu().numpy() * _UNCLUSTERED_COLOR[0]
                empty = mip.max(axis=2) == 0
                mip[empty] = under[empty]
                colorful.mask = torch.ones(len(rows), dtype=torch.bool)
            self._mips.append(mip)
        if self._panels is not None:
            for k in range(len(self._panel_names)):
                self._refresh_panel(k)
            self._refresh_stills()

    def _pointer_down(self, ev):
        # pygfx reports any two quick presses on one panel as a double-click, however far apart
        self._same_spot = (
            self._press is not None and abs(ev.x - self._press[0]) + abs(ev.y - self._press[1]) <= CLICK_SLOP
        )
        self._press = (ev.x, ev.y)

    def neuron_selection(self,
                         panel: int,
                         ev):
        if not self._same_spot:
            return
        curr_ac = self.ac_arrays[self._panel_session[panel]]
        rows = self._rows_by_session[self._panel_session[panel]]
        neuron = component_at_pixel(curr_ac.spatial_demixed,
                                    curr_ac.centers,
                                    curr_ac.shape[1:],
                                    ev.pick_info['index'],
                                    mask=torch.from_numpy(rows >= 0))
        if neuron is None:
            return
        row = int(rows[neuron])
        self._order.reveal(row)
        self._scroll_to_current = True
        self.select_cluster(row)

    def select_cluster(self, row: int | None):
        """Highlight displayed cluster ``row`` (an index into cluster_ids) in every session, or clear with None."""
        self._active = row
        for j, selector in enumerate(self._image_highlight_selectors):
            local = -1 if row is None else int(self._first_member[row, j])
            selector.selection = None if local < 0 else local
        self._update_traces()
        if row is not None and self._follow:
            self._center_on(row)

    def _update_traces(self, fit: bool = True):
        """
        The trace panel gets the selected cluster's ROIs in each checked session, only those on screen unless "all
        sessions" is on, in the session's color, on the plot's x samples; ``fit`` refits the axes to them.
        """
        lines = []
        on_screen = set(self._panel_session)
        colors = self._session_colors()
        for j, name in enumerate(self.session_names):
            if self._active is None or not self._trace_shown[j]:
                continue
            if not self._traces_all_sessions and j not in on_screen:
                continue
            temporal = self.ac_arrays[j].temporal_demixed
            members = np.flatnonzero(self._rows_by_session[j] == self._active)
            for local in members:
                trace = temporal[:, int(local)].float().cpu().numpy()
                resampled = np.interp(self._trace_x, self._session_x[j], trace, left=np.nan, right=np.nan)
                label = name if len(members) == 1 else f"{name} roi {local}"
                lines.append((label, resampled, colors[j]))
        self._traces.set("traces", lines, fit=fit)

    def _session_colors(self) -> list[tuple]:
        """
        Each session's trace color, as the demixing viewer colors grouped traces: the sessions on screen take the
        contrasting colors first in panel order, so they always differ; the rest follow in session order.
        """
        order = self._on_screen() + [j for j in range(self.num_sessions_displayed) if j not in self._panel_session]
        colors = [None] * self.num_sessions_displayed
        for i, j in enumerate(order):
            colors[j] = GROUP_COLORS[i % len(GROUP_COLORS)]
        return colors

    def _center_on(self, row: int):
        """
        Pan every panel to the cluster's footprints. Zoomed out to the whole fov, this also zooms in on them with
        some context; once zoomed in, the zoom is the user's and only the center moves. The Static images row is
        centered by the same rule, on its own zoom.
        """
        points = [
            np.asarray(self.ac_arrays[j].contours[local])
            for j, local in enumerate(self._first_member[row])
            if local >= 0
        ]
        points = [p for p in points if p.size]
        if not points:
            return
        points = np.concatenate(points)
        (y0, x0), (y1, x1) = points.min(axis=0), points.max(axis=0)
        cy, cx = (y0 + y1) / 2, (x0 + x1) / 2
        span = max(max(y1 - y0, x1 - x0, 1.0) * 4.0, 80.0)
        camera = self._ndw.figure[self._panel_names[0]].camera
        fov_height, fov_width = self.ac_arrays[0].shape[1:]
        if camera.width >= fov_width or camera.height >= fov_height:
            width = height = span
        else:
            width, height = camera.width, camera.height
        for subplot in self._ndw.figure:
            subplot.camera.show_rect(cx - width / 2, cx + width / 2, cy - height / 2, cy + height / 2)
        # a pixel's index is its top-left corner in the stills
        self._summary.center_on(cy + 0.5, cx + 0.5, span)

    def _reset_view(self):
        for subplot in self._ndw.figure:
            subplot.auto_scale()

    def _toggle_follow(self):
        self._follow = not self._follow
        if self._follow and self._active is not None:
            self._center_on(self._active)

    def _toggle_trace_follow(self):
        self._traces.follow = not self._traces.follow

    def _set_contours(self, show: bool):
        """``show`` draws every other footprint's contour at the contour opacity; the selection's has its own pair."""
        self._show_contours = show
        for j, selector in enumerate(self._image_highlight_selectors):
            selector.options_alpha = self._contour_opacity if show and self._masks_shown[j] else 0.0

    def _set_selected_contours(self, show: bool):
        self._show_selected_contours = show
        for j, selector in enumerate(self._image_highlight_selectors):
            selector.alpha = self._selected_contour_opacity if show and self._masks_shown[j] else 0.0

    def _step(self, delta: int):
        if self._active is None:
            self._order.pos = 0 if delta > 0 else len(self._order.order) - 1
        elif not self._order.step(delta):
            return
        if self._order.current is not None:
            self._scroll_to_current = True
            self.select_cluster(self._order.current)

    def _set_color_by(self, name: str):
        self._color_by = name
        self._apply_consistent_coloring()
        self._update_traces()

    def _set_unclustered(self, show: bool):
        self._show_unclustered = show
        self._apply_consistent_coloring()

    def show_in_every_panel(self, name: str):
        """Show source ``name`` in every panel that has it."""
        for panel, array in self._panels.items():
            if name in array.sources:
                self._set_source(panel, name)

    def _handle_keys(self):
        if imgui.get_io().want_text_input:
            return
        stride = 10 if imgui.get_io().key_shift else 1
        if pressed(MULTISESSION["down"]):
            self._step(stride)
        if pressed(MULTISESSION["up"]):
            self._step(-stride)
        if pressed(MULTISESSION["right"]):
            self._shift_sessions(1)
        if pressed(MULTISESSION["left"]):
            self._shift_sessions(-1)
        if pressed(MULTISESSION["page_back"]):
            self._shift_sessions(-len(self._panel_names))
        if pressed(MULTISESSION["page_next"]):
            self._shift_sessions(len(self._panel_names))
        if pressed(MULTISESSION["contours"]):
            self._set_contours(not self._show_contours)
        if pressed(MULTISESSION["follow"]):
            self._toggle_follow()
        if pressed(MULTISESSION["trace_follow"]):
            self._toggle_trace_follow()
        if pressed(MULTISESSION["unclustered"]):
            self._set_unclustered(not self._show_unclustered)
        if pressed(MULTISESSION["reset"]):
            self._reset_view()
        if pressed(MULTISESSION["escape"]):
            if self._keybinds_open or self._options_open:
                self._keybinds_open = self._options_open = False
            else:
                self.select_cluster(None)
        if pressed(MULTISESSION["keybinds"]):
            self._keybinds_open = not self._keybinds_open

    def _draw_side_panel(self):
        """
        Docked at "right" (the NDWidget owns "bottom", the traces "top"): a File menu, the Panels and Static images
        buttons with the keybinds on their row, then the clusters, display and sessions tabs.
        """
        opaque_popups()
        self._handle_keys()
        if draw_options_menu():
            self._options_open = True
        if imgui.button(PANELS_LABEL):
            self._panels_open = True
        if imgui.is_item_hovered():
            imgui.set_tooltip(PANELS_TIP)
        imgui.same_line()
        if imgui.button(f"{fa.ICON_FA_IMAGE} Static images"):
            if self._stills is None:
                self._stills = self._static_images()
            self._summary.set_images(self._stills)
            self._summary.open()
        tooltip("the panels' sessions side by side, one still at a time, in the aligned space: zoom, colormap, contrast, pixel values")
        # the keybinds button right-aligned on the same row, or on a row of its own when it is too narrow
        keys_w = hint_button_width(KEYBINDS_LABEL, "(k)")
        imgui.same_line()
        if imgui.get_content_region_avail().x < keys_w + em(0.6):
            imgui.new_line()
        imgui.set_cursor_pos_x(imgui.get_cursor_pos_x() + imgui.get_content_region_avail().x - keys_w)
        self._keybinds_open = draw_keybinds_button(self._keybinds_open)
        # each tab's body is a child that scrolls on its own: the buttons and the tabs stay put
        if imgui.begin_tab_bar("##side"):
            if imgui.begin_tab_item("Clusters")[0]:
                imgui.begin_child("##clusters_tab")
                self._draw_clusters_tab()
                imgui.end_child()
                imgui.end_tab_item()
            if imgui.begin_tab_item("Display")[0]:
                imgui.begin_child("##display_tab")
                self._draw_display_tab()
                imgui.end_child()
                imgui.end_tab_item()
            if imgui.begin_tab_item("Sessions")[0]:
                imgui.begin_child("##sessions_tab")
                self._draw_sessions_tab()
                imgui.end_child()
                imgui.end_tab_item()
            imgui.end_tab_bar()
        self._keybinds_open = draw_keybinds_popup(MULTISESSION, self._keybinds_open)
        self._options_open = draw_options_popup(self._ndw.figure, self._options_open)
        self._panels_open = draw_panels_popup(self._panels, self._panels_open, self._set_source)
        self._summary.draw()

    def _draw_display_tab(self):
        g = grid(_CAPTIONS)
        # sliders: seven tenths of the panel, or what their row has left before the (?) mark
        right = imgui.get_cursor_pos_x() + imgui.get_content_region_avail().x
        slider_w = min(0.7 * imgui.get_window_width(), right - g.cell_x[0] - g.mark_w)

        section("OVERLAY")
        changed, show = imgui.checkbox("contours", self._show_contours)
        if changed:
            self._set_contours(show)
        g.cell(0)
        imgui.set_next_item_width(slider_w)
        changed, self._contour_opacity = imgui.slider_float("##contour-opacity", self._contour_opacity, 0.05, 1.0, "%.2f")
        if changed and self._show_contours:
            self._set_contours(True)
        help_mark("every other footprint's contour at this opacity (c)")
        changed, show = imgui.checkbox("sel contours", self._show_selected_contours)
        if changed:
            self._set_selected_contours(show)
        g.cell(0)
        imgui.set_next_item_width(slider_w)
        changed, self._selected_contour_opacity = imgui.slider_float(
            "##selected-contour-opacity", self._selected_contour_opacity, 0.05, 1.0, "%.2f"
        )
        if changed and self._show_selected_contours:
            self._set_selected_contours(True)
        help_mark("the selected cluster's contour in every session, at this opacity")

        options = ["cluster", "sessions", *self._quality]
        g.row("color by")
        imgui.set_next_item_width(g.w)
        changed, index = imgui.combo("##color_by", options.index(self._color_by), options)
        if changed and options[index] != self._color_by:
            self._set_color_by(options[index])
        help_mark(
            f"color the footprints and the table's ids: a random color per cluster, or a column mapped onto "
            f"{_SCALAR_CMAP}, low to high\n"
            "- similarity: ROICaT's mean similarity within the cluster\n"
            "- silhouette: how much closer the cluster's ROIs are to each other than to other clusters"
        )
        g.row("unclustered")
        changed, show = imgui.checkbox("##unclustered", self._show_unclustered)
        if changed:
            self._set_unclustered(show)
        help_mark("the ROIs no cluster took, in gray under the tracked ones (u)")

        g.row("traces")
        with button_colors(THEME.accent, THEME.accent, (0.05, 0.05, 0.05), on=self._traces.follow):
            if imgui.button(f"{fa.ICON_FA_ARROWS_LEFT_RIGHT_TO_LINE}##traces_follow", imgui.ImVec2(em(3.2), 0)):
                self._toggle_trace_follow()
        tooltip(
            "Center: keep the current frame in the middle of the traces as the movie plays or the slider moves, the "
            "zoom kept; near either end of the recording the view stops at that end (t)"
        )
        imgui.same_line(0, em(0.6))
        changed, self._traces_all_sessions = imgui.checkbox("all sessions", self._traces_all_sessions)
        if changed:
            self._update_traces()
        tooltip("traces from every checked session, not only those on screen (Sessions tab)")

        section("SELECTION")
        size = imgui.ImVec2(em(3.2), imgui.get_frame_height() * 1.2)
        with button_colors(THEME.accent, THEME.accent, (0.05, 0.05, 0.05), on=self._follow):
            if imgui.button(f"{fa.ICON_FA_LOCATION_CROSSHAIRS}##follow", size):
                self._toggle_follow()
        tooltip("Center: every panel and the Static images on the selected cluster, following it as the selection moves (f)")
        imgui.same_line(0, em(0.6))
        if imgui.button(f"{fa.ICON_FA_EXPAND}##reset", size):
            self._reset_view()
        tooltip("zoom every panel out to the whole fov (r)")
        imgui.same_line(0, em(0.6))
        imgui.begin_disabled(self._active is None)
        if imgui.button(f"{fa.ICON_FA_XMARK}##deselect", size):
            self.select_cluster(None)
        imgui.end_disabled()
        tooltip("deselect (esc)")

    def _draw_clusters_tab(self):
        imgui.text_disabled(f"{len(self.cluster_ids)} clusters over {self.num_sessions_displayed} sessions")
        footer = imgui.get_frame_height_with_spacing() * 2.5
        if imgui.begin_child("##cluster_table", imgui.ImVec2(0, -footer)):
            columns = ("cluster", "sessions", *self._quality, *(self.session_names[j] for j in self._on_screen()))
            self._scroll_to_current = draw_roi_table(
                self._order,
                columns,
                {name: partial(self._format_cell, name) for name in columns[1:]},
                self._scroll_to_current,
                table_id="clusters",
                cursor=self._active is not None,
                on_select=self._table_select,
                row_color=lambda row: self._current_colors[row],
                row_label=lambda row: f"{self.cluster_ids[row]}",
                fit_headers=True,
            )
        imgui.end_child()
        imgui.separator()
        imgui.push_text_wrap_pos(0)
        imgui.text_disabled(self._selection_status())
        imgui.pop_text_wrap_pos()

    def _table_select(self, row: int):
        self.select_cluster(None if row == self._active else row)

    def _format_cell(self, name: str, row: int) -> str:
        if name == "sessions":
            return f"{int(self._order.columns[name][row])}"
        if name in self._quality:
            value = self._quality[name][row]
            return "" if np.isnan(value) else f"{value:.2f}"
        j = self.session_names.index(name)
        local = self._first_member[row, j]
        if local < 0:
            return "-"
        return f"{local}+" if self._member_count[row, j] > 1 else f"{local}"

    def _selection_status(self) -> str:
        if self._active is None:
            return "select a cluster in the table, or double-click a footprint in any panel"
        found = [
            f"{name} roi {self._first_member[self._active, j]}"
            for j, name in enumerate(self.session_names)
            if self._first_member[self._active, j] >= 0
        ]
        return f"cluster {self.cluster_ids[self._active]}: " + ", ".join(found)

    def _draw_sessions_tab(self):
        num_panels = len(self._panel_names)
        if imgui.button(f"{fa.ICON_FA_ANGLE_LEFT}##page_back"):
            self._shift_sessions(-num_panels)
        tooltip(f"every panel back {num_panels} sessions ([)")
        imgui.same_line()
        if imgui.button(f"{fa.ICON_FA_ANGLE_RIGHT}##page_next"):
            self._shift_sessions(num_panels)
        tooltip(f"every panel on {num_panels} sessions (])")
        imgui.same_line(0, em(1.2))
        for label, shown in (("show all", True), ("hide all", False)):
            if imgui.button(label):
                self._trace_shown[:] = shown
                self._update_traces()
            imgui.same_line()
        imgui.new_line()
        flags = imgui.TableFlags_.row_bg | imgui.TableFlags_.borders_inner_h | imgui.TableFlags_.scroll_x
        if not imgui.begin_table("##sessions", 6 + num_panels, flags):
            return
        for k in range(num_panels):
            imgui.table_setup_column(f"{k + 1}", imgui.TableColumnFlags_.width_fixed)
        for name in ("traces", "masks", "session", "frames", "rois", "tracked"):
            imgui.table_setup_column(name, imgui.TableColumnFlags_.width_fixed)
        imgui.table_headers_row()
        colors = self._session_colors()
        for j, sess_id in enumerate(self.session_ids):
            imgui.table_next_row()
            # a panel column per panel: the radio picks the session that panel shows
            for k in range(num_panels):
                imgui.table_next_column()
                if imgui.radio_button(f"##panel{k}-{j}", self._panel_session[k] == j):
                    self.set_session(k, j)
                tooltip(f"show {self.session_names[j]} in panel {k + 1}")
            imgui.table_next_column()
            r, g, b = colors[j]
            imgui.push_style_color(imgui.Col_.check_mark, imgui.ImVec4(r, g, b, 1.0))
            changed, shown = imgui.checkbox(f"##trace{j}", bool(self._trace_shown[j]))
            imgui.pop_style_color()
            if changed:
                self._trace_shown[j] = shown
                self._update_traces()
            tooltip(
                f"{self.session_names[j]}'s traces in the trace panel, in this color, while a panel shows it "
                "(always with Display's traces \"all sessions\")"
            )
            imgui.table_next_column()
            changed, shown = imgui.checkbox(f"##masks{j}", bool(self._masks_shown[j]))
            if changed:
                self._masks_shown[j] = shown
                self._set_contours(self._show_contours)
                self._set_selected_contours(self._show_selected_contours)
            tooltip(f"{self.session_names[j]}'s contours on the panels showing it, as Display's contour checkboxes set them")
            imgui.table_next_column()
            imgui.text_colored(imgui.ImVec4(r, g, b, 1.0), self.session_names[j])
            tooltip(str(self.tracking_results.session_files[sess_id]))
            imgui.table_next_column()
            imgui.text(f"{self.ac_arrays[j].shape[0]}")
            imgui.table_next_column()
            imgui.text(f"{len(self._rows_by_session[j])}")
            imgui.table_next_column()
            imgui.text(f"{int((self._rows_by_session[j] >= 0).sum())}")
        imgui.end_table()

    def _validate_session_ids(self, session_ids: np.ndarray):
        for k in range(len(session_ids)):
            if not 0 <= session_ids[k] < self.tracking_results.num_sessions:
                raise ValueError(
                    f"Your tracking results contain {self.tracking_results.num_sessions}, all session ids must be a nonnegative integer less than this value")
        return True

    @property
    def coloring(self) -> np.ndarray:
        """
        This is the coloring scheme used to color in neural components that are matched across sessions
        The coloring is a (num_clusters, 3) np.ndarray
        """
        return self._coloring

    @coloring.setter
    def coloring(self, new_coloring: np.ndarray):
        if not self._coloring.shape == new_coloring.shape:
            raise ValueError(
                f"The new coloring must be same shape as old coloring. New coloring had shape {new_coloring.shape}, old coloring had shape {self._coloring.shape}")
        self._coloring = np.asarray(new_coloring, dtype=np.float32)
        self._apply_consistent_coloring()
        self._update_traces()

    @property
    def ac_arrays(self) -> list[masknmf.SignalsArray]:
        return self._ac_arrays

    @property
    def colorful_ac_arrays(self) -> list[masknmf.ColorfulSignalsArray]:
        return self._colorful_ac_arrays

    @property
    def reference_ranges(self):
        return self._reference_ranges

    @property
    def reference_index(self) -> fpl.ReferenceIndices:
        return self._ndw.indices

    @property
    def reference_range_timeaxis(self) -> str:
        return self._reference_range_timeaxis

    @property
    def session_frame_timings(self):
        return self._session_frame_timings

    @property
    def session_names(self) -> list[str]:
        return self._session_names

    @property
    def figure_shape(self) -> tuple[int, int]:
        return self._figure_shape

    @property
    def clustering_mat(self) -> np.ndarray:
        """
        Returns a binary membership matrix of dimensions (len(self.cluster_ids), num_sessions_displayed)
        """
        return self._clustering_mat

    @property
    def session_ids(self) -> np.ndarray:
        return self._session_ids

    @property
    def num_sessions_displayed(self) -> int:
        return len(self.session_ids)

    @property
    def tracking_results(self) -> RoicatTrackingResults:
        return self._tracking_results

    @property
    def device(self):
        return self._device

    @property
    def cluster_ids(self) -> np.ndarray:
        """
        These are the cluster ids of the tracking results that are being displayed
        """
        return self._cluster_ids

    @property
    def selected_cluster(self) -> int | None:
        """The tracking results' id of the selected cluster, or None."""
        return None if self._active is None else int(self.cluster_ids[self._active])

    @property
    def fov_widget(self) -> fpl.NDWidget:
        return self._ndw

    @property
    def traces(self) -> TracePlot:
        return self._traces

    def show(self):
        return self._ndw.show()

    def close(self):
        self._summary.cleanup()
        self._ndw.close()
