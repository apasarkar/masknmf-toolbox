from typing import *
from functools import partial

import numpy as np
import fastplotlib as fpl
from fastplotlib.widgets.nd_widget._async import run_sync
from cmap import Colormap
from imgui_bundle import imgui
import h5py
import torch

import masknmf.arrays
from masknmf.arrays import SwitchableArray
from masknmf.multisession import RoicatTrackingResults
from masknmf.visualization.imgui import (
    RoiOrder,
    SourceRightClickMenu,
    TracePlot,
    component_at_pixel,
    draw_keybinds_button,
    draw_keybinds_popup,
    draw_range_filter,
    draw_roi_table,
    grid,
    help_mark,
    opaque_popups,
    section,
    tooltip,
)
from masknmf.visualization.imgui.keybinds import MULTISESSION, pressed

_UNCLUSTERED_COLOR = (0.45, 0.45, 0.45)
_SCALAR_CMAP = "viridis"
# ROICaT's quality_metrics key per table column, one value per cluster label
_QUALITY = {"similarity": "cluster_intra_means", "silhouette": "cluster_silhouette"}
_CAPTIONS = ("panels", "color by", "unclustered")


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

        One window: a panel per session, laid out by figure_shape (rows, cols; default one row), the selected
        cluster's traces above them and a Tools panel on the right listing the clusters. Each panel shows the
        signals, their MIP, the aligned FOV image or that FOV overlaid on another session's (right-click a panel,
        or pick for every panel in Tools). Selecting a cluster, from the table or by double-clicking a footprint,
        highlights it in every session.

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
            self._figure_shape = (1, self.num_sessions_displayed)
        else:
            if (figure_shape[0] * figure_shape[1]) < self.num_sessions_displayed:
                raise ValueError(
                    f"The figure shape is {figure_shape[0]} x {figure_shape[1]} which is too small to display {self.num_sessions_displayed} sessions")
            self._figure_shape = tuple(figure_shape)

        if isinstance(clusters, Callable):
            self._cluster_ids = self.tracking_results.select(clusters)
        elif clusters is not None:
            self._cluster_ids = np.asarray(clusters).astype('int')
        else:
            self._cluster_ids = np.arange(self.tracking_results.num_clusters).astype('int')
        self._cluster_ids = np.unique(self._cluster_ids)

        self._clustering_mat = self.tracking_results.presence[np.ix_(self.cluster_ids, self.session_ids)].astype(
            np.float32)

        self._ac_arrays = []
        self._colorful_ac_arrays = []
        for index, sess_id in enumerate(self.session_ids):
            fpath = self.tracking_results.session_files[sess_id]
            with h5py.File(fpath, "r") as f:
                c = torch.from_numpy(f["DemixingResults/temporal_demixed"][:])
                curr_shape = tuple([int(i) for i in f["DemixingResults/shape"][:]])

            curr_a = masknmf.demixing.demixing_utils.scipy_sparse_to_torch(
                self.tracking_results.aligned_rois[sess_id]).coalesce()

            self._ac_arrays.append(masknmf.SignalsArray.from_tensors(curr_shape[1:],
                                                                     curr_a.to(self.device),
                                                                     c.to(self.device)))
            self._colorful_ac_arrays.append(masknmf.ColorfulSignalsArray.from_tensors(curr_shape[1:],
                                                                                      curr_a.to(self.device),
                                                                                      c.to(self.device)))

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
            reference_ranges = dict()
            reference_range_timeaxis = "time" if reference_range_timeaxis is None else reference_range_timeaxis
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

        self._panels = []
        for j in range(self.num_sessions_displayed):
            sources = {"signals": self.colorful_ac_arrays[j], "MIP": self._mips[j]}
            if self._fovs is not None:
                sources["aligned FOV"] = np.repeat(self._fovs[j][..., None], 3, axis=2)
                if self.num_sessions_displayed > 1:
                    sources["FOV overlay"] = self._overlay(j)
            self._panels.append(SwitchableArray(sources, (*self.colorful_ac_arrays[j].shape[:3], 3)))

        self._ndw = fpl.NDWidget(self.reference_ranges,
                                 extents=self._extents(),
                                 names=[*self.session_names],
                                 controller_ids=[tuple(self.session_names)],
                                 size=(1500, 1000))

        self._nd_image_graphics = []
        self._signal_clims = []
        for j in range(self.num_sessions_displayed):
            timings = self.session_frame_timings[j]
            graphic = self._ndw[self.session_names[j]].add_nd_image(
                self._panels[j],
                (self.reference_range_timeaxis, "m", "n", "c"),
                ("m", "n", "c"),
                rgb_dim="c",
                compute_histogram=False,
                slider_maps=None if timings is None else {self.reference_range_timeaxis: timings},
                name=self.session_names[j],
            )
            self._nd_image_graphics.append(graphic)
            self._signal_clims.append((graphic.graphic.vmin, graphic.graphic.vmax))
            self._set_title(j)
        self._ndw.figure.set_imgui_right_click(SourceRightClickMenu(self._panel_choices, self._set_source))

        self._image_highlight_selectors = []
        for j in range(self.num_sessions_displayed):
            selector = fpl.ImageHighlightSelector(lut="tab10",
                                                  lut_wrap="repeat",
                                                  selection_options={"pixels": self.ac_arrays[j].contours},
                                                  options_color="w",
                                                  options_alpha=0.0,
                                                  alpha=0.95)
            selector.add_graphic(self._nd_image_graphics[j].graphic)
            selector.selection = None
            self._image_highlight_selectors.append(selector)
            self._nd_image_graphics[j].graphic.add_event_handler(partial(self.neuron_selection, j), "double_click")

        for subplot in self._ndw.figure:
            subplot.tooltip.enabled = False
            subplot.toolbar = False

        self._trace_x, self._session_x = self._trace_axes()
        self._traces = TracePlot(
            self.session_names,
            len(self._trace_x),
            None if self.session_frame_timings[0] is None else self._trace_x,
            autofit=False,
        )
        self._traces.dock(self._ndw.figure, size=min(90 + 110 * self.num_sessions_displayed, 520))
        if self.reference_range_timeaxis in self.reference_index.ref_ranges:
            self._traces.link(self.reference_index, dim=self.reference_range_timeaxis)

        columns = {"sessions": self.clustering_mat.sum(axis=1), **self._quality}
        columns.update({
            name: np.where(self._first_member[:, j] >= 0, self._first_member[:, j], np.nan)
            for j, name in enumerate(self.session_names)
        })
        self._order = RoiOrder(columns, len(self.cluster_ids))
        self._order.set_range_column("sessions")
        self._order.rebuild()
        self._active = None
        self._scroll_to_current = False
        self._keybinds_open = False

        self._ndw.figure.add_imgui_window(
            self._draw_side_panel,
            location="right",
            size=round(30 * self._ndw.figure.default_imgui_font.legacy_size),
            title="Tools",
        )

    def _extents(self) -> dict:
        """One panel per session on the figure_shape grid."""
        rows, cols = self.figure_shape
        extents = {}
        for k, name in enumerate(self.session_names):
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
        fovs = []
        for sess_id in self.session_ids:
            image = np.nan_to_num(np.asarray(images[sess_id], dtype=np.float32))
            top = np.percentile(image, 99.5)
            fovs.append(np.clip(image / top if top > 0 else image, 0, 1))
        return fovs

    def _overlay_partner(self, j: int) -> int:
        """The session a panel's FOV overlay compares against: the first displayed one, or the second for the first."""
        return 1 if j == 0 else 0

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

    def _set_title(self, j: int):
        current = self._panels[j].current
        title = f"{self.session_names[j]} - {current}"
        if current == "FOV overlay":
            title += f" (green: {self.session_names[self._overlay_partner(j)]})"
        self._ndw.figure[self.session_names[j]].title = title

    def _panel_choices(self, panel: str) -> tuple[list, str]:
        """What ``panel`` can show and what it shows, for its right-click menu."""
        array = self._panels[self.session_names.index(panel)]
        return list(array.sources), array.current

    def _set_source(self, panel: str, name: str):
        """Show ``name`` in ``panel``; the zoom and the highlighted cluster stay."""
        j = self.session_names.index(panel)
        self._panels[j].current = name
        self._set_title(j)
        self._refresh_panel(j)

    def _refresh_panel(self, j: int):
        """Re-slice panel j now, with the color limits of what it shows: the signals' own, 0-1 for the stills."""
        graphic = self._nd_image_graphics[j]
        run_sync(graphic._set_indices_())
        vmin, vmax = self._signal_clims[j] if self._panels[j].current == "signals" else (0.0, 1.0)
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
            for j, array in enumerate(self._panels):
                array.sources["MIP"] = self._mips[j]
                self._refresh_panel(j)

    def neuron_selection(self,
                         display_sess_index: int,
                         ev):
        curr_ac = self.ac_arrays[display_sess_index]
        rows = self._rows_by_session[display_sess_index]
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

    def select_cluster(self, row: int | None, center: bool = True):
        """Highlight displayed cluster ``row`` (an index into cluster_ids) in every session, or clear with None."""
        self._active = row
        for j, selector in enumerate(self._image_highlight_selectors):
            local = -1 if row is None else int(self._first_member[row, j])
            selector.selection = None if local < 0 else local
        self._update_traces()
        if row is not None and center:
            self._center_on(row)

    def _update_traces(self):
        """Every panel of the trace plot gets its session's ROIs of the selected cluster, on the plot's x samples."""
        for j, name in enumerate(self.session_names):
            if self._active is None:
                self._traces.set(name, [])
                continue
            temporal = self.ac_arrays[j].temporal_demixed
            lines = []
            members = np.flatnonzero(self._rows_by_session[j] == self._active)
            for local in members:
                trace = temporal[:, int(local)].float().cpu().numpy()
                resampled = np.interp(self._trace_x, self._session_x[j], trace, left=np.nan, right=np.nan)
                # a lone line named after its panel draws no legend; the status line names its roi
                label = name if len(members) == 1 else f"roi {local}"
                lines.append((label, resampled, tuple(self._current_colors[self._active])))
            self._traces.set(name, lines, fit=True)

    def _center_on(self, row: int):
        """
        Pan every panel to the cluster's footprints. Zoomed out to the whole fov, this also zooms in on them with
        some context; once zoomed in, the zoom is the user's and only the center moves.
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
        camera = self._ndw.figure[self.session_names[0]].camera
        fov_height, fov_width = self.ac_arrays[0].shape[1:]
        if camera.width >= fov_width or camera.height >= fov_height:
            width = height = max(max(y1 - y0, x1 - x0, 1.0) * 4.0, 80.0)
        else:
            width, height = camera.width, camera.height
        for subplot in self._ndw.figure:
            subplot.camera.show_rect(cx - width / 2, cx + width / 2, cy - height / 2, cy + height / 2)

    def _reset_view(self):
        for subplot in self._ndw.figure:
            subplot.auto_scale()

    def _step(self, delta: int):
        if self._active is None:
            self._order.pos = 0 if delta > 0 else len(self._order.order) - 1
        elif not self._order.step(delta):
            return
        if self._order.current is not None:
            self._scroll_to_current = True
            self.select_cluster(self._order.current)

    def _step_frame(self, delta: int):
        """Move the time index by delta; every movie follows."""
        axis = self.reference_range_timeaxis
        index = self.reference_index
        step = index.ref_ranges[axis].step if axis in index.ref_ranges else 1
        index.set({axis: index[axis] + delta * step})

    def _set_color_by(self, name: str):
        self._color_by = name
        self._apply_consistent_coloring()
        self._update_traces()

    def _set_unclustered(self, show: bool):
        self._show_unclustered = show
        self._apply_consistent_coloring()

    def show_in_every_panel(self, name: str):
        """Show source ``name`` in every panel that has it."""
        for j, array in enumerate(self._panels):
            if name in array.sources:
                self._set_source(self.session_names[j], name)

    def _handle_keys(self):
        if imgui.get_io().want_text_input:
            return
        stride = 10 if imgui.get_io().key_shift else 1
        if pressed(MULTISESSION["down"]):
            self._step(stride)
        if pressed(MULTISESSION["up"]):
            self._step(-stride)
        if pressed(MULTISESSION["right"]):
            self._step_frame(stride)
        if pressed(MULTISESSION["left"]):
            self._step_frame(-stride)
        if pressed(MULTISESSION["center"]) and self._active is not None:
            self._center_on(self._active)
        if pressed(MULTISESSION["reset"]):
            self._reset_view()
        if pressed(MULTISESSION["unclustered"]):
            self._set_unclustered(not self._show_unclustered)
        if pressed(MULTISESSION["escape"]):
            if self._keybinds_open:
                self._keybinds_open = False
            else:
                self.select_cluster(None)
        if pressed(MULTISESSION["keybinds"]):
            self._keybinds_open = not self._keybinds_open

    def _draw_side_panel(self):
        """Docked at "right" (the NDWidget owns "bottom", the traces "top"): the view options, then the clusters and sessions tabs."""
        opaque_popups()
        self._handle_keys()
        right = imgui.get_cursor_pos_x() + imgui.get_content_region_avail().x
        imgui.align_text_to_frame_padding()
        imgui.text_disabled(f"{len(self.cluster_ids)} clusters, {self.num_sessions_displayed} sessions")
        self._keybinds_open = draw_keybinds_button(self._keybinds_open, right=right)
        self._keybinds_open = draw_keybinds_popup(MULTISESSION, self._keybinds_open)
        self._draw_view_options()
        if imgui.begin_tab_bar("##side"):
            if imgui.begin_tab_item("Clusters")[0]:
                imgui.begin_child("##clusters_tab")
                self._draw_clusters_tab()
                imgui.end_child()
                imgui.end_tab_item()
            if imgui.begin_tab_item("Sessions")[0]:
                imgui.begin_child("##sessions_tab")
                self._draw_sessions_tab()
                imgui.end_child()
                imgui.end_tab_item()
            imgui.end_tab_bar()

    def _draw_view_options(self):
        section("View")
        g = grid(_CAPTIONS)
        names = list(self._panels[0].sources)
        currents = {array.current for array in self._panels}
        g.row("panels")
        imgui.set_next_item_width(g.span)
        preview = currents.pop() if len(currents) == 1 else "mixed"
        if imgui.begin_combo("##panels", preview):
            for name in names:
                if imgui.selectable(name, name == preview)[0]:
                    self.show_in_every_panel(name)
            imgui.end_combo()
        imgui.same_line()
        help_mark(
            "what every panel shows; right-click a panel to change just that one\n"
            "- signals: the tracked ROIs' activity, in their cluster colors\n"
            "- MIP: each ROI at its peak\n"
            "- aligned FOV: the session's FOV image after ROICaT's registration\n"
            "- FOV overlay: that FOV in magenta over another session's in green; aligned structure reads white"
        )

        options = ["cluster", "sessions", *self._quality]
        g.row("color by")
        imgui.set_next_item_width(g.span)
        if imgui.begin_combo("##color_by", self._color_by):
            for name in options:
                if imgui.selectable(name, name == self._color_by)[0] and name != self._color_by:
                    self._set_color_by(name)
            imgui.end_combo()
        imgui.same_line()
        help_mark(
            f"cluster: a random color per cluster; the others map the column onto {_SCALAR_CMAP}, low to high\n"
            "- similarity: ROICaT's mean similarity within the cluster\n"
            "- silhouette: how much closer the cluster's ROIs are to each other than to other clusters"
        )

        g.row("unclustered")
        changed, show = imgui.checkbox("show in gray##unclustered", self._show_unclustered)
        if changed:
            self._set_unclustered(show)
        imgui.dummy(imgui.ImVec2(0, imgui.get_style().item_spacing.y))

    def _draw_clusters_tab(self):
        if self._order.range_span[0] < self._order.range_span[1]:
            imgui.align_text_to_frame_padding()
            imgui.text_disabled("sessions")
            imgui.same_line()
            help_mark("show the clusters found in this many of the displayed sessions")
            imgui.same_line()
            if draw_range_filter(self._order, "sessions"):
                self._order.rebuild()
        footer = imgui.get_frame_height_with_spacing() * 2.5
        if imgui.begin_child("##cluster_table", imgui.ImVec2(0, -footer)):
            columns = ("cluster", "sessions", *self._quality, *self.session_names)
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
        flags = imgui.TableFlags_.row_bg | imgui.TableFlags_.resizable | imgui.TableFlags_.borders_inner_h
        if not imgui.begin_table("##sessions", 4, flags):
            return
        for name in ("session", "frames", "rois", "tracked"):
            imgui.table_setup_column(name)
        imgui.table_headers_row()
        for j, sess_id in enumerate(self.session_ids):
            imgui.table_next_row()
            imgui.table_next_column()
            imgui.text(self.session_names[j])
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
        self._ndw.close()
