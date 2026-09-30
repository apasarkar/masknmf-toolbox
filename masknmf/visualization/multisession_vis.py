from typing import *
from functools import partial

import numpy as np
import fastplotlib as fpl
from imgui_bundle import imgui
import h5py
import torch

import masknmf.arrays
from masknmf.multisession import RoicatTrackingResults
from masknmf.visualization.imgui import (
    RoiOrder,
    component_at_pixel,
    draw_keybinds_button,
    draw_keybinds_popup,
    draw_range_filter,
    draw_roi_table,
    help_mark,
    opaque_popups,
    tooltip,
)
from masknmf.visualization.imgui.keybinds import MULTISESSION, pressed


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

        One window: each session's movie, its MIP under it, and a Tools panel on the right listing the clusters.
        Selecting a cluster, from the table or by double-clicking a footprint, highlights it in every session.

        figure_shape (rows, cols) lays out the sessions of each block, the movies above the MIPs; by default one row.

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

        coloring = np.random.uniform(low=30, high=255, size=self.cluster_ids.shape[0] * 3).reshape(
            self.cluster_ids.shape[0], 3).astype('float32')
        coloring /= np.amax(coloring, axis=1, keepdims=True)
        self._coloring = coloring.astype('float32')
        self._apply_consistent_coloring()

        self._build_members()

        self._mip_session_names = ["MIP " + elt for elt in self.session_names]
        self._ndw = fpl.NDWidget(self.reference_ranges,
                                 extents=self._extents(),
                                 names=[*self.session_names, *self.mip_session_names],
                                 controller_ids=[tuple([*self.session_names, *self.mip_session_names])],
                                 size=(1500, 900))

        self._nd_image_graphics = []
        self._nd_mip_graphics = []
        for k in range(self.num_sessions_displayed):
            timings = self.session_frame_timings[k]
            graphic = self._ndw[self.session_names[k]].add_nd_image(
                self.colorful_ac_arrays[k],
                (self.reference_range_timeaxis, "m", "n", "c"),
                ("m", "n", "c"),
                rgb_dim="c",
                compute_histogram=False,
                slider_maps=None if timings is None else {self.reference_range_timeaxis: timings},
                name=self.session_names[k],
            )
            self._nd_image_graphics.append(graphic)
            self._ndw.figure[self.session_names[k]].title = self.session_names[k]

            mip = self._ndw[self.mip_session_names[k]].add_nd_image(
                self.colorful_ac_arrays[k].compute_mip().cpu().numpy(),
                ("m", "n", "c"),
                ("m", "n", "c"),
                rgb_dim="c",
                compute_histogram=False,
                name=self.mip_session_names[k],
            )
            self._nd_mip_graphics.append(mip)
            self._ndw.figure[self.mip_session_names[k]].title = self.mip_session_names[k]

        self._image_highlight_selectors = []
        for index in range(self.num_sessions_displayed):
            selector = fpl.ImageHighlightSelector(lut="tab10",
                                                  lut_wrap="repeat",
                                                  selection_options={"pixels": self.ac_arrays[index].contours},
                                                  options_color="w",
                                                  options_alpha=0.0,
                                                  alpha=0.95)
            selector.add_graphic(self._nd_image_graphics[index].graphic)
            selector.add_graphic(self._nd_mip_graphics[index].graphic)
            selector.selection = None
            self._image_highlight_selectors.append(selector)

        for index in range(self.num_sessions_displayed):
            self._nd_image_graphics[index].graphic.add_event_handler(partial(self.neuron_selection, index), "double_click")
            self._nd_mip_graphics[index].graphic.add_event_handler(partial(self.neuron_selection, index), "double_click")

        for subplot in self._ndw.figure:
            subplot.tooltip.enabled = False
            subplot.toolbar = False

        self._order = RoiOrder(
            {"sessions": self.clustering_mat.sum(axis=1), **{
                name: np.where(self._first_member[:, j] >= 0, self._first_member[:, j], np.nan)
                for j, name in enumerate(self.session_names)
            }},
            len(self.cluster_ids),
        )
        self._order.set_range_column("sessions")
        self._order.rebuild()
        self._active = None
        self._scroll_to_current = False
        self._keybinds_open = False

        self._ndw.figure.add_imgui_window(
            self._draw_side_panel,
            location="right",
            size=round(28 * self._ndw.figure.default_imgui_font.legacy_size),
            title="Tools",
        )

    def _extents(self) -> dict:
        """The session grid of figure_shape twice over, the movies in the top half and the MIPs under them."""
        rows, cols = self.figure_shape
        extents = {}
        for k in range(self.num_sessions_displayed):
            row, col = divmod(k, cols)
            x0, x1 = col / cols, (col + 1) / cols
            extents[self.session_names[k]] = (x0, x1, row / (2 * rows), (row + 1) / (2 * rows))
            extents[self.mip_session_names[k]] = (x0, x1, 0.5 + row / (2 * rows), 0.5 + (row + 1) / (2 * rows))
        return extents

    def _build_members(self):
        """_first_member[row, j] is the first ROI of displayed cluster row in displayed session j (-1: none); _member_count counts them."""
        row_of = np.full(self.tracking_results.num_clusters, -1, dtype=np.int64)
        row_of[self.cluster_ids] = np.arange(len(self.cluster_ids))
        self._row_of = row_of
        self._first_member = np.full((len(self.cluster_ids), self.num_sessions_displayed), -1, dtype=np.int64)
        self._member_count = np.zeros((len(self.cluster_ids), self.num_sessions_displayed), dtype=np.int64)
        for j, sess_id in enumerate(self.session_ids):
            labels = self.tracking_results.labels_by_session[sess_id]
            rows = np.where(labels >= 0, row_of[np.maximum(labels, 0)], -1)
            for local in np.flatnonzero(rows >= 0)[::-1]:
                self._first_member[rows[local], j] = local
            np.add.at(self._member_count[:, j], rows[rows >= 0], 1)

    def neuron_selection(self,
                         display_sess_index: int,
                         ev):
        curr_ac = self.ac_arrays[display_sess_index]
        labels = self.tracking_results.labels_by_session[self.session_ids[display_sess_index]]
        tracked = torch.as_tensor(np.isin(labels, self.cluster_ids), dtype=torch.bool)
        neuron = component_at_pixel(curr_ac.spatial_demixed,
                                    curr_ac.centers,
                                    curr_ac.shape[1:],
                                    ev.pick_info['index'],
                                    mask=tracked)
        if neuron is None:
            return
        row = int(self._row_of[labels[neuron]])
        self._order.reveal(row)
        self._scroll_to_current = True
        self.select_cluster(row)

    def select_cluster(self, row: int | None, center: bool = True):
        """Highlight displayed cluster ``row`` (an index into cluster_ids) in every session, or clear with None."""
        self._active = row
        for j, selector in enumerate(self._image_highlight_selectors):
            local = -1 if row is None else int(self._first_member[row, j])
            selector.selection = None if local < 0 else local
        if row is not None and center:
            self._center_on(row)

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
        if pressed(MULTISESSION["escape"]):
            if self._keybinds_open:
                self._keybinds_open = False
            else:
                self.select_cluster(None)
        if pressed(MULTISESSION["keybinds"]):
            self._keybinds_open = not self._keybinds_open

    def _draw_side_panel(self):
        """Docked at "right" (the NDWidget owns "bottom"): the keybinds button, then the clusters and sessions tabs."""
        opaque_popups()
        self._handle_keys()
        right = imgui.get_cursor_pos_x() + imgui.get_content_region_avail().x
        imgui.align_text_to_frame_padding()
        imgui.text_disabled(f"{len(self.cluster_ids)} clusters, {self.num_sessions_displayed} sessions")
        self._keybinds_open = draw_keybinds_button(self._keybinds_open, right=right)
        self._keybinds_open = draw_keybinds_popup(MULTISESSION, self._keybinds_open)
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
            columns = ("cluster", "sessions", *self.session_names)
            self._scroll_to_current = draw_roi_table(
                self._order,
                columns,
                {name: partial(self._format_cell, name) for name in columns[1:]},
                self._scroll_to_current,
                table_id="clusters",
                cursor=self._active is not None,
                on_select=self._table_select,
                row_color=lambda row: self.coloring[row],
                row_label=lambda row: f"{self.cluster_ids[row]}",
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
            labels = self.tracking_results.labels_by_session[sess_id]
            imgui.table_next_row()
            imgui.table_next_column()
            imgui.text(self.session_names[j])
            tooltip(str(self.tracking_results.session_files[sess_id]))
            imgui.table_next_column()
            imgui.text(f"{self.ac_arrays[j].shape[0]}")
            imgui.table_next_column()
            imgui.text(f"{len(labels)}")
            imgui.table_next_column()
            imgui.text(f"{int(np.isin(labels, self.cluster_ids).sum())}")
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
        self._coloring = new_coloring

    @property
    def ac_arrays(self) -> list[masknmf.SignalsArray]:
        return self._ac_arrays

    @property
    def colorful_ac_arrays(self) -> list[masknmf.ColorfulSignalsArray]:
        return self._colorful_ac_arrays

    def _apply_consistent_coloring(self):
        cluster_id_to_index = np.zeros((self.tracking_results.num_clusters,)).astype('int')
        cluster_id_to_index[self.cluster_ids] = np.arange(len(self.cluster_ids)).astype('int')
        for index, sess_id in enumerate(self.session_ids):
            curr_ac_array = self.ac_arrays[index]
            curr_colorful_ac_array = self.colorful_ac_arrays[index]
            curr_labels = self.tracking_results.labels_by_session[sess_id]
            mask = np.isin(curr_labels, self.cluster_ids)
            curr_ac_array.mask = torch.from_numpy(mask)
            curr_colorful_ac_array.mask = torch.from_numpy(mask)

            curr_coloring = np.zeros((int(curr_ac_array.spatial_demixed.shape[1]), 3)).astype('float32')
            cluster_indices = cluster_id_to_index[curr_labels[mask]]
            curr_coloring[mask, :] = self.coloring[cluster_indices, :]
            curr_colorful_ac_array.colors = torch.from_numpy(curr_coloring).float()

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
    def mip_session_names(self) -> list[str]:
        return self._mip_session_names

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

    def show(self):
        return self._ndw.show()

    def close(self):
        self._ndw.close()
