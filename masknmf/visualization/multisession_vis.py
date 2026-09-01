from typing import *
import numpy as np
import fastplotlib as fpl
from imgui_bundle import imgui
from fastplotlib import ui
import pygfx
import torch
from collections import OrderedDict
import masknmf.arrays
from masknmf.utils import display
from functools import partial
import h5py
from fastplotlib.widgets.nd_widget._index import ReferenceIndex
from masknmf.multisession import RoicatTrackingResults
from masknmf.visualization.imgui import (
    close_figure,
    component_at_pixel,
    contours_to_bbox,
    zoom_to_bbox,
    is_notebook_canvas,
)


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

        This visualizer will show the tracked sessions and the relevant clusters

        reference_ranges and synchronization:
        In this viewer, time reference space specifies units of time relative to the start of each session. So t = 1 refers to 1 unit after the start of a session.
        With this convention, all temporal data from all sessions can be synchronized along a single time axis

        If a reference range is provided, the user needs to specify reference_range_timeaxis as the key in the reference range dictionary that corresponds to the time axis.
        """

        self._device = device
        self._tracking_results = tracking_results

        ## Validate and set session ids
        if session_ids is None:
            session_ids = np.arange(self.tracking_results.num_sessions).astype('int')
        self._validate_session_ids(session_ids)
        self._session_ids = np.array(session_ids)

        ## Validate and set session names
        if session_names is None:
            session_names = [f'Session_{i}' for i in self.session_ids]
        if len(session_names) != self.num_sessions_displayed:
            raise ValueError(
                f"You provided {len(session_names)} session names there are {self.num_sessions_displayed} sessions being visualized")

        self._session_names = session_names

        ## Set the figure shape
        if figure_shape is None:
            self._figure_shape = (1, self.num_sessions_displayed)
        else:
            if (figure_shape[0] * figure_shape[1]) < self.num_sessions_displayed:
                raise ValueError(
                    f"The figure shape is {figure_shape[0]} x {figure_shape[1]} which is too small to display {self.num_sessions_displayed} sessions")
            self._figure_shape = figure_shape

        ## Determine the cluster ids to visualize
        if isinstance(clusters, Callable):
            cluster_ids = []
            for k in self.tracking_results.num_clusters:
                if clusters(k):
                    cluster_ids.append(k)
            self._cluster_ids = np.array(cluster_ids).astype('int')
        elif isinstance(clusters, np.ndarray):
            self._cluster_ids = clusters.astype('int')
        else:
            self._cluster_ids = np.arange(self.tracking_results.num_clusters).astype('int')

        ## Remove all duplicate cluster ids
        self._cluster_ids = np.unique(self._cluster_ids)

        ##Now that you know the cluster ids (rows) and session ids (columns), you can just display this subset of rows/columns of the clustering matrix
        self._clustering_mat = self.tracking_results.presence[np.ix_(self.cluster_ids, self.session_ids)].astype(
            np.float32)  ##Cast needed for visualization

        self._ac_arrays = []
        self._colorful_ac_arrays = []
        self._nd_image_graphics = []

        ## Set up the arrays that we want to visualize
        for index, sess_id in enumerate(self.session_ids):
            fpath = self.tracking_results.session_files[sess_id]
            with h5py.File(fpath, "r") as f:
                c = torch.from_numpy(f["DemixingResults/c"][:])
                curr_shape = tuple([int(i) for i in f["DemixingResults/shape"][:]])

            curr_c = c
            curr_a = masknmf.demixing.demixing_utils.scipy_sparse_to_torch(
                self.tracking_results.aligned_rois[sess_id]).coalesce()

            curr_ac_array = masknmf.ACArray.from_tensors(curr_shape[1:],
                                                         curr_a.to(self.device),
                                                         curr_c.to(self.device))

            curr_colorful_ac_array = masknmf.ColorfulACArray.from_tensors(curr_shape[1:],
                                                                          curr_a.to(self.device),
                                                                          curr_c.to(self.device))

            self._ac_arrays.append(curr_ac_array)
            self._colorful_ac_arrays.append(curr_colorful_ac_array)

        ##Make the reference ranges for the time axes
        """
        Below code standardizes all input and makes public properties for reference_ranges, reference_range_timeaxis, session_frame_timings
        """
        if reference_ranges is not None:
            if reference_range_timeaxis is not None:
                if reference_range_timeaxis not in reference_ranges:
                    raise ValueError(
                        f"reference_range_timeaxis key {reference_range_timeaxis} must be a key in reference_ranges")
            else:
                raise ValueError(
                    "If you provide your own reference_ranges, you need to specify which key in reference range represents the time axis in ``reference_range_timeaxis``")

            if session_frame_timings is not None:
                ## Check that each frame timing array shape matches the number of frames
                if len(session_frame_timings) != self.num_sessions_displayed:
                    raise ValueError(
                        f"Provide exactly one frame timing array for each of the {self.num_sessions_displayed} session(s) being visualized. You provided {len(session_frame_timings)}.")
                for index, elt in session_frame_timings:
                    if elt.shape[0] != self.demixing_results[index].shape[0]:
                        raise ValueError(
                            f"session_frame_timings for {self.session_ids[index]} has shape {elt.shape[0]}, but the video for that session has {self.demixing_results[index].shape[0]} frames.")

            else:
                raise ValueError(
                    "If you provide your own reference_ranges, you need to provide frame timings for each session via the session_frame_timings parameter")

        else:
            reference_ranges = dict()
            reference_range_timeaxis = "time" if reference_range_timeaxis is None else reference_range_timeaxis
            if session_frame_timings is not None:
                ## Check that each frame timing array shape matches the number of frames
                if len(session_frame_timings) != self.num_sessions_displayed:
                    raise ValueError(
                        f"Provide exactly one frame timing array for each of the {self.num_sessions_displayed} session(s) being visualized. You provided {len(session_frame_timings)}.")
                for index, elt in session_frame_timings:
                    if elt.shape[0] != self.demixing_results[index].shape[0]:
                        raise ValueError(
                            f"session_frame_timings for {self.session_ids[index]} has shape {elt.shape[0]}, but the video for that session has {self.demixing_results[index].shape[0]} frames.")
            else:
                session_frame_timings = [None for elt in range(self.num_sessions_displayed)]

        self._reference_ranges = reference_ranges
        self._session_frame_timings = session_frame_timings
        self._reference_range_timeaxis = reference_range_timeaxis

        trace_subplot_names = self.session_names

        coloring = np.random.uniform(low=30, high=255, size=self.cluster_ids.shape[0] * 3).reshape(
            self.cluster_ids.shape[0], 3).astype('float32')
        coloring /= np.amax(coloring, axis=1, keepdims=True)
        self._coloring = coloring.astype('float32')

        # Apply a consistent coloring to all arrays so matched neurons across sessions have same color
        self._apply_consistent_coloring()

        self._nd_image_graphics = []
        ## Now let's construct the NDGraphic just for the image
        self._ndw_videos = fpl.NDWidget(self.reference_ranges,
                                        shape=self.figure_shape,  ## SHAPE UPDATE
                                        names=[*self.session_names],
                                        controller_ids=[tuple(self.session_names)],
                                        size=(500, 300))

        self._mip_session_names = ["MIP " + elt for elt in self.session_names]

        ### UPDATE SHAPE HERE
        self._ndw_mip = fpl.NDWidget(shape=self.figure_shape,
                                     names=[*self.mip_session_names],
                                     controller_ids=[tuple(self.mip_session_names)],
                                     size=(500, 300))

        self._nd_mip_graphics = []
        for k in range(self.num_sessions_displayed):
            curr_data = self.colorful_ac_arrays[k].compute_mip().cpu().numpy()
            dims = ("m", "n", "c")
            spatial_dims = ("m", "n", "c")
            curr_graphic = self._ndw_mip[self.mip_session_names[k]].add_nd_image(curr_data,
                                                                                 dims,
                                                                                 spatial_dims,
                                                                                 rgb_dim="c",
                                                                                 name=self.mip_session_names)
            self._nd_mip_graphics.append(curr_graphic)

        for k in range(self.num_sessions_displayed):
            curr_data = self.colorful_ac_arrays[k]
            dims = (self.reference_range_timeaxis, "m", "n", "c")
            spatial_dims = ("m", "n", "c")
            curr_graphic = self._ndw_videos[self.session_names[k]].add_nd_image(curr_data,
                                                                                dims,
                                                                                spatial_dims,
                                                                                rgb_dim="c",
                                                                                slider_dim_transforms=
                                                                                self.session_frame_timings[k],
                                                                                name=self.session_names[k])
            self._nd_image_graphics.append(curr_graphic)

        common_camera = self.ndw_videos.figure[0].camera
        for subplot in self.ndw_mip.figure:
            subplot.camera = common_camera
        for subplot in self.ndw_videos.figure:  ##
            subplot.camera = common_camera

        self._ndw_cluster_map = fpl.NDWidget(names=['raster'],
                                             shape=(1, 1),
                                             size=(200, 300))

        self._ndw_raster_graphic = self.ndw_cluster_map['raster'].add_nd_image(self.clustering_mat,
                                                                               ("height", "width"),
                                                                               ("height", "width"),
                                                                               name='trace_raster')

        self._selection_vector = fpl.SelectionVector()
        self._cluster_map_selector = fpl.ImageHighlightSelector(lut="tab10",
                                                                lut_wrap="repeat",
                                                                selection_options={"rows": [i for i in range(
                                                                    self.clustering_mat.shape[0])]},
                                                                options_color="w",
                                                                options_alpha=0.1,
                                                                alpha=0.95,
                                                                )
        self._cluster_map_selector.add_graphic(self._ndw_raster_graphic.graphic)
        self._cluster_map_selector.selection = None
        self._selection_vector.add_selector(
            self._cluster_map_selector)  # The global and local indices for this selector are identical
        self._ndw_raster_graphic.graphic.add_event_handler(partial(self.raster_selection), "double_click")

        ## Now let's add the individual contour selectors for each session
        self._image_highlight_selectors = []
        for index, sess_id in enumerate(self.session_ids):
            curr_selector = fpl.ImageHighlightSelector(lut="tab10",
                                                       lut_wrap="repeat",
                                                       selection_options={"pixels": self.ac_arrays[index].contours},
                                                       options_color="w",
                                                       options_alpha=0.0,
                                                       alpha=0.95)
            curr_selector.add_graphic(self._nd_image_graphics[index].graphic)
            curr_selector.add_graphic(self._nd_mip_graphics[index].graphic)
            curr_selector.selection = None
            ## Provide a dictionary specifying global indices --> local indices for each labeling
            curr_map = self._construct_global_to_local_map(sess_id)
            self._selection_vector.add_selector((curr_selector, curr_map))
            self._image_highlight_selectors.append(curr_selector)

        ## Add double click events to all the ndimage graphics:
        for index, _ in enumerate(self.session_ids):
            curr_nd_image_graphic = self._nd_image_graphics[index].graphic
            curr_nd_image_graphic.add_event_handler(partial(self.neuron_selection, index), "double_click")
            curr_nd_mip_graphic = self._nd_mip_graphics[index].graphic
            curr_nd_mip_graphic.add_event_handler(partial(self.neuron_selection, index), "double_click")

        # Turn off tooltip
        for subplot in self.ndw_videos.figure:
            subplot.tooltip.enabled = False
        for subplot in self.ndw_mip.figure:
            subplot.tooltip.enabled = False

    def neuron_selection(self,
                         display_sess_index: int,
                         ev):
        curr_ac = self.ac_arrays[display_sess_index]
        ## Mask out cells that never got tracked
        tracked = torch.as_tensor(
            self.tracking_results.labels_by_session[self.session_ids[display_sess_index]] > 0,
            dtype=torch.bool)
        neuron = component_at_pixel(curr_ac.a,
                                    curr_ac.centers,
                                    curr_ac.shape[1:],
                                    ev.pick_info['index'],
                                    mask=tracked)
        if neuron is None:
            return
        self._image_highlight_selectors[display_sess_index].selection = neuron

    def raster_selection(self, ev):
        col, row = ev.pick_info['index']

        ## Select the right neuron
        self._cluster_map_selector.selection = int(row)

        min_lb, max_ub = None, None
        for index, selector in enumerate(self._image_highlight_selectors):
            ## Access the ID of the neuron belonging to this session
            neuron_id = selector.selection[0]
            if neuron_id is not None:
                curr_ac_contour = self.ac_arrays[index].contours[neuron_id]
                lb, ub = contours_to_bbox(self.ac_arrays[index].shape[1:], curr_ac_contour)
                if min_lb is None:
                    min_lb = list(lb)
                else:
                    if min_lb[0] > lb[0]:
                        min_lb[0] = lb[0]
                    if min_lb[1] > lb[1]:
                        min_lb[1] = lb[1]
                    min_lb[2] = 1

                if max_ub is None:
                    max_ub = list(ub)
                else:
                    if max_ub[0] < ub[0]:
                        max_ub[0] = ub[0]
                    if max_ub[1] < ub[1]:
                        max_ub[1] = ub[1]
                    max_ub[2] = 1

        ## If no neurons were found
        if min_lb is None or max_ub is None:
            return

            # These are now the spatial bounds to apply uniformly across all FOV
        lb_apply = tuple(min_lb)
        ub_apply = tuple(max_ub)

        for index, selector in enumerate(self._image_highlight_selectors):
            ## Access the ID of the neuron belonging to this session
            neuron_id = selector.selection[0]
            ## Apply below crop regardless of whether or not neuron exists
            # curr_ac_contour = self.ac_arrays[index].contours[neuron_id]

            curr_subplot = self.ndw_videos.figure[index]
            for graphic in curr_subplot.graphics:
                zoom_to_bbox(curr_subplot, graphic, lb_apply, ub_apply)

    def _construct_global_to_local_map(self, sess_id: int) -> dict:
        ## Make an inverse map:
        inverse_map = {int(value): int(index) for index, value in enumerate(self.cluster_ids)}
        return {inverse_map[value]: int(index) for index, value in
                enumerate(self.tracking_results.labels_by_session[sess_id]) if
                int(value) >= 0 and int(value) in self.cluster_ids}

    def _validate_session_ids(self, session_ids: np.ndarray):
        for k in range(len(session_ids)):
            if not 0 <= session_ids[k] < self.tracking_results.num_sessions:
                raise ValueError(
                    f"Your tracking results contain {self.tracking_results.num_sessions}, all session ids must be a nonnegative integer less than this value")
        return True

    def _session_timeaxis_name(self, session_id: int):
        """
        standardized way to make a time axis for each individual
        """
        return f"time sess {session_id}"

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
    def ac_arrays(self) -> list[masknmf.ACArray]:
        return self._ac_arrays

    @property
    def colorful_ac_arrays(self) -> list[masknmf.ColorfulACArray]:
        return self._colorful_ac_arrays

    def _apply_consistent_coloring(self):
        ## Load the AC Array data now, using a common coloring scheme etc.
        cluster_id_to_index = np.zeros((len(self.cluster_ids),)).astype('int')
        cluster_id_to_index[self.cluster_ids] = np.arange(len(self.cluster_ids)).astype('int')
        for index, sess_id in enumerate(self.session_ids):
            ## Let's define a mask for both arrays
            curr_ac_array = self.ac_arrays[index]
            curr_colorful_ac_array = self.colorful_ac_arrays[index]
            curr_labels = tracking_results.labels_by_session[sess_id]
            mask = np.isin(curr_labels, self.cluster_ids)
            curr_ac_array.mask = torch.from_numpy(mask)
            curr_colorful_ac_array.mask = torch.from_numpy(mask)

            ## Now define the coloring scheme
            curr_coloring = np.zeros((int(curr_ac_array.a.shape[1]), 3)).astype('float32')
            clusters_present = curr_labels[mask]
            cluster_indices = cluster_id_to_index[clusters_present]
            curr_coloring[mask, :] = self.coloring[cluster_indices, :]
            curr_colorful_ac_array.colors = torch.from_numpy(curr_coloring).float()

    @property
    def reference_ranges(self):
        return self._reference_ranges

    @property
    def reference_index(self):
        return self.ndw_videos.indices

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
        This will be displayed so the user can click on rows (clusters) and see the corresponding neural signals across sessions
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

    ## Below are properties exposing the widgets + the show function
    @property
    def ndw_videos(self) -> fpl.NDWidget:
        return self._ndw_videos

    @property
    def ndw_cluster_map(self) -> fpl.NDWidget:
        return self._ndw_cluster_map

    @property
    def ndw_mip(self) -> fpl.NDWidget:
        return self._ndw_mip

    def show(self):

        if is_notebook_canvas(self.ndw_videos.figure):
            from ipywidgets import HBox, VBox
            return HBox(
                [VBox([self.ndw_videos.show(), self.ndw_mip.show()]), self.ndw_cluster_map.show(maintain_aspect=False)])
        else:
            return self.ndw_videos.show(), self.ndw_mip.show(), self.ndw_cluster_map.show(maintain_aspect=False)

    def close(self):
        for widget in (self.ndw_videos, self.ndw_mip, self.ndw_cluster_map):
            close_figure(widget.figure)

