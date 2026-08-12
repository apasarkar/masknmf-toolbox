from typing import *
import numpy as np
from fastplotlib.widgets import ImageWidget
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


class MultiSessionDemixingVis:

    def __init__(self,
                 tracking_results: masknmf.multisession.RoicatTrackingResults,
                 session_ids: np.ndarray | list | None = None,
                 clusters: np.ndarray | Callable | None = None,
                 session_names: list[str] | None = None,
                 reference_ranges: dict | None = None,
                 reference_range_timeaxis: str | None = None,
                 session_frame_timings: list[np.ndarray] | None = None,
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
