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
from fastplotlib.widgets.nd_widget._index import ReferenceIndex

class SingleSessionDemixingVis:
    """
    This is a general viewer for analyzing demixing results a single imaging field of view (say, frames x height x width)
    """
    def __init__(
        self,
        demixing_results: masknmf.DemixingResults | List[masknmf.DemixingResults],
        frame_timings: Optional[np.ndarray | List[np.ndarray]] = None,
        ref_range: Optional[dict] = None,
        roi_radius: int = 1,
        device='cpu'
    ):
        self._roi_radius = roi_radius
        if device=='cpu':
            display("Using CPU; it will be much slower. Use CUDA for much faster rendering")
        self._demixing_results = demixing_results
        self._device = device

        self._demixing_results.to(self.device)

        if frame_timings is not None:
            if ref_range is None:
                ref_range = {
                    "time": (
                        0,
                        np.amax(frame_timings),
                        np.amin(frame_timings[1:] - frame_timings[:-1]),
                    )
                }
        else:
            if ref_range is not None:
                raise ValueError(
                    "If you provide a reference range, you need to provide the imaging frame timings (per frame) within that range"
                )
            else:
                ref_range = {"time": (0, self.demixing_results.shape[0], 1)}
                frame_timings = np.arange(self.demixing_results.shape[0])




        self._video_panels = ("compressed+denoised",
                        "signals",
                        "background",
                        "residual",
                        "colorful_signals",
                        "residual corr img")

        self._local_signal_panels = ("signals",
                                     "traces")

        self._pmd_array = self.demixing_results.pmd_array
        self._fluctuating_background_array = self.demixing_results.fluctuating_background_array
        self._residual_array = self.demixing_results.residual_array
        self._colorful_ac_array = self.demixing_results.colorful_ac_array
        self._ac_array = self.demixing_results.ac_array

        self._trace_panels = ("compressed trace",
                        "demixed trace",
                        "background trace",
                        "residual trace")

        self._video_extents =  {
                self._video_panels[0]: (0, 0.333, 0.0, 0.5),
                self._video_panels[1]: (0.33, 0.666, 0.0, 0.5),
                self._video_panels[2]: (0.666, 1, 0.0, 0.5),
                self._video_panels[3]: (0.0, 0.333, 0.5, 1.0),
                self._video_panels[4]: (0.333, 0.666, 0.5, 1.0),
                self._video_panels[5]: (0.666, 1, 0.5, 1.0)}
        self._trace_extents = {
                self._trace_panels[0]: (0, 1, 0.0, 0.25),
                self._trace_panels[1]: (0, 1, 0.25, 0.5),
                self._trace_panels[2]: (0, 1, 0.5, 0.75),
                self._trace_panels[3]: (0, 1, 0.75, 1.0)
            }
        self._local_signal_extents = {
            self._local_signal_panels[0]: (0, 0.2, 0, 1),
            self._local_signal_panels[1]: (0.2, 1, 0, 1)
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

        self._ac_graphic = self._ndw_fov[self._video_panels[1]].add_nd_image(
            self._ac_array,
            movie_dims,
            movie_spatial_dims,
            slider_dim_transforms=movie_index_mapping.copy(),
            name=self._video_panels[1],
        )

        self._background_graphic = self._ndw_fov[self._video_panels[2]].add_nd_image(
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
        movie_spatial_dims_rgb= ["m", "n", "c"]
        movie_index_mapping = {"time": frame_timings}
        self._colorful_signal_graphic = self._ndw_fov[self._video_panels[4]].add_nd_image(
            self._colorful_ac_array,
            movie_dims_rgb,
            movie_spatial_dims_rgb,
            slider_dim_transforms=movie_index_mapping.copy(),
            rgb_dim="c",
            name=self._video_panels[4],
        )

        self._residual_correlation_graphic = self._ndw_fov[self._video_panels[5]].add_nd_image(
            self.demixing_results.global_residual_correlation_image.cpu().numpy(),
            ["m", "n"],
            ["m", "n"],
            name=self._video_panels[5],
        )

        self._ndw_traces = fpl.NDWidget(
            ref_ranges=self.reference_index.ref_ranges,
            ref_index=self.reference_index,
            extents=self._trace_extents,
            names=[*self._trace_panels],
            controller_ids=[
                tuple(self._trace_panels),
            ],
            size=(1200, 1200),
        )

        #Traces for the denoised data
        self._pmd_trace_graphic = self._ndw_traces[self._trace_panels[0]].add_nd_timeseries(
                None,
                ("l", "time", "d"),
                ("l", "time", "d"),
                slider_dim_transforms=movie_index_mapping.copy(),
                max_display_datapoints=5000,
                x_range_mode="auto",
                display_window=None,
                name=self._trace_panels[0],
            )

        #Traces for the color-matched signals
        self._colorful_signal_trace_graphic = self._ndw_traces[self._trace_panels[1]].add_nd_timeseries(
            None,
            ("l", "time", "d"),
            ("l", "time", "d"),
            slider_dim_transforms=movie_index_mapping.copy(),
            x_range_mode="auto",
            max_display_datapoints=5000,
            display_window=None,
            name=self._trace_panels[1],
        )

        #Traces for the background
        self._fluctuating_background_trace_graphic = self._ndw_traces[self._trace_panels[2]].add_nd_timeseries(
            None,
            ("l", "time", "d"),
            ("l", "time", "d"),
            slider_dim_transforms=movie_index_mapping.copy(),
            x_range_mode="auto",
            max_display_datapoints=5000,
            display_window=None,
            name=self._trace_panels[2],
        )

        #Traces for the residual
        self._residual_trace_graphic = self._ndw_traces[self._trace_panels[3]].add_nd_timeseries(
            None,
            ("l", "time", "d"),
            ("l", "time", "d"),
            slider_dim_transforms=movie_index_mapping.copy(),
            x_range_mode="auto",
            max_display_datapoints=5000,
            display_window=None,
            name=self._trace_panels[3],
        )

        self._ndw_local_signals = fpl.NDWidget(
            ref_ranges=self.reference_index.ref_ranges,
            ref_index=self.reference_index,
            extents=self._local_signal_extents,
            names=[*self._local_signal_panels],
            controller_ids=[
                tuple(self._local_signal_panels),
            ],
            size=(1200, 1200),
        )

        for name in self._video_panels:
            self._ndw_fov[name][name].graphic.add_event_handler(partial(self._click_update), "double_click")

        for subplot in self._ndw_fov.figure:
            subplot.toolbar = False

        for subplot in self._ndw_traces.figure:
            subplot.toolbar = False

    ## Let's make a dummy click event for now
    def _click_update(self, ev: pygfx.PointerEvent):
        num_frames, height, width = self.demixing_results.shape
        x_data = np.arange(num_frames)
        col, row = ev.pick_info["index"]

        col_start, col_stop = max(0, col - self._roi_radius), min(width, col + self._roi_radius + 1)
        row_start, row_stop = max(0, row - self._roi_radius), min(height, row + self._roi_radius + 1)
        ## For each array, add the appropriate data

        pmd_trace = np.mean(self._pmd_array[:, row_start:row_stop, col_start:col_stop], axis = (1,2))
        residual_trace = np.mean(self._residual_array[:, row_start:row_stop, col_start:col_stop], axis = (1, 2))
        background_trace = np.mean(self._fluctuating_background_array[:, row_start:row_stop, col_start:col_stop], axis = (1, 2))

        max_pmd_trace = np.amax(pmd_trace)
        min_pmd_trace = np.amin(pmd_trace)

        self._pmd_trace_graphic.data = fpl.utils.heatmap_to_positions(pmd_trace[None, :], x_data)
        self._ndw_traces.figure[self._trace_panels[0]].y_range = (min_pmd_trace, max_pmd_trace)

        #Pull out colorful signals
        separated_ac_signals, separated_colors = extract_per_trace_roi_averages(self._colorful_ac_array,
                                                              slice(row_start, row_stop),
                                                              slice(col_start, col_stop))

        if separated_ac_signals is not None:
            colorful_signals_to_display = fpl.utils.heatmap_to_positions(separated_ac_signals, x_data)
            self._colorful_signal_trace_graphic.data = colorful_signals_to_display
            self._ndw_traces.figure[self._trace_panels[1]].y_range = (min_pmd_trace, max_pmd_trace)
            self._colorful_signal_trace_graphic.graphic.colors = separated_colors
            self._ndw_traces.figure[self._trace_panels[1]].title = f"{separated_ac_signals.shape[0]} signals"
        else:
            colorful_signals_to_display = fpl.utils.heatmap_to_positions(np.ones((1, self.demixing_results.shape[0])), x_data)
            self._colorful_signal_trace_graphic.data = colorful_signals_to_display
            self._ndw_traces.figure[self._trace_panels[1]].title = "No signals here"


        self._fluctuating_background_trace_graphic.data = fpl.utils.heatmap_to_positions(background_trace[None, :], x_data)
        self._ndw_traces.figure[self._trace_panels[2]].y_range = (min_pmd_trace, max_pmd_trace)

        self._ndw_traces[self._trace_panels[3]][self._trace_panels[3]].data = fpl.utils.heatmap_to_positions(residual_trace[None, :], x_data)
        self._ndw_traces.figure[self._trace_panels[3]].y_range = (min_pmd_trace, max_pmd_trace)

    @property
    def roi_radius(self) -> int:
        return self._roi_radius

    @roi_radius.setter
    def roi_radius(self, new_radius):
        self._roi_radius = new_radius

    @property
    def device(self) -> str:
        return self._device

    @property
    def demixing_results(self) -> masknmf.DemixingResults:
        return self._demixing_results

    @property
    def fov_widget(self) -> fpl.NDWidget:
        return self._ndw_fov

    @property
    def trace_widget(self) -> fpl.NDWidget:
        return self._ndw_traces

    @property
    def local_signal_widget(self) -> fpl.NDWidget:
        return self._ndw_local_signals

    @property
    def reference_index(self) -> ReferenceIndex:
        return self._reference_index

    def show(self):

        # parse based on canvas type
        if self.fov_widget.figure.canvas.__class__.__name__ == "JupyterRenderCanvas":
            from ipywidgets import VBox
            return VBox([self.fov_widget.show(), self.trace_widget.show()])
        else:
            return self.fov_widget.show(), self.trace_widget.show()


def extract_per_trace_roi_averages(colorful_ac_array: masknmf.ACArray,
                                   rowslice: slice,
                                   colslice: slice):
    """

    Args:
        ac_array (masknmf.ACArray): The signal array that contains the factorized signals
        coloring (torch.tensor): Shape (num_neurons, 3) #Each row is RGB coloring
    """
    device = colorful_ac_array.device
    num_frames, height, width, _ = colorful_ac_array.shape
    a = colorful_ac_array.a.coalesce() #Shape (num_pixels, num_signals)
    c = colorful_ac_array.c #Shape (num_frames, num_signals)

    pixel_space = torch.arange(height * width, device = device).reshape(height, width).long()
    good_row_values = pixel_space[rowslice, colslice].flatten()
    num_pixels = good_row_values.shape[0]

    row, col = a.indices()
    values = a.values()

    valid_indices = torch.isin(row, good_row_values)
    if torch.count_nonzero(valid_indices) == 0:
        return None, None
    else:
        filtered_rows = row[valid_indices]
        filtered_col = col[valid_indices]
        filtered_values = values[valid_indices]

        reduce_tensor = torch.zeros(a.shape[1], device=device)
        reduce_tensor.scatter_reduce_(0, filtered_col, filtered_values, reduce="sum")
        reduce_tensor = reduce_tensor / num_pixels

        unique_signals = torch.unique(filtered_col)
        unique_means = reduce_tensor[unique_signals]

        weighted_signals = unique_means[None, :] * c[:, unique_signals] #Shape (num_frames, neural_signals)
        colors = colorful_ac_array.colors[unique_signals, :] #(neural_signals, 3)

        return weighted_signals.T.cpu().numpy(), colors.cpu().numpy()







