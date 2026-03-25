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

        demixing_colors = self.demixing_results.colorful_ac_array.colors



        self._video_panels = ("compressed+denoised",
                        "signals",
                        "background",
                        "residual",
                        "colorful_signals",
                        "residual corr img")

        self._pmd_array = self.demixing_results.pmd_array
        self._fluctuating_background_array = self.demixing_results.fluctuating_background_array
        self._residual_array = self.demixing_results.residual_array
        self._colorful_ac_array = self.demixing_results.colorful_ac_array
        self._ac_array = self.demixing_results.ac_array

        self._trace_panels = ("compressed trace",
                        "demixed trace",
                        "background trace",
                        "residual trace")

        self._extents =  {
                self._video_panels[0]: (0, 0.333, 0.0, 0.2),
                self._video_panels[1]: (0.33, 0.666, 0.0, 0.2),
                self._video_panels[2]: (0.666, 1, 0.0, 0.2),
                self._video_panels[3]: (0.0, 0.333, 0.2, 0.4),
                self._video_panels[4]: (0.333, 0.666, 0.2, 0.4),
                self._video_panels[5]: (0.666, 1, 0.2, 0.4),
                self._trace_panels[0]: (0, 1, 0.4, 0.55),
                self._trace_panels[1]: (0, 1, 0.55, 0.7),
                self._trace_panels[2]: (0, 1, 0.7, 0.85),
                self._trace_panels[3]: (0, 1, 0.85, 1.0)
            }

        self._ndw = fpl.NDWidget(
            ref_range,
            extents=self._extents,
            names=[*self._video_panels,*self._trace_panels],
            controller_ids=[
                tuple(self._video_panels),
                tuple(self._trace_panels),
            ],
            size=(1200, 1200),
        )

        movie_dims = ["time", "m", "n"]
        movie_spatial_dims = ["m", "n"]
        movie_index_mapping = {"time": frame_timings}
        self._ndw[self._video_panels[0]].add_nd_image(
            self._pmd_array,
            movie_dims,
            movie_spatial_dims,
            slider_dim_transforms=movie_index_mapping.copy(),
            name=self._video_panels[0],
        )

        self._ndw[self._video_panels[1]].add_nd_image(
            self._ac_array,
            movie_dims,
            movie_spatial_dims,
            slider_dim_transforms=movie_index_mapping.copy(),
            name=self._video_panels[1],
        )

        self._ndw[self._video_panels[2]].add_nd_image(
            self._fluctuating_background_array,
            movie_dims,
            movie_spatial_dims,
            slider_dim_transforms=movie_index_mapping.copy(),
            name=self._video_panels[2],
        )

        self._ndw[self._video_panels[3]].add_nd_image(
            self._residual_array,
            movie_dims,
            movie_spatial_dims,
            slider_dim_transforms=movie_index_mapping.copy(),
            name=self._video_panels[3],
        )

        movie_dims_rgb = ["time", "m", "n", "c"]
        movie_spatial_dims_rgb= ["m", "n", "c"]
        movie_index_mapping = {"time": frame_timings}
        self._ndw[self._video_panels[4]].add_nd_image(
            self._colorful_ac_array,
            movie_dims_rgb,
            movie_spatial_dims_rgb,
            slider_dim_transforms=movie_index_mapping.copy(),
            rgb_dim="c",
            name=self._video_panels[4],
        )

        self._ndw[self._video_panels[5]].add_nd_image(
            self.demixing_results.global_residual_correlation_image.cpu().numpy(),
            ["m", "n"],
            ["m", "n"],
            name=self._video_panels[5],
        )

        #Traces for the denoised data
        self._ndw[self._trace_panels[0]].add_nd_timeseries(
                None,
                ("l", "time", "d"),
                ("l", "time", "d"),
                slider_dim_transforms=movie_index_mapping.copy(),
                x_range_mode="auto",
                display_window=10000000000.0,
                name=self._trace_panels[0],
            )

        #Traces for the color-matched signals
        self._ndw[self._trace_panels[1]].add_nd_timeseries(
            None,
            ("l", "time", "d"),
            ("l", "time", "d"),
            slider_dim_transforms=movie_index_mapping.copy(),
            x_range_mode="auto",
            display_window=10000000000.0,
            name=self._trace_panels[1],
        )

        #Traces for the background
        self._ndw[self._trace_panels[2]].add_nd_timeseries(
            None,
            ("l", "time", "d"),
            ("l", "time", "d"),
            slider_dim_transforms=movie_index_mapping.copy(),
            x_range_mode="auto",
            display_window=10000000000.0,
            name=self._trace_panels[2],
        )

        #Traces for the residual
        self._ndw[self._trace_panels[3]].add_nd_timeseries(
            None,
            ("l", "time", "d"),
            ("l", "time", "d"),
            slider_dim_transforms=movie_index_mapping.copy(),
            x_range_mode="auto",
            display_window=10000000000.0,
            name=self._trace_panels[3],
        )

        for name in self._video_panels:
            print(type(self._ndw[name][name]))

            self._ndw[name][name].graphic.add_event_handler(partial(self._click_update), "double_click")

        for subplot in self._ndw.figure:
            subplot.toolbar = False

    ## Let's make a dummy click event for now
    def _click_update(self, ev: pygfx.PointerEvent):
        num_frames, height, width = self.demixing_results.shape
        x_data = np.arange(num_frames)
        col, row = ev.pick_info["index"]

        colslice = slice(max(0, col - self._roi_radius), min(width, col + self._roi_radius + 1))
        rowslice = slice(max(0, row - self._roi_radius), min(height, row + self._roi_radius + 1))
        ## For each array, add the appropriate data

        pmd_trace = np.mean(self._pmd_array[:, rowslice, colslice], axis = (1,2))
        residual_trace = np.mean(self._residual_array[:, rowslice, colslice], axis = (1, 2))
        background_trace = np.mean(self._fluctuating_background_array[:, rowslice, colslice], axis = (1, 2))
        ac_trace = np.mean(self._ac_array[:, rowslice, colslice], axis = (1, 2))


        self._ndw[self._trace_panels[0]][self._trace_panels[0]].data = format_timeseries(num_frames, x_data, pmd_trace)
        self._ndw[self._trace_panels[1]][self._trace_panels[1]].data = format_timeseries(num_frames, x_data, ac_trace)
        self._ndw[self._trace_panels[2]][self._trace_panels[2]].data = format_timeseries(num_frames, x_data, background_trace)
        self._ndw[self._trace_panels[3]][self._trace_panels[3]].data = format_timeseries(num_frames, x_data, residual_trace)



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
    def widget(self) -> fpl.NDWidget:
        return self._ndw

    def show(self):
        return self.widget.show()

def format_timeseries(data_shape: int,
                      x_data: np.ndarray,
                      y_data: np.ndarray):
    series = np.zeros((1, data_shape, 2))
    series[0, :, 0] = x_data
    series[0, :, 1] = y_data
    return series