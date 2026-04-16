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


class MotionCorrectionVis:
    def __init__(
        self,
        raw_stack: masknmf.LazyFrameLoader,
        registration_array: masknmf.RegistrationArray,
        ref_range: Optional[dict] = None,
        frame_timings: Optional[np.ndarray] = None,
    ):
        self._raw_stack = raw_stack
        self._registration_array = registration_array
        display(
            "Extracting shifts, this may take a moment if shifts were not precomputed"
        )
        self._shifts = self.registration_array.shifts[:]

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
                ref_range = {"time": (0, registration_array.shape[0], 1)}
                frame_timings = np.arange(registration_array.shape[0])

        if self.shifts.ndim == 4:  # This is piecewise rigid registration
            rigid_shifts = False
        elif self.shifts.ndim == 2:
            rigid_shifts = True
        else:
            raise ValueError("Shifts can either have ndim 2 or 4")

        self._extents = {
            "raw data": (0, 0.5, 0.0, 0.6),  # raw data
            "motion corrected": (0.5, 1.0, 0.0, 0.6),  # motion correction
            "applied shifts (height)": (0.0, 1, 0.6, 0.8),  # traces y axis
            "applied shifts (width)": (0.0, 1, 0.8, 1.0),  # traces x axis
        }

        self._ndw = fpl.NDWidget(
            ref_range,
            extents=self._extents,
            names=[
                "raw data",
                "motion corrected",
                "applied shifts (height)",
                "applied shifts (width)",
            ],
            controller_ids=[
                ["raw data", "motion corrected"],
                ["applied shifts (height)"], ["applied shifts (width)"],
            ],
            size=(1200, 1200),
        )

        movie_dims = ["time", "m", "n"]
        movie_spatial_dims = ["m", "n"]
        movie_index_mapping = {"time": frame_timings}
        self._ndw["raw data"].add_nd_image(
            self.raw_stack,
            movie_dims,
            movie_spatial_dims,
            slider_dim_transforms=movie_index_mapping.copy(),
            name="raw data",
        )

        if not rigid_shifts:
            vector_dims = ["time", "num vecs", "vec dim", "stack dim"]
            spatial_dims = ["num vecs", "vec dim", "stack dim"]
            vector_data = pwrigid_shifts_to_ndvector(self.shifts, self.registration_array.block_centers)
            self._ndvec = self._ndw['raw data'].add_nd_vectors(
                vector_data,
                vector_dims,
                spatial_dims,
                name="vectors",
                size=5
            )
            self._ndw.figure['raw data'].title = "Raw Data + Applied Shift Vectors"
        else:
            self._ndvec = None

        self._ndw["motion corrected"].add_nd_image(
            self.registration_array,
            movie_dims,
            movie_spatial_dims,
            slider_dim_transforms=movie_index_mapping.copy(),
            name="motion corrected",
        )


        self._ndw.figure['raw data'].tooltip.enabled = False
        self._ndw.figure['motion corrected'].tooltip.enabled = False

        #No matter what method was used, we construct a summary shift time series, one for each spatial dim (height, width)
        if rigid_shifts:
            summary_shifts = self.shifts
            height_message = "applied rigid shifts height"
            width_message = "applied rigid shifts width"
        else:
            summary_shifts = np.amax(np.abs(self.shifts), axis = (1, 2))
            height_message = "max pwrigid shift height"
            width_message = "max pwrigid shift width"

        height_shift_data = np.zeros((1, summary_shifts.shape[0], 2))
        height_shift_data[0, :, 0] = np.arange(summary_shifts.shape[0])
        height_shift_data[0, :, 1] = summary_shifts[:, 0]
        self._ndw["applied shifts (height)"].add_nd_timeseries(
            height_shift_data,
            ("l", "time", "d"),
            ("l", "time", "d"),
            slider_dim_transforms=movie_index_mapping.copy(),
            x_range_mode="auto",
            display_window=50.0,
            name="applied shifts (height)",
        )

        self._ndw.figure['applied shifts (height)'].title = height_message

        width_shift_data = np.zeros((1, summary_shifts.shape[0], 2))
        width_shift_data[0, :, 0] = np.arange(summary_shifts.shape[0])
        width_shift_data[0, :, 1] = summary_shifts[:, 1]
        self._ndw["applied shifts (width)"].add_nd_timeseries(
            width_shift_data,
            ("l", "time", "d"),
            ("l", "time", "d"),
            slider_dim_transforms=movie_index_mapping.copy(),
            x_range_mode="auto",
            display_window=50.0,
            name="applied shifts (width)",
        )

        self._ndw.figure['applied shifts (width)'].title = width_message

        #Link the traces in X but not in Y
        camera_height = self.widget.figure['applied shifts (height)'].camera
        camera_width = self.widget.figure['applied shifts (width)'].camera

        controller_height = self.widget.figure['applied shifts (height)'].controller
        controller_width = self.widget.figure['applied shifts (width)'].controller

        controller_height.add_camera(camera_width, include_state={"x", "width"})
        controller_width.add_camera(camera_height, include_state={"x", "width"})

        for subplot in self.widget.figure:
            subplot.toolbar = False



    @property
    def raw_stack(self) -> masknmf.LazyFrameLoader:
        return self._raw_stack

    @property
    def registration_array(self) -> masknmf.RegistrationArray:
        return self._registration_array

    @property
    def shifts(self) -> np.ndarray:
        return self._shifts

    @property
    def widget(self) -> fpl.NDWidget:
        return self._ndw

    def show(self):
        return self.widget.show()


def pwrigid_shifts_to_ndvector(shifts, block_centers):
    """
    shifts (np.ndarray): Shape (num_frames, height blocks, width blocks, 2)
    block_centers (np.ndarray): Shape (height blocks, width_blocks, 2)

    Returns a dataset of shape (num_frames, height_blocks*width_blocks, 2, 2) to construct ndvectors graphic
    """
    final_output = np.zeros((shifts.shape[0], shifts.shape[1]*shifts.shape[2], 2, 2))
    shift_data = shifts.reshape(shifts.shape[0], -1, 2)
    shift_data = shift_data - np.mean(shift_data, axis = 1, keepdims = True)
    final_output[:, :, 1, :] = shift_data #shifts.reshape(shifts.shape[0], -1, 2)
    final_output[:, :, 0, :] = block_centers.reshape(-1, 2)[None, :, :]
    return final_output