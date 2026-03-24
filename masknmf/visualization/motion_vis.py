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
            raise ValueError("Not fully supported yet")
            ## TODO: Make a nicer vector field graphic here
            # self._ndw = fpl.NDWidget(ref_range,
            #                          extents=self._extents,
            #                          names=['raw data',
            #                                 'piecewise rigid motion correction',
            #                                 'max applied shift (height)',
            #                                 'max applied shift (width)'],
            #                          controller_ids=[('raw data', 'rigid motion correction'),
            #                                          ('applied shifts (height)', 'applied shifts (width)')],
            #                          size=(1200, 1200))

        elif self.shifts.ndim == 2:
            self._extents = {
                "raw data": (0, 0.5, 0.0, 0.5),  # raw data
                "rigid motion correction": (0.5, 1.0, 0.0, 0.5),  # motion correction
                "applied shifts (height)": (0.0, 1, 0.5, 0.75),  # traces y axis
                "applied shifts (width)": (0.0, 1, 0.75, 1.0),  # traces x axis
            }

            self._ndw = fpl.NDWidget(
                ref_range,
                extents=self._extents,
                names=[
                    "raw data",
                    "rigid motion correction",
                    "applied shifts (height)",
                    "applied shifts (width)",
                ],
                controller_ids=[
                    ("raw data", "rigid motion correction"),
                    ("applied shifts (height)", "applied shifts (width)"),
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

            self._ndw["rigid motion correction"].add_nd_image(
                self.registration_array,
                movie_dims,
                movie_spatial_dims,
                slider_dim_transforms=movie_index_mapping.copy(),
                name="motion correction",
            )

            height_shift_data = np.zeros((1, self.shifts.shape[0], 2))
            height_shift_data[0, :, 0] = np.arange(self.shifts.shape[0])
            height_shift_data[0, :, 1] = self.shifts[:, 0]
            self._ndw["applied shifts (width)"].add_nd_timeseries(
                height_shift_data,
                ("l", "time", "d"),
                ("l", "time", "d"),
                slider_dim_transforms=movie_index_mapping.copy(),
                x_range_mode="auto",
                display_window=50.0,
                name="applied shifts (height)",
            )

            width_shift_data = np.zeros((1, self.shifts.shape[0], 2))
            width_shift_data[0, :, 0] = np.arange(self.shifts.shape[0])
            width_shift_data[0, :, 1] = self.shifts[:, 1]
            self._ndw["applied shifts (height)"].add_nd_timeseries(
                width_shift_data,
                ("l", "time", "d"),
                ("l", "time", "d"),
                slider_dim_transforms=movie_index_mapping.copy(),
                x_range_mode="auto",
                display_window=50.0,
                name="applied shifts (height)",
            )

        for subplot in self._ndw.figure:
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
