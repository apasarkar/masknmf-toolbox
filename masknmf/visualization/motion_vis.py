import math
from functools import partial
from typing import *
import numpy as np
import torch
import fastplotlib as fpl
from tqdm import tqdm
import masknmf.arrays
from masknmf.arrays.array_interfaces import ArrayLike
from masknmf.utils import display
from masknmf.visualization.imgui import resolve_time_reference
from masknmf.motion_correction import BaseRegistrationArray

def compute_mean_subtract(mean: np.ndarray | torch.Tensor,
                  frame: np.ndarray | torch.Tensor):
    """
    Basic function to do on the fly mean subtraction of ophys data
    Args:
        mean (np.ndarray). Shape (height, width)
        frame (np.ndarray). Shape (height, width)
    """
    if isinstance(mean, torch.Tensor) and isinstance(frame, torch.Tensor):
        return frame - mean.to(frame.device)
    elif isinstance(mean, np.ndarray) and isinstance(frame, np.ndarray):
        return frame - mean
    else:
        raise ValueError("both inputs must be torch tensors or np arrays")


class MotionCorrectionVis:
    def __init__(
        self,
        registration_array: BaseRegistrationArray,
        ref_range: Optional[dict] = None,
        frame_timings: Optional[np.ndarray] = None,
        mean_subtract: bool = False,
        num_frames_mean_sketch: int = 1000,
    ):
        self._registration_array = registration_array
        self._raw_stack = self.registration_array.input_movie

        if mean_subtract:
            display(f"Sketching the mean of each stack with {num_frames_mean_sketch} frames")
            raw_mean, register_mean = self._sketch_means(num_frames=num_frames_mean_sketch)
            register_mean = torch.as_tensor(register_mean,
                                            device=self.registration_array.output_device,
                                            dtype=self.registration_array.dtype)
            spatial_func_raw = partial(compute_mean_subtract, raw_mean)
            spatial_func_register = partial(compute_mean_subtract, register_mean)
        else:
            spatial_func_raw = None
            spatial_func_register = None

        self._shifts = self.registration_array.shifts

        ref_range, frame_timings = resolve_time_reference(
            registration_array.shape[0], frame_timings, ref_range
        )

        if self.shifts.ndim == 4:  # This is piecewise rigid registration
            rigid_shifts = False
        elif self.shifts.ndim == 2:
            rigid_shifts = True
        else:
            raise ValueError("Shifts can either have ndim 2 or 4")

        names = [
            "raw data" if not mean_subtract else "raw data mean 0",
            "motion corrected" if not mean_subtract else "motion corrected mean 0",
            "applied shifts (height)",
            "applied shifts (width)",
        ]

        self._extents = {
            names[0]: (0, 0.5, 0.0, 0.6),  # raw data
            names[1]: (0.5, 1.0, 0.0, 0.6),  # motion correction
            names[2]: (0.0, 1, 0.6, 0.8),  # traces y axis
            names[3]: (0.0, 1, 0.8, 1.0),  # traces x axis
        }


        self._ndw = fpl.NDWidget(
            ref_range,
            extents=self._extents,
            names=names,
            controller_ids=[
                [names[0], names[1]],
                [names[2]], [names[3]],
            ],
            size=(1200, 1200),
        )

        movie_dims = ["time", "m", "n"]
        movie_spatial_dims = ["m", "n"]
        movie_index_mapping = {"time": frame_timings}
        self._ndw[names[0]].add_nd_image(
            self.raw_stack,
            movie_dims,
            movie_spatial_dims,
            spatial_func=spatial_func_raw,
            slider_dim_transforms=movie_index_mapping.copy(),
            name=names[0],
        )

        if not rigid_shifts:
            vector_dims = ["time", "num vecs", "vec dim", "stack dim"]
            spatial_dims = ["num vecs", "vec dim", "stack dim"]
            vector_data = pwrigid_shifts_to_ndvector(self.shifts.cpu().numpy(), self.registration_array.block_centers.cpu().numpy())
            ndvec_graphic_kwargs = {'size': 5}
            self._ndvec = self._ndw[names[0]].add_nd_vectors(
                vector_data,
                vector_dims,
                spatial_dims,
                name="vectors",
                graphic_kwargs=ndvec_graphic_kwargs
            )
            self._ndw.figure[names[0]].title = names[0] + " Applied Shift Vectors"
        else:
            self._ndvec = None

        self._ndw[names[1]].add_nd_image(
            self.registration_array,
            movie_dims,
            movie_spatial_dims,
            spatial_func=spatial_func_register,
            slider_dim_transforms=movie_index_mapping.copy(),
            name=names[1],
        )


        self._ndw.figure[names[0]].tooltip.enabled = False
        self._ndw.figure[names[1]].tooltip.enabled = False

        #No matter what method was used, we construct a summary shift time series, one for each spatial dim (height, width)
        if rigid_shifts:
            summary_shifts = self.shifts.cpu().numpy()
            height_message = "applied rigid shifts height"
            width_message = "applied rigid shifts width"
        else:
            summary_shifts = np.amax(np.abs(self.shifts.cpu().numpy()), axis = (1, 2))
            height_message = "max pwrigid shift height"
            width_message = "max pwrigid shift width"

        height_shift_data = np.zeros((1, summary_shifts.shape[0], 2))
        height_shift_data[0, :, 0] = np.arange(summary_shifts.shape[0])
        height_shift_data[0, :, 1] = summary_shifts[:, 0]
        self._ndw[names[2]].add_nd_timeseries(
            height_shift_data,
            ("l", "time", "d"),
            ("l", "time", "d"),
            slider_dim_transforms=movie_index_mapping.copy(),
            x_range_mode="auto",
            display_window=50.0,
            name=names[2],
        )

        self._ndw.figure[names[2]].title = height_message

        width_shift_data = np.zeros((1, summary_shifts.shape[0], 2))
        width_shift_data[0, :, 0] = np.arange(summary_shifts.shape[0])
        width_shift_data[0, :, 1] = summary_shifts[:, 1]
        self._ndw[names[3]].add_nd_timeseries(
            width_shift_data,
            ("l", "time", "d"),
            ("l", "time", "d"),
            slider_dim_transforms=movie_index_mapping.copy(),
            x_range_mode="auto",
            display_window=50.0,
            name=names[3],
        )

        self._ndw.figure[names[3]].title = width_message

        #Link the traces in X but not in Y
        camera_height = self.widget.figure[names[2]].camera
        camera_width = self.widget.figure[names[3]].camera

        controller_height = self.widget.figure[names[2]].controller
        controller_width = self.widget.figure[names[3]].controller

        controller_height.add_camera(camera_width, include_state={"x", "width"})
        controller_width.add_camera(camera_height, include_state={"x", "width"})

        for subplot in self.widget.figure:
            subplot.toolbar = False


    def _sketch_means(self, num_frames: int = 1000):
        num_frames_used = min(num_frames, self.raw_stack.shape[0])
        frame_indices = np.random.choice(np.arange(self.raw_stack.shape[0]), size=num_frames_used, replace=False)
        raw_mean = np.zeros((self.raw_stack.shape[1], self.raw_stack.shape[2]))
        register_mean = torch.zeros(self.registration_array.shape[1],
                                    self.registration_array.shape[2],
                                    device=self.registration_array.output_device,
                                    dtype=self.registration_array.dtype)
        batch_size = self.registration_array.strategy.batch_size
        num_iters = math.ceil(num_frames_used / batch_size)
        for k in tqdm(range(num_iters)):
            start = k * batch_size
            end = min(start + batch_size, frame_indices.shape[0])
            raw_frames = self.raw_stack[frame_indices[start:end]]
            reg_frames = self.registration_array[frame_indices[start:end]]
            raw_mean += (np.sum(raw_frames, axis = 0) / num_frames_used)
            register_mean += (torch.sum(reg_frames, dim = 0) / num_frames_used)
        return raw_mean, register_mean

    @property
    def raw_stack(self) -> masknmf.LazyFrameLoader:
        return self.registration_array.input_movie

    @property
    def registration_array(self) -> ArrayLike:
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
    final_output[:, :, 0, :] = block_centers.reshape(-1, 2)[None, :, ::-1]
    return final_output