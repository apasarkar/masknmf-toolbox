import math
from typing import Optional, Callable, Union, Tuple
import numpy as np
from numpy.typing import DTypeLike

import torch
import masknmf
from masknmf.arrays.array_interfaces import LazyFrameLoader, ArrayLike
from .strategies import MotionCorrectionStrategy, RigidMotionCorrector, GradientMotionCorrector, PiecewiseRigidMotionCorrector, DummyMotionCorrector
from .registration_methods import compute_pwrigid_patch_midpoints
from masknmf.utils import Serializer
from pathlib import Path
import h5py
import os
from tqdm import tqdm


class Shifts(ArrayLike):
    def __init__(self,
                 reg_arr: ArrayLike,
                 shift_dims: tuple):
        self._reg = reg_arr
        self._shape = (reg_arr.shape[0], *shift_dims)

    @property
    def dtype(self) -> str:
        return self._reg.dtype

    @property
    def shape(self) -> tuple[int, int, int]:
        return self._shape

    @property
    def ndim(self) -> int:
        return self._reg.ndim

    def __getitem__(self, ind):
        return self._reg._index_frames_tensor(ind)[1].squeeze()


class RegistrationArray(LazyFrameLoader, Serializer):

    _motion_export_name = "motion_corrected"
    _shifts_export_name = "shifts"
    _block_centers_name = "block_centers"

    def __init__(
        self,
        reference_movie: LazyFrameLoader,
        strategy: MotionCorrectionStrategy | None = None,
        target_movie: Optional[LazyFrameLoader] = None,
        shifts: Shifts | np.ndarray | None = None,
        dtype: Union[str, np.dtype] = np.float32
    ):
        """
        Array-like motion correction representation that support on-the-fly motion correction

        Args:
            reference_movie (LazyFrameLoader): Image stack that we use to compute motion correction transform relative to template
            strategy (masknmf.MotionCorrectionStrategy): The method used to register each frame to the template.
                Can initialize as ``None``, but must be set before slicing frames
            target_movie (Optional[LazyFrameLoader]): Once we learn the motion correction transform by aligning reference_dataset
                with template, we actually apply the transform to target_dataset, if it is specified. If None, we apply the
                transform to reference_dataset
        """
        self._reference_movie = reference_movie

        if strategy is None:
            self.strategy = DummyMotionCorrector()
        else:
            self.strategy = strategy

        self._target_movie = target_movie
        self._shape = self.target_movie.shape if self._target_movie is not None else self.reference_movie.shape
        self._ndim = self.reference_movie.ndim

        if shifts is None:
            if isinstance(self.strategy, DummyMotionCorrector):
                shift_shape = (2,)
            elif isinstance(self.strategy, RigidMotionCorrector):
                shift_shape = (2,)
            elif isinstance(self.strategy, GradientMotionCorrector):
                shift_shape = (2,)
            elif isinstance(self.strategy, PiecewiseRigidMotionCorrector):
                shift_shape = self.block_centers.shape
            else:
                raise ValueError("Invalid strategy object provided")
            self._shifts = Shifts(self, shift_shape)
        else:
            #Here the shifts are pre-computed
            self._shifts = shifts

        self._dtype = dtype

    @property
    def ndim(self) -> int:
        return self._ndim

    @property
    def shape(self) -> tuple[int, int, int]:
        return self._shape

    @property
    def dtype(self) ->  Union[str, np.dtype]:
        return self._dtype

    @property
    def reference_movie(self) -> LazyFrameLoader:
        return self._reference_movie

    @property
    def target_movie(self) -> LazyFrameLoader | None:
        return self._target_movie

    @property
    def shifts(self) -> Shifts | np.ndarray:
        """An array-like object that either lazily or directly exposes the per-frame estimated rigid/piecewise rigid shifts"""
        return self._shifts

    @property
    def strategy(self) -> PiecewiseRigidMotionCorrector | RigidMotionCorrector | None:
        return self._strategy

    @strategy.setter
    def strategy(self, corrector: PiecewiseRigidMotionCorrector | RigidMotionCorrector | DummyMotionCorrector | None):
        self._strategy = corrector

        if isinstance(self.strategy, PiecewiseRigidMotionCorrector):
            self._block_centers = compute_pwrigid_patch_midpoints(
                num_blocks=self.strategy.num_blocks,
                overlaps=self.strategy.overlaps,
                fov_height=self.reference_movie.shape[1],
                fov_width=self.reference_movie.shape[2]
            )
        else:
            self._block_centers = None

    @property
    def block_centers(self) -> None | np.ndarray:
        """centers of the blocks when using ``PiecewiseRigidMotionCorrector``, ``None`` otherwise"""
        return self._block_centers

    @block_centers.setter
    def block_centers(self, centers: np.ndarray | None):
        self._block_centers = centers

    def _compute_at_indices(self, indices: list | int | slice) -> np.ndarray:
        """
        Lazy computation logic goes here to return frames. Slices the array over time (dimension 0) at the desired indices.

        Args:
            indices: Union[list, int, slice] the user's desired way of picking frames, either an int, list of ints, or slice
                i.e. slice object or int passed from `__getitem__()`

        Returns:
            np.ndarray: array at the indexed slice
        """
        return self._index_frames_tensor(indices)[0]

    def __getitem__(self, idx):
        if isinstance(self.strategy, masknmf.DummyMotionCorrector):
            return self.reference_movie.__getitem__(idx)
        else:
            return super().__getitem__(idx)

    def _index_frames_tensor(
        self,
        idx: int | list | np.ndarray | tuple[int | np.ndarray | slice | range],
    ) -> tuple[np.ndarray, np.ndarray]:
        """(corrected_frames, shifts) at index `idx`."""

        reference_data_frames = self.reference_movie[idx]
        target_data_frames = None if self.target_movie is None else self.target_movie[idx]

        #Ensure that we pass in (num_frames, height, width) data to the correction code
        if reference_data_frames.ndim == 2:
            reference_data_frames = reference_data_frames[None, ...]
        if target_data_frames is not None:
            if target_data_frames.ndim == 2:
                target_data_frames = target_data_frames[None, ...]

        return self.strategy.correct(
            reference_movie_frames=reference_data_frames,
            target_movie_frames=target_data_frames,
        )

    def export(self, path: str | Path):
        data_output_shape = self.shape
        if isinstance(self.strategy, masknmf.DummyMotionCorrector):
            shifts_output_shape = None
        else:
            shifts_output_shape = self.shifts.shape



        if os.path.isfile(path):
            raise FileExistsError

        with h5py.File(path, 'w') as f:
            num_frames = self.shape[0]
            moco_dset = f.create_dataset(self._motion_export_name,
                                         data_output_shape,
                                         dtype=self.dtype)
            if shifts_output_shape is not None:
                shifts_dset = f.create_dataset(self._shifts_export_name, shifts_output_shape)
            else:
                shifts_dset = None

            if isinstance(self.strategy, masknmf.PiecewiseRigidMotionCorrector):
                block_centers = self.block_centers
                block_centers_dst = f.create_dataset(self._block_centers_name, block_centers.shape)
                block_centers_dst[:] = block_centers

            batch_size = self.strategy.batch_size
            for k in tqdm(range(math.ceil(num_frames / batch_size))):
                start = k * batch_size
                end = min(start + batch_size, num_frames)
                moco_subset, shifts_subset = self._index_frames_tensor(slice(start, end))
                moco_dset[start:end, :, :] = moco_subset.astype(self.dtype)
                if shifts_dset is not None:
                    shifts_dset[start:end, ...] = shifts_subset

    @classmethod
    def from_hdf5(cls, path, **kwargs):
        """Load result from a hdf5 file. Any additional kwargs are passed to the constructor"""
        registered_array = h5py.File(path, "r")[cls._motion_export_name]
        with h5py.File(path, "r") as f:
            if cls._shifts_export_name in f:
                shifts = f[cls._shifts_export_name][()]
            else:
                shifts = None
            if cls._block_centers_name in f:
                block_centers = f[cls._block_centers_name][()]
            else:
                block_centers = None

        class_from_disk = cls(reference_movie=registered_array,
                   shifts=shifts)
        class_from_disk.block_centers = block_centers
        return class_from_disk

class FilteredArray(LazyFrameLoader):
    def __init__(
        self,
        raw_data_loader: LazyFrameLoader,
        filter_function: Callable,
        batching: int = 100,
        device: str = "cpu",
    ):
        """
        Class for loading and filtering data; this is broadly useful because we often want to spatially filter
        data to expose salient signals. We use this filtered version of the data to estimate shifts
        Args:
                raw_data_loader (LazyFrameLoader): An object that supports the lazy_data_loader interface.
                    This can be for e.g. a custom object that reads data from disk, an array in RAM (like a numpy ndarray)
                    or anything else

                filter_function (Callable): A function that applies a spatial filter to every frame of a data array. It takes
                    an input movie of type torch.Tensor with shape (frames, fov dim 1, fov dim 2) and returns a
                    filtered movie of type torch.Tensor with the same shape.

                batching (int): Max number of frames we process on GPU at a time, used to avoid OOM errors.

                device (str): The device on which computations are performed ('cpu' or 'cuda')
        """

        self._raw_data_loader = raw_data_loader
        self._filter = filter_function
        self._batching = batching
        self._device = device

    @property
    def raw_data_loader(self) -> LazyFrameLoader:
        return self._raw_data_loader

    @property
    def device(self) -> str:
        return self._device

    @device.setter
    def device(self, new_device: str):
        self._device = new_device

    @property
    def filter_function(self) -> Callable:
        return self._filter

    @property
    def batching(self):
        return self._batching

    @batching.setter
    def batching(self, new_batch: int):
        self._batching = new_batch

    @property
    def dtype(self) -> str:
        """
        data type
        """
        return np.float32

    @property
    def shape(self) -> tuple[int, int, int]:
        """
        Array shape (n_frames, dims_x, dims_y)
        """
        return self.raw_data_loader.shape

    @property
    def ndim(self) -> int:
        """
        Number of dimensions
        """
        return len(self.shape)

    def _compute_at_indices(self, indices: list | int | slice) -> np.ndarray:
        """
        Lazy computation logic goes here to return frames. Slices the array over time (dimension 0) at the desired indices.

        Parameters
        ----------
        indices: Union[list, int, slice]
            the user's desired way of picking frames, either an int, list of ints, or slice
             i.e. slice object or int passed from `__getitem__()`

        Returns
        -------
        np.ndarray
            array at the indexed slice
        """
        frames = torch.from_numpy(self.raw_data_loader[indices]).float()
        if frames.ndim == 2:
            frames = frames[None, :, :]
        if frames.shape[0] <= self.batching:
            frames = frames.to(self.device)
            return self.filter_function(frames).cpu().numpy()
        else:
            batches = list(range(0, frames.shape[0], self.batching))
            output = []
            for k in range(len(batches)):
                start = batches[k]
                end = min(frames.shape[0], start + self.batching)
                curr_frames = frames[start:end].to(self.device)
                if curr_frames.ndim == 2:
                    curr_frames = curr_frames[None, :, :]
                output.append(self.filter_function(curr_frames).cpu())

            return torch.concatenate(output, dim=0).numpy()


class VoltageArray(masknmf.ArrayLike):

    def __init__(self,
                 dataset: ArrayLike,
                 negative_indicator: bool = True,
                 include_mean: bool = True,
                 device='cuda',
                 batch_size: int = 200):
        """
        Array-like object for viewing inverted, mean subtracted, and/or raw voltage data
        Args:
            dataset (masknmf.ArrayLike): Shape (num_frames, height, width)
            negative_indicator (bool): True if indicator is negatively tuned, else False
            include_mean (bool): If True, includes the mean into the "getitem" call. If false, getitem shows the "mean subtracted" movie
            device (str): Which device to perform computations/return the tensor from getitem
        """
        self._dataset = dataset
        self._device = device
        self._batch_size = batch_size
        self._negative_indicator = negative_indicator
        self._compute_mean()
        self._include_mean = include_mean

    @property
    def batch_size(self) -> int:
        return self._batch_size

    @batch_size.setter
    def batch_size(self, new_size: int):
        self._batch_size = new_size

    def _compute_mean(self):
        num_frames = self.shape[0]
        cumulated_mean = torch.zeros(self.shape[1], self.shape[2], dtype=self.dtype, device=self.device)
        num_batches = math.ceil(self.shape[0] / self.batch_size)
        for ind in range(num_batches):
            start_pt = ind * self.batch_size
            end_pt = min(self.shape[0], start_pt + self.batch_size)
            data_subset = self._dataset[start_pt:end_pt]
            if isinstance(data_subset, np.ndarray):
                subset = torch.from_numpy(data_subset).to(self.device).to(self.dtype)
            elif isinstance(data_subset, torch.Tensor):
                subset = data_subset.to(self.device).to(self.dtype)
            else:
                raise ValueError("Calling getitem on dataset should return either a torch tensor or np.ndarray")
            cumulated_mean += torch.sum(subset, dim=0) / num_frames
        self._mean_image = cumulated_mean

    @property
    def include_mean(self) -> bool:
        return self._include_mean

    @include_mean.setter
    def include_mean(self, new_flag: bool):
        self._include_mean = new_flag

    @property
    def negative_indicator(self) -> bool:
        return self._negative_indicator

    @property
    def device(self) -> str:
        return self._device

    @property
    def mean_image(self) -> torch.Tensor:
        return self._mean_image

    @property
    def dtype(self) -> torch.dtype:
        """
        data type
        """
        return torch.float32

    @property
    def shape(self) -> Tuple[int, int, int]:
        return self._dataset.shape

    @property
    def nbytes(self) -> int:
        return math.prod(self.shape) * self.dtype.itemsize

    def __getitem__(self,
                    item: Union[int, list, np.ndarray, Tuple[Union[int, np.ndarray, slice, range]]]) -> torch.Tensor:
        return self._get(item, include_mean=self.include_mean)

    def _get(self,
             item: Union[int, list, np.ndarray, Tuple[Union[int, np.ndarray, slice, range]]],
             include_mean: bool | None = None) -> torch.Tensor:
        ## Private helper method designed to avoid race conditions associated with state variables like include_mean
        if include_mean is None:
            include_mean = self.include_mean

        frame_indexer, item = self._parse_indices(item)
        data_subset = torch.as_tensor(self._dataset[item]).to(self.device).to(self.dtype) ## Parse indices first to ensure consistent output

        # Check if spatial cropping occurred, deal with mean image accordingly
        if isinstance(item, tuple):
            mean_crop = self._mean_image[item[1:]]
        else:
            mean_crop = self._mean_image

        if data_subset.ndim > mean_crop.ndim: #This means that there is a temporal dimension, which means we need to broadcast
            mean_crop = mean_crop[None, ...]

        if self.negative_indicator:
            data_subset *= -1
            if include_mean:
                data_subset += 2 * mean_crop
            else:
                data_subset += mean_crop
        else:
            if include_mean:
                pass
            else:
                data_subset -= mean_crop
        return data_subset




class GradientRegistrationArray(ArrayLike):
    """
    This is a special registration array which provides fast slicing in all dimensions (pixels AND frames)
    The key idea is that the per-frame correction (which uses the gradient of the template image) is low-rank,
    so we can adaptively compute frames/pixels of this movie very quickly if we just store the low-rank info

    This array is not meant to be serialized
    """
    def __init__(self,
                 raw_data: VoltageArray,
                 strategy: GradientMotionCorrector,
                 output_device: str | None = None):

        self._dataset = raw_data
        self._strategy = strategy
        self._output_device = self.strategy.device if output_device is None else output_device
        self._gradient_steps = None
        self._gradient_steps_mean = None
        self._gradient = None
        self._include_mean = True
        self._extract_lowrank_shifts()

    @property
    def shape(self) -> Tuple[int, int, int]:
        return self.raw_data.shape

    @property
    def dtype(self) -> torch.dtype:
        return self.strategy.dtype

    @property
    def output_device(self) -> str:
        return self._output_device

    @output_device.setter
    def output_device(self, new_device:str):
        self._output_device = new_device

    @property
    def nbytes(self) -> int:
        return math.prod(self.shape) * self.dtype.itemsize

    @property
    def raw_data(self) -> VoltageArray:
        return self._dataset

    @property
    def include_mean(self) -> bool:
        return self._include_mean

    @include_mean.setter
    def include_mean(self, new_flag: bool):
        self._include_mean = new_flag

    @property
    def strategy(self) -> GradientMotionCorrector:
        return self._strategy

    @property
    def gradient_steps(self) -> torch.Tensor | None:
        return self._gradient_steps


    @property
    def gradient(self) -> torch.Tensor | None:
        return self._gradient
    def _extract_lowrank_shifts(self):
        batch_size = self.strategy.batch_size
        num_iters = math.ceil(self.shape[0] / batch_size)

        gradient_steps = []
        for k in tqdm(range(num_iters)):
            start = k * batch_size
            end = min(start + batch_size, self.shape[0])
            subset = torch.as_tensor(self.raw_data._get(slice(start,end), include_mean=True), device=self.strategy.device, dtype=self.dtype)
            frames, shifts, scale = self.strategy.correction_routine(subset)
            gradient_step = self.strategy.compute_gradient_step(shifts, scale)
            gradient_steps.append(gradient_step)

        self._gradient_steps = torch.concatenate(gradient_steps, dim=0) #Shape (frames, 2)
        self._gradient_steps_mean = torch.mean(self._gradient_steps, dim = 0)
        self._gradient = self.strategy.gradient.reshape(self.shape[1], self.shape[2], 2)

    def __getitem__(self,
                    item: Union[int, list, np.ndarray, Tuple[Union[int, np.ndarray, slice, range]]]) -> torch.Tensor:

        data_subset = torch.as_tensor(self.raw_data._get(item, include_mean=self.include_mean), device=self.output_device, dtype=self.dtype)
        frame_indexer, item = self._parse_indices(item)

        gradient_step_used = self.gradient_steps[frame_indexer, :]

        if not self.include_mean:
            if gradient_step_used.ndim > self._gradient_steps_mean.ndim:
                grad_sub = self._gradient_steps_mean[None, :]
            else:
                grad_sub = self._gradient_steps_mean
            gradient_step_used -= grad_sub

        # Check if spatial cropping occurred, deal with factorized tensors appropriately
        if isinstance(item, tuple):
            gradient_crop = self.gradient[item[1:]]
        else:
            gradient_crop = self.gradient

        grad_movie = gradient_crop @ gradient_step_used.T
        grad_movie = grad_movie.permute(2, 0, 1).to(self.output_device)

        return data_subset - grad_movie

