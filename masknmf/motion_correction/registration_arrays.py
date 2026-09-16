import math
from typing import Optional, Callable, Union, Tuple
import numpy as np
from numpy.typing import DTypeLike

import torch
import masknmf
from masknmf.arrays.array_interfaces import LazyFrameLoader, ArrayLike
from .strategies import MotionCorrectionStrategy, DummyMotionCorrector
from .registration_methods import compute_pwrigid_patch_midpoints
from masknmf.utils import Serializer
from pathlib import Path
import h5py
import os
from tqdm import tqdm
from abc import ABC, abstractmethod
from masknmf.utils._serialization import save_dict, load_dict


class BaseRegistrationArray(ArrayLike, Serializer, ABC):
    _strategy_cls: type = None

    @property
    @abstractmethod
    def shifts(self): ...

    @property
    @abstractmethod
    def input_movie(self): ...

    @property
    @abstractmethod
    def strategy(self) -> Serializer: ...

    @property
    @abstractmethod
    def output_device(self): ...

    def export(self, path: str | Path):
        d_array = self._to_dict()
        d_strategy = self.strategy._to_dict()
        save_dict(d_array, filename=path, group=self.__class__.__name__)
        save_dict(d_strategy, filename=path, exists_ok=True, group=self._strategy_cls.__name__)

    @classmethod
    def from_hdf5(cls,
                  path,
                  input_movie: ArrayLike,
                  **kwargs):
        if cls._strategy_cls is None:
            raise NotImplementedError(
                f"{cls.__name__} must set `_strategy_cls` to enable from_hdf5"
            )
        strat = cls._strategy_cls(**load_dict(path, cls._strategy_cls.__name__))
        reg_arr_dict = load_dict(path, cls.__name__)
        return cls(input_movie=input_movie, strategy=strat, **reg_arr_dict)





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


class OphysArray(ArrayLike):

    def __init__(self,
                 dataset: ArrayLike,
                 negative_indicator: bool = True,
                 include_mean: bool = True,
                 device='cuda',
                 batch_size: int = 200):
        """
        Array-like object for viewing inverted, mean subtracted, and/or raw optical physiology data
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





