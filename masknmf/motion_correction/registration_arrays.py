from masknmf.arrays.array_interfaces import LazyFrameLoader
import torch
from typing import *
from .strategies import MotionCorrectionStrategy
import math
import numpy as np


class RegistrationArray(LazyFrameLoader):
    def __init__(
        self,
        reference_dataset: LazyFrameLoader,
        strategy: MotionCorrectionStrategy,
        device: str = "cpu",
        target_dataset: Optional[LazyFrameLoader] = None,
    ):
        """
        Array-like motion correction representation that support on-the-fly motion correction

        Args:
            reference_dataset (LazyFrameLoder): Image stack that we use to compute motion correction transform relative to template
            strategy (masknmf.MotionCorrectionStrategy): The method used to register each frame to the template
            device (torch.tensor): The device on which computations are performed (for e.g. 'cuda' or 'cpu')
            batch_size (int): The number of frames we load onto the computation device at a time to do motion correction.
            target_dataset (Optional[LazyFrameLoader]): Once we learn the motion correction transform by aligning reference_dataset
                with template, we actually apply the transform to target_dataset, if it is specified. If None, we apply the
                transform to reference_dataset
        """
        self._reference_dataset = reference_dataset
        self._strategy = strategy
        self._template = strategy.template
        self._device = device
        self._target_dataset = target_dataset

        self._shape = self.reference_dataset.shape
        self._ndim = self.reference_dataset.ndim
        self._shifts = self._Shifts(self)


    @property
    def ndim(self) -> int:
        return self._ndim

    @property
    def shape(self) -> Tuple[int, int, int]:
        return self._shape

    @property
    def dtype(self) -> str:
        return "float32"

    @property
    def reference_dataset(self) -> LazyFrameLoader:
        return self._reference_dataset

    @property
    def target_dataset(self) -> Optional[LazyFrameLoader]:
        return self._target_dataset

    @property
    def shifts(self) -> "_Shifts":
        return self._shifts

    @property
    def strategy(self) -> MotionCorrectionStrategy:
        return self._strategy

    @property
    def template(self) -> torch.tensor:
        return self._template

    @property
    def device(self) -> str:
        return self._device

    @device.setter
    def device(self, new_device: str):
        self._device = new_device

    def _compute_at_indices(self, indices: Union[list, int, slice]) -> np.ndarray:
        """
        Lazy computation logic goes here to return frames. Slices the array over time (dimension 0) at the desired indices.

        Args:
            indices: Union[list, int, slice] the user's desired way of picking frames, either an int, list of ints, or slice
                i.e. slice object or int passed from `__getitem__()`

        Returns:
            np.ndarray: array at the indexed slice
        """
        return self._index_frames_tensor(indices)[0]

    def _index_frames_tensor(
        self,
        idx: Union[int, list, np.ndarray, Tuple[Union[int, np.ndarray, slice, range]]],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Retrieve motion-corrected frame at index `idx`."""
        reference_data_frames = self._reference_dataset[idx]
        target_data_frames = None if self.target_dataset is None else self.target_dataset[idx]

        return self.strategy.correct(reference_frames=reference_data_frames,
                                     target_frames=target_data_frames,
                                     device=self.device)

    class _Shifts:
        def __init__(self, reg_arr):
            self.reg = reg_arr

        def __getitem__(self, ind):
            return self.reg._index_frames_tensor(ind)[1]


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
        return self.raw_data_loader.dtype

    @property
    def shape(self) -> Tuple[int, int, int]:
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

    def _compute_at_indices(self, indices: Union[list, int, slice]) -> np.ndarray:
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

