from typing import Optional, Callable

import torch
import numpy as np

from masknmf.arrays.array_interfaces import LazyFrameLoader, ArrayLike
from .strategies import MotionCorrectionStrategy, RigidMotionCorrector, PiecewiseRigidMotionCorrector
from .registration_methods import compute_pwrigid_patch_midpoints


class Shifts(ArrayLike):
    def __init__(self, reg_arr):
        self._reg = reg_arr

    @property
    def dtype(self) -> str:
        return self._reg.dtype

    @property
    def shape(self) -> tuple[int, int, int]:
        return self._reg.shape

    @property
    def ndim(self) -> int:
        return self._reg.ndim

    def __getitem__(self, ind):
        return self._reg._index_frames_tensor(ind)[1].squeeze()


class RegistrationArray(LazyFrameLoader):
    def __init__(
        self,
        reference_movie: LazyFrameLoader,
        strategy: MotionCorrectionStrategy | None = None,
        target_movie: Optional[LazyFrameLoader] = None,
    ):
        """
        Array-like motion correction representation that support on-the-fly motion correction

        Args:
            reference_movie (LazyFrameLoder): Image stack that we use to compute motion correction transform relative to template
            strategy (masknmf.MotionCorrectionStrategy): The method used to register each frame to the template.
                Can initialize as ``None``, but must be set before slicing frames
            target_movie (Optional[LazyFrameLoader]): Once we learn the motion correction transform by aligning reference_dataset
                with template, we actually apply the transform to target_dataset, if it is specified. If None, we apply the
                transform to reference_dataset
        """
        self._reference_movie = reference_movie

        if not isinstance(strategy, MotionCorrectionStrategy):
            raise TypeError(
                f"`strategy` must be a `MotionCorrectionStrategy` type. "
                f"Usually `RigidMotionCorrector` or `PiecewiseRigidMotionCorrector`"
            )

        self.strategy = strategy

        self._target_movie = target_movie
        self._shape = self.reference_movie.shape
        self._ndim = self.reference_movie.ndim
        self._shifts = Shifts(self)

    @property
    def ndim(self) -> int:
        return self._ndim

    @property
    def shape(self) -> tuple[int, int, int]:
        return self._shape

    @property
    def dtype(self) -> str:
        return "float32"

    @property
    def reference_movie(self) -> LazyFrameLoader:
        return self._reference_movie

    @property
    def target_movie(self) -> LazyFrameLoader | None:
        return self._target_movie

    @property
    def shifts(self) -> Shifts:
        return self._shifts

    @property
    def strategy(self) -> PiecewiseRigidMotionCorrector | RigidMotionCorrector | None:
        return self._strategy

    @strategy.setter
    def strategy(self, corrector: PiecewiseRigidMotionCorrector | RigidMotionCorrector | None):
        self._strategy = corrector

        if isinstance(self.strategy, PiecewiseRigidMotionCorrector):
            self._block_centers = compute_pwrigid_patch_midpoints(
                num_blocks=self.strategy.num_blocks,
                overlaps=self.strategy.overlaps,
                fov_height=self.strategy.template.shape[0],
                fov_width=self.strategy.template.shape[1]
            )
        else:
            self._block_centers = None

    @property
    def block_centers(self) -> np.ndarray | None:
        """centers of the blocks when using ``PiecewiseRigidMotionCorrector``, ``None`` otherwise"""
        return self._block_centers

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

    def _index_frames_tensor(
        self,
        idx: int | list | np.ndarray | tuple[int | np.ndarray | slice | range],
    ) -> tuple[np.ndarray, np.ndarray]:
        """(corrected_frames, shifts) at index `idx`."""

        reference_data_frames = self.reference_movie[idx]
        target_data_frames = None if self.target_movie is None else self.target_movie[idx]

        return self.strategy.correct(
            reference_movie_frames=reference_data_frames,
            target_movie_frames=target_data_frames,
        )


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
