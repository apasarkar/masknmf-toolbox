import math
from collections.abc import Callable
import numpy as np

import torch
import masknmf
from masknmf.arrays.array_interfaces import LazyFrameLoader, ArrayLike
from masknmf.utils import Serializer
from pathlib import Path
from abc import ABC, abstractmethod
from masknmf.utils._serialization import save_dict, load_dict


class BaseRegistrationArray(ArrayLike, Serializer, ABC):
    _strategy_cls: type = None

    @property
    @abstractmethod
    def shifts(self): ...

    @property
    @abstractmethod
    def dtype(self) -> torch.dtype: ...

    @property
    @abstractmethod
    def input_movie(self): ...

    @property
    @abstractmethod
    def strategy(self) -> Serializer: ...

    @property
    def output_device(self) -> torch.device:
        return self._output_device

    @output_device.setter
    def output_device(self, value: str | torch.device):
        self._output_device = torch.device(value)

    @abstractmethod
    def __getitem__(self, idx) -> torch.Tensor:
        """
        Index into the array.

        Always returns a ``torch.Tensor`` on ``self.output_device`` with dtype
        ``self.dtype``. Never returns a numpy array, for any index pattern,
        including empty selections.
        """

    def export(self, path: str | Path, prefix: str = ""):
        d_array = self._to_dict()
        d_strategy = self.strategy._to_dict()
        prefix = f"{prefix}/" if prefix else ""
        save_dict(d_array, filename=path, exists_ok=True, group=prefix + self.__class__.__name__)
        save_dict(d_strategy, filename=path, exists_ok=True, group=prefix + self._strategy_cls.__name__)

    @classmethod
    def from_hdf5(cls,
                  path,
                  input_movie: ArrayLike,
                  prefix: str = "",
                  **kwargs):
        if cls._strategy_cls is None:
            raise NotImplementedError(
                f"{cls.__name__} must set `_strategy_cls` to enable from_hdf5"
            )
        prefix = f"{prefix}/" if prefix else ""
        strat = cls._strategy_cls(**load_dict(path, prefix + cls._strategy_cls.__name__))
        reg_arr_dict = load_dict(path, prefix + cls.__name__)
        return cls(input_movie=input_movie, strategy=strat, **reg_arr_dict, **kwargs)





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
        indices: list | int | slice
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





