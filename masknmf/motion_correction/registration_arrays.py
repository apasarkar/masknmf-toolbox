import math
from collections.abc import Callable
import h5py
import numpy as np

import torch
import masknmf
from masknmf import MotionCorrectionStrategy
from masknmf.arrays.array_interfaces import LazyFrameLoader, ArrayLike
from masknmf.utils import Serializer
from pathlib import Path
from abc import ABC, abstractmethod
from masknmf.utils._serialization import save_dict, load_dict
from masknmf.utils import torch_select_device


class BaseRegistrationArray(ArrayLike, Serializer, ABC):
    _strategy_cls: type = None
    _default_batch_size: int = 200

    def __init__(self,
                 strategy: Serializer | None = None,
                 batch_size: int | None = None,
                 compute_device: str | torch.device | None = None,
                 output_device: str | torch.device = "cpu"):
        self._strategy = strategy
        self._batch_size = self._resolve("batch_size", batch_size, self._default_batch_size)
        self._compute_device = torch_select_device(self._resolve("device", compute_device, "auto"))
        self._output_device = torch_select_device(output_device)


    def _resolve(self, strategy_attr: str, explicit, default):
        if explicit is not None:
            return explicit
        if self._strategy is not None and getattr(self._strategy, strategy_attr, None) is not None:
            return getattr(self._strategy, strategy_attr)
        return default

    @property
    def batch_size(self) -> int:
        return self._batch_size

    @batch_size.setter
    def batch_size(self, new_batch_size: int):
        self._batch_size = new_batch_size

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
    def shape(self) -> tuple[int, int, int]:
        return self.input_movie.shape

    @property
    def dtype(self) -> torch.dtype:
        return torch.float32

    @property
    def strategy(self):
        return self._strategy

    @property
    def output_device(self) -> torch.device:
        return self._output_device

    @output_device.setter
    def output_device(self, value: str | torch.device):
        self._output_device = torch.device(value)

    @property
    def compute_device(self) -> torch.device:
        return self._compute_device

    @compute_device.setter
    def compute_device(self, new_compute_device: str | torch.device):
        self._compute_device = torch.device(new_compute_device)

    @abstractmethod
    def __getitem__(self, idx) -> torch.Tensor:
        """
        Index into the array.

        Always returns a ``torch.Tensor`` on ``self.output_device`` with dtype
        ``self.dtype``. Never returns a numpy array, for any index pattern,
        including empty selections.
        """

    def export(self, path: str | Path):
        """
        Write the registration to path, replacing any registration the file already holds, of any kind. Raises when
        the file holds a compression or demixing, which were computed from the registration it holds.
        """
        if Path(path).is_file():
            with h5py.File(path, "a") as f:
                later = [name for name in ("CompressionArray", "DemixingResults") if name in f]
                later += [f"{name}/DemixingResults" for name in f
                          if isinstance(f[name], h5py.Group) and "DemixingResults" in f[name]]
                if len(later) > 0:
                    raise ValueError(f"{path} holds {', '.join(later)}, computed from its registration; "
                                     f"write the new registration to another file")
                for cls in BaseRegistrationArray.__subclasses__():
                    for name in (cls.__name__, cls._strategy_cls.__name__):
                        if name in f:
                            del f[name]
        save_dict(self._to_dict(), filename=path, exists_ok=True, group=self.__class__.__name__)
        if self.strategy is not None:
            save_dict(self.strategy._to_dict(), filename=path, exists_ok=True, group=self._strategy_cls.__name__)

    @classmethod
    def from_hdf5(cls,
                  path,
                  input_movie: ArrayLike,
                  device: str | torch.device | None = None,
                  **kwargs):
        """Load a stored registration and apply it to input_movie; device is where the shifts are applied."""
        if cls._strategy_cls is None:
            raise NotImplementedError(
                f"{cls.__name__} must set `_strategy_cls` to enable from_hdf5"
            )
        with h5py.File(path, "r") as f:
            has_strategy = cls._strategy_cls.__name__ in f
        if has_strategy:
            kwargs_strategy = load_dict(path, cls._strategy_cls.__name__)
            if device is not None:
                kwargs_strategy["device"] = device

            strat = cls._strategy_cls(**kwargs_strategy)
        else:
            strat = None
            if device is not None:
                kwargs["device"] = device
        reg_arr_dict = load_dict(path, cls.__name__)
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
        frames = torch.as_tensor(self.raw_data_loader[indices]).float()
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





