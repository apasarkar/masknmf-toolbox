from copy import copy as copy_func
from typing import *
from abc import ABC, abstractmethod
import numpy as np

import torch
def is_arraylike(obj):
    """Returns if the object is sufficiently array-like for lazy compute or loading"""
    for attr in ["dtype", "shape", "ndim", "__getitem__"]:
        if not hasattr(obj, attr):
            raise TypeError(
                f"The object you have passed is not sufficiently array like, "
                f"it lacks the following property or method: {attr}."
            )


class ArrayLike(ABC):
    """
    The most general class capturing the minimum functionality a general array needs to support
    """
    def __array__(self, dtype=None, copy=None):
        # required for minimal xarray compatability
        if copy:
            return copy_func(self)

        return self

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        # required for minimal xarray compatability, doesn't actually have to do anything
        raise NotImplementedError

    def __array_function__(self, func, types, *args, **kwargs):
        # required for minimal xarray compatability, doesn't actually have to do anything
        raise NotImplementedError

    @property
    @abstractmethod
    def dtype(self) -> Union[str, np.dtype]:
        """
        data type
        """
        pass

    @property
    @abstractmethod
    def shape(self) -> Tuple[int, int, int]:
        """
        Array shape (n_frames, height, width)
        """
        pass


    @property
    def ndim(self) -> int:
        """
        Number of dimensions
        """
        return len(self.shape)

    @property
    def nbytes(self) -> int:
        """
        int
            number of bytes for the array if it were fully computed
        """
        return np.prod(self.shape + (np.dtype(self.dtype).itemsize,), dtype=np.int64)

    def _parse_indices(self, item: list | int | np.ndarray | tuple[int, np.ndarray | slice | range]):
        # Step 1: index the frames (dimension 0)

        if isinstance(item, tuple):
            # if the last item is Ellipsis, remove it. This probably came from xarray's indexer
            if item[-1] is Ellipsis:
                item = item[:-1]

            if len(item) > len(self.shape):
                raise IndexError(
                    f"Cannot index more dimensions than exist in the array. "
                    f"You have tried to index with <{len(item)}> dimensions, "
                    f"only <{len(self.shape)}> dimensions exist in the array"
                )
            frame_indexer = item[0]
        else:
            frame_indexer = item

        # Step 2: Do some basic error handling for frame_indexer before using it to slice

        if isinstance(frame_indexer, np.ndarray):
            pass

        if isinstance(frame_indexer, list):
            pass

        elif isinstance(frame_indexer, int):
            pass

        # numpy int scaler
        elif isinstance(frame_indexer, np.integer):
            frame_indexer = frame_indexer.item()

        # treat slice and range the same
        elif isinstance(frame_indexer, (slice, range)):
            start = frame_indexer.start
            stop = frame_indexer.stop
            step = frame_indexer.step

            if start is not None:
                if start > self.shape[0]:
                    raise IndexError(
                        f"Cannot index beyond `n_frames`.\n"
                        f"Desired frame start index of <{start}> "
                        f"lies beyond `n_frames` <{self.shape[0]}>"
                    )
            if stop is not None:
                if stop > self.shape[0]:
                    raise IndexError(
                        f"Cannot index beyond `n_frames`.\n"
                        f"Desired frame stop index of <{stop}> "
                        f"lies beyond `n_frames` <{self.shape[0]}>"
                    )

            if step is None:
                step = 1

            # convert indexer to slice if it was a range, allows things like decord.VideoReader slicing
            frame_indexer = slice(start, stop, step)  # in case it was a range object

        else:
            raise IndexError(
                f"Invalid indexing method, " f"you have passed a: <{type(item)}>"
            )

        return frame_indexer, item

    @abstractmethod
    def __getitem__(
        self,
        item: Union[int, list, np.ndarray, Tuple[Union[int, np.ndarray, slice, range]]],
    ):
        # Step 1: index the frames (dimension 0)
        pass

class TensorFlyWeight:
    """
    Generic class for managing a collection of tensors across multiple objects
    """
    def __init__(self, **kwargs):
        device = None
        for name, value in kwargs.items():
            if isinstance(value, torch.Tensor):
                if device is None:
                    device = value.device
                setattr(self, name, value.to(device))
            ##Useful to track None types
            if value is None:
                setattr(self, name, None)

            else:
                raise ValueError(f"field {name} is not a torch Tensor object")

    @property
    def device(self) -> str | None:
        device = None
        for name in vars(self):
            data = getattr(self, name)
            if data is None:
                continue
            if device is None:
                device = data.device
            if data.device != device:
                raise ValueError("Not all attributes on same device")
        return device

    def list_tensor_attributes(self) -> dict[str, torch.Tensor | None]:
        return vars(self)

    def validate_attributes(self, attr_list):
        for name in attr_list:
            if not hasattr(self, name):
                raise ValueError(f"Required attribute: {name} missing from constructor")

    def to(self, device: str):
        for name in vars(self):
            curr_tensor = getattr(self, name)
            if curr_tensor is not None:
                new_values = curr_tensor.to(device)
                setattr(self, name, new_values)


class LazyFrameLoader(ArrayLike):
    """
    An array-like object that only supports fast slicing in the temporal domain. Used for motion correction algorithms.

    Key: To implement support for a new file type, you just need to specify the key properties below (dtype, shape, ndim)
    and then implement the function _compute_at_indices.
    Adapted from mesmerize core: https://github.com/nel-lab/mesmerize-core/blob/master/mesmerize_core/arrays/_base.py
    """

    def __getitem__(
        self,
        item: Union[int, list, np.ndarray, slice, range, Tuple[Union[int, np.ndarray, slice, range]]],
    ):
        frame_indexer, item = self._parse_indices(item)
        # Step 3: Now slice the data with frame_indexer (careful: if the ndims has shrunk, add a dim)
        frames = self._compute_at_indices(frame_indexer)
        if len(frames.shape) < len(self.shape):
            frames = np.expand_dims(frames, axis=0)

        # Step 4: Deal with remaining indices after lazy computing the frame(s)
        if isinstance(item, tuple):
            if len(item) == 2:
                frames = frames[:, item[1]]
            elif len(item) == 3:
                frames = frames[:, item[1], item[2]]

        # Only squeeze at axis = 0 (time dimension) in case one of the spatial dimensions is actually 1
        if frames.shape[0] == 1:
            return frames.squeeze(axis=0).astype(self.dtype)
        else:
            return frames.astype(self.dtype)

    @abstractmethod
    def _compute_at_indices(self, indices: Union[list, int, slice]) -> np.ndarray:
        """
        Lazy computation logic goes here to return frames. Slices the array over time (dimension 0) at the desired indices.

        Args:
            indices: Union[list, int, slice] the user's desired way of picking frames, either an int, list of ints, or slice
                i.e. slice object or int passed from `__getitem__()`

        Returns:
            np.ndarray: array at the indexed slice
        """
        pass
