from typing import *
from abc import ABC, abstractmethod
import numpy as np


class FactorizedVideo(ABC):
    """
    This captures the numpy array-like functionality for factorized videos in our NMF model.
    """

    @property
    @abstractmethod
    def dtype(self) -> str:
        """
        data type
        """
        pass

    @property
    @abstractmethod
    def shape(self) -> Tuple[int, int, int]:
        """
        Array shape (n_frames, dims_x, dims_y)
        """
        pass

    @property
    def ndim(self) -> int:
        """
        Number of dimensions
        """
        return len(self.shape)

    @abstractmethod
    def __getitem__(
        self,
        item: Union[int, list, np.ndarray, Tuple[Union[int, np.ndarray, slice, range]]],
    ):
        # Step 1: index the frames (dimension 0)
        pass


class LazyFrameLoader(ABC):
    """
    This captures the numpy array-like functionality that all data loaders for motion correction need to contain

    Key: To implement support for a new file type, you just need to specify the key properties below (dtype, shape, ndim)
    and then implement the function _compute_at_indices.
    Adapted from mesmerize core: https://github.com/nel-lab/mesmerize-core/blob/master/mesmerize_core/arrays/_base.py
    """

    @property
    @abstractmethod
    def dtype(self) -> str:
        """
        data type
        """
        pass

    @property
    @abstractmethod
    def shape(self) -> Tuple[int, int, int]:
        """
        Array shape (n_frames, dims_x, dims_y)
        """
        pass

    @property
    def ndim(self) -> int:
        """
        Number of dimensions
        """
        return len(self.shape)

    def __getitem__(
        self,
        item: Union[int, list, np.ndarray, Tuple[Union[int, np.ndarray, slice, range]]],
    ):
        # Step 1: index the frames (dimension 0)

        if isinstance(item, tuple):
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
            frame_indexer = frame_indexer.tolist()

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

        #Only squeeze at axis = 0 (time dimension) in case one of the spatial dimensions is actually 1
        if frames.shape[0] == 1:
            return frames.squeeze(axis=0)
        else:
            return frames

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
