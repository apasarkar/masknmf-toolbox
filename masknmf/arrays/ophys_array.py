import math
import threading

import numpy as np
import torch
from tqdm import tqdm

from masknmf.arrays.array_interfaces import ArrayLike

class OphysArray(ArrayLike):
    """
    Lazily evaluated view of an optical physiology movie in which the indicator always appears positively tuned.

    Whatever indicator was recorded, frames returned by getitem show activity as an *increase* in fluorescence:
        - Positively tuned indicator (negative_indicator=False): frames are returned as recorded.
        - Negatively tuned indicator (negative_indicator=True): each frame is reflected about the mean image,
          2 * mean_image - frame. The baseline (mean image) is unchanged, but every dip below it (activity) becomes
          a rise above it. Downstream code can therefore always assume a positive indicator.

    On top of that view:
        - include_mean=False returns the mean-subtracted movie (the positive view minus mean_image).
        - good_frames crops the movie in time. Only frames labeled 1 exist as far as this array is concerned:
          shape, all temporal indexing and the mean image refer to the cropped movie.

    The mean image costs a full pass over the data, so it is only computed when the view depends on it:
        - At construction, if negative_indicator=True or include_mean=False.
        - Otherwise on first use: accessing mean_image, or a getitem after include_mean is set to False.
    With a positive indicator and include_mean=True the mean is never computed, and getitem returns the data with
    no extra arithmetic.
    """

    def __init__(self,
                 dataset: ArrayLike,
                 negative_indicator: bool = True,
                 include_mean: bool = True,
                 device='cpu',
                 batch_size: int = 200,
                 good_frames: np.ndarray | torch.Tensor | list | None = None):
        """
        Args:
            dataset (masknmf.ArrayLike): Shape (num_frames, height, width)
            negative_indicator (bool): True if the recorded indicator is negatively tuned. The view is then inverted
                so it reads as positively tuned (see class docstring).
            include_mean (bool): If True, getitem returns frames with their baseline (the mean image). If False,
                getitem returns the mean-subtracted movie.
            device (str): Which device to perform computations/return the tensor from getitem
            batch_size (int): Number of frames loaded at a time when computing the mean image
            good_frames (np.ndarray | torch.Tensor | list | None): Optional 1D binary vector of length dataset.shape[0].
                Only frames labeled 1 are exposed: shape[0] becomes good_frames.sum(), and all temporal indexing
                and the mean image refer to this temporally cropped movie. None keeps every frame.
        """
        self._dataset = dataset
        self._device = device
        self._batch_size = batch_size
        self._negative_indicator = negative_indicator
        self._include_mean = include_mean
        self._set_good_frames(good_frames)
        self._mean_image = None
        self._mean_lock = threading.Lock()
        if negative_indicator or not include_mean:
            self._get_mean()  # every frame read will need it, so pay for it now rather than on the first frame

    def _set_good_frames(self, good_frames):
        num_raw_frames, height, width = (int(d) for d in self._dataset.shape)
        self._good_frames = None
        self._frame_map = None   # cropped frame index -> dataset frame index. None means identity (no remapping cost)
        self._run_offset = None  # frame_map - arange: constant within each contiguous run of good frames
        if good_frames is not None:
            if isinstance(good_frames, torch.Tensor):
                good_frames = good_frames.detach().cpu().numpy()
            mask = np.asarray(good_frames).astype(bool)
            if mask.shape != (num_raw_frames,):
                raise ValueError(f"good_frames must be a 1D vector of length {num_raw_frames}, got shape {mask.shape}")
            if not mask.any():
                raise ValueError("good_frames must label at least one frame with 1")
            self._good_frames = mask
            if not mask.all():  # an all-ones mask is the identity, so keep the zero-overhead path
                self._frame_map = np.flatnonzero(mask)
                self._run_offset = self._frame_map - np.arange(self._frame_map.size)
        num_frames = num_raw_frames if self._frame_map is None else int(self._frame_map.size)
        self._shape = (num_frames, height, width)

    def _to_raw(self, idx: int | slice | list | np.ndarray) -> int | slice | list | np.ndarray:
        """
        Map a temporal indexer on the cropped movie to the equivalent indexer on self._dataset.
        Expects the frame_indexer produced by ArrayLike._parse_indices (int, slice, list or np.ndarray).
        """
        if isinstance(idx, int):
            # numpy handles negative indices and raises IndexError when out of bounds
            return int(self._frame_map[idx])

        if isinstance(idx, slice):
            start, stop, step = idx.indices(self._shape[0])
            length = len(range(start, stop, step))
            if length == 0:
                return slice(0, 0)
            last = start + (length - 1) * step
            offset = int(self._run_offset[start])
            if offset == self._run_offset[last]:
                # Every requested frame lies in one contiguous run of good frames (always true for a single
                # crop window), so the request stays a slice and the dataset keeps its fast contiguous-read path. O(1).
                raw_stop = last + offset + (1 if step > 0 else -1)
                return slice(start + offset, raw_stop if raw_stop >= 0 else None, step)
            return self._frame_map[idx]  # spans a gap between good frames: integer array (a view, no copy)

        # list or np.ndarray: integer indices (incl. negative) or a boolean mask of length shape[0]; numpy handles both
        arr = np.asarray(idx)
        if arr.size == 0:
            return idx
        return self._frame_map[arr]

    @property
    def good_frames(self) -> np.ndarray | None:
        """Binary vector over the full dataset of the frames this array exposes (None if no cropping was requested)"""
        return self._good_frames

    @property
    def frame_indices(self) -> np.ndarray:
        """Index in the full dataset of each frame of this array"""
        return np.arange(self._shape[0]) if self._frame_map is None else self._frame_map

    @property
    def batch_size(self) -> int:
        return self._batch_size

    @batch_size.setter
    def batch_size(self, new_size: int):
        self._batch_size = new_size

    def _compute_mean(self) -> torch.Tensor:
        """Full pass over the good frames of the raw data. Use mean_image / _get_mean, which cache the result."""
        num_frames = self.shape[0]
        cumulated_mean = torch.zeros(self.shape[1], self.shape[2], dtype=self.dtype, device=self.device)
        num_batches = math.ceil(self.shape[0] / self.batch_size)
        # Visible progress: this pass can take a while and may start in the middle of a getitem / mean_image call
        progress = tqdm(range(num_batches), unit="batch",
                        desc=f"OphysArray: computing mean image over {num_frames} frames")
        for ind in progress:
            start_pt = ind * self.batch_size
            end_pt = min(self.shape[0], start_pt + self.batch_size)
            frames = slice(start_pt, end_pt)
            data_subset = self._dataset[frames if self._frame_map is None else self._to_raw(frames)]
            if isinstance(data_subset, np.ndarray):
                subset = torch.as_tensor(data_subset, device=device, dtype=self.dtype)
            elif isinstance(data_subset, torch.Tensor):
                subset = data_subset.to(self.device).to(self.dtype)
            else:
                raise ValueError("Calling getitem on dataset should return either a torch tensor or np.ndarray")
            cumulated_mean += torch.sum(subset, dim=0) / num_frames
        return cumulated_mean

    def _get_mean(self) -> torch.Tensor:
        mean = self._mean_image
        if mean is None:
            with self._mean_lock:  # if several threads hit this at once, only one of them pays for the pass
                if self._mean_image is None:
                    self._mean_image = self._compute_mean()
                mean = self._mean_image
        return mean

    @property
    def mean_image(self) -> torch.Tensor:
        """
        Mean over the good frames, shape (height, width). Identical for the recorded data and the positive view.
        Computed (and cached) on first access if construction did not need it.
        """
        return self._get_mean()

    @property
    def include_mean(self) -> bool:
        return self._include_mean

    @include_mean.setter
    def include_mean(self, new_flag: bool):
        # Switching to False defers the mean to the next getitem (or mean_image access) if it isn't computed yet
        self._include_mean = new_flag

    @property
    def negative_indicator(self) -> bool:
        """True if the recorded indicator is negatively tuned. getitem always returns the positively tuned view."""
        return self._negative_indicator

    @property
    def device(self) -> str:
        return self._device

    @property
    def dtype(self) -> torch.dtype:
        """
        data type
        """
        return torch.float32

    @property
    def shape(self) -> tuple[int, int, int]:
        return self._shape

    @property
    def nbytes(self) -> int:
        return math.prod(self.shape) * self.dtype.itemsize

    def __getstate__(self):
        state = self.__dict__.copy()
        del state["_mean_lock"]  # locks can't be pickled or deep-copied; give each copy its own
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._mean_lock = threading.Lock()

    def __getitem__(self,
                    item: int | list | np.ndarray | tuple[int | np.ndarray | slice | range]) -> torch.Tensor:
        """
        Index the positively tuned (and, if include_mean is False, mean-subtracted) view of the good frames.
        Temporal indices refer to the cropped movie, e.g. with good_frames = [0, 0, 1, 1, 1, 0, 0], index 0 is dataset frame 2.
        """
        return self._get(item, include_mean=self.include_mean)

    def _get(self,
             item: int | list | np.ndarray | tuple[int | np.ndarray | slice | range],
             include_mean: bool | None = None) -> torch.Tensor:
        ## Private helper method designed to avoid race conditions associated with state variables like include_mean
        if include_mean is None:
            include_mean = self.include_mean

        frame_indexer, item = self._parse_indices(item)  # validates and normalizes against the cropped shape

        if self._frame_map is not None:
            # Translate the (already normalized) temporal index from the cropped movie to the underlying dataset
            frame_indexer = self._to_raw(frame_indexer)
            item = (frame_indexer,) + item[1:] if isinstance(item, tuple) else frame_indexer

        data_subset = torch.as_tensor(self._dataset[item]).to(self.device).to(self.dtype) ## Parse indices first to ensure consistent output

        if not self.negative_indicator and include_mean:
            return data_subset  # already the positive view with its baseline: no mean needed

        # Check if spatial cropping occurred, deal with mean image accordingly
        mean_image = self._get_mean()
        mean_crop = mean_image[item[1:]] if isinstance(item, tuple) else mean_image

        if data_subset.ndim > mean_crop.ndim: #This means that there is a temporal dimension, which means we need to broadcast
            mean_crop = mean_crop[None, ...]

        # Out-of-place ops: one full-size pass instead of two, and never writes into the dataset's own memory
        # (torch.as_tensor and .to() return the dataset's buffer when no conversion is needed)
        if self.negative_indicator:
            if include_mean:
                return 2 * mean_crop - data_subset  # reflect about the mean: dips become rises, baseline unchanged
            return mean_crop - data_subset
        return data_subset - mean_crop