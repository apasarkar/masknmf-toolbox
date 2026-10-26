from typing import *
import numpy as np
from masknmf.arrays.array_interfaces import ArrayLike, TensorFlyWeight
import torch
from masknmf.arrays.array_utils import normalize_spatial_indexer, test_spatial_crop_effect


class StaticBackgroundArray(ArrayLike):
    """
    Class for managing the static baseline movie, a (frames, height, width) video where every frame is the same
    All indexing into this array is returned as a view for memory efficiency
    """

    def __init__(
            self,
            flyweight: TensorFlyWeight,
            num_frames: int,
            rescale: bool = False,
    ):
        if num_frames <= 0:
            raise ValueError(f"The number of frames must be positive, provided {num_frames}")
        self._flyweight=flyweight
        self.flyweight.validate_attributes(["baseline_image"])
        self._shape = (int(num_frames), *self.flyweight.baseline_image.shape)
        self._rescale=rescale

        self._default_normalizer = torch.ones_like(self.baseline_image, device=self.device).float()
        if hasattr(self.flyweight, "normalizer") and getattr(self.flyweight, "normalizer", None) is not None:
            if self.flyweight.normalizer.shape[0] != self.shape[1] or self.flyweight.normalizer.shape[1] != self.shape[
                2]:
                raise ValueError("Normalizer from flyweight had dimensions not equal to the fov dimensions")

        self._normalizer_baseline = self.normalizer * self.baseline_image

    @property
    def flyweight(self) -> TensorFlyWeight:
        return self._flyweight

    @classmethod
    def from_tensors(cls,
                     baseline_image: torch.Tensor,
                     num_frames: int,
                     normalizer: torch.Tensor | None = None,
                     rescale: bool=False):
        """
        Constructor for static baseline class
        Args:
            baseline_image (torch.Tensor): Shape (height, width)
            num_frames (int): The number of frames in this static movie
            normalizer (torch.Tensor | None): Shape (height, width)
        """

        flyweight = TensorFlyWeight(baseline_image=baseline_image, normalizer=normalizer)
        return cls(flyweight,
                   num_frames,
                   rescale=rescale)

    @classmethod
    def from_flyweight(cls,
                       flyweight: TensorFlyWeight,
                       num_frames: int,
                       rescale: bool = False):
        return cls(flyweight,
                   num_frames,
                   rescale=rescale)

    @property
    def device(self) -> str:
        return self.flyweight.device

    def to(self, new_device: str):
        if self.flyweight.device != new_device:
            self.flyweight.to(new_device)
        self._move_local_tensors(new_device)

    def _move_local_tensors(self, new_device: str):
        self._default_normalizer = self._default_normalizer.to(new_device)
        self._normalizer_baseline = self._normalizer_baseline.to(new_device)

    @property
    def shape(self) -> tuple[int, int]:
        return self._shape

    @property
    def dtype(self) -> str:
        """
        data type, default np.float32
        """
        return np.float32

    @property
    def normalizer(self) -> torch.Tensor:
        normalizer = getattr(self.flyweight, "normalizer", None)
        return normalizer if normalizer is not None else self._default_normalizer

    @property
    def rescale(self):
        return self._rescale

    @rescale.setter
    def rescale(self, new_value: bool):
        self._rescale = new_value

    @property
    def baseline_image(self) -> torch.Tensor:
        return self.flyweight.baseline_image

    def _count_frames(self, indexer) -> tuple[int, bool]:
        """Number of frames selected, and whether the frame axis is dropped. O(1) for ints and slices."""
        n = self._shape[0]
        if isinstance(indexer, int):
            if not -n <= indexer < n:
                raise IndexError(f"Frame index {indexer} out of bounds for {n} frames")
            return 1, True
        if isinstance(indexer, slice):
            return len(range(*indexer.indices(n))), False
        idx = np.asarray(indexer)
        if idx.dtype == bool:
            if idx.shape != (n,):
                raise IndexError(f"Boolean frame mask has shape {idx.shape}, expected ({n},)")
            return int(idx.sum()), False
        if idx.size and (idx.min() < -n or idx.max() >= n):
            raise IndexError(f"Frame indices out of bounds for {n} frames")
        return int(idx.size), False

    def __getitem__(
        self,
        item: int | list | np.ndarray | tuple[int | np.ndarray | slice | range],
    ) -> torch.Tensor:
        frame_indexer, item = self._parse_indices(item)
        num_frames, drop_frames = self._count_frames(frame_indexer)

        spatial = normalize_spatial_indexer(item, self.shape[1:])

        if self.rescale:
            final_image = self._normalizer_baseline[spatial]
        else:
            final_image = self.baseline_image[spatial]

        if drop_frames:
            return final_image.unsqueeze(0)
        elif num_frames == 1:
            return final_image.unsqueeze(0)
        else:
            return final_image.expand(num_frames, *final_image.shape)