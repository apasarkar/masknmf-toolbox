import numpy as np
import torch
from masknmf.arrays.array_interfaces import ArrayLike


class StaticArray(ArrayLike):
    """One image repeated over every frame. Indexing returns broadcast views"""

    def __init__(self,
                 image: np.ndarray | torch.Tensor,
                 num_frames: int,
                 device: str | torch.device = "cpu"):
        num_frames = int(num_frames)
        if num_frames <= 0:
            raise ValueError(f"Need a positive number of frames, you provided {num_frames}")
        self._image = torch.as_tensor(image, device=device)
        if self._image.ndim != 2:
            raise ValueError(f"image must be 2D (height, width), got shape {tuple(self._image.shape)}")
        self._shape = (num_frames, *self._image.shape)

    @property
    def dtype(self) -> torch.dtype:
        return self._image.dtype

    @property
    def device(self) -> torch.device:
        return self._image.device

    def to(self, new_device: str | torch.device) -> None:
        self._image = self._image.to(torch.device(new_device))

    @property
    def shape(self) -> tuple[int, int, int]:
        return self._shape

    @property
    def image(self) -> torch.Tensor:
        return self._image

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

    def __getitem__(self, item) -> torch.Tensor:
        """
        Provides a view of the static array for memory efficiency. In-place operations might not work
        """
        frame_indexer, item = self._parse_indices(item)
        num_frames, drop_frames = self._count_frames(frame_indexer)

        spatial = item[1:] if isinstance(item, tuple) else ()
        image = self._image[spatial] if len(spatial) > 0 else self._image

        if drop_frames:
            return image
        elif num_frames == 1:
            return image.unsqueeze(0)      # stride-0 size-1 view would alias self._image
        else:
            return image.expand(num_frames, *image.shape)