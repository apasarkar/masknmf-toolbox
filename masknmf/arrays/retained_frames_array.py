import numpy as np
import torch

from .array_interfaces import ArrayLike, LazyFrameLoader


class RetainedFramesArray(LazyFrameLoader):
    """
    A movie computed on a subset of a recording's frames, laid out on the recording's frame axis: a retained frame
    comes from that movie, every other frame from a fallback movie of the whole recording.
    """

    def __init__(self, retained_movie: ArrayLike, retained_frames: np.ndarray, fallback: ArrayLike):
        """
        Args:
            retained_movie (ArrayLike): Shape (retained frames, height, width)
            retained_frames (np.ndarray): Shape (retained frames,). The recording frame of each frame of retained_movie
            fallback (ArrayLike): Shape (frames, height, width). Read at the frames that were not retained
        """
        if retained_movie.shape[0] != retained_frames.shape[0]:
            raise ValueError(f"{retained_movie.shape[0]} frames for {retained_frames.shape[0]} retained frames")
        if tuple(retained_movie.shape[1:]) != tuple(fallback.shape[1:]):
            raise ValueError(f"the movies have frames of {tuple(retained_movie.shape[1:])} and {tuple(fallback.shape[1:])}")
        self._retained_movie = retained_movie
        self._fallback = fallback
        self._position = np.full(fallback.shape[0], -1, dtype=np.int64)  # recording frame -> frame of retained_movie
        self._position[retained_frames] = np.arange(retained_frames.shape[0])

    @property
    def dtype(self) -> str:
        return np.float32

    @property
    def shape(self) -> tuple[int, int, int]:
        return tuple(int(n) for n in self._fallback.shape)

    def _compute_at_indices(self, indices: int | list | np.ndarray | slice) -> np.ndarray:
        frames = np.atleast_1d(np.arange(self.shape[0])[indices])
        positions = self._position[frames]
        retained = positions >= 0
        out = np.empty((frames.shape[0], *self.shape[1:]), dtype=np.float32)
        for movie, index, target in ((self._retained_movie, positions[retained], retained),
                                     (self._fallback, frames[~retained], ~retained)):
            if index.size > 0:
                chunk = movie[index]
                out[target] = chunk.cpu().numpy() if isinstance(chunk, torch.Tensor) else np.asarray(chunk)
        return out
