import os
import re
from typing import Union

import numpy as np

from masknmf.arrays.array_interfaces import LazyFrameLoader, ArrayLike


class InterleavedChannelArray(LazyFrameLoader):
    """
    Lazy single-channel view of a recording whose channels are interleaved in time, as in widefield imaging that
    alternates excitation light (e.g. 470 nm calcium-dependent and 405 nm isosbestic/hemodynamic frames).

    The source can be:
        - 4D, shape (num_cycles, num_channels, height, width): one row per acquisition cycle, e.g. the memmap wfield
          writes (``motioncorrect_<C>_<H>_<W>_uint16.bin``, see :func:`load_interleaved_binary`).
        - 3D, shape (num_cycles * num_channels, height, width): frames in acquisition order, channel ``c`` at raw frames
          ``c, c + C, c + 2C, ...``. A trailing incomplete cycle is dropped, so every channel has the same frame count.

    Frame k of this array is the channel's frame in cycle k. Nothing is read until frames are indexed.
    """

    def __init__(self,
                 data: np.ndarray | ArrayLike,
                 channel: int,
                 num_channels: int = 2,
                 num_frames: int | None = None):
        """
        Args:
            data (np.ndarray | ArrayLike): The interleaved recording, 4D (cycles, channels, height, width) or 3D
                (frames in acquisition order, height, width). A numpy memmap is read lazily.
            channel (int): Which channel this view exposes, 0 <= channel < num_channels.
            num_channels (int): Channels per acquisition cycle. Must equal data.shape[1] for 4D data.
            num_frames (int | None): Expose only the first num_frames cycles. None exposes all complete cycles.
        """
        if num_channels < 1:
            raise ValueError(f"num_channels must be positive, got {num_channels}")
        if not 0 <= channel < num_channels:
            raise ValueError(f"channel must be in [0, {num_channels}), got {channel}")
        if len(data.shape) == 4:
            if data.shape[1] != num_channels:
                raise ValueError(f"4D data has {data.shape[1]} channels on axis 1 but num_channels is {num_channels}")
            num_cycles = data.shape[0]
            fov = tuple(data.shape[2:])
        elif len(data.shape) == 3:
            num_cycles = data.shape[0] // num_channels
            fov = tuple(data.shape[1:])
        else:
            raise ValueError(f"data must be 3D or 4D, got shape {tuple(data.shape)}")
        if num_frames is not None:
            if num_frames < 1 or num_frames > num_cycles:
                raise ValueError(f"num_frames must be in [1, {num_cycles}], got {num_frames}")
            num_cycles = int(num_frames)
        if num_cycles < 1:
            raise ValueError("data holds no complete acquisition cycle")
        self._data = data
        self._channel = int(channel)
        self._num_channels = int(num_channels)
        self._four_d = len(data.shape) == 4
        self._shape = (num_cycles, fov[0], fov[1])
        self._dtype = data.dtype

    @property
    def channel(self) -> int:
        return self._channel

    @property
    def num_channels(self) -> int:
        return self._num_channels

    @property
    def dtype(self):
        return self._dtype

    @property
    def shape(self) -> tuple[int, int, int]:
        return self._shape

    @property
    def ndim(self) -> int:
        return 3

    def __getitem__(self, item):
        # ArrayLike keeps every axis an int indexes; LazyFrameLoader's spatial step would drop it, so turn spatial
        # ints into length-1 slices first
        if isinstance(item, tuple):
            spatial = []
            for axis, index in enumerate(item[1:], start=1):
                if isinstance(index, (int, np.integer)):
                    index = int(index) % self.shape[axis]
                    index = slice(index, index + 1)
                spatial.append(index)
            item = (item[0], *spatial)
        return super().__getitem__(item)

    def raw_frame_indices(self) -> np.ndarray:
        """Position of each of this channel's frames in the acquisition order (cycle * num_channels + channel)."""
        return np.arange(self.shape[0]) * self._num_channels + self._channel

    def _compute_at_indices(self, indices: Union[list, int, slice, range, np.ndarray]) -> np.ndarray:
        cycles = np.arange(self.shape[0])[indices]
        scalar = np.ndim(cycles) == 0
        cycles = np.atleast_1d(cycles)
        if self._four_d:
            if cycles.size > 1 and np.all(np.diff(cycles) == 1):
                frames = self._data[int(cycles[0]):int(cycles[-1]) + 1, self._channel]
            else:
                frames = self._data[cycles, self._channel]
        else:
            raw = cycles * self._num_channels + self._channel
            if raw.size > 1 and np.all(np.diff(raw) == self._num_channels):
                frames = self._data[int(raw[0]):int(raw[-1]) + 1:self._num_channels]
            else:
                frames = self._data[raw]
        frames = np.asarray(frames)
        return frames[0] if scalar else frames


_WFIELD_BIN = re.compile(r"_(\d+)_(\d+)_(\d+)_([a-z]+\d+)\.bin$")


def load_interleaved_binary(path: str | os.PathLike,
                            num_channels: int | None = None,
                            height: int | None = None,
                            width: int | None = None,
                            dtype: str | np.dtype | None = None) -> np.memmap:
    """
    Read-only memmap of a raw interleaved binary stack, shape (num_cycles, num_channels, height, width).

    Shape and dtype come from the arguments, or from the wfield file name convention
    ``<name>_<num_channels>_<height>_<width>_<dtype>.bin`` (e.g. ``motioncorrect_2_540_640_uint16.bin``) for any that
    are None. Trailing bytes that do not fill a whole cycle are ignored.

    Args:
        path: The .bin file.
        num_channels, height, width, dtype: The layout; None reads it from the file name.
    Returns:
        np.memmap: Shape (num_cycles, num_channels, height, width).
    """
    path = os.fspath(path)
    match = _WFIELD_BIN.search(os.path.basename(path))
    if match is not None:
        parsed = {"num_channels": int(match.group(1)), "height": int(match.group(2)),
                  "width": int(match.group(3)), "dtype": match.group(4)}
    else:
        parsed = {}
    num_channels = num_channels if num_channels is not None else parsed.get("num_channels")
    height = height if height is not None else parsed.get("height")
    width = width if width is not None else parsed.get("width")
    dtype = dtype if dtype is not None else parsed.get("dtype")
    missing = [n for n, v in (("num_channels", num_channels), ("height", height), ("width", width), ("dtype", dtype))
               if v is None]
    if missing:
        raise ValueError(f"{path}: give {', '.join(missing)} (the file name does not follow "
                         f"<name>_<channels>_<height>_<width>_<dtype>.bin)")
    dtype = np.dtype(dtype)
    cycle_bytes = int(num_channels) * int(height) * int(width) * dtype.itemsize
    num_cycles = os.path.getsize(path) // cycle_bytes
    if num_cycles < 1:
        raise ValueError(f"{path} is smaller than one acquisition cycle")
    return np.memmap(path, dtype=dtype, mode="r", shape=(num_cycles, int(num_channels), int(height), int(width)))
