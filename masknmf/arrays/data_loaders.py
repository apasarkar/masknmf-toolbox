from .array_interfaces import LazyFrameLoader
import tifffile
from collections import defaultdict
import numpy as np
import h5py
from typing import *


class TiffSeriesLoader(LazyFrameLoader):
    def __init__(self,
                 file_paths: list[str],
                 memmap: bool=False,):
        self._arrays = [TiffArray(p, memmap=memmap) for p in file_paths]

        total_frames = sum(arr.shape[0] for arr in self._arrays)
        self._frame_map = np.empty((total_frames, 3), dtype=np.int64)

        start = 0
        for file_id, arr in enumerate(self._arrays):
            n_frames = arr.shape[0]
            end = start + n_frames
            self._frame_map[start:end, 0] = np.arange(start, end)
            self._frame_map[start:end, 1] = np.arange(n_frames)
            self._frame_map[start:end, 2] = file_id
            start = end

    @property
    def dtype(self):
        return self._arrays[0].dtype

    @property
    def ndim(self) -> int:
        return 3

    @property
    def shape(self) -> tuple:
        _, h, w = self._arrays[0].shape
        return (len(self._frame_map), h, w)

    def _compute_at_indices(self, indices) -> np.ndarray:
        if isinstance(indices, int):
            indices = [indices]
        elif isinstance(indices, slice):
            indices = list(range(*indices.indices(self.shape[0])))
        else:
            indices = list(indices)

        rows = self._frame_map[indices, :]
        frames = np.empty((len(rows), self.shape[1], self.shape[2]), dtype=self.dtype)

        active_file_ids = np.unique(rows[:, 2])
        for file_id in active_file_ids:
            mask = rows[:, 2] == file_id
            out_indices = np.where(mask)[0]
            local_indices = rows[mask, 1].tolist()
            frames[out_indices] = self._arrays[file_id][local_indices]

        return frames

class TiffArray(LazyFrameLoader):
    def __init__(self, filename, memmap: bool = False):
        """
        TiffArray data loading object. Supports loading data from multipage tiff files.

        Args:
            filename (str): Path to file

        """
        self._memmap = memmap
        if self.memmap:
            self.filename = tifffile.memmap(filename)
        else:
            self.filename = filename

    @property
    def dtype(self) -> str:
        """
        str
            data type
        """
        return np.float32

    @property
    def shape(self) -> Tuple[int, int, int]:
        """
        Tuple[int]
            (n_frames, dims_x, dims_y)
        """
        if self.memmap:
            return self.filename.shape
        else:
            with tifffile.TiffFile(self.filename) as tffl:
                num_frames = len(tffl.pages)
                for page in tffl.pages[0:1]:
                    image = page.asarray()
                x, y = page.shape
            return num_frames, x, y

    @property
    def ndim(self) -> int:
        """
        int
            Number of dimensions
        """
        return len(self.shape)

    @property
    def memmap(self) -> bool:
        return self._memmap

    def _compute_at_indices(self, indices: Union[list, int, slice]) -> np.ndarray:
        if isinstance(indices, int):
            data = tifffile.imread(self.filename, key=[indices])
        elif isinstance(indices, list):
            data = tifffile.imread(self.filename, key=indices)
        else:
            indices_list = list(
                range(
                    indices.start or 0, indices.stop or self.shape[0], indices.step or 1
                )
            )
            data = tifffile.imread(self.filename, key=indices_list)
        return data.astype(self.dtype)

    def __getitem__(
            self,
            item: Union[int, list, np.ndarray, slice, range, Tuple[Union[int, np.ndarray, slice, range]]],
    ):
        if self.memmap:
            data = self.filename.__getitem__(item).copy()
            return data.astype(self.dtype)
        else:
            return super().__getitem__(item)


class Hdf5Array(LazyFrameLoader):
    def __init__(self, filename: str, field: str) -> None:
        """
        Generic lazy loader for Hdf5 files video files, where data is stored as (T, x, y). T is number of frames,
        x and y are the field of view dimensions (height and width).

        Args:
            filename (str): Path to filename
            field (str): Field of hdf5 file containing data
        """
        if not isinstance(field, str):
            raise ValueError("Field must be a string")
        self.filename = filename
        self.field = field
        with h5py.File(self.filename, "r") as file:
            # Access the 'field' dataset
            field_dataset = file[self.field]

            # Get the shape of the array
            self._shape = field_dataset.shape

    @property
    def dtype(self) -> str:
        """
        str
            data type
        """
        return np.float32

    @property
    def shape(self) -> Tuple[int, int, int]:
        """
        Tuple[int]
            (n_frames, dims_x, dims_y)
        """
        return self._shape

    @property
    def ndim(self) -> int:
        """
        int
            Number of dimensions
        """
        return len(self.shape)

    def _compute_at_indices(self, indices: Union[list, int, slice]) -> np.ndarray:
        with h5py.File(self.filename, "r") as file:
            # Access the 'field' dataset
            field_dataset = file[self.field]
            if isinstance(indices, int):
                data = field_dataset[indices, :, :]
            elif isinstance(indices, list):
                data = field_dataset[indices, :, :]
            else:
                indices_list = list(
                    range(
                        indices.start or 0,
                        indices.stop or self.shape[0],
                        indices.step or 1,
                    )
                )
                data = field_dataset[indices_list, :, :]
        return data.astype(self.dtype)
