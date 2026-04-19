from .array_interfaces import LazyFrameLoader
import tifffile
from collections import defaultdict
import numpy as np
import h5py
from typing import *
import os


class TiffSeriesLoader(LazyFrameLoader):
    def __init__(self,
                 file_paths: list[str],
                 memmap: bool = False):
        self._arrays = [TiffArray(p, memmap=memmap) for p in file_paths]
        self._array_shapes = [arr.shape for arr in self._arrays]
        self._dtype = self._arrays[0].dtype

        self._n_frames = sum(k[0] for k in self._array_shapes)
        self._height = self._arrays[0].shape[1]
        self._width = self._arrays[0].shape[2]

        self._frame_map = np.zeros((self._n_frames, 3), dtype=np.int64)

        start = 0
        for file_id, arr in enumerate(self._arrays):
            curr_frames = self._arrays[file_id].shape[0]
            end = start + curr_frames
            self._frame_map[start:end, 0] = np.arange(start, end)
            self._frame_map[start:end, 1] = np.arange(curr_frames)
            self._frame_map[start:end, 2] = file_id
            start = end

    @property
    def dtype(self):
        return self._dtype

    @property
    def ndim(self) -> int:
        return 3

    @property
    def shape(self) -> tuple:
        return self._n_frames, self._height, self._width

    def _compute_at_indices(self, indices) -> np.ndarray:
        if isinstance(indices, int):
            indices = [indices]
        elif isinstance(indices, slice):
            indices = list(range(*indices.indices(self.shape[0])))
        else:
            indices = list(indices)

        rows = self._frame_map[indices, :]

        chunks = []  # list of (out_indices, frames) tuples
        insertion_order = np.zeros(len(rows), dtype=np.int64)

        active_file_ids = np.unique(rows[:, 2])
        pos = 0
        for file_id in active_file_ids:
            mask = rows[:, 2] == file_id
            out_indices = np.where(mask)[0]
            local_indices = rows[mask, 1].tolist()

            file_frames = self._arrays[file_id][local_indices]
            if file_frames.ndim == 2:
                file_frames = file_frames[None, :, :]
            chunks.append(file_frames)

            insertion_order[pos:pos + len(out_indices)] = out_indices
            pos += len(out_indices)

        # Single stack, then single argsort-based permutation on axis 0
        stacked = np.concatenate(chunks, axis=0)  # one allocation
        perm = np.argsort(insertion_order)  # where each frame should go
        return stacked[perm]  # single fancy-index read, not write

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
            self._dtype =self.filename.dtype
            self._shape = self.filename.shape
        else:
            self.filename = filename
            self._dtype = self._get_dtype(self.filename)
            self._shape = self._get_shape(self.filename)

    @staticmethod
    def _get_dtype(filename):
        with tifffile.TiffFile(filename) as tffl:
            num_frames = len(tffl.pages)
            for page in tffl.pages[0:1]:
                image = page.asarray()
                return image.dtype

    @staticmethod
    def _get_shape(file: str | np.memmap):
        if isinstance(file, np.memmap):
            return file.shape
        else:
            with tifffile.TiffFile(file) as tffl:
                num_frames = len(tffl.pages)
                for page in tffl.pages[0:1]:
                    image = page.asarray()
                x, y = page.shape
            return num_frames, x, y

    @property
    def dtype(self) -> str:
        """
        str
            data type
        """
        return self._dtype

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
            if data.ndim == 2:
                data = data[None, ...]
        return data.astype(self.dtype, copy=False)

    def __getitem__(
            self,
            item: Union[int, list, np.ndarray, slice, range, Tuple[Union[int, np.ndarray, slice, range]]],
    ):
        if self.memmap:
            data = self.filename.__getitem__(item).copy()
            return data.astype(self.dtype, copy=False)
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

            # Get the dtype of the array
            self._dtype = field_dataset.dtype

    @property
    def dtype(self) -> str:
        """
        str
            data type
        """
        return self._dtype

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
            if data.ndim == 2:
                data = data[None, :, :]
        return data.astype(self.dtype, copy=False)
