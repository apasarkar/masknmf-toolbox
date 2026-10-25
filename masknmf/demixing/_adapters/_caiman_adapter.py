import os

import h5py
import numpy as np
import scipy.sparse as sp
import torch

from masknmf.arrays.array_interfaces import ArrayLike, LazyFrameLoader
from masknmf.demixing._base_results import BaseResults
from masknmf.demixing.demixing_arrays import SignalsArray
from masknmf.demixing.demixing_utils import scipy_sparse_to_torch
from masknmf.utils import SparseCOOTensor


def _is_none(value) -> bool:
    """caiman writes None as the string 'NoneType'"""
    return value is None or (isinstance(value, (str, bytes)) and value in ("NoneType", b"NoneType"))


def _read_dataset(group: h5py.Group, key: str):
    """The value stored under key, or None if it is missing or was saved as None"""
    if key not in group:
        return None
    value = group[key][()]
    return None if _is_none(value) else value


def _read_csc(group: h5py.Group, key: str) -> sp.csc_matrix | None:
    """caiman saves its scipy sparse matrices (A, W, ...) as a group holding data, indices, indptr and shape"""
    if key not in group or not isinstance(group[key], h5py.Group):
        return None
    sub = group[key]
    return sp.csc_matrix((sub["data"][:], sub["indices"][:], sub["indptr"][:]), tuple(sub["shape"][:]))


def caiman_f_to_c_pixel_order(A: sp.spmatrix, fov_shape: tuple[int, int]) -> sp.csc_matrix:
    """
    caiman flattens a (height, width) image in Fortran order (pixel index = y + height * x); masknmf's SignalsArray
    uses C order (pixel index = y * width + x). Returns A with its rows permuted from the first to the second.

    Args:
        A (scipy.sparse): shape (height * width, n_components), rows in caiman (F) order
        fov_shape (tuple[int, int]): (height, width)
    """
    height, width = fov_shape
    if A.shape[0] != height * width:
        raise ValueError(f"A has {A.shape[0]} pixels but the fov is {height}x{width}")
    # for each C-order pixel (y, x) the row of A that holds it
    f_row_of_c_pixel = (np.arange(height)[:, None] + height * np.arange(width)[None, :]).ravel()
    return sp.csc_matrix(A.tocsr()[f_row_of_c_pixel])


def load_caiman_memmap(filename: str | os.PathLike) -> np.ndarray:
    """
    Opens a memory mapped movie written by caiman's save_memmap (the .mmap file caiman fits CNMF on) 
    """
    name = os.path.splitext(os.path.basename(filename))[0]
    tokens = name.split("_")[1:]
    fields = {}
    for key in ("d1", "d2", "d3", "order", "frames"):
        for i in range(len(tokens) - 2, -1, -1):
            if tokens[i] == key:
                fields[key] = tokens[i + 1]
                break
    # like caiman: the number of frames is the last token of the name, or after "frames" if the name ends in "_"
    last = tokens[-1] if tokens else ""
    num_frames_text = last if last != "" else fields.get("frames", "")
    if not all(k in fields for k in ("d1", "d2", "d3", "order")) or not num_frames_text.isdigit():
        raise ValueError(f"{filename} does not look like a caiman memmap file name "
                         "(.._d1_<h>_d2_<w>_d3_1_order_<C|F>_..._<frames>.mmap)")
    height, width, depth = int(fields["d1"]), int(fields["d2"]), int(fields["d3"])
    if depth != 1:
        raise ValueError("only 2D movies are supported (d3 must be 1)")
    num_frames = int(num_frames_text)
    Yr = np.memmap(filename, mode="r", dtype=np.float32, order=fields["order"], shape=(height * width, num_frames))
    return Yr.T.reshape((num_frames, height, width), order="F")      # caiman's pixel order is Fortran


def caiman_decimation_matrix(fov_shape: tuple[int, int], sub: int) -> sp.csc_matrix:
    """
    The sparse matrix that averages sub x sub blocks of a flattened (Fortran order) image, so (pixels_ds, pixels) @
    image.ravel(order="F") is the image downsampled by sub. Same matrix as caiman's decimation_matrix.
    """
    height, width = fov_shape
    height_ds = 1 + (height - 1) // sub
    width_ds = 1 + (width - 1) // sub
    y, x = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
    ds_index = (y // sub + height_ds * (x // sub)).ravel(order="F")        
    counts = np.bincount(ds_index, minlength=height_ds * width_ds)
    data = (1.0 / counts[ds_index]).astype(np.float32)
    return sp.csc_matrix((data, ds_index, np.arange(1 + height * width)),
                         shape=(height_ds * width_ds, height * width), dtype=np.float32)


class _CaimanRingModel:
    """
    caiman's CNMF-E ring-model background, computed for a few frames at a time (caiman's
    Estimates.compute_background needs the whole movie in memory). With Y the movie, A and C the footprints and
    traces of every component the model was fit with, and b0, W the ring model's baseline and weights:
        B = b0 + W (Y - A C - b0)
    When the model was fit on a spatially downsampled movie (ssub_B > 1), the term W (...) is computed on the
    downsampled pixels and upsampled back, as in caiman.
    Everything here is in caiman's pixel order (Fortran, pixel = y + height * x).
    """

    def __init__(self, movie, W: sp.spmatrix, b0: np.ndarray, A: sp.spmatrix, C: np.ndarray,
                 fov_shape: tuple[int, int]):
        height, width = fov_shape
        n_pixels = height * width
        if tuple(movie.shape[1:]) != (height, width):
            raise ValueError(f"movie is {tuple(movie.shape)} but the fov is {height}x{width}")
        if movie.shape[0] != C.shape[1]:
            raise ValueError(f"movie has {movie.shape[0]} frames but the traces have {C.shape[1]}")
        b0 = np.asarray(b0, np.float32).ravel()
        if b0.size != n_pixels or A.shape[0] != n_pixels:
            raise ValueError("b0 and A must have one entry per pixel")
        self._movie = movie
        self._fov_shape = (height, width)
        self._num_frames = int(movie.shape[0])
        self._W = sp.csr_matrix(W, dtype=np.float32)
        self._b0 = b0
        self._A = sp.csc_matrix(A, dtype=np.float32)
        self._C = np.asarray(C, np.float32)
        self._ssub = int(np.round(np.sqrt(n_pixels / self._W.shape[0])))
        if self._ssub < 1 or self._W.shape[0] != (1 + (height - 1) // self._ssub) * (1 + (width - 1) // self._ssub):
            raise ValueError(f"W is {self._W.shape}, which does not fit a {height}x{width} fov at any downsampling")
        if self._ssub > 1:
            ds = caiman_decimation_matrix(self._fov_shape, self._ssub)
            self._ds = sp.csr_matrix(ds)
            self._ds_A = sp.csr_matrix(ds @ self._A)
            self._ds_b0 = np.asarray(ds @ b0, np.float32)

    def _to_pixels(self, frames: np.ndarray) -> np.ndarray:
        """(k, height, width) -> (pixels, k), pixels in caiman order"""
        k = frames.shape[0]
        return np.ascontiguousarray(frames.transpose(2, 1, 0).reshape(-1, k))

    def _to_frames(self, pixels: np.ndarray) -> np.ndarray:
        """(pixels, k) in caiman order -> (k, height, width)"""
        height, width = self._fov_shape
        return np.ascontiguousarray(pixels.reshape(width, height, -1).transpose(2, 1, 0))

    def movie_pixels(self, frame_ids: np.ndarray) -> np.ndarray:
        frames = np.asarray(self._movie[np.asarray(frame_ids)], dtype=np.float32)
        return self._to_pixels(frames.reshape((-1,) + self._fov_shape))

    def background_pixels(self, Y: np.ndarray, frame_ids: np.ndarray) -> np.ndarray:
        """The background at the given frames; Y is the movie there, (pixels, k)"""
        C = self._C[:, frame_ids]
        if self._ssub == 1:
            return self._b0[:, None] + self._W @ (Y - self._A @ C - self._b0[:, None])
        height, width = self._fov_shape
        height_ds, width_ds = 1 + (height - 1) // self._ssub, 1 + (width - 1) // self._ssub
        low = self._ds @ Y - self._ds_A @ C - self._ds_b0[:, None]
        low = (self._W @ low).reshape((height_ds, width_ds, -1), order="F")
        up = np.repeat(np.repeat(low, self._ssub, 0), self._ssub, 1)[:height, :width]
        return self._b0[:, None] + up.reshape((height * width, -1), order="F")

    def background_frames(self, frame_ids: np.ndarray) -> np.ndarray:
        return self._to_frames(self.background_pixels(self.movie_pixels(frame_ids), frame_ids))


def _frame_indices(frame_indexer, num_frames: int) -> np.ndarray:
    if isinstance(frame_indexer, slice):
        return np.arange(*frame_indexer.indices(num_frames))
    return np.atleast_1d(np.asarray(frame_indexer, dtype=np.int64)) % num_frames


class _ChunkedFrames(LazyFrameLoader):
    """Frames that are computed on request, chunk_size at a time, cropped to the requested pixels before they are kept,
    so that e.g. a region's average trace over every frame needs only one chunk at a time in memory"""

    chunk_size = 100

    def _frames(self, frame_ids: np.ndarray) -> np.ndarray:  # (k, height, width)
        raise NotImplementedError

    @property
    def dtype(self):
        return np.float32

    def __getitem__(self, item):
        frame_indexer, item = self._parse_indices(item)
        frame_ids = _frame_indices(frame_indexer, self.shape[0])
        crop = item[1:] if isinstance(item, tuple) else ()
        pieces = []
        for start in range(0, len(frame_ids), self.chunk_size):
            chunk = self._frames(frame_ids[start:start + self.chunk_size])
            pieces.append(chunk[(slice(None),) + tuple(crop)] if crop else chunk)
        out = np.concatenate(pieces, axis=0) if pieces else np.zeros((0,) + self.shape[1:], np.float32)
        return out.astype(np.float32, copy=False)

    def _compute_at_indices(self, indices):
        return self._frames(_frame_indices(indices, self.shape[0]))


class CaimanBackgroundArray(_ChunkedFrames):
    """caiman's fluctuating (ring-model) background in masknmf's pixel order"""

    def __init__(self, ring_model: _CaimanRingModel, mean_zero: bool = False):
        """mean_zero: subtract each pixel's average over all frames """
        self._ring = ring_model
        self._mean_zero = mean_zero
        self._mean_frame = None

    @property
    def shape(self) -> tuple[int, int, int]:
        return (self._ring._num_frames, *self._ring._fov_shape)

    def _frames(self, frame_ids):
        frames = self._ring.background_frames(frame_ids)
        if not self._mean_zero:
            return frames
        if self._mean_frame is None:                       # average over all frames, a chunk at a time
            total = np.zeros(self.shape[1:], np.float64)
            for start in range(0, self.shape[0], self.chunk_size):
                total += self._ring.background_frames(np.arange(start, min(start + self.chunk_size, self.shape[0]))).sum(0)
            self._mean_frame = (total / self.shape[0]).astype(np.float32)
        return frames - self._mean_frame


class CaimanResidualArray(_ChunkedFrames):
    """
    The movie minus the components minus the ring-model background, lazily: Y - A C - B, which is the residual that
    caiman's Estimates.play_movie shows (Y_res = imgs - Y_rec - B)
    """

    def __init__(self, ring_model: _CaimanRingModel, A_accepted: sp.spmatrix, C_accepted: np.ndarray):
        self._ring = ring_model
        self._A = sp.csc_matrix(A_accepted, dtype=np.float32)
        self._C = np.asarray(C_accepted, np.float32)

    @property
    def shape(self) -> tuple[int, int, int]:
        return (self._ring._num_frames, *self._ring._fov_shape)

    def _frames(self, frame_ids):
        Y = self._ring.movie_pixels(frame_ids)
        B = self._ring.background_pixels(Y, frame_ids)
        return self._ring._to_frames(Y - self._A @ self._C[:, frame_ids] - B)


class CaimanResults(BaseResults):
    """
    Loads caiman (CNMF / CNMF-E) results into masknmf

    Reads the .hdf5 file written by caiman's CNMF.save directly with h5py.
    Needs the footprints estimates/A (sparse, pixels x components, caiman's Fortran pixel order), the temporal
    traces estimates/C (components x frames) and the fov estimates/dims.

    """

    def __init__(self,
                 results_file: str | os.PathLike,
                 device: str | torch.device = "cpu",
                 accepted_only: bool = True,
                 include_residual_traces: bool = True,
                 movie: ArrayLike | np.ndarray | None = None,
                 residual_components: str = "all",
                 background_mean_zero: bool = False):
        """
        Args:
            results_file (str | os.PathLike): the hdf5 file written by caiman's cnmf.save
            device (str | torch.device): device on which the data is stored/used for computation
            accepted_only (bool): keep only the components caiman accepted (estimates/idx_components, set by evaluate_components). 
            movie (ArrayLike | np.ndarray | None): the movie caiman was fit on (its motion corrected, memory mapped movie), shaped (frames, height, width)
            residual_components (str): which components the residual subtracts, "all" or "accepted". "all" (the default)
            background_mean_zero (bool): subtract each pixel's average over all frames from the fluctuating background,
            include_residual_traces (bool): if estimates/YrA was saved, expose C + YrA (caiman's raw trace of each
                component, i.e. its denoised trace plus the residual projected onto its footprint) as
                temporal_demixed_raw
        """
        results_file = os.path.abspath(results_file)
        with h5py.File(results_file, "r") as f:
            if "estimates" not in f:
                raise KeyError(f"{results_file} has no 'estimates' group.")
            est = f["estimates"]
            A = _read_csc(est, "A")
            C = _read_dataset(est, "C")
            dims = _read_dataset(est, "dims")
            if A is None or C is None or dims is None:
                raise KeyError("estimates/A, estimates/C and estimates/dims are all needed")
            idx = _read_dataset(est, "idx_components")
            YrA = _read_dataset(est, "YrA") if include_residual_traces else None
            W = _read_csc(est, "W")
            b0 = _read_dataset(est, "b0")
            Cn = _read_dataset(est, "Cn")

        height, width = (int(d) for d in dims) 
        C = np.asarray(C, np.float32)
        if A.shape[1] != C.shape[0]:
            raise ValueError(f"A has {A.shape[1]} components but C has {C.shape[0]}")

        if accepted_only and idx is not None and np.size(idx) > 0:
            self._component_ids = np.asarray(idx, dtype=np.int64).ravel()
        else:
            self._component_ids = np.arange(A.shape[1], dtype=np.int64)

        A_f_all, C_all = A, C                                   
        A_f_accepted = A[:, self._component_ids]                 
        A = caiman_f_to_c_pixel_order(A_f_accepted, (height, width))
        C = C[self._component_ids]

        self._fov_shape = (height, width)
        self._num_frames = C.shape[1]
        self._spatial_demixed = scipy_sparse_to_torch(A).coalesce()
        self._temporal_demixed = torch.from_numpy(np.ascontiguousarray(C.T)) 

        self._temporal_demixed_raw = None
        if YrA is not None:
            YrA = np.asarray(YrA, np.float32)  
            if YrA.ndim == 2 and YrA.shape[1] == self._num_frames and YrA.shape[0] > self._component_ids.max():
                self._temporal_demixed_raw = torch.from_numpy(
                    np.ascontiguousarray((C + YrA[self._component_ids]).T))

        self._correlation_image = None
        if Cn is not None and np.shape(Cn) in ((height, width), (width, height)):
            Cn = np.asarray(Cn, np.float32)
            self._correlation_image = torch.from_numpy(Cn if Cn.shape == (height, width) else Cn.T.copy())

        self._signals_array = SignalsArray.from_tensors(self._fov_shape,
                                                        self._spatial_demixed,
                                                        self._temporal_demixed)
        self._ring_model = None
        self._background_array = None
        self._residual_array = None
        if residual_components not in ("all", "accepted"):
            raise ValueError("residual_components must be 'all' or 'accepted'")
        self._A_f_accepted = A_f_accepted
        self._C_accepted = C
        self._A_f_residual, self._C_residual = ((A_f_all, C_all) if residual_components == "all"
                                                else (A_f_accepted, C))
        self._roi_averages_cache = None
        self._background_mean_zero = bool(background_mean_zero)
        if movie is not None:
            if W is None or b0 is None or np.size(b0) == 0:
                raise ValueError("a movie was given, but the file has no ring-model background ")
            self._ring_model = _CaimanRingModel(movie, W, b0, A_f_all, C_all, self._fov_shape)
            self._background_array = CaimanBackgroundArray(self._ring_model, self._background_mean_zero)
            self._residual_array = CaimanResidualArray(self._ring_model, self._A_f_residual, self._C_residual)

        self._device = torch.device("cpu")
        self.to(device)

    def to(self, new_device: str | torch.device):
        updated_device = torch.device(new_device)
        self._spatial_demixed = self._spatial_demixed.to(updated_device)
        self._temporal_demixed = self._temporal_demixed.to(updated_device)
        if self._temporal_demixed_raw is not None:
            self._temporal_demixed_raw = self._temporal_demixed_raw.to(updated_device)
        if self._correlation_image is not None:
            self._correlation_image = self._correlation_image.to(updated_device)
        self._signals_array.to(updated_device)
        self._device = updated_device

    @property
    def device(self) -> str | torch.device:
        return self._device

    @property
    def shape(self) -> tuple[int, int, int]:
        return (self._num_frames, *self._fov_shape)

    @property
    def component_ids(self) -> np.ndarray:
        return self._component_ids

    @property
    def spatial_demixed(self) -> SparseCOOTensor:
        return self._spatial_demixed

    @property
    def temporal_demixed(self) -> torch.Tensor:
        return self._temporal_demixed

    @property
    def temporal_demixed_raw(self) -> torch.Tensor | None:
        return self._temporal_demixed_raw

    @property
    def signals_array(self) -> SignalsArray:
        return self._signals_array

    @property
    def correlation_image(self) -> torch.Tensor | None:
        return self._correlation_image

    @property
    def global_residual_correlation_image(self) -> torch.Tensor | None:
        return None

    @property
    def fluctuating_background_array(self) -> CaimanBackgroundArray | None:
        return self._background_array

    @property
    def residual_array(self) -> CaimanResidualArray | None:
        return self._residual_array

    def _compute_roi_averages(self) -> dict[str, np.ndarray]:
        """Each signal's footprint-weighted average (weights sum to one) of the movie, of the movie's background and
        of the residual, shape (signals, frames)"""
        if self._roi_averages_cache is None:
            A = self._A_f_accepted
            column_sums = np.asarray(A.sum(axis=0)).ravel()
            weights = sp.csc_matrix(A @ sp.diags(1.0 / np.maximum(column_sums, 1e-12)))   # (pixels, signals)
            num_signals, num_frames = A.shape[1], self._num_frames
            movie_avg = np.zeros((num_signals, num_frames), np.float32)
            background_avg = np.zeros((num_signals, num_frames), np.float32)
            step = _ChunkedFrames.chunk_size
            for start in range(0, num_frames, step):
                ids = np.arange(start, min(start + step, num_frames))
                Y = self._ring_model.movie_pixels(ids)
                B = self._ring_model.background_pixels(Y, ids)
                movie_avg[:, ids] = weights.T @ Y
                background_avg[:, ids] = weights.T @ B
            subtracted_avg = np.asarray((weights.T @ self._A_f_residual) @ self._C_residual, np.float32)
            self._roi_averages_cache = {"background": background_avg,
                                        "residual": movie_avg - subtracted_avg - background_avg}
        return self._roi_averages_cache

    @property
    def fluctuating_background_roi_averages(self) -> torch.Tensor | None:
        if self._ring_model is None:
            return None
        average = self._compute_roi_averages()["background"]
        if self._background_mean_zero:
            average = average - average.mean(axis=1, keepdims=True)
        return torch.from_numpy(average).to(self._device)

    @property
    def residual_roi_averages(self) -> torch.Tensor | None:
        if self._ring_model is None:
            return None
        return torch.from_numpy(self._compute_roi_averages()["residual"]).to(self._device)