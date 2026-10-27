"""
miniALI results (``ALI_Result_new.mat``, written by mini_ali_run / mini_ali_8bit_script with ``-v7.3``) as a masknmf
results object

miniALI's model of the movie, which the arrays here reproduce:
    registered[t] = raw[t] - dfdx * shift[0, t] - dfdy * shift[1, t]      (mini_ali_regenerateMotionCorrectedSubmovie)
    registered    = Background + footprints @ cell_traces + residual        (mini_ali_trace_generator)

For negatively tuned indicators we invert this appropriately

Conventions on the masknmf side: frames come first, a pixel (row, col) is index row * width + col (C order, as in
SignalsArray), and every index is 0-based. MATLAB's 1-based, column-major indices are converted on load.
"""
import os
import warnings
from pathlib import Path
from typing import Literal

import h5py
import numpy as np
import scipy.sparse as sp
import torch
from tqdm import tqdm

from masknmf.arrays.array_interfaces import ArrayLike
from masknmf.arrays import OphysArray
from masknmf.arrays import StaticArray
from masknmf.motion_correction import GradientRegistrationArray, GradientMotionCorrector
from masknmf.compression.compression_strategies import CompressStrategy
from masknmf.demixing._base_results import BaseResults
from masknmf.demixing.demixing_arrays import SignalsArray
from masknmf.demixing.demixing_utils import scipy_sparse_to_torch
from masknmf.utils import SparseCOOTensor, torch_select_device
RESULT_FILE_NAME = "ALI_Result_new.mat"

# ---- reading MATLAB -v7.3 files ----------------------------------------------------------------------------------
def _matlab_array(dataset: h5py.Dataset) -> np.ndarray:
    """
    A numeric MATLAB variable as MATLAB sees it: h5py holds an m x n MATLAB array as (n, m), so the axes are
    reversed. MATLAB writes an empty array as its size vector with a MATLAB_empty attribute.
    """
    if dataset.attrs.get("MATLAB_empty", 0):
        return np.zeros(tuple(int(d) for d in np.asarray(dataset[()]).ravel()))
    data = np.asarray(dataset[()])
    return data.transpose(tuple(reversed(range(data.ndim))))


def _matlab_cell(file: h5py.File, dataset: h5py.Dataset) -> list:
    """The elements of a MATLAB cell array, in MATLAB's linear (column-major) order, each an h5py object."""
    if dataset.attrs.get("MATLAB_empty", 0):
        return []
    refs = _matlab_array(dataset)
    return [file[ref] for ref in refs.ravel(order="F")]


def _optional(file: h5py.File, name: str) -> np.ndarray | None:
    return _matlab_array(file[name]) if name in file else None


def _footprint_pixels(idx: np.ndarray,
                      values: np.ndarray,
                      fov_shape: tuple[int, int],
                      fix_orientation: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """
    Made via claude, fixes a bug in the matlab code. Context here is that ALI outputs per ROI a rectangular patch
    of nonzero pixels.
    miniALI builds idx with meshgrid(selection_x, selection_y), which runs over the columns of the patch fastest,
    but takes values from footprint(:), which runs over its rows fastest. So as saved, value k lands on the
    patch pixel mirrored across the diagonal (scrambled when an image edge clips the patch to a non-square), and
    mini_ali_trace_generator fit cell_traces with footprints in that layout. fix_orientation puts each value
    back on the pixel it was averaged at instead.
    """
    nrow, ncol = fov_shape
    rows, cols = idx % nrow, idx // nrow  # MATLAB linear index (0-based here) is column-major
    if not fix_orientation:
        return rows * ncol + cols, values
    patch_rows, patch_cols = np.unique(rows), np.unique(cols)
    if patch_rows.size * patch_cols.size != idx.size:
        raise ValueError("a footprint's pixels do not form the rectangular patch miniALI saves")
    image = values.reshape(patch_rows.size, patch_cols.size, order="F")  # footprint(:) is column-major
    return (patch_rows[:, None] * ncol + patch_cols[None, :]).ravel(), image.ravel()


def mean_invert_temporal_vectors(temporal_data: torch.Tensor):
    """
    Use case: you have temporal vectors (num_frames, num_vectors), and you want to invert each vector with respect
    to its mean
    Args:
        temporal_data (torch.Tensor): Shape (num_frames, num_vectors). The data to invert (column-wise)
    """
    means = torch.mean(temporal_data, dim = 0, keepdim=True)
    return 2 * means - temporal_data



def miniali_background(reference_image: np.ndarray) -> np.ndarray:
    """
    miniALI's static background, recomputed from the reference image the way mini_ali_run does it:
    imgaussfilt(imopen(reference_image, strel('disk', 50)), 20). Approximate: MATLAB's strel('disk', 50) and
    scikit-image's decomposed disk are both polygonal approximations of a disk, and not the same one. Save
    "Background" from MATLAB to use the exact one.
    """
    from scipy.ndimage import gaussian_filter
    from skimage.morphology import disk, opening

    opened = opening(np.asarray(reference_image, np.float32), disk(50, decomposition="sequence"))
    # imgaussfilt(_, 20): a kernel of 2 * ceil(2 * sigma) + 1 taps with replicated edges
    return gaussian_filter(opened, sigma=20, truncate=2.0, mode="nearest").astype(np.float32)


def _as_numpy(frames) -> np.ndarray:
    if isinstance(frames, torch.Tensor):
        return frames.detach().cpu().numpy()
    return np.asarray(frames)

class MiniALIResidualArray(ArrayLike):
    """registered - background - signals: what miniALI's model leaves unexplained."""

    def __init__(self, registered: GradientRegistrationArray, background: StaticArray, signals: SignalsArray):
        self._registered = registered
        self._background = background
        self._signals = signals

    @property
    def dtype(self) -> torch.dtype:
        return torch.float32

    @property
    def shape(self) -> tuple[int, int, int]:
        return self._signals.shape

    def __getitem__(self, item: int | list | np.ndarray | tuple[int | np.ndarray | slice | range]) -> torch.Tensor:
        explained = self._background[item] + self._signals[item]
        output = torch.as_tensor(self._registered[item], dtype=explained.dtype, device=explained.device) - explained
        return output.to(self.dtype)


class MiniALIResults(BaseResults):

    def __init__(self,
                 path: str | os.PathLike,
                 indicator_polarity: Literal["positive", "negative"],
                 good_frame_indicator: np.ndarray | torch.Tensor,
                 raw_video: ArrayLike | np.ndarray,
                 device: str | torch.device = "auto",
                 background: np.ndarray | None = None,
                 fix_footprint_orientation: bool = False,
                 residual_compress: bool = False,
                 frame_batch_size: int = 256):
        """
        Args:
            path (str | os.PathLike): miniALI's ALI_Result_new.mat, or the folder holding it. It must hold footprint,
                cell_traces, reference_image, dfdx, dfdy and shift (mini_ali_run saves all of them).
            indicator_polarity (Literal["positive", "negative"]): Required specification whether data is negatively or positively tuned.
                If the data is negatively tuned, this class will "invert" the results so that when viewed, positive fluctuations indicate activity
            good_frame_indicator (np.ndarray | torch.Tensor): 1D boolean tensor or numpy.ndarray, length equal to number of frames
                Value 1 if the frame is included in the analysis, 0 if it is not included in the analysis.
            raw_video (ArrayLike | np.ndarray): The movie miniALI read, shape (frames, height, width). This should be ALL frames, even the
                ones that were not analyzed. This is a required parameter because so many voltage imaging analyses carefully required validation
                with respect to the raw data.
            device (str | torch.device): Where the tensors live
            background (np.ndarray | None): miniALI's static background, shape (height, width). When None, the file's
                "Background" is used if it saved one (add it to the save call to get the exact one), else it is
                recomputed from reference_image (see miniali_background).
            fix_footprint_orientation (bool): Put each footprint value on the pixel it was averaged at; miniALI saves
                the patch mirrored across its diagonal (see _footprint_pixels). Off by default because cell_traces
                were fit with the footprints as saved.
            residual_compress (bool): Compress the residual movie with masknmf's vanilla compressor, to expose
                signal miniALI missed. Needs raw_movie, and loads the whole residual onto the device at once.
            frame_batch_size (int): Frames read at a time for the registered movie's ROI averages
        """
        self._device = torch_select_device(device)

        path = Path(path)
        if not path.is_file():
            raise ValueError(f"path: {path} either does not exist or is not a file")
        self._path = path

        if indicator_polarity.lower() in {"positive", "negative"}:
            self._indicator_polarity = indicator_polarity.lower()
        else:
            raise ValueError("indicator polarity can only be positive or negative, nothing else")

        good_frame_indicator = torch.as_tensor(good_frame_indicator).bool()
        self._num_frames_analyzed = int(torch.count_nonzero(good_frame_indicator))
        self._num_frames_raw = int(good_frame_indicator.numel())

        self._raw_array = OphysArray(raw_video,
                                     negative_indicator = True if self._indicator_polarity == "negative" else False,
                                     device=self.device,
                                     good_frames = good_frame_indicator)

        #This code loads the gradients,
        with h5py.File(path, "r") as f:
            missing = [name for name in ("footprint", "cell_traces", "reference_image", "dfdx", "dfdy", "shift")
                       if name not in f]
            if missing:
                raise ValueError(f"{path} holds no {', '.join(missing)}; is it miniALI's standard .mat file output")
            self._reference_image = _matlab_array(f["reference_image"]).astype(np.float32)  # (height, width)
            dfdx = _matlab_array(f["dfdx"]).astype(np.float32)
            dfdy = _matlab_array(f["dfdy"]).astype(np.float32)
            self._gradient = torch.as_tensor(np.stack([dfdx, dfdy], axis=-1), dtype=torch.float32)  # (height, width, 2)
            self._shifts = torch.as_tensor(_matlab_array(f["shift"]).astype(np.float32).T, dtype=torch.float32)  # MATLAB 2 x frames -> (frames, 2)
            if self.indicator_polarity == "negative":
                self._shifts = mean_invert_temporal_vectors(self._shifts) ## This will give gradient shifts for the inverted dataset

            if self._shifts.shape[0] != self.num_frames_analyzed:
                raise ValueError(f"The number of shifts, {int(self._shifts.shape[0])} does not match the number"
                                 f"of frames analyzed {self._num_frames_analyzed}")


            footprints = [[_matlab_array(part).ravel(order="F") for part in _matlab_cell(f, entry)]
                          for entry in _matlab_cell(f, f["footprint"])]
            self._num_sources = len(footprints)
            traces = _matlab_array(f["cell_traces"]).astype(np.float32)  # MATLAB num_cells x frames
            traces = traces.reshape(self.num_sources, self.num_frames_analyzed) if traces.size == 0 else traces

            saved_background = _optional(f, "Background")

        if traces.shape != (self.num_sources, self.num_frames_analyzed):
            raise ValueError(f"cell_traces has shape {traces.shape}; {traces.shape[0]} footprints and "
                             f"{traces.shape[1]} shifts need ({self.num_sources}, {self.num_frames_analyzed})")
        if self._gradient.shape[:2] != self.fov_shape:
            raise ValueError(f"dfdx has shape {self._gradient.shape[:2]}, reference_image {self.fov_shape}")

        self._fix_footprint_orientation = fix_footprint_orientation

        # footprints: MATLAB {idx (1-based, column-major), values} per cell -> (pixels, cells), zeros dropped
        rows, cols, vals = [], [], []
        for k, (idx, values) in enumerate(footprints):
            pixels, values = _footprint_pixels(idx.astype(np.int64) - 1, values.astype(np.float32), self.fov_shape,
                                               fix_footprint_orientation)
            keep = values != 0
            rows.append(pixels[keep])
            cols.append(np.full(int(keep.sum()), k, np.int64))
            vals.append(values[keep])
        num_pixels = self.fov_shape[0] * self.fov_shape[1]
        self._spatial_scipy = sp.csc_matrix(
            (np.concatenate(vals) if vals else np.zeros(0, np.float32),
             (np.concatenate(rows) if rows else np.zeros(0, np.int64),
              np.concatenate(cols) if cols else np.zeros(0, np.int64))),
            shape=(num_pixels, len(footprints)), dtype=np.float32)
        self._spatial_demixed = scipy_sparse_to_torch(self._spatial_scipy).coalesce().float()
        self._temporal_demixed = torch.as_tensor(traces).T # (frames, cells)
        if self.indicator_polarity == "negative":
            self._temporal_demixed = mean_invert_temporal_vectors(self._temporal_demixed) #(frames, cells)

        if background is None:
            background = saved_background if saved_background is not None else miniali_background(self._reference_image)
        self._background = np.asarray(background, np.float32)
        if self._background.shape != self.fov_shape:
            raise ValueError(f"background has shape {self._background.shape}, the field of view is {self.fov_shape}")

        # Averages over each footprint's support, (cells, frames), as the viewer's trace panel draws them
        self._roi_average = roi_average_operator(self._spatial_scipy)
        self._signal_roi_averages = torch.from_numpy(
            np.asarray((self._roi_average @ self._spatial_scipy) @ traces, np.float32))
        self._background_roi_averages = torch.from_numpy(
            (self._roi_average @ self._background.ravel()).astype(np.float32))

        self._registered_array = None
        self._residual_array = None
        self._registered_roi_averages = None


        """
        Gradient Registration Array has (among others) these inputs: shifts and gradient steps.
        The gradient steps are actually applied to the gradient image to produce the per-frame motion correction
        The ``steps`` are a photobleaching-normalized version of the gradient steps that gives the user a sense for 
        the size of shifts in each frame (in arbitrary units). miniALI does not generate this, so we just 
        provide the regular gradient steps in place of the gradient shifts.
        """
        self._registered_array = GradientRegistrationArray(self.raw_array,
                                                           self._shifts,
                                                           self._gradient.reshape(-1, 2),
                                                           self._shifts)
        self._signals_array = SignalsArray.from_tensors(self.fov_shape, self._spatial_demixed, self._temporal_demixed)
        self._background_array = StaticArray(self._background, self.shape[0])

        residual = MiniALIResidualArray(self._registered_array, self._background_array, self._signals_array)
        if residual_compress:
            residual = CompressStrategy(block_sizes=[32, 32]).compress(residual)
        self._residual_array = residual


        self._device = torch.device(device)
        self.to(self._device)

    def to(self, new_device: str | torch.device):
        device = torch.device(new_device)
        self._spatial_demixed = self._spatial_demixed.to(device)
        self._temporal_demixed = self._temporal_demixed.to(device)
        self._signal_roi_averages = self._signal_roi_averages.to(device)
        self._background_roi_averages = self._background_roi_averages.to(device)
        self._signals_array.to(device)
        self._background_array.to(device)
        self._device = device


    @property
    def indicator_polarity(self) -> Literal["positive", "negative"]:
        return self._indicator_polarity.lower()

    @property
    def raw_array(self) -> OphysArray:
        """
        This is the raw movie, temporally restricted to the frames that were actually analyzed.
        If the indicator is negative, this provides an inverted view for visualization purposes

        """
        return self._raw_array

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def path(self) -> Path:
        return self._path

    @property
    def shape(self) -> tuple[int, int, int]:
        """
        The shape of the movie that was analyzed (ignoring the excluded frames as specified in good_frames_indicator)
        :return:
        """
        return self.raw_array.shape

    @property
    def fov_shape(self) -> tuple[int, int]:
        return self.shape[1:]

    @property
    def num_frames_raw(self) -> int:
        return self._num_frames_raw

    @property
    def num_frames_analyzed(self) -> int:
        return self._num_frames_analyzed

    @property
    def num_sources(self) -> int:
        return self._num_sources

    @property
    def spatial_demixed(self) -> SparseCOOTensor:
        """Footprints, (pixels, cells), pixel row * width + col. miniALI's are the mean -dF of each cluster's spike frames."""
        return self._spatial_demixed

    @property
    def temporal_demixed(self) -> torch.Tensor:
        """
        cell_traces, (frames, cells): the registered movie minus the background regressed on each footprint.
        Spikes are as recorded (negative going for a negative indicator); negate to compare with masknmf traces.
        """
        return self._temporal_demixed

    @property
    def temporal_demixed_raw(self) -> None:
        """None: miniALI's traces are already fit to the registered raw movie, with no compression or denoising."""
        return None

    @property
    def reference_image(self) -> np.ndarray:
        """(height, width), the mean of the movie's first 300 frames."""
        return self._reference_image

    @property
    def device(self) -> torch.device | str:
        """
        The device on which slices of the array results are returned, also used for computational purposes
        """
        return self._device

    @property
    def background_image(self) -> np.ndarray:
        """(height, width), the static background subtracted before fitting the traces."""
        return self._background

    @property
    def gradient(self) -> np.ndarray:
        """(height, width, 2), the reference image's dfdx (along columns) and dfdy (along rows)."""
        return self._gradient

    @property
    def shifts(self) -> np.ndarray:
        """
        (frames, 2), the coefficients of dfdx and dfdy subtracted from each frame. Same layout as
        GradientRegistrationArray.shifts; miniALI fits them by least squares to the raw frames (A \\ frames), where
        masknmf projects out the template and rescales by each frame's brightness, so the two differ in scale.
        """
        return self._shifts

    @property
    def registered_array(self) -> OphysArray:
        return self._registered_array

    @property
    def global_residual_correlation_image(self) -> torch.Tensor | None:
        return None

    @property
    def signals_array(self) -> SignalsArray:
        """footprints @ traces, as a movie"""
        return self._signals_array

    @property
    def fluctuating_background_array(self) -> StaticArray:
        """miniALI's background is static: background_image on every frame."""
        return self._background_array

    @property
    def residual_array(self) -> ArrayLike | None:
        """registered - background - signals; None without raw_movie."""
        return self._residual_array

    # ROI averages, all (cells, frames): each movie averaged uniformly over the support of spatial_demixed. Unlike
    # suite2p's, miniALI's 13 x 13 footprints can overlap, so the signal average includes neighbors.
    @property
    def signal_roi_averages(self) -> torch.Tensor:
        return self._signal_roi_averages

    @property
    def fluctuating_background_roi_averages(self) -> torch.Tensor:
        return self._background_roi_averages[:, None].expand(-1, self.shape[0])

    @property
    def residual_roi_averages(self) -> torch.Tensor | None:
        if self._registered_roi_averages is None:
            return None
        return self._registered_roi_averages - self._signal_roi_averages - self._background_roi_averages[:, None]

def roi_average_operator(A: sp.spmatrix) -> sp.csr_matrix:
    """(cells, pixels): averages a flattened image uniformly over each column of A's nonzero support."""
    support = (A != 0).astype(np.float32).T.tocsr()
    return sp.diags(1.0 / np.maximum(support.sum(axis=1).A1, 1)) @ support