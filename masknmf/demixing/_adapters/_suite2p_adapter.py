import os
import pathlib
import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch
from tqdm import tqdm
import torch

from masknmf.arrays.array_interfaces import ArrayLike
from masknmf.demixing._base_results import BaseResults
from masknmf.demixing.demixing_arrays import SignalsArray, StaticBackgroundArray
from masknmf.demixing.demixing_utils import (
    ndarray_to_torch_sparse_coo,
    scipy_sparse_to_torch,
    torch_sparse_to_scipy_coo,
)
from masknmf.compression.compression_strategies import CompressStrategy
from masknmf.utils import SparseCOOTensor

class Suite2pResidualArray(ArrayLike):
    """
    Factorized video for the spatial and temporal extracted sources from the data
    """

    def __init__(
            self,
            motion_corrected_data: ArrayLike,
            ac_array: SignalsArray,
            fluctuating_background_array: SignalsArray,
            # Not a typo, this is effectively what the suite2p background model is (it's a per-cell rank-1 correction)
    ):
        """
        Args:
            motion_corrected_data (ArrayLike)
            ac_array (SignalsArray)
            fluctuating_array (FluctuatingBackgroundArray)
        """

        self._motion_corrected_data = motion_corrected_data
        self._ac_array = ac_array
        self._fluctuating_background_array = fluctuating_background_array

        self._shape = self.ac_array.shape

    @property
    def device(self) -> str | torch.device:
        if self.ac_array.device == self.fluctuating_background_array.device:
            return self.ac_array.device
        else:
            raise ValueError("Not all arrays are on same device")

    @property
    def dtype(self) -> str:
        """
        data type, default np.float32
        """
        return self.ac_array.dtype

    @property
    def motion_corrected_data(self) -> ArrayLike:
        return self._motion_corrected_data

    @property
    def ac_array(self) -> SignalsArray:
        return self._ac_array

    @property
    def fluctuating_background_array(self) -> SignalsArray:
        return self._fluctuating_background_array

    @property
    def shape(self) -> tuple[int, int, int]:
        """
        Array shape (n_frames, dims_x, dims_y)
        """
        return self._shape

    @property
    def ndim(self) -> int:
        """
        Number of dimensions
        """
        return len(self.shape)

    def __getitem__(
            self,
            item: int | list | np.ndarray | tuple[int | np.ndarray | slice | range],
    ):
        intermediate = self.ac_array.getitem_tensor(item) + self.fluctuating_background_array.getitem_tensor(item)
        intermediate = intermediate.cpu().numpy()
        final = np.asarray(self.motion_corrected_data[item], dtype=intermediate.dtype) - intermediate
        return final


class MotionBinDataset:
    """Load a suite2p data.bin imaging registration file."""

    def __init__(self,
                 data_path: str | pathlib.Path,
                 metadata_path: str | pathlib.Path):
        """
        Load a suite2p data.bin imaging registration file.

        Parameters
        ----------
        data_path (str, pathlib.Path): The session path containing preprocessed data.
        metadata_path (str, pathlib.Path): The metadata_path to load.
        """
        self.bin_path = Path(data_path)
        self.ops_path = Path(metadata_path)
        self._dtype = np.int16
        self._shape = self._compute_shape()
        self.data = np.memmap(self.bin_path, mode='r', dtype=self.dtype, shape=self.shape)

    @property
    def dtype(self) -> np.dtype:
        return self._dtype

    @property
    def shape(self):
        """
        This property should return the shape of the dataset, in the form: (d1, d2, T) where d1
        and d2 are the field of view dimensions and T is the number of frames.

        Returns
        -------
        (int, int, int)
            The number of y pixels, number of x pixels, number of frames.
        """
        return self._shape

    @property
    def ndim(self):
        return len(self.shape)

    def _compute_shape(self):
        """
        Loads the suite2p ops file to retrieve the dimensions of the data.bin file. This is now lazily loaded from a
        zip file

        Returns
        -------
        (int, int, int)
            number of frames, number of y pixels, number of x pixels.
        """
        _, ext_path = os.path.splitext(self.ops_path)
        if ext_path == ".zip":
            s2p_ops = np.load(self.ops_path, allow_pickle=True)['ops'].item()
        elif ext_path == ".npy":
            s2p_ops = np.load(self.ops_path, allow_pickle=True).item()
        else:
            raise ValueError("The file name should either be zip or npy")
        return s2p_ops['nframes'], s2p_ops['Ly'], s2p_ops['Lx']

    def __getitem__(self, item: int | list | np.ndarray | tuple[int | np.ndarray | slice | range]):
        return self.data[item].copy()


def load_bin_file(s2p_zip_path: str | bytes | os.PathLike,
                  alf_bin_path: str | bytes | os.PathLike) -> np.ndarray:
    my_data = MotionBinDataset(alf_bin_path, s2p_zip_path)
    return my_data


def s2p_setting(ops: dict, old_key: str, new_key: str, default):
    """suite2p <= 0.14 keeps flat keys in ops.npy; newer versions nest them under ops['extraction']."""
    if old_key in ops:
        return ops[old_key]
    return ops.get("extraction", {}).get(new_key, default)


def suite2p_spatial_matrices(stat, Ly: int, Lx: int, allow_overlap: bool = False):
    """
    Three (Ly*Lx, n_rois) scipy CSC matrices on one shared support; pixel index is y * Lx + x (C order),
    matching SignalsArray.
      W  suite2p's readout weights: W.T @ Y reproduces F.npy
      A  signal footprints, A[:, k] = w_k / ||w_k||^2 = lam * lam.sum() / (lam**2).sum(), so W.T @ A = I
      B  neuropil footprints, 1.0 on every ROI pixel, so W.T @ B = I (r*Fneu is a flat offset under the ROI)
    Then W.T @ (Y - A @ (F - r*Fneu) - B @ (r*Fneu)) == 0: suite2p's own extraction finds nothing in the residual.
    """
    if allow_overlap:
        raise NotImplementedError(
            "allow_overlap=True: supports overlap, so signal footprints become W @ inv(W.T @ W) and the flat "
            "neuropil terms would double count in shared pixels")
    rows, cols, w_vals, a_vals = [], [], [], []
    for k, s in enumerate(stat):
        keep = ~np.asarray(s["overlap"], bool) if "overlap" in s else np.ones(len(s["ypix"]), bool)
        lam = np.asarray(s["lam"], np.float64)[keep]
        if lam.size == 0 or lam.sum() <= 0:
            continue  # every pixel shared: suite2p's F for this ROI is 0, so its column stays empty
        w = lam / lam.sum()
        rows.append(np.ravel_multi_index((s["ypix"][keep], s["xpix"][keep]), (Ly, Lx)))
        cols.append(np.full(lam.size, k))
        w_vals.append(w)
        a_vals.append(w / (w @ w))
    rows, cols = np.concatenate(rows), np.concatenate(cols)

    def build(vals):
        return sp.csc_matrix((np.concatenate(vals), (rows, cols)), shape=(Ly * Lx, len(stat)))

    return build(w_vals), build(a_vals), build([np.ones_like(w) for w in w_vals])


class Suite2pResults(BaseResults):

    def __init__(self,
                 folder: str | os.PathLike,
                 device: str | torch.device ="cpu",
                 motion_corrected_data: ArrayLike | None = None,
                 residual_compress: bool = True):
        """
        Assumes that folder contains the following suite2p results:
        F.npy
        Fneu.npy
        stat.npy
        ops.npy
        Args:
            folder (str | os.PathLike): The folder containing suite2p results
            device (str | torch.device): Device on which data should be stored/used for computation
            motion_corrected_data (ArrayLike | None): An array for the motion corrected data, optional
            residual_compress (bool): If the motion corrected data is provided, the residual (formed by subtracting the suite2p
                signal and background estimates from the motion correction) is compressed using the masknmf vanilla compressor,
                which runs a patchwise adaptive PCA to denoise the data. The goal here is to expose any remaining signal and demixing errors
        """
        folder = os.path.abspath(folder)
        self._temporal_demixed = torch.from_numpy(np.load(os.path.join(folder, "F.npy"), allow_pickle=True))
        self._temporal_neuropil_demixed = torch.from_numpy(np.load(os.path.join(folder, "Fneu.npy"), allow_pickle=True))
        self.stat = np.load(os.path.join(folder, "stat.npy"), allow_pickle=True)
        self.ops = np.load(os.path.join(folder, 'ops.npy'), allow_pickle=True).item()

        # Now let's do neuropil correction (suite2p <= 0.14: ops['neucoeff']; 1.x: ops['extraction']['neuropil_coefficient'])
        neucoeff = float(s2p_setting(self.ops, "neucoeff", "neuropil_coefficient", 0.7))
        self._temporal_demixed = self._temporal_demixed - neucoeff * self._temporal_neuropil_demixed
        # Define the neuropil estimate (at each cell) as a scaled version of the neuropil mask ROI average:
        self._temporal_neuropil_demixed = neucoeff * self._temporal_neuropil_demixed

        self._temporal_demixed = self._temporal_demixed.T
        self._temporal_neuropil_demixed = self._temporal_neuropil_demixed.T

        self.make_masks_from_suite2p_statfile()

        print("DONE")
        self._signals_array = SignalsArray.from_tensors(self.shape[1:],
                                                                self.spatial_demixed,
                                                                self.temporal_demixed)

        print("DONE")
        self._fluctuating_background_array = SignalsArray.from_tensors(self.shape[1:],
                                                                               self._neuropil_spatial_demixed,
                                                                               self._temporal_neuropil_demixed)

        self._motion_corrected_data = motion_corrected_data
        self._temporal_demixed_raw = None  # (num_frames, num_neurons), the registered movie's ROI averages
        if self._motion_corrected_data is not None:
            self._temporal_demixed_raw = torch.from_numpy(
                registered_roi_averages(self._motion_corrected_data, self._spatial_scipy)).T

            residual_array = Suite2pResidualArray(self.motion_corrected_data,
                                                  self.signals_array,
                                                  self.fluctuating_background_array)
            if residual_compress:
                compress_strat = CompressStrategy(block_sizes=[32, 32])
                residual_array = compress_strat.compress(residual_array)
            self._residual_array = residual_array

        else:
            self._temporal_demixed_raw = None
            self._residual_array = None

        self._device = torch.device(device)
        self.to(self._device)

    ## Optimizing with flyweights is overkill since this is mainly a diagnostic tool
    def to(self, new_device: str | torch.device):
        updated_device = torch.device(new_device)
        self._temporal_demixed = self._temporal_demixed.to(updated_device)
        self._temporal_neuropil_demixed = self._temporal_neuropil_demixed.to(updated_device)
        self._spatial_demixed = self._spatial_demixed.to(updated_device)
        self._neuropil_spatial_demixed = self._neuropil_spatial_demixed.to(updated_device)
        self._signal_roi_scale = self._signal_roi_scale.to(updated_device)
        self._background_roi_scale = self._background_roi_scale.to(updated_device)
        self._signals_array.to(updated_device)
        self._fluctuating_background_array.to(updated_device)
        if self._temporal_demixed_raw is not None:
            self._temporal_demixed_raw = self._temporal_demixed_raw.to(updated_device)
        self._device = updated_device

    @property
    def device(self) -> str | torch.device:
        return self._device

    @property
    def spatial_demixed(self) -> SparseCOOTensor:
        return self._spatial_demixed

    @property
    def shape(self) -> tuple[int, int, int]:
        return (self._temporal_demixed.shape[0], self.ops['Ly'], self.ops['Lx'])

    @property
    def shifts(self) -> np.ndarray | None:
        """
        suite2p's registration as the (height, width) shift applied to each frame, shape (num_frames, 2), or with
        nonrigid registration to each block of it, shape (num_frames, height blocks, width blocks, 2). suite2p
        stores how far each frame sat from the reference (yoff, xoff) and how far each block still sat after that
        was corrected (yoff1, xoff1), so the shift applied is minus their sum. None when suite2p did not register
        """
        if 'yoff' not in self.ops:
            return None
        rigid = np.stack([self.ops['yoff'], self.ops['xoff']], axis=-1).astype(np.float32)
        if not self.ops['nonrigid']:
            return -rigid
        # suite2p's blocks run row by row, ceil(1.5 * L / block size) of them along an axis (nonrigid.calculate_nblocks)
        blocks = [1 if size >= L else int(np.ceil(1.5 * L / size)) for L, size in
                  zip(self.shape[1:], self.ops['block_size'])]
        nonrigid = np.stack([self.ops['yoff1'], self.ops['xoff1']], axis=-1).reshape(-1, *blocks, 2)
        return -(rigid[:, None, None] + nonrigid)

    @property
    def temporal_demixed(self) -> torch.Tensor:
        return self._temporal_demixed

    @property
    def temporal_demixed_raw(self) -> torch.Tensor | None:
        return self._temporal_demixed_raw

    @property
    def motion_corrected_data(self) -> ArrayLike | None:
        return self._motion_corrected_data

    @property
    def global_residual_correlation_image(self) -> torch.Tensor | None:
        return None

    @property
    def signals_array(self) -> SignalsArray | None:
        """Across pipelines this term should be a factorized product (spatial components @ temporal components)"""
        return self._signals_array

    @property
    def fluctuating_background_array(self) -> SignalsArray | None:
        return self._fluctuating_background_array

    @property
    def residual_array(self) -> ArrayLike | None:
        return self._residual_array

    # ROI averages, all (num_neurons, num_frames): each movie averaged uniformly over the support of
    # spatial_demixed (roi_average_operator). The supports don't share pixels, so a footprint only contributes to
    # its own ROI's average, scaled by (sum of its values over the support) / (support size).
    @property
    def signal_roi_averages(self) -> torch.Tensor:
        return self._signal_roi_scale[:, None] * self._temporal_demixed.T

    @property
    def fluctuating_background_roi_averages(self) -> torch.Tensor:
        # The neuropil footprint is 1.0 on the support, so this is r*Fneu itself (0 for an ROI with no support)
        return self._background_roi_scale[:, None] * self._temporal_neuropil_demixed.T

    @property
    def residual_roi_averages(self) -> torch.Tensor | None:
        """Registered movie minus signal minus background, ROI-averaged. Needs motion_corrected_data (data.bin)."""
        if self._temporal_demixed_raw is None:
            return None
        return self._temporal_demixed_raw.T - self.signal_roi_averages - self.fluctuating_background_roi_averages

    def make_masks_from_suite2p_statfile(self):
        Ly, Lx = int(self.ops["Ly"]), int(self.ops["Lx"])
        allow_overlap = bool(s2p_setting(self.ops, "allow_overlap", "allow_overlap", False))
        self._readout_weights, A, B = suite2p_spatial_matrices(self.stat, Ly, Lx, allow_overlap)
        self._spatial_scipy = A  # for registered_roi_averages
        self._spatial_demixed = scipy_sparse_to_torch(A).coalesce()
        self._neuropil_spatial_demixed = scipy_sparse_to_torch(B).coalesce()

        # Per-ROI factors turning a temporal trace into its ROI average (see the ROI-average properties)
        support_size = np.maximum((A != 0).sum(axis=0).A1, 1)
        self._signal_roi_scale = torch.from_numpy((A.sum(axis=0).A1 / support_size).astype(np.float32))
        self._background_roi_scale = torch.from_numpy((B.sum(axis=0).A1 / support_size).astype(np.float32))


# ---- trace panel: the viewer averages each movie uniformly over the support of spatial_demixed ------
def roi_average_operator(A: sp.spmatrix) -> sp.csr_matrix:
    support = (A != 0).astype(np.float32).T.tocsr()
    return sp.diags(1.0 / np.maximum(support.sum(axis=1).A1, 1)) @ support


def registered_roi_averages(registered, A: sp.spmatrix, batch: int = 1000) -> np.ndarray:
    """
    One pass over data.bin -> (n_rois, n_frames): the registered-movie line, averaged the way the viewer
    averages the other movies. With the footprints above, the background's average is exactly r*Fneu, the
    signal's is (A.sum(0).A1 / npix)[:, None] * (F - r*Fneu), and residual = this - signal - background.
    """
    R = roi_average_operator(A)
    n_frames = registered.shape[0]
    out = np.empty((A.shape[1], n_frames), np.float32)
    for t0 in tqdm(range(0, n_frames, batch)):
        Y = np.asarray(registered[t0:t0 + batch], np.float32).reshape(-1, A.shape[0])
        out[:, t0:t0 + Y.shape[0]] = R @ Y.T
    return out
