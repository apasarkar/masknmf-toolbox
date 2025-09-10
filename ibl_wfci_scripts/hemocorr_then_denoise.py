import masknmf
import masknmf
import torch
import os
import wfield
import numpy as np
import fastplotlib as fpl
from typing import *
import cv2
from scipy.ndimage import label
import math
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.interpolate import interp1d
from omegaconf import DictConfig, OmegaConf
import hydra
from masknmf import display


class MemmapArray(masknmf.ArrayLike):
    """
    Array-like wrapper around a NumPy memmap file.
    """

    def __init__(self, filepath: str, shape: Tuple[int, int, int], dtype: Union[str, np.dtype], mode: str = "r"):
        """
        Parameters
        ----------
        filepath : str
            Path to the memmap .bin/.dat file
        shape : tuple
            Shape of the array (n_frames, height, width)
        dtype : str or np.dtype
            Data type of the array
        mode : str
            Memmap mode, e.g. 'r' (read-only), 'r+' (read/write), 'w+' (create/overwrite)
        """
        self._filepath = filepath
        self._shape = shape
        self._dtype = np.dtype(dtype)
        self._mm = np.memmap(filepath, dtype=self._dtype, mode=mode, shape=self._shape)

    @property
    def dtype(self) -> np.dtype:
        return self._dtype

    @property
    def shape(self) -> Tuple[int, int, int]:
        return self._shape

    def __getitem__(
            self,
            item: Union[int, list, np.ndarray, Tuple[Union[int, np.ndarray, slice, range]]],
    ):
        return np.asarray(self._mm[item]).copy()

    def __len__(self) -> int:
        return self._shape[0]

    def __array__(self):
        """Allows np.asarray() to work on this object"""
        return np.asarray(self._mm)

    def flush(self):
        """Ensure data is written to disk (if writable)"""
        self._mm.flush()


def write_lazy_to_memmap(lazy_array, out_path, chunk_size=1000, dtype=np.float32):
    """
    Load a lazy array in chunks and write to a numpy memmap on disk.

    Parameters
    ----------
    lazy_array : array-like (lazy, e.g. dask, h5py, zarr, custom loader)
        The source array, must support slicing.
    out_path : str
        Path to the memmap file to be created.
    chunk_size : int
        Number of frames to load and write per iteration.
    dtype : np.dtype
        Data type of the memmap (must match or be compatible).
    """
    # Shape of the array
    n_frames, *spatial_dims = lazy_array.shape

    # Create an empty memmap file
    memmap_arr = np.memmap(out_path, mode="w+", dtype=dtype, shape=(n_frames, *spatial_dims))

    # Loop over chunks with a progress bar
    for start in tqdm(range(0, n_frames, chunk_size), desc="Writing to memmap"):
        end = min(start + chunk_size, n_frames)

        # Load chunk from lazy array
        chunk = lazy_array[start:end]  # triggers actual load

        # Write into memmap at correct slice
        memmap_arr[start:end] = np.asarray(chunk, dtype=dtype)

        # Flush to ensure data is written to disk
        memmap_arr.flush()

    return memmap_arr, (n_frames, spatial_dims[0], spatial_dims[1])  # can reopen later with np.memmap

def per_pixel_lstsq(u_gcamp: torch.sparse_coo_tensor,
                    u_hemo: torch.sparse_coo_tensor,
                    v_signal: torch.tensor,
                    v_hemo_interp: torch.tensor,
                    batch_size: int = 600,
                    device="cpu"):
    u_gcamp = u_gcamp.to(device)
    u_hemo = u_hemo.to(device)
    v_hemo_interp = v_hemo_interp.to(device)
    v_signal = v_signal.to(device)
    num_pixels = u_gcamp.shape[0]
    num_iters = math.ceil(u_gcamp.shape[0] / batch_size)

    outputs = torch.zeros(num_pixels, dtype=u_gcamp.dtype, device=device)
    for k in tqdm(range(num_iters)):
        start_elt = batch_size * k
        end_elt = min(num_pixels, start_elt + batch_size)
        i_select = torch.arange(start_elt, end_elt, device=device).long()
        u_subset_gcamp = torch.index_select(u_gcamp, 0, i_select).coalesce()
        u_subset_hemo = torch.index_select(u_hemo, 0, i_select).coalesce()
        signal = torch.sparse.mm(u_subset_gcamp, v_signal)
        blood = torch.sparse.mm(u_subset_hemo, v_hemo_interp)

        outputs[start_elt:end_elt] = torch.sum(signal * blood, dim=1) / (1e-6 + torch.sum(blood * blood, dim=1))

    return outputs


def compute_blood_basis(u_gcamp: torch.sparse_coo_tensor,
                        u_projector_gcamp: torch.sparse_coo_tensor,
                        u_hemo: torch.sparse_coo_tensor,
                        v_hemo_interp: torch.tensor,
                        coefficients: torch.tensor,
                        device="cpu"):
    u_gcamp = u_gcamp.to(device)
    u_project_gcamp = u_projector_gcamp.to(device)
    u_hemo = u_hemo.to(device)
    v_hemo_interp = v_hemo_interp.to(device)
    cofficients = coefficients.to(device)
    indices = torch.arange(len(coefficients)).unsqueeze(0).repeat(2, 1)  # [[0,1,2..],[0,1,2..]]
    size = (len(coefficients), len(coefficients))
    sparse_diag = torch.sparse_coo_tensor(indices.to(device), coefficients.to(device), size).to(device)

    uts = torch.sparse.mm(u_projector_gcamp.t().to(device), sparse_diag)
    utsu = torch.sparse.mm(uts, u_hemo)

    utsuv = torch.sparse.mm(utsu, v_hemo_interp)

    return utsuv


def hemocorr_basis(blood_basis,
                   gcamp_basis,
                   device='cpu'):
    return gcamp_basis.to(device) - blood_basis.to(device)


class ucla_wf_singlechannel(masknmf.ArrayLike):
    def __init__(self,
                 my_memmap: np.memmap,
                 dtype = np.uint16,
                 channel: int = 0,
                 mask: Optional[np.ndarray] = None,
                 num_frames: Optional[int] = None):
        self._channel = channel
        self._dtype = dtype
        self._mmap = my_memmap[:num_frames] if num_frames is not None else my_memmap
        frames = self._mmap.shape[0]
        self._shape = (self._mmap.shape[0], self._mmap.shape[2], self._mmap.shape[3])
        self._mask = mask.astype(dtype) if mask is not None else None

    @property
    def shape(self):
        return self._shape

    @property
    def dtype(self):
        return self._dtype

    @property
    def ndim(self):
        return 3


    def __getitem__(self, item: Union[int, list, np.ndarray, Tuple[Union[int, np.ndarray, slice, range]]]) -> np.ndarray:
        if isinstance(item, (int, slice, np.ndarray, range)):
            return np.asarray(self._mmap[item, self._channel, :, :]).copy()
        elif isinstance(item, list) or isinstance(item, tuple):
            if len(item) == 1:
                return np.asarray(self._mmap[item[0], self._channel, :, :]).copy()
            elif len(item) == 2:
                return np.asarray(self._mmap[item[0], self._channel, item[1], :]).copy()
            elif len(item) == 3:
                return np.asarray(self._mmap[item[0], self._channel, item[1], item[2]]).copy()


def _run_pipeline(cfg: DictConfig) -> None:
    out_save_dir = os.path.abspath(cfg.out_folder)
    data_path = os.path.abspath(cfg.bin_file_path)
    video_obj = wfield.load_stack(data_path, nchannels=2)
    mask = None
    gcamp_channel = ucla_wf_singlechannel(video_obj, channel=0, mask=mask, num_frames=cfg.num_frames_used)
    blood_channel = ucla_wf_singlechannel(video_obj, channel=1, mask=mask, num_frames=cfg.num_frames_used)


    pixel_weighting = None #Update this with mask later
    block_sizes = [cfg.block_size_dim1, cfg.block_size_dim2]

    pmd_gcamp_no_nn = masknmf.compression.pmd_decomposition(gcamp_channel,
                                                            block_sizes,
                                                            gcamp_channel.shape[0],
                                                            max_components=cfg.max_components,
                                                            max_consecutive_failures=cfg.max_consecutive_failures,
                                                            temporal_avg_factor=10,
                                                            spatial_avg_factor=1,
                                                            device=cfg.device,
                                                            temporal_denoiser=None,
                                                            frame_batch_size=1024,
                                                            pixel_weighting=pixel_weighting)
    np.savez(os.path.join(out_save_dir, "gcamp_no_nn.npz"), pmd=pmd_gcamp_no_nn)

    pmd_hemo_no_nn = masknmf.compression.pmd_decomposition(blood_channel,
                                                           block_sizes,
                                                           blood_channel.shape[0],
                                                           max_components=20,
                                                           max_consecutive_failures=1,
                                                           temporal_avg_factor=1,
                                                           spatial_avg_factor=1,
                                                           device=cfg.device,
                                                           temporal_denoiser=None,
                                                           frame_batch_size=1024,
                                                           pixel_weighting=pixel_weighting)
    np.savez(os.path.join(out_save_dir, "hemo_no_nn.npz"), pmd=pmd_hemo_no_nn)

    pmd_gcamp_no_nn.to(cfg.device)
    pmd_gcamp_no_nn.rescale = False
    pmd_hemo_no_nn.to(cfg.device)
    pmd_hemo_no_nn.rescale = False

    blood_indices = np.array([i * 2 + 1 for i in range(pmd_hemo_no_nn.shape[0])])
    gcamp_indices = np.array([i * 2 for i in range(pmd_gcamp_no_nn.shape[0])])

    t = np.arange(pmd_hemo_no_nn.shape[0] * 2 + 1)
    v_hemo_interp = interp1d(blood_indices,
                             pmd_hemo_no_nn.v.cpu().numpy(),
                             axis=1,
                             fill_value='extrapolate')(
        gcamp_indices)
    v_hemo_interp = torch.from_numpy(v_hemo_interp).to(pmd_hemo_no_nn.v.dtype)

    regression_coefficients = per_pixel_lstsq(pmd_gcamp_no_nn.u,
                                              pmd_hemo_no_nn.u,
                                              pmd_gcamp_no_nn.v,
                                              v_hemo_interp,
                                              batch_size=300,
                                              device="cuda")

    blood_basis = compute_blood_basis(pmd_gcamp_no_nn.u,
                                      pmd_gcamp_no_nn.u_local_projector,
                                      pmd_hemo_no_nn.u,
                                      v_hemo_interp,
                                      regression_coefficients,
                                      device="cpu")

    v_hemocorr = hemocorr_basis(blood_basis,
                                pmd_gcamp_no_nn.v,
                                device='cuda')

    # You really want to subtract the regression coeff * blood channel (that is the estimate of blood in gcamp channel) from raw
    hemo_corr_est = masknmf.PMDArray((pmd_gcamp_no_nn.shape[0], pmd_gcamp_no_nn.shape[1], pmd_gcamp_no_nn.shape[2]),
                                     pmd_gcamp_no_nn.u.cpu(),
                                     blood_basis.cpu(),
                                     pmd_gcamp_no_nn.mean_img.cpu(),
                                     pmd_gcamp_no_nn.var_img.cpu(),
                                     u_local_projector=pmd_gcamp_no_nn.u_local_projector.cpu(),
                                     device="cpu",
                                     )

    np.savez(os.path.join(out_save_dir, "hemo_corr_est.npz"), pmd=hemo_corr_est)

    hemo_corr_est.rescale = True
    hemo_corr_est.to('cuda')

    raw_hemocorr = masknmf.PMDResidualArray(gcamp_channel,
                                            hemo_corr_est)
    # Let's check how the memmap writeout works
    file_writeout = os.path.join(os.path.abspath(cfg.out_folder), "hemocorr_stack.bin")
    _, _ = write_lazy_to_memmap(raw_hemocorr, file_writeout, chunk_size = 1000, dtype = np.float32)

    memmap_load = MemmapArray(file_writeout, gcamp_channel.shape, np.float32, mode="r")

    pmd_hemocorr_no_nn = masknmf.compression.pmd_decomposition(memmap_load,
                                                               block_sizes,
                                                               memmap_load.shape[0],
                                                               max_components=cfg.max_components,
                                                               max_consecutive_failures=cfg.max_consecutive_failures,
                                                               temporal_avg_factor=10,
                                                               spatial_avg_factor=1,
                                                               device=cfg.device,
                                                               temporal_denoiser=None,
                                                               frame_batch_size=1024,
                                                               pixel_weighting=pixel_weighting)

    np.savez(os.path.join(out_save_dir, "pmd_hemocorr_no_nn.npz"), pmd=pmd_hemocorr_no_nn)

    v = pmd_hemocorr_no_nn.v.cpu()
    trained_model, _ = masknmf.compression.denoising.train_total_variance_denoiser(v,
                                                                                   max_epochs=5,
                                                                                   batch_size=128,
                                                                                   learning_rate=1e-4)

    trained_nn_module = masknmf.compression.PMDTemporalDenoiser(trained_model)

    pmd_hemocorr_with_nn = masknmf.compression.pmd_decomposition(memmap_load,
                                                               block_sizes,
                                                               memmap_load.shape[0],
                                                               max_components=cfg.max_components,
                                                               max_consecutive_failures=cfg.max_consecutive_failures,
                                                               temporal_avg_factor=10,
                                                               spatial_avg_factor=1,
                                                               device=cfg.device,
                                                               temporal_denoiser=trained_nn_module,
                                                               frame_batch_size=1024,
                                                               pixel_weighting=pixel_weighting)
    np.savez(os.path.join(out_save_dir, "pmd_hemocorr_with_nn.npz"), pmd=pmd_hemocorr_with_nn)

if __name__ == "__main__":
    config_dict = {
        'bin_file_path': '/path/to/data/frames.bin',
        'mask_file_path': '/path/to/mask/file/',
        'num_frames_used': 60000,
        'out_folder': '/path/to/output/folder/',
        'block_size_dim1': 32,
        'block_size_dim2': 32,
        'max_components': 20,
        'max_consecutive_failures': 1,
        'spatial_avg_factor': 1,
        'temporal_avg_factor': 1,
        'device': 'cpu',
        'frame_batch_size': 1024,
    }

    cfg = OmegaConf.create(config_dict)
    cli_conf = OmegaConf.from_cli()
    cfg = OmegaConf.merge(cfg, cli_conf)

    _run_pipeline(cfg)
