import os
import sys
import sys
import masknmf
from masknmf import display
import tifffile
import torch
import numpy as np

import matplotlib.pyplot as plt
import time
from typing import *
import pathlib
from pathlib import Path

from omegaconf import DictConfig, OmegaConf
import hydra


class MotionBinDataset:
    """Load a suite2p data.bin imaging registration file."""

    def __init__(self,
                 data_path: Union[str, pathlib.Path],
                 metadata_path: Union[str, pathlib.Path]):
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
            s2p_ops = np.load(self.ops_path, allow_pickle = True)['ops'].item()
        elif ext_path == ".npy":
            s2p_ops = np.load(self.ops_path, allow_pickle = True).item()
        else:
            raise ValueError("The file name should either be zip or npy")
        return s2p_ops['nframes'], s2p_ops['Ly'], s2p_ops['Lx']

    def __getitem__(self, item: Union[int, list, np.ndarray, Tuple[Union[int, np.ndarray, slice, range]]]):
        return self.data[item].copy()

def load_bin_file(s2p_zip_path: Union[str, bytes, os.PathLike],
                   alf_bin_path: Union[str, bytes, os.PathLike]) -> np.ndarray:
    my_data = MotionBinDataset(alf_bin_path, s2p_zip_path)
    return my_data




@hydra.main()
def compress_and_denoise(cfg: DictConfig) -> None:
    my_data = load_bin_file(cfg.ops_file_path, cfg.bin_file_path)

    # Zero out border pixels
    binary_mask = np.zeros((my_data.shape[1], my_data.shape[2]), dtype=my_data.dtype)
    binary_mask[3:-3, 3:-3] = 1.0

    block_sizes = [cfg.block_size_dim1, cfg.block_size_dim2]

    
    pmd_no_denoise = masknmf.compression.pmd_decomposition(my_data,
                                                           block_sizes,
                                                           max_components=cfg.max_components,
                                                           max_consecutive_failures=cfg.max_consecutive_failures,
                                                           temporal_avg_factor=cfg.temporal_avg_factor,
                                                           spatial_avg_factor=cfg.spatial_avg_factor,
                                                           temporal_denoiser=None,  # Turn off denoiser
                                                           frame_batch_size=cfg.frame_batch_size,
                                                           pixel_weighting = binary_mask)

    if cfg.neural_network is not None:
        net_path = os.path.abspath(cfg.neural_network)
        trained_model = np.load(net_path, allow_pickle=True)['model'].item()
        
    else:
        v = pmd_no_denoise.v.cpu()
        trained_model, _ = masknmf.compression.denoising.train_total_variance_denoiser(v,
                                                               max_epochs = 5,
                                                               batch_size = 128,
                                                                learning_rate=1e-4)
    
    curr_temporal_denoiser = masknmf.compression.PMDTemporalDenoiser(trained_model, 0.7)

    
    pmd_denoised = masknmf.compression.pmd_decomposition(my_data,
                                                         block_sizes,
                                                         max_components=cfg.max_components,
                                                         max_consecutive_failures=cfg.max_consecutive_failures,
                                                         temporal_avg_factor=cfg.temporal_avg_factor,
                                                         spatial_avg_factor=cfg.spatial_avg_factor,
                                                         temporal_denoiser=curr_temporal_denoiser,
                                                         frame_batch_size=cfg.frame_batch_size,
                                                         pixel_weighting=binary_mask)
        

    display(
        f"Processing complete. The rank of PMD with denoiser is {pmd_denoised.pmd_rank}. The rank of PMD without denoiser is {pmd_no_denoise.pmd_rank}")

    out_path = os.path.abspath(cfg.out_path)

    pmd_denoised.export(out_path)
    display("Results saved")


if __name__ == "__main__":
    config_dict = {
        'bin_file_path': '/path/to/data/frames.bin',
        'ops_file_path': '/path/to/ibl_outputs.zip',
        'out_path': '.',
        'block_size_dim1': 32,
        'block_size_dim2': 32,
        'max_components': 20,
        'max_consecutive_failures': 1,
        'spatial_avg_factor': 1,
        'temporal_avg_factor': 1,
        'frame_batch_size': 1024,
        'neural_network': None
    }

    cfg = OmegaConf.create(config_dict)
    cli_conf = OmegaConf.from_cli()
    cfg = OmegaConf.merge(cfg, cli_conf)

    compress_and_denoise(cfg)