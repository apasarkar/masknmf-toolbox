import os

import sys
import masknmf
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

from masknmf import display

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
        s2p_ops = np.load(self.ops_path, allow_pickle = True)['ops'].item()
        return s2p_ops['nframes'], s2p_ops['Ly'], s2p_ops['Lx']

    def __getitem__(self, item: Union[int, list, np.ndarray, Tuple[Union[int, np.ndarray, slice, range]]]):
        return self.data[item].copy()

def load_bin_file(s2p_zip_path: Union[str, bytes, os.PathLike],
                   alf_bin_path: Union[str, bytes, os.PathLike]) -> np.ndarray:
    my_data = MotionBinDataset(alf_bin_path, s2p_zip_path)[:]
    return my_data


@hydra.main()
def train_denoiser(cfg: DictConfig) -> None:

    my_data = load_bin_file(cfg.zip_file_path, cfg.bin_file_path)
    my_data = my_data[:, 10:-10, 10:-10]
    display(f"post crop the shape is {my_data.shape}")
    block_sizes = [cfg.block_size_dim1, cfg.block_size_dim2]

    device = cfg.device
    if device == 'cpu':
        display("Running PMD to generate training data on CPU")
    
    pmd_obj = masknmf.compression.pmd_decomposition(my_data,
                                                    block_sizes,
                                                    my_data.shape[0],
                                                    max_components = cfg.max_components,
                                                    max_consecutive_failures = cfg.max_consecutive_failures,
                                                    temporal_avg_factor = cfg.temporal_avg_factor,
                                                    spatial_avg_factor = cfg.spatial_avg_factor,
                                                    device = device,
                                                    frame_batch_size = cfg.frame_batch_size)
    
    v = pmd_obj.v.cpu()
    trained_model, _ = masknmf.compression.denoising.train_total_variance_denoiser(v,
                                                           max_epochs = 5,
                                                           batch_size = 128,
                                                            learning_rate=1e-4)
    save_path = os.path.abspath(cfg.outfile)
    np.savez(save_path, model=trained_model)
    display("Results saved successfully")
    

    

if __name__ == "__main__":
    config_dict = {
        'bin_file_path': '/path/to/data/frames.bin',
        'zip_file_path': '/path/to/ibl_outputs.zip',
        'outfile': '/path/to/output.npz',
        'block_size_dim1': 32,
        'block_size_dim2': 32,
        'max_components':20,
        'max_consecutive_failures': 1,
        'spatial_avg_factor': 1,
        'temporal_avg_factor': 1,
        'device': 'cpu',
        'frame_batch_size': 1024,
        ## For training the network:
        'epochs': 5,
        'learning_rate': 1e-4,
    }

    cfg = OmegaConf.create(config_dict)
    cli_conf = OmegaConf.from_cli()
    cfg = OmegaConf.merge(cfg, cli_conf)

    train_denoiser(cfg)