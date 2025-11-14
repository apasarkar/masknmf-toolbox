import os
import sys
import sys
import masknmf
from masknmf import display
import tifffile
import torch
import numpy as np
import argparse

import matplotlib.pyplot as plt
import time
from typing import *
import pathlib
from pathlib import Path

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



def compress_and_denoise(hdf5_file_path: str | Path | None,
                         ops_file_path: str | Path | None,
                         bin_file_path: str | Path | None,
                         block_size_dim1: int,
                         block_size_dim2: int,
                         max_components: int,
                         max_consecutive_failures: int,
                         temporal_avg_factor: int,
                         spatial_avg_factor: int,
                         frame_batch_size: int,
                         noise_variance_quantile: float,
                         out_path: str | Path) -> None:
    if hdf5_file_path is not None:
        my_data = masknmf.RegistrationArray.from_hdf5(hdf5_file_path)
    elif bin_file_path is not None and ops_file_path is not None:
        my_data = load_bin_file(ops_file_path, bin_file_path)
    else:
        raise ValueError("invalid set of file info provided")

    # Zero out border pixels
    binary_mask = np.zeros((my_data.shape[1], my_data.shape[2]), dtype=my_data.dtype)
    binary_mask[3:-3, 3:-3] = 1.0

    block_sizes = [block_size_dim1, block_size_dim2]

    compress_strat = masknmf.CompressDenoiseStrategy(my_data,
                                                     block_sizes=block_sizes,
                                                     max_components=max_components,
                                                     max_consecutive_failures=max_consecutive_failures,
                                                     temporal_avg_factor=temporal_avg_factor,
                                                     spatial_avg_factor=spatial_avg_factor,
                                                     frame_batch_size=frame_batch_size,
                                                     pixel_weighting=binary_mask,
                                                     noise_variance_quantile=noise_variance_quantile
                                                     )

    pmd_denoised = compress_strat.compress()

    out_path = os.path.abspath(out_path)
    pmd_denoised.export(out_path)
    display("Results saved")


if __name__ == "__main__":
    config_dict = {
        'hdf5_file_path': None,
        'bin_file_path': None,
        'ops_file_path': None,
        'out_path': '.',
        'block_size_dim1': 32,
        'block_size_dim2': 32,
        'max_components': 20,
        'max_consecutive_failures': 1,
        'spatial_avg_factor': 1,
        'temporal_avg_factor': 1,
        'frame_batch_size': 1024,
        'noise_variance_quantile': 0.7,
    }

    parser = argparse.ArgumentParser()
    for key in config_dict.keys():
        curr_key = "--"+key
        parser.add_argument("--"+key)

    args = vars(parser.parse_args())
    #Delete none values.
    print(args)
    args = {key:val for key, val in args.items() if val is not None}
    final_inputs = {**config_dict, **args}
    compress_and_denoise(**final_inputs)