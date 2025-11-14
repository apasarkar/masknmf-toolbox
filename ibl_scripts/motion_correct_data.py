import os
import sys
import sys
import masknmf
from masknmf import display
import tifffile
import torch
import numpy as np
import argparse

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



def run_motion_correction(bin_file_path: str | Path,
                          ops_file_path: str | Path,
                          out_path: str | Path,
                          num_blocks_dim1: int,
                          num_blocks_dim2: int,
                          overlaps_dim1: int,
                          overlaps_dim2: int,
                          max_rigid_shifts_dim1: int,
                          max_rigid_shifts_dim2: int,
                          max_deviation_rigid_dim1: int,
                          max_deviation_rigid_dim2: int,
                          frame_batch_size: int):

    data = load_bin_file(ops_file_path, bin_file_path)
    num_blocks = [num_blocks_dim1, num_blocks_dim2]
    overlaps = [overlaps_dim1, overlaps_dim2]
    max_rigid_shifts = [max_rigid_shifts_dim1, max_rigid_shifts_dim2]
    max_deviation_rigid = [max_deviation_rigid_dim1, max_deviation_rigid_dim2]


    pwrigid_strategy = masknmf.PiecewiseRigidMotionCorrector(
        num_blocks=num_blocks,
        overlaps=overlaps,
        max_rigid_shifts=max_rigid_shifts,
        max_deviation_rigid=max_deviation_rigid,
        batch_size=frame_batch_size
    )

    pwrigid_strategy.compute_template(data)
    moco_results = masknmf.RegistrationArray(data, pwrigid_strategy)


    out_path = os.path.abspath(out_path)
    moco_results.export(out_path)


if __name__ == "__main__":
    config_dict = {
        'bin_file_path': '/path/to/data/frames.bin',
        'ops_file_path': '/path/to/ibl_outputs.zip',
        'out_path': '.',
        'num_blocks_dim1': 10,
        'num_blocks_dim2': 10,
        'overlaps_dim1': 5,
        'overlaps_dim2': 5,
        'max_rigid_shifts_dim1': 15,
        'max_rigid_shifts_dim2': 15,
        'max_deviation_rigid_dim1': 2,
        'max_deviation_rigid_dim2': 2,
        'frame_batch_size': 500
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
    run_motion_correction(**final_inputs)