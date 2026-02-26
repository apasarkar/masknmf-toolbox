import torch

import masknmf
from masknmf.motion_correction import RigidMotionCorrector
from masknmf.compression import CompressStrategy, CompressDenoiseStrategy
from masknmf.arrays import LazyFrameLoader, ArrayLike
from masknmf.motion_correction import RegistrationArray
from masknmf.utils import display
from typing import *
import numpy as np
import os

def widefield_singlechannel_pipeline(data: Union[np.ndarray, LazyFrameLoader, ArrayLike],
                            max_rigid_shifts: Tuple[int, int],
                            compression_block_sizes: Tuple[int, int] = [50, 50],
                            keep_intermediates: bool = True,
                            outpath_motion_correction: Optional[str] = None,
                            outpath_compression: Optional[str] = None,
                            load_into_ram: bool = True,
                            binary_compression_mask: Optional[np.ndarray] = None,
                            frame_batch_size = 300) -> Tuple[masknmf.RegistrationArray, masknmf.PMDArray]:
    """
    Pipeline for motion correcting, compressing + denoising widefield single-channel data

    Args:
        data : The (frames, height, width) imaging data

    Returns:
        - masknmf.RegistrationArray: An array describing the motion correction output
        - masknmf.PMDArray: An array describing the compression output
    """

    display("Motion Correction")
    max_rigid_shifts = max_rigid_shifts
    moco_strat = RigidMotionCorrector(max_rigid_shifts,
                                              batch_size=frame_batch_size)
    moco_strat.compute_template(data)

    if outpath_motion_correction is None:
        outpath_motion_correction = os.path.abspath("moco.hdf5")

    moco_results = RegistrationArray(data, moco_strat)
    if load_into_ram:
        if keep_intermediates:
            moco_results.export(outpath_motion_correction)
            compression_input = masknmf.RegistrationArray.from_hdf5(outpath_motion_correction)[:]
        else:
            compression_input = moco_results[:]
    else:
        moco_results.export(outpath_motion_correction)
        if keep_intermediates:
            #In this we don't have to worry about deleting the file
            moco_results = masknmf.RegistrationArray.from_hdf5(outpath_motion_correction)
            compression_input = moco_results
        else:
            compression_input = masknmf.RegistrationArray.from_hdf5(outpath_motion_correction) #Allows fast loadin in all dims now

    display("Now compressing and denoising the data")
    compress_strat = masknmf.CompressDenoiseStrategy(compression_input,
                                                     block_sizes=compression_block_sizes,
                                                     max_components= 20,
                                                     max_consecutive_failures=1,
                                                     temporal_avg_factor=1,
                                                     spatial_avg_factor=1,
                                                     frame_batch_size=frame_batch_size,
                                                     pixel_weighting=binary_compression_mask,
                                                     noise_variance_quantile=0.3,
                                                     num_epochs=10)
    pmd_result = compress_strat.compress()
    if outpath_compression is None:
        outpath_compression = os.path.abspath("compressed_widefield_calcium.hdf5")
    else:
        outpath_compression = os.path.abspath(outpath_compression)
    display(f"Exporting to {outpath_compression}")
    pmd_result.export(outpath_compression)

    if os.path.exists(outpath_motion_correction) and not keep_intermediates:
        os.remove(outpath_motion_correction)
        display("Removed motion corrected intermediate file")
    return moco_results, pmd_result


