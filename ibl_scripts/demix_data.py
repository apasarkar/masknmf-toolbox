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

from masknmf.utils import display
from datetime import datetime

import argparse


def demix(data_path: str | Path,
          frame_batch_size: int,
          spatial_hp_sigma: tuple[float, float],
          ring_radius: int,
          background_downsampling_factor: int,
          out_path: str | Path,
          device: str) -> None:

    data_file = os.path.abspath(data_path)
    if not os.path.exists(data_file):
        raise ValueError(f"the path {data_file} does not seem to exist")
    pmd_denoise = masknmf.PMDArray.from_hdf5(data_path)

    # Generate a spatially filtered version of the PMD data
    spatial_filt_pmd = masknmf.demixing.filters.spatial_filter_pmd(pmd_denoise,
                                                       batch_size=frame_batch_size,
                                                       filter_sigma=spatial_hp_sigma,
                                                       device = device)

    display("Processing spatially-highpass filtered PMD array")
    device = 'cuda'
    highpass_pmd_demixer = masknmf.demixing.signal_demixer.SignalDemixer(
                                                    spatial_filt_pmd,
                                                    device=device,
                                                    frame_batch_size=frame_batch_size)

    init_kwargs = {
        'mad_correlation_threshold':0.8,

        #Mostly stable
        'mad_threshold':1,
        'residual_threshold': 0.3,
        'patch_size':(40, 40),
    }

    highpass_pmd_demixer.initialize_signals(**init_kwargs, is_custom = False)
    if highpass_pmd_demixer.results is not None:
        display(f"Identified {highpass_pmd_demixer.results[0].shape[1]} candidate neural signals at initialization step.")

    ## Demixing State
    num_iters = 25
    ## Now run demixing...
    localnmf_params = {
        'maxiter':num_iters,
        'support_threshold':np.linspace(0.95, 0.7, num_iters).tolist(),
        'deletion_threshold':0.2,
        'ring_model_start_pt':num_iters + 1,
        'merge_threshold':0.6,
        'merge_overlap_threshold':0.3,
        'update_frequency':4,
        'c_nonneg':True,
        'denoise':False,
        'plot_en': False
    }

    with torch.no_grad():
        highpass_pmd_demixer.demix(**localnmf_params)
    display(f"After demixing the high-pas data, there were {highpass_pmd_demixer.results.a.shape[1]} signals identified")

    display("Using ROIs from the high-pass data to demix the unfiltered PMD data now")

    a_init = highpass_pmd_demixer.results.ac_array.export_a()
    c_init = highpass_pmd_demixer.results.ac_array.export_c()


    device = 'cuda'
    unfiltered_pmd_demixer = masknmf.demixing.signal_demixer.SignalDemixer(
                                                    pmd_denoise,
                                                    device=device,
                                                    frame_batch_size=frame_batch_size)

    unfiltered_pmd_demixer.initialize_signals(is_custom=True,
                                              spatial_footprints=a_init,
                                              temporal_footprints=c_init,
                                              c_nonneg = True)

    ## Demixing State
    num_iters = 25
    ## Now run demixing...
    localnmf_params = {
        'maxiter':num_iters,
        'support_threshold':np.linspace(0.95, 0.5, num_iters).tolist(),
        'deletion_threshold':0.2,
        'ring_model_start_pt':0,
        'ring_radius': ring_radius,
        'background_downsampling_factor': background_downsampling_factor,
        'merge_threshold':0.8,
        'merge_overlap_threshold':0.8,
        'update_frequency':4,
        'c_nonneg':True,
        'denoise':False,
        'plot_en': False
    }

    with torch.no_grad():
        unfiltered_pmd_demixer.demix(**localnmf_params)
    display(f"after this step {unfiltered_pmd_demixer.results.a.shape[1]} signals identified")

    init_kwargs = {
        'mad_correlation_threshold':0.4,
    
        #Mostly stable
        'mad_threshold':1,
        'residual_threshold': 0.3,
        'patch_size':(40, 40),
    }
    
    unfiltered_pmd_demixer.initialize_signals(**init_kwargs, carry_background=True, is_custom = False)
    if unfiltered_pmd_demixer.results is not None:
        display(f"Identified {unfiltered_pmd_demixer.results[0].shape[1]} candidate neural signals at initialization step.")

    num_iters = 25
    ## Now run demixing...
    localnmf_params = {
        'maxiter':num_iters,
        'support_threshold':np.linspace(0.95, 0.2, num_iters).tolist(),
        'deletion_threshold':0.2,
        'ring_model_start_pt':0,
        'ring_radius':ring_radius,
        'background_downsampling_factor': background_downsampling_factor,
        'merge_threshold':0.8,
        'merge_overlap_threshold':0.8,
        'update_frequency':4,
        'c_nonneg':True,
        'denoise':False,
        'plot_en': False
    }

    with torch.no_grad():
        unfiltered_pmd_demixer.demix(**localnmf_params)
    display(f"after this step {unfiltered_pmd_demixer.results.a.shape[1]} signals identified")

    
    out_path = os.path.abspath(out_path)

    display(f"Saving demixing results to {out_path}")
    unfiltered_pmd_demixer.results.export(out_path)

if __name__ == "__main__":
    config_dict = {
        'data_path': '/path/to/data/',
        'out_path': '.',
        'spatial_hp_sigma': 4,
        'frame_batch_size':400,
        'device': 'cpu',
        'background_downsampling_factor': 20,
        'ring_radius': 15,
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
    demix(**final_inputs)