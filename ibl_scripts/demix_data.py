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
from masknmf.utils import display
from datetime import datetime


@hydra.main()
def demix(cfg: DictConfig) -> None:
    data_file = os.path.abspath(cfg.data_path)
    if not os.path.exists(data_file):
        raise ValueError(f"the path {data_file} does not seem to exist")
    pmd_denoise = np.load(data_file, allow_pickle=True)[cfg.data_field].item()

    # Generate a spatially filtered version of the PMD data
    spatial_filt_pmd = masknmf.demixing.filters.spatial_filter_pmd(pmd_denoise,
                                                       batch_size=cfg.frame_batch_size,
                                                       filter_sigma=cfg.spatial_hp_sigma,
                                                       device = 'cuda')

    display("Processing spatially-highpass filtered PMD array")
    num_frames, fov_dim1, fov_dim2 = spatial_filt_pmd.shape
    device = 'cuda'
    highpass_pmd_demixer = masknmf.demixing.signal_demixer.SignalDemixer(
                                                    spatial_filt_pmd,
                                                    device=device,
                                                    frame_batch_size=cfg.frame_batch_size)

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

    highpass_pmd_demixer.lock_results_and_continue()
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


    num_frames, fov_dim1, fov_dim2 = pmd_denoise.shape
    device = 'cuda'
    unfiltered_pmd_demixer = masknmf.demixing.signal_demixer.SignalDemixer(
                                                    pmd_denoise,
                                                    device=device,
                                                    frame_batch_size=cfg.frame_batch_size)

    unfiltered_pmd_demixer.initialize_signals(is_custom=True,
                                              spatial_footprints=a_init,
                                              temporal_footprints=c_init,
                                              c_nonneg = True)
    unfiltered_pmd_demixer.lock_results_and_continue()

        ## Demixing State
    num_iters = 25
    ## Now run demixing...
    localnmf_params = {
        'maxiter':num_iters,
        'support_threshold':np.linspace(0.95, 0.5, num_iters).tolist(),
        'deletion_threshold':0.2,
        'ring_model_start_pt':0,
        'ring_radius':cfg.ring_radius,
        'background_downsampling_factor': cfg.background_downsampling_factor,
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


    unfiltered_pmd_demixer.lock_results_and_continue(carry_background=True)

    init_kwargs = {
        'mad_correlation_threshold':0.4,
    
        #Mostly stable
        'mad_threshold':1,
        'residual_threshold': 0.3,
        'patch_size':(40, 40),
    }
    
    unfiltered_pmd_demixer.initialize_signals(**init_kwargs, is_custom = False)
    if unfiltered_pmd_demixer.results is not None:
        display(f"Identified {unfiltered_pmd_demixer.results[0].shape[1]} candidate neural signals at initialization step.")

    unfiltered_pmd_demixer.lock_results_and_continue()

    num_iters = 25
    ## Now run demixing...
    localnmf_params = {
        'maxiter':num_iters,
        'support_threshold':np.linspace(0.95, 0.2, num_iters).tolist(),
        'deletion_threshold':0.2,
        'ring_model_start_pt':0,
        'ring_radius':cfg.ring_radius,
        'background_downsampling_factor': cfg.background_downsampling_factor,
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

    
    out_path = os.path.abspath(cfg.out_path)

    display(f"Saving demixing results to {out_path}")
    config_to_save = OmegaConf.to_container(cfg, resolve=True)
    np.savez(out_path,
             results=unfiltered_pmd_demixer.results,
             metadata = config_to_save)

    
    

if __name__ == "__main__":
    config_dict = {
        'data_path': '/path/to/data/',
        'data_field': 'pmd',
        'out_path': '.',
        'spatial_hp_sigma': 4,
        'frame_batch_size':400,
        'device': 'cpu',
        'background_downsampling_factor': 20,
        'ring_radius': 15,
    }

    cfg = OmegaConf.create(config_dict)
    cli_conf = OmegaConf.from_cli()
    cfg = OmegaConf.merge(cfg, cli_conf)

    demix(cfg)