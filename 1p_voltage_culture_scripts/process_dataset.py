import masknmf
from masknmf import utils
from masknmf import display
import os
import torch
import numpy as np
import sys
from tqdm import tqdm
import math
import fastplotlib as fpl
import h5py
from omegaconf import DictConfig, OmegaConf
import hydra
from typing import *

def get_fov_shape(mat_file):
    with h5py.File(mat_file, "r") as f:
        # 1) Check if 'Device_Data' is in the file
        if "Device_Data" not in f:
            raise ValueError("'Device_Data' not found in the .mat file.")

        device_data = f["Device_Data"]

        # 2) Read the cell array references 
        device_data_refs = device_data[()] 

        # 3) Find the 'Camera' entry
        camera_entry = None
        rows, cols = device_data_refs.shape
        for i in range(rows):
            ref = device_data_refs[i, 0]  
            entry = f[ref]  # dereference the actual group

            if "deviceType" not in entry:
                continue

            device_type_data = entry["deviceType"][()]  # read the data
            device_type_data = device_type_data.flatten()
            device_type = "".join(chr(x) for x in device_type_data)

            if device_type == "Camera":
                camera_entry = entry
                break

        if camera_entry is None:
            raise ValueError("No entry in Device_Data has deviceType == 'Camera'.")

        # 4) Extract ROI dimensions: ROI(2) -> index 1, ROI(4) -> index 3
        roi_data = camera_entry["ROI"][()]  # e.g. shape (4,1) or (1,4)
        roi_data = roi_data.flatten()
        dim1 = int(roi_data[3])
        dim2 = int(roi_data[1])
        frames_requested = int(camera_entry["frames_requested"][()])
        return frames_requested, dim1, dim2


def _crop_and_motion_correct(data: np.ndarray,
                            batch_size:int = 800) -> Tuple[np.ndarray, np.ndarray]:

    if cfg.frames_crop * 2 >= data.shape[0]:
        display(f"No cropping since the cropping value, {cfg.frames_crop} is more than half the number of frames")
    else:
        data = data[cfg.frames_crop:cfg.frames_crop*-1]
        
    my_template = torch.from_numpy(data[0])
    rigid_strategy = masknmf.RigidMotionCorrector(max_shifts = [5, 5],
                                                  template = my_template,
                                                  pixel_weighting=None)
                                                        
    full_moco_arr = masknmf.motion_correction.RegistrationArray(data,
                                                           strategy = rigid_strategy,
                                                           device = cfg.device,
                                                           batch_size = 800)
    
    full_moco_dense, full_moco_dense_shifts = [i.numpy() for i in full_moco_arr._index_frames_tensor(slice(0, data.shape[0]))]
    return full_moco_dense, full_moco_dense_shifts

def _train_network(mov: np.ndarray,
                   cfg: DictConfig) -> torch.nn.Module:
    pass

def _compress_data(mov: np.ndarray,
                   network: torch.nn.Module) -> masknmf.PMDArray:
    pass

def _load_data(cfg: DictConfig) -> np.ndarray:
    
    binfile = os.path.join(cfg.bin_file)
    if cfg.mat_file is None:
        if cfg.nrows is None or cfg.ncols is None:
            raise ValueError("Not enough info to infer shape of bin file. Either provide a .mat file with metadata or specify the rows and cols")
        else:
            data = np.fromfile(binfile, dtype= np.uint16).astype("float").reshape(-1, cfg.nrows, cfg.ncols)
    else:
        _, nrows, ncols = get_fov_shape(cfg.mat_file)
        data = np.fromfile(binfile, dtype= np.uint16).astype("float").reshape(-1, nrows, ncols)

    
    if cfg.negatively_tuned:
        data = data * -1

    
    return data


def train_denoiser(my_data: np.ndarray,
                   cfg: DictConfig) -> torch.nn.Module:

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
                                                    background_rank = cfg.background_rank,
                                                    device = device,
                                                    frame_batch_size = cfg.frame_batch_size)
    
    v = pmd_obj.v.cpu()
    trained_model, _ = masknmf.compression.denoising.train_total_variance_denoiser(v,
                                                           max_epochs = 5,
                                                           batch_size = 128,
                                                            learning_rate=1e-4)
    temporal_denoiser = masknmf.compression.PMDTemporalDenoiser(trained_model)
    return temporal_denoiser



def compress_and_denoise_data(my_data: np.ndarray,
                              denoiser: torch.nn.Module,
                              cfg: DictConfig) -> masknmf.PMDArray:

    block_sizes = [cfg.block_size_dim1, cfg.block_size_dim2]

    device = cfg.device
    if device == 'cpu':
        display("Compressing data on CPU")
    
    pmd_obj = masknmf.compression.pmd_decomposition(my_data,
                                                    block_sizes,
                                                    my_data.shape[0],
                                                    max_components = cfg.max_components,
                                                    max_consecutive_failures = cfg.max_consecutive_failures,
                                                    temporal_avg_factor = cfg.temporal_avg_factor,
                                                    spatial_avg_factor = cfg.spatial_avg_factor,
                                                    background_rank = cfg.background_rank,
                                                    device = device,
                                                    frame_batch_size = cfg.frame_batch_size,
                                                    temporal_denoiser = denoiser)
    
    return pmd_obj


def _demix_video(pmd_obj: np.ndarray, 
                 cfg: DictConfig) -> masknmf.DemixingResults: 
    
    # Demix
    cutoff_freq = (5/800)*cfg.frame_rate
    v_matrix = pmd_obj.v.cpu().numpy()
    
    new_v =  masknmf.demixing.filters.high_pass_filter_batch(v_matrix, 
                                                            cutoff_freq, 
                                                            cfg.frame_rate)
    
    pmd_obj.to('cpu')
    threshold_pmd = masknmf.PMDArray(
            pmd_obj.shape,
            pmd_obj.u,
            torch.from_numpy(new_v),
            pmd_obj.mean_img,
            pmd_obj.var_img,
            u_local_projector = pmd_obj.u_local_projector,
            u_global_projector = pmd_obj.u_global_projector,
            device = "cpu",
            rescale = True,
        )

    num_frames, fov_dim1, fov_dim2 = threshold_pmd.shape
    highpass_pmd_demixer = masknmf.demixing.signal_demixer.SignalDemixer(
                                                    threshold_pmd,
                                                    device=cfg.device,
                                                    frame_batch_size = cfg.frame_batch_size)

    init_kwargs = {
    #Worth modifying
    'mad_correlation_threshold':0.8,

    #Mostly stable
    'robust_corr_term':1,
    'mad_threshold':4,
    'residual_threshold': 0.5,
    'patch_size':(30, 30),
    'plot_en':False,
    'text':False,
     }
        
    highpass_pmd_demixer.initialize_signals(**init_kwargs, is_custom = False)

    highpass_pmd_demixer.lock_results_and_continue()

    num_iters = 25
    ## Now run demixing...
    localnmf_params = {
        'maxiter':num_iters,
        'support_threshold':np.linspace(0.9, 0.8, num_iters).tolist(),
        'deletion_threshold':0.2,
        'ring_model_start_pt':num_iters + 1,  #No ring model
        'ring_radius':10,
        'merge_threshold':0.6,
        'merge_overlap_threshold':0.6,
        'update_frequency':4,
        'c_nonneg': False, #Voltage can fluctuate above/below baseline
        'denoise':False,
        'plot_en': False
    }
    

    with torch.no_grad():
        highpass_pmd_demixer.demix(**localnmf_params)
    display(f"{highpass_pmd_demixer.results.a.shape[1]} signals identified from highpass filtered movie")

    display("Running demixing on unfiltered data") 
    num_frames, fov_dim1, fov_dim2 = pmd_obj.shape
    device = 'cuda'
    full_pmd_demixer = masknmf.demixing.signal_demixer.SignalDemixer(
                                                    pmd_obj,
                                                    device=device,
                                                    frame_batch_size=cfg.frame_batch_size)

    full_pmd_demixer.initialize_signals(is_custom=True, 
                                    spatial_footprints=highpass_pmd_demixer.results.a,
                                    temporal_footprints=highpass_pmd_demixer.results.c)
    full_pmd_demixer.lock_results_and_continue()
    
    ## Now run demixing...
    num_iters = 25
    localnmf_params = {
        'maxiter':num_iters,
        'support_threshold':np.linspace(0.8, 0.6, num_iters).tolist(),
        'deletion_threshold':0.5,
        'ring_model_start_pt': 0, #Use ring model needed
        'ring_radius':10,
        'merge_threshold':0.8,
        'merge_overlap_threshold':0.8,
        'update_frequency':4, #No support updates
        'c_nonneg':False,
        'denoise':False,
        'plot_en': False
    }
    
    with torch.no_grad():
        full_pmd_demixer.demix(**localnmf_params)
    display(f"Identified {full_pmd_demixer.results.a.shape[1]} neurons after full NMF")

    return full_pmd_demixer.results



@hydra.main()
def register_and_compress(cfg: DictConfig):
    
    #Load and crop data:
    data = _load_data(cfg)
    display("loaded data")

    # Crop + Motion Correct
    display("Running Motion Correction")
    data, shifts = _crop_and_motion_correct(data,
                             cfg)
    
    display("Motion Correction Complete")
    # Compress + Denoise

    display("Running neural net training")
    denoiser = train_denoiser(data, cfg)

    display("Running PMD")
    pmd_obj = compress_and_denoise_data(data, denoiser, cfg)

    display("Compression finished")

    display("Running demixing")
    _demix_video(pmd_obj,
                 cfg)
    
    

if __name__ == "__main__":
    config_dict = {
        ## Dataset and I/O details: 
        'negatively_tuned': True,
        'bin_file': '/path/to/data/',
        'mat_file': None,
        'nrows': None,
        'ncols': None,
        'outdir': '.',
        #Which device to use, batch size, temporal cropping
        'device': 'cpu',
        'frame_batch_size': 1024,
        'frames_crop': 200,
        'frame_rate': 800,


        ##Motion Correction Details
        'max_rigid_shift': 5,
        'max_rigid_shift': 5,

        #Compression details
        'block_size_dim1': 32,
        'block_size_dim2': 32,
        'background_rank': 0,
        'max_components': 20,
        'max_consecutive_failures': 1,
        'spatial_avg_factor': 1,
        'temporal_avg_factor': 4,        
        
    }

    cfg = OmegaConf.create(config_dict)
    cli_conf = OmegaConf.from_cli()
    cfg = OmegaConf.merge(cfg, cli_conf)

    register_and_compress(cfg)