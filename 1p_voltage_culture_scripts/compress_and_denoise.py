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


def _crop_and_motion_correct(mov: np.ndarray,
                            batch_size:int = 800) -> Tuple[np.ndarray, np.ndarray]:

    if cfg.frames_crop * 2 >= mov.shape[0]:
        display(f"No cropping since the cropping value, {cfg.frames_crop} is more than half the number of frames")
    else:
        mov = mov[cfg.frames_crop:cfg.frames_crop*-1]
        
    my_template = torch.from_numpy(data[0])
    rigid_strategy = masknmf.RigidMotionCorrection(max_shifts = [5, 5],
                                                   template = my_template,
                                                  pixel_weighting=None)
                                                        
    full_moco_arr = masknmf.motion_correction.RegistrationArray(data,
                                                           strategy = rigid_strategy,
                                                           device = cfg.device,
                                                           batch_size = 800)
    
    full_moco_dense, full_moco_dense_shifts = [i.numpy() for i in full_moco_arr.index_frames_tensor(slice(0, data.shape[0]))]
    return full_moco_dense, full_moco_dense_shifts

def _train_network(mov: np.ndarray,
                   cfg: DictConfig) -> torch.nn.Module:
    pass

def _compress_data(mov: np.ndarray,
                   network: torch.nn.Module) -> masknmf.PMDArray:
    pass

def _load_and_crop(cfg: DictConfig):
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

def register_and_compress(cfg: DictConfig):
    #Load and crop data:
    data = _load_and_crop(cfg)
    display("loaded data")

    # Crop + Motion Correct

    # Compress + Denoise

    # Demix

    

if __name__ == "__main__":
    config_dict = {
        'negatively_tuned': True,
        'bin_file': '/path/to/data/',
        'mat_file': None,
        'nrows': None,
        'ncols': None,
        'outdir': '.',
        'max_rigid_shift': 5,
        'max_rigid_shift': 5,
        'block_size_dim1': 32,
        'block_size_dim2': 32,
        'background_rank': 0,
        'max_components': 20,
        'max_consecutive_failures': 1,
        'spatial_avg_factor': 1,
        'temporal_avg_factor': 4,
        'device': 'cpu',
        'frame_batch_size': 1024,
        'frames_crop': 200,
    }

    cfg = OmegaConf.create(config_dict)
    cli_conf = OmegaConf.from_cli()
    cfg = OmegaConf.merge(cfg, cli_conf)

    register_and_compress(cfg)