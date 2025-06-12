from pathlib import Path

import numpy as np

import torch
import tifffile
import ffmpeg
import matplotlib

import masknmf

ic.enable()


def get_device():
    
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    ic(DEVICE)
    ic(f"Using CUDA with device: {torch.cuda.get_device_name(0)}")

    rigid_strategy = masknmf.RigidMotionCorrection(
        max_shifts=(5, 5)
    )
    pwrigid_strategy = masknmf.PiecewiseRigidMotionCorrection(
        num_blocks=(32, 32),
        overlaps=(5, 5),
        max_rigid_shifts=[5, 5],
        max_deviation_rigid=[2, 2]
    )


