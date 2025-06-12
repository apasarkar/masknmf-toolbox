from pathlib import Path

import numpy as np
import torch
import masknmf


def get_device():
    
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(DEVICE)

    rigid_strategy = masknmf.RigidMotionCorrection(
        max_shifts=(5, 5)
    )
    pwrigid_strategy = masknmf.PiecewiseRigidMotionCorrection(
        num_blocks=(32, 32),
        overlaps=(5, 5),
        max_rigid_shifts=[5, 5],
        max_deviation_rigid=[2, 2]
    )


