from dataclasses import dataclass, field
import numpy as np
from typing import *

"""
Data classes store algorithm parameters only (no batch sizes, pytorch device specifications, etc.). 
"""
@dataclass
class RigidMotionCorrectionConfig:
    max_shifts: tuple[int, int] = (15, 15)
    pixel_weighting: Optional[np.ndarray] = None
    template: Optional[np.ndarray] = None

@dataclass
class PiecewiseRigidMotionCorrectionConfig:
    num_blocks: tuple[int, int] = (15, 15)
    overlaps: tuple[int, int] = (5, 5)
    max_rigid_shifts: tuple[int, int] = (15, 15)
    max_deviation_rigid: tuple[int, int] = (2, 2)
    pixel_weighting: Optional[np.ndarray] = None
    template: Optional[np.ndarray] = None

