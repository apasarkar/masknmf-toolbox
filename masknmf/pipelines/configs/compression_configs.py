from dataclasses import dataclass, field
import numpy as np
from typing import *
from masknmf.compression.preprocessing import SplineDetrenderBase


@dataclass
class CompressConfig:
    block_sizes: tuple[int, int] = (20, 20)
    frame_range: int | None = None
    max_components: int = 20
    sim_conf: int = 5
    frame_batch_size: int = 10000
    max_consecutive_failures: int = 1
    spatial_avg_factor: int = 1
    temporal_avg_factor: int = 1
    compute_normalizer: bool | None = True
    pixel_weighting: np.ndarray | None = None
    frame_weighting: np.ndarray | None = None
    detrender: SplineDetrenderBase | None = None

@dataclass
class CompressDenoiseConfig:
    block_sizes: tuple[int, int] = (20, 20)
    frame_range: int | None = None
    max_components: int = 20
    sim_conf: int = 5
    max_consecutive_failures: int = 1
    spatial_avg_factor: int = 1
    temporal_avg_factor: int = 1
    compute_normalizer: bool | None = True
    pixel_weighting: np.ndarray | None = None
    frame_weighting: np.ndarray | None = None
    noise_variance_quantile: float = 0.3
    num_epochs: int = 10
    detrender: SplineDetrenderBase | None = None