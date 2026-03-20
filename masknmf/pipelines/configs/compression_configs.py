from dataclasses import dataclass, field
import numpy as np
from typing import *


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
    compute_normalizer: Optional[bool] = True
    pixel_weighting: Optional[np.ndarray] = None

@dataclass
class CompressDenoiseConfig:
    block_sizes: tuple[int, int] = (20, 20)
    frame_range: int | None = None
    max_components: int = 20
    sim_conf: int = 5
    max_consecutive_failures: int = 1
    spatial_avg_factor: int = 1
    temporal_avg_factor: int = 1
    compute_normalizer: Optional[bool] = True
    pixel_weighting: Optional[np.ndarray] = None
    noise_variance_quantile: float = 0.3
    num_epochs: int = 10