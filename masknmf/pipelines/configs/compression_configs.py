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
    max_consecutive_failures: int = 1
    spatial_avg_factor: int = 1
    temporal_avg_factor: int = 1
    compute_normalizer: bool | None = True
    pixel_weighting: np.ndarray | None = None
    frame_weighting: np.ndarray | None = None
    detrender: SplineDetrenderBase | None = None
    # Where compression holds its tiles: "cpu" (host RAM), "device" (the compute device), or None to let
    # the planner choose by read cost. "device" suits data already in RAM; None or "cpu" suits data read from disk.
    tile_device: Literal["cpu", "device"] | None = None

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
    num_epochs: int = 30
    detrender: SplineDetrenderBase | None = None
    # Where compression holds its tiles: "cpu" (host RAM), "device" (the compute device), or None to let
    # the planner choose by read cost. "device" suits data already in RAM; None or "cpu" suits data read from disk.
    tile_device: Literal["cpu", "device"] | None = None

# every config a pipeline's compression step can take; see MotionCorrectionConfigs
CompressionConfigs = CompressConfig | CompressDenoiseConfig