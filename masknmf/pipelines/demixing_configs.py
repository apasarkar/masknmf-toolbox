from dataclasses import dataclass, field
import numpy as np
from typing import *

@dataclass
class SuperpixelInitConfig:
    mad_correlation_threshold: float = 0.8
    min_peak_distance: int = 3
    mad_threshold: float =  1
    residual_threshold: float = 0.3
    patch_size: tuple[int, int] = (40, 40)

@dataclass
class CustomInitConfig:
    is_custom: bool = field(default=True, init=False)
    spatial_footprints: np.ndarray
    temporal_footprints: np.ndarray
    c_nonneg: bool = True

@dataclass
class NMFConfig:
    maxiter: int = 40
    support_threshold: tuple[int, int] = (0.95, 0.8)
    deletion_threshold: float = 0.2
    ring_model_start_pt: Optional[int] = 0
    ring_radius: int = 10
    background_downsampling_factor: int = 30
    merge_threshold: float = 0.8
    merge_overlap_threshold: float = 0.8
    update_frequency: int = 4
    c_nonneg: bool = True
    denoise: bool = False
    plot_en: bool = False

@dataclass
class SinglepassDemixingConfig:
    InitConfig: SuperpixelInitConfig
    NMFConfig: NMFConfig

@dataclass
class MultipassDemixingConfig:
    DemixingConfigs: list[SinglepassDemixingConfig]

#### Below are configs for the different decimation strategies used to initialize cells
@dataclass
class SpatialHighpassConfig:
    """
    Config for the filtering we do on the compressed data to produced a new dataset that exposes signals (for the NMF initialization step)
    If this step ever becomes more complicated, add fields here
    """
    filter_sigma: float=4.0

