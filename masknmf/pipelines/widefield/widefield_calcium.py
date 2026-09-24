import torch
from masknmf.arrays import LazyFrameLoader, ArrayLike
from masknmf.utils import display

from masknmf.pipelines._base import BasePipeline
from masknmf.pipelines.configs.motion_correction_configs import RigidMotionCorrectionConfig, MotionCorrectionConfigs
from masknmf.pipelines.configs.compression_configs import CompressDenoiseConfig, CompressionConfigs
import torch
from typing import *
import numpy as np
from pathlib import Path


class WidefieldSinglechannelPipeline(BasePipeline):
    def __init__(self,
                 motion_correct_config: MotionCorrectionConfigs | Literal["skip"] | None = None,
                 compress_config: CompressionConfigs | None = None,
                 output_folder: str | Path | None = None,
                 frame_batch_size: int = 300,
                 device: Literal["auto", "cuda", "cpu"] = "auto"
                 ):
        """
        Args:
            motion_correct_config: Config object specifying parameters for motion correcting the data. If None,
                uses the one from default_configs(). If "skip", skips motion correction entirely.
            compress_config: Config object specifying parameters for compressing the data. If None, uses the one from
                default_configs()
            output_folder: Every stage is written to ``<output_folder>/<timestamp>_widefield-singlechannel/results.hdf5``,
                one hdf5 group per stage. None uses the working directory
            frame_batch_size (int): Number of frames to load into GPU at a time for processing
            device (str): Indicates which device pytorch runs on
        """
        super().__init__(output_folder=output_folder, frame_batch_size=frame_batch_size, device=device,
                         motion_correct_config=motion_correct_config, compress_config=compress_config)

    @classmethod
    def default_configs(cls) -> dict:
        """
        Rigid motion correction and compression with denoising.
        """
        return {'motion_correct_config': RigidMotionCorrectionConfig(),
                'compress_config': CompressDenoiseConfig()}

    def run(self, data: np.ndarray | ArrayLike, exclude_border_radius: int = 0) -> Path:
        """
        Uses the API to run rigid motion correction, compression (with denoising)
        """
        results_path = self.results_path()
        moco_data, shift_mask = self.motion_correct(data, self.motion_correct_config, results_path,
                                                    exclude_border_radius)

        display("Running Compression")
        compress_strategy = self.compress_strategy(self.compress_config, shift_mask)

        compressed_results = compress_strategy.compress(moco_data)

        compressed_results.export(results_path)
        return Path(results_path).parent

