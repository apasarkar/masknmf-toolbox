import torch
from masknmf.arrays import LazyFrameLoader, ArrayLike
from masknmf.utils import display

from masknmf.pipelines._base import BasePipeline
from masknmf.pipelines.configs.motion_correction_configs import RigidMotionCorrectionConfig, PiecewiseRigidMotionCorrectionConfig
from masknmf.pipelines.configs.compression_configs import CompressConfig, CompressDenoiseConfig
import torch
from typing import *
import numpy as np
from pathlib import Path


class WidefieldSinglechannelPipeline(BasePipeline):
    def __init__(self,
                 motion_correct_config: RigidMotionCorrectionConfig | PiecewiseRigidMotionCorrectionConfig | Literal[
                     "skip"] | None = None,
                 compress_config: CompressConfig | CompressDenoiseConfig | None = None,
                 output_folder: str | Path | None = None,
                 frame_batch_size: int = 300,
                 device: Literal["auto", "cuda", "cpu"] = "auto"
                 ):
        """
        The pipeline takes the compressed data and filters to suppress background and identify signal. After demixing
        this filtered data, it returns to the unfiltered data to further demix.
        Args:
            motion_correct_config: Config object specifying parameters for motion correcting the data. If None,
                uses RigidMotionCorrectionConfig defaults. If "skip", skips motion correction entirely.
            compress_config: Config object specifying parameters for compressing the data.
                If None is specified, the joint compression + denoising code is run
            output_folder: Every stage is written to ``<output_folder>/<timestamp>_widefield-singlechannel/results.hdf5``,
                one hdf5 group per stage. None uses the working directory
            frame_batch_size (int): Number of frames to load into GPU at a time for processing
            device (str): Indicates which device pytorch runs on
        """
        self._motion_correct_config = motion_correct_config
        self._compress_config = compress_config
        super().__init__(output_folder, frame_batch_size, device)

    @property
    def motion_correct_config(self) -> RigidMotionCorrectionConfig | PiecewiseRigidMotionCorrectionConfig | Literal[
        "skip"] | None:
        return self._motion_correct_config

    @property
    def compress_config(self) -> CompressConfig | CompressDenoiseConfig | None:
        return self._compress_config

    @property
    def config(self):
        return {'motion_correct_config': self.motion_correct_config,
                'compress_config': self.compress_config,
                'output_folder': self.output_folder,
                'frame_batch_size': self.frame_batch_size,
                'device': self.device}

    def run(self, data: np.ndarray | ArrayLike, exclude_border_radius: int = 0):
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
        return compressed_results

