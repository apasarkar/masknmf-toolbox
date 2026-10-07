import torch
from masknmf.arrays import LazyFrameLoader, ArrayLike

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
                 compress_config: CompressionConfigs | Literal["skip"] | None = None,
                 output_folder: str | Path | None = None,
                 frame_batch_size: int = 300,
                 device: Literal["auto", "cuda", "cpu"] = "auto",
                 log_level: Literal["debug", "info", "warning"] = "info",
                 load_into_ram: bool = False
                 ):
        """
        Args:
            motion_correct_config: Config object specifying parameters for motion correcting the data. If None,
                uses the one from default_configs(). If "skip", skips motion correction entirely.
            compress_config: Config object specifying parameters for compressing the data. If None, uses the one from
                default_configs(). If "skip", the run ends after motion correction
            output_folder: Every stage is written to ``<output_folder>/<timestamp>_widefield-singlechannel/results.hdf5``,
                one hdf5 group per stage. None uses the working directory
            frame_batch_size (int): Number of frames to load into GPU at a time for processing
            device (str): Indicates which device pytorch runs on
            log_level (str): How much the run logs, to the console and to the run folder's .log file
            load_into_ram (bool): Read the whole movie into RAM before motion correction, so no stage reads it from disk
        """
        super().__init__(output_folder=output_folder, frame_batch_size=frame_batch_size, device=device,
                         log_level=log_level,
                         load_into_ram=load_into_ram,
                         motion_correct_config=motion_correct_config, compress_config=compress_config)

    @classmethod
    def default_configs(cls) -> dict:
        """
        Rigid motion correction and compression with denoising.
        """
        return {'motion_correct_config': RigidMotionCorrectionConfig(),
                'compress_config': CompressDenoiseConfig()}

    def run(self, data: np.ndarray | ArrayLike, exclude_border_radius: int = 0,
            stop_after: Literal["registration", "compression"] = "compression",
            resume_from: str | Path | None = None) -> Path:
        """
        Uses the API to run rigid motion correction, compression (with denoising). With stop_after "registration" the
        run ends once the shifts and template are written; resume_from, an earlier results file, is copied into the new
        run folder and its registration replayed instead of estimating one.
        """
        ends_at_registration = stop_after == "registration" or isinstance(self.compress_config, str)
        if ends_at_registration and isinstance(self.motion_correct_config, str) and resume_from is None:
            raise ValueError('a run ending after registration with motion_correct_config "skip" has nothing to write')
        resume_from, base = self.resume_source(resume_from, reuse_compression=False)
        self.run_config = {"exclude_border_radius": exclude_border_radius, "stop_after": stop_after,
                           "resume_from": None if resume_from is None else str(resume_from)}
        results_path = self.results_path(base)
        stored = None if resume_from is None else self.resume(resume_from, results_path, reuse_compression=False)
        if self.load_into_ram:
            data = self.read_into_ram(data)
        moco_data, shift_mask = self.motion_correct(data, self.motion_correct_config, results_path,
                                                    exclude_border_radius, stored)
        if ends_at_registration:
            return self.finish()

        compress_strategy = self.compress_strategy(self.compress_config, shift_mask)

        with self.step("compression"):
            compressed_results = compress_strategy.compress(moco_data)
            compressed_results.export(results_path)
        return self.finish()

