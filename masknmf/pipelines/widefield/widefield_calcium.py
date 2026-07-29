import torch
from dataclasses import asdict
import masknmf
from masknmf.motion_correction import RigidMotionCorrector, PiecewiseRigidMotionCorrector
from masknmf.compression import CompressStrategy, CompressDenoiseStrategy
from masknmf.arrays import LazyFrameLoader, ArrayLike
from masknmf.motion_correction import BaseRegistrationArray
from masknmf.utils import display

from masknmf.pipelines._base import BasePipeline
from masknmf.pipelines.configs.motion_correction_configs import RigidMotionCorrectionConfig, PiecewiseRigidMotionCorrectionConfig
from masknmf.pipelines.configs.compression_configs import CompressConfig, CompressDenoiseConfig
import torch
from typing import *
import numpy as np
import os


class WidefieldSinglechannelPipeline(BasePipeline):
    def __init__(self,
                 motion_correct_config: RigidMotionCorrectionConfig | PiecewiseRigidMotionCorrectionConfig | Literal[
                     "skip"] | None = None,
                 compress_config: CompressConfig | CompressDenoiseConfig | None = None,
                 outpath_motion_correction: Optional[str] = "motion_correction.hdf5",
                 outpath_compression: Optional[str] = "compression.hdf5",
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
            outpath_motion_correction (Optional[str]): Where to write out the motion corrected stack
            outpath_compression (Optional[str]): Where to write out the compression + results
            frame_batch_size (int): Number of frames to load into GPU at a time for processing
            device (str): Indicates which device pytorch runs on
        """
        self._motion_correct_config = motion_correct_config
        self._compress_config = compress_config
        self._outpath_motion_correction = outpath_motion_correction
        self._outpath_compression = outpath_compression
        self._frame_batch_size = frame_batch_size
        self._device = device

    @property
    def motion_correct_config(self) -> RigidMotionCorrectionConfig | PiecewiseRigidMotionCorrectionConfig | Literal[
        "skip"] | None:
        return self._motion_correct_config

    @property
    def compress_config(self) -> CompressConfig | CompressDenoiseConfig | None:
        return self._compress_config

    @property
    def outpath_motion_correction(self) -> Optional[str]:
        return self._outpath_motion_correction

    @property
    def outpath_compression(self) -> Optional[str]:
        return self._outpath_compression

    @property
    def frame_batch_size(self) -> int:
        return self._frame_batch_size

    @property
    def device(self) -> Literal["auto", "cuda", "cpu"]:
        return self._device

    @property
    def config(self):
        return {'motion_correct_config': self.motion_correct_config,
                'compress_config': self.compress_config,
                'outpath_motion_correction': self.outpath_motion_correction,
                'outpath_compression': self.outpath_compression,
                'frame_batch_size': self.frame_batch_size,
                'device': self.device}

    def run(self, data: np.ndarray | ArrayLike):
        """
        Uses the API to run rigid motion correction, compression (with denoising)
        """
        ## Decide whether to motion correct data or not
        if self.motion_correct_config is None:
            moco_strategy = RigidMotionCorrector(**asdict(RigidMotionCorrectionConfig()), device=self.device,
                                                 batch_size=self.frame_batch_size)

        elif isinstance(self.motion_correct_config, RigidMotionCorrectionConfig):
            moco_strategy = RigidMotionCorrector(**asdict(self.motion_correct_config), device=self.device,
                                                 batch_size=self.frame_batch_size)
        elif isinstance(self.motion_correct_config, PiecewiseRigidMotionCorrectionConfig):
            moco_strategy = PiecewiseRigidMotionCorrector(**asdict(self.motion_correct_config), device=self.device,
                                                          batch_size=self.frame_batch_size)
        else:
            moco_strategy = None

        if isinstance(self.motion_correct_config, str):
            if self.motion_correct_config.lower() == "skip":
                moco_data = data
                display("Not Running Motion Correction")
            else:
                raise ValueError("Invalid MotionCorrectionConfig input")
        elif moco_strategy is None:
            raise ValueError("Invalid MotionCorrectionConfig input")
        else:  ## If motion correction is meant to be run, this branch must execute
            ##Compute template if one is not provided
            if moco_strategy.template is None:
                moco_strategy.compute_template(data)
            moco_data = moco_strategy.motion_correct(data)
            moco_data.output_device = moco_data.strategy.device
            moco_data.export(os.path.abspath(self.outpath_motion_correction))

        if isinstance(moco_data, BaseRegistrationArray):
            shift_mask = masknmf.motion_correction.moco_preprocessing.construct_moco_template(
                moco_data.shifts.cpu().numpy(),
                moco_data.shape[1:]).astype(
                "float")
        else:
            shift_mask = np.ones((moco_data.shape[1], moco_data.shape[2])).astype("float")
        if exclude_border_radius > 0:
            shift_mask[:exclude_border_radius, :] = 0
            shift_mask[:, :exclude_border_radius] = 0
            shift_mask[-1 * exclude_border_radius:, :] = 0
            shift_mask[:, -1 * exclude_border_radius:] = 0

        display("Running Compression")
        if self.compress_config is None:
            curr_config = CompressDenoiseConfig()
            curr_config.pixel_weighting = shift_mask
            compress_strategy = CompressDenoiseStrategy(device=self.device, **asdict(curr_config))
        elif isinstance(self.compress_config, CompressConfig):
            curr_config = asdict(self.compress_config)
            if self.compress_config.pixel_weighting is not None:
                curr_config['pixel_weighting'] = curr_config['pixel_weighting'] * shift_mask
            else:
                curr_config['pixel_weighting'] = shift_mask
            compress_strategy = CompressStrategy(device=self.device, **curr_config)
        elif isinstance(self.compress_config, CompressDenoiseConfig):
            curr_config = asdict(self.compress_config)
            if self.compress_config.pixel_weighting is not None:
                curr_config['pixel_weighting'] = curr_config['pixel_weighting'] * shift_mask
            else:
                curr_config['pixel_weighting'] = shift_mask
            compress_strategy = CompressDenoiseStrategy(device=self.device, **asdict(self.compress_config))
        else:
            raise ValueError("Invalid compression config")

        compressed_results = compress_strategy.compress(moco_data)

        compressed_results.export(self.outpath_compression)
        return compressed_results

