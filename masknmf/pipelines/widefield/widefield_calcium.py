from masknmf.arrays import ArrayLike

from masknmf.pipelines._base import BasePipeline
from masknmf.pipelines.stages import (
    build_pixel_weighting,
    compress as compress_stage,
    register as register_stage,
)
from masknmf.pipelines.configs.motion_correction_configs import RigidMotionCorrectionConfig, PiecewiseRigidMotionCorrectionConfig
from masknmf.pipelines.configs.compression_configs import CompressConfig, CompressDenoiseConfig
from typing import *
import numpy as np


class WidefieldSinglechannelPipeline(BasePipeline):
    def __init__(self,
                 motion_correct_config: RigidMotionCorrectionConfig | PiecewiseRigidMotionCorrectionConfig | Literal[
                     "skip"] | None = None,
                 compress_config: CompressConfig | CompressDenoiseConfig | None = None,
                 outpath_motion_correction: Optional[str] = "results.hdf5",
                 outpath_compression: Optional[str] = "results.hdf5",
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
            outpath_compression (Optional[str]): Where to write out the compression + results. The two outpaths
                default to one file holding one hdf5 group per stage; give them different names for one file per stage
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

    def run(self, data: np.ndarray | ArrayLike, exclude_border_radius: int = 0):
        """
        Uses the API to run rigid motion correction, compression (with denoising)

        Args:
            data (Union[np.ndarray, ArrayLike]): The raw (frames, height, width) data stack
            exclude_border_radius (int): Zero this many pixels at each edge when compressing
        """
        moco_data = register_stage(
            data,
            self.motion_correct_config,
            device=self.device,
            batch_size=self.frame_batch_size,
            outpath=self.outpath_motion_correction,
        )
        shift_mask = build_pixel_weighting(moco_data, exclude_border_radius)
        return compress_stage(
            moco_data,
            self.compress_config,
            pixel_weighting=shift_mask,
            device=self.device,
            outpath=self.outpath_compression,
        )
