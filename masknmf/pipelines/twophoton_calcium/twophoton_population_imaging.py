import masknmf
from masknmf.arrays import ArrayLike
from masknmf.utils import display, has_group, drop_group

from masknmf.pipelines._base import BasePipeline
from masknmf.pipelines.stages import (
    build_detrender,
    build_pixel_weighting,
    compress as compress_stage,
    demix_two_phase,
    register as register_stage,
)
from masknmf.pipelines.configs.motion_correction_configs import RigidMotionCorrectionConfig, PiecewiseRigidMotionCorrectionConfig
from masknmf.pipelines.configs.compression_configs import CompressConfig, CompressDenoiseConfig
from masknmf.pipelines.configs.demixing_configs import SpatialHighpassConfig, MultipassDemixingConfig

from typing import *
import numpy as np
import os


class TwoPhotonCalciumPipeline(BasePipeline):
    def __init__(self,
                 motion_correct_config: RigidMotionCorrectionConfig | PiecewiseRigidMotionCorrectionConfig | Literal[
                     "skip"] | None = None,
                 compress_config: CompressConfig | CompressDenoiseConfig | Literal["skip"] | None = None,
                 spatial_highpass_config: SpatialHighpassConfig | None = None,
                 filtered_demixing_config: MultipassDemixingConfig | None = None,
                 unfiltered_demixing_config: MultipassDemixingConfig | None = None,
                 outpath_motion_correction: Optional[str] = "results.hdf5",
                 outpath_compression: Optional[str] = "results.hdf5",
                 outpath_demixing: Optional[str] = "results.hdf5",
                 frame_batch_size: int = 300,
                 device: Literal["auto", "cuda", "cpu"] = "auto"
                 ):
        self._motion_correct_config = motion_correct_config
        self._compress_config = compress_config
        self._spatial_highpass_config = spatial_highpass_config
        self._filtered_demixing_config = filtered_demixing_config
        self._unfiltered_demixing_config = unfiltered_demixing_config
        self._outpath_motion_correction = outpath_motion_correction
        self._outpath_compression = outpath_compression
        self._outpath_demixing = outpath_demixing
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
    def spatial_highpass_config(self) -> SpatialHighpassConfig | None:
        return self._spatial_highpass_config

    @property
    def filtered_demixing_config(self) -> MultipassDemixingConfig:
        return self._filtered_demixing_config

    @property
    def unfiltered_demixing_config(self) -> MultipassDemixingConfig | None:
        return self._unfiltered_demixing_config

    @property
    def outpath_motion_correction(self) -> Optional[str]:
        return self._outpath_motion_correction

    @property
    def outpath_compression(self) -> Optional[str]:
        return self._outpath_compression

    @property
    def outpath_demixing(self) -> Optional[str]:
        return self._outpath_demixing

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
                'spatial_highpass_config': self.spatial_highpass_config,
                'filtered_demixing_config': self.filtered_demixing_config,
                'unfiltered_demixing_config': self.unfiltered_demixing_config,
                'outpath_motion_correction': self.outpath_motion_correction,
                'outpath_compression': self.outpath_compression,
                'outpath_demixing': self.outpath_demixing,
                'frame_batch_size': self.frame_batch_size,
                'device': self.device}

    def run(self,
            data: np.ndarray | ArrayLike,
            frame_rate: float,
            exclude_border_radius: int = 0,
            remove_intermediates: bool = True):
        """
                Uses the API to run rigid motion correction, compression (with denoising), and demixing.

                The pipeline takes the compressed data and filters to suppress background and identify signal. After demixing
                this filtered data, it returns to the unfiltered data to further demix.
                Args:
                    data (Union[np.ndarray, ArrayLike]): The raw (frames, height, width) data stack
                    frame_rate (float): Acquisition rate, in Hz
                    exclude_border_radius (int): Zero this many pixels at each edge when compressing
                    remove_intermediates (bool): delete the motion correction and compression files once demixing
                        is done; in one results file, drop its PMDArray group instead (the demixing results carry
                        the pmd) and keep the registration shifts
                """

        pmd_source = os.path.abspath(self.outpath_compression)
        if isinstance(self.compress_config, str):
            if self.compress_config.lower() == "skip":
                # a previous run's compression: at outpath_compression, else an old compression.hdf5 beside it
                if not has_group(pmd_source, "PMDArray"):
                    pmd_source = os.path.join(os.path.dirname(pmd_source), "compression.hdf5")
                if not has_group(pmd_source, "PMDArray"):
                    raise ValueError("You specified that compression should be skipped but there is no compression at "
                                     "outpath_compression or in a compression.hdf5 beside it")
            else:
                raise ValueError(f"If compress_config is a string, it can only be `skip`")
        else:
            moco_data = register_stage(
                data,
                self.motion_correct_config,
                device=self.device,
                batch_size=self.frame_batch_size,
                outpath=self.outpath_motion_correction,
            )
            shift_mask = build_pixel_weighting(moco_data, exclude_border_radius)
            detrender = build_detrender(
                data.shape[0], frame_rate, 40.0, 25.0, self.device
            )
            compress_stage(
                moco_data,
                self.compress_config,
                pixel_weighting=shift_mask,
                detrender=detrender,
                device=self.device,
                outpath=self.outpath_compression,
            )

        pmd_denoise = masknmf.PMDArray.from_hdf5(pmd_source)
        latest_demix_results = demix_two_phase(
            pmd_denoise,
            frame_rate,
            filtered_config=self.filtered_demixing_config,
            unfiltered_config=self.unfiltered_demixing_config,
            spatial_highpass_config=self.spatial_highpass_config,
            device=self.device,
            frame_batch_size=self.frame_batch_size,
            num_frames=data.shape[0],
        )

        final = os.path.abspath(self.outpath_demixing)
        if remove_intermediates:
            display("Removing intermediates")
            for path in (os.path.abspath(self.outpath_motion_correction), os.path.abspath(self.outpath_compression)):
                if path != final and os.path.exists(path):
                    os.remove(path)
        latest_demix_results.export(final)
        if remove_intermediates:
            # in one results file the pmd group only duplicates what the demixing results carry; the shifts stay
            drop_group(final, "PMDArray")
        return latest_demix_results
