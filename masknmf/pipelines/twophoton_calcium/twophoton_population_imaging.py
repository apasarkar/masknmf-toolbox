from dataclasses import asdict
import masknmf
from masknmf.compression import CompressStrategy, CompressDenoiseStrategy
from masknmf.arrays import LazyFrameLoader, ArrayLike
from masknmf.motion_correction import RegistrationArray, DummyMotionCorrector, RigidMotionCorrector, PiecewiseRigidMotionCorrector
from masknmf.utils import display

from masknmf.pipelines._base import BasePipeline
from masknmf.pipelines.configs.motion_correction_configs import RigidMotionCorrectionConfig, PiecewiseRigidMotionCorrectionConfig
from masknmf.pipelines.configs.compression_configs import CompressConfig, CompressDenoiseConfig
from masknmf.pipelines.configs.demixing_configs import NMFConfig, CustomInitConfig, SuperpixelInitConfig, SpatialHighpassConfig, SinglepassDemixingConfig, MultipassDemixingConfig

from masknmf.utils import torch_select_device
from typing import *
import numpy as np
import os


def run_singlepass_demixing(demixing_obj: masknmf.SignalDemixer,
                            singlepass_config: SinglepassDemixingConfig) -> masknmf.SignalDemixer:
    init_config = singlepass_config.InitConfig
    nmf_config = singlepass_config.NMFConfig

    demixing_obj.initialize_signals(**asdict(init_config))
    demixing_obj.demix(**asdict(nmf_config))
    return demixing_obj


class TwoPhotonCalciumPipeline(BasePipeline):
    def __init__(self,
                 motion_correct_config: RigidMotionCorrectionConfig | PiecewiseRigidMotionCorrectionConfig | Literal[
                     "skip"] | None = None,
                 compress_config: CompressConfig | CompressDenoiseConfig | None = None,
                 spatial_highpass_config: SpatialHighpassConfig | None = None,
                 filtered_demixing_config: MultipassDemixingConfig | None = None,
                 unfiltered_demixing_config: MultipassDemixingConfig | None = None,
                 outpath_motion_correction: Optional[str] = "motion_correction.hdf5",
                 outpath_compression: Optional[str] = "compression.hdf5",
                 outpath_demixing: Optional[str] = "demixing_results.hdf5",
                 load_into_ram: bool = False,
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
        self._load_into_ram = load_into_ram
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
    def load_into_ram(self) -> bool:
        return self._load_into_ram

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
                'load_into_ram': self.load_into_ram,
                'frame_batch_size': self.frame_batch_size,
                'device': self.device}

    def run(self, data: np.ndarray | LazyFrameLoader | ArrayLike):
        """
                Uses the API to run rigid motion correction, compression (with denoising), and demixing.

                The pipeline takes the compressed data and filters to suppress background and identify signal. After demixing
                this filtered data, it returns to the unfiltered data to further demix.
                Args:
                    data (Union[np.ndarray, LazyFrameLoader, ArrayLike]): The raw (frames, height, width) data stack
                    motion_correct_config: Config object specifying parameters for motion correcting the data. If None,
                        uses RigidMotionCorrectionConfig defaults. If "skip", skips motion correction entirely.
                    compress_config: Config object specifying parameters for compressing the data.
                        If None is specified, the joint compression + denoising code is run
                    DemixConfig: Config object specifying parameters for demixing the data
                    outpath_motion_correction (Optional[str]): Where to write out the motion corrected stack
                    outpath_compression (Optional[str]): Where to write out the compression + results
                    load_into_ram (bool): Whether or not to load the full dataset into RAM for faster processing
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
        elif isinstance(self.motion_correct_config, str):
            if self.motion_correct_config.lower() == "skip":
                moco_strategy = DummyMotionCorrector()
                display("Not Running Motion Correction")
            else:
                raise ValueError("Invalid MotionCorrectionConfig input")
        else:
            raise ValueError("Invalid MotionCorrectionConfig input")

        ##Compute template if one is not provided
        if moco_strategy.template is None:
            moco_strategy.compute_template(data)

        full_moco_arr = RegistrationArray(data, strategy=moco_strategy)
        # Export the motion correction to a new file
        full_moco_arr.export(os.path.abspath(self.outpath_motion_correction))

        moco_data = masknmf.RegistrationArray.from_hdf5(self.outpath_motion_correction)
        shift_mask = masknmf.motion_correction.moco_preprocessing.construct_moco_template(moco_data.shifts,
                                                                                          moco_data.shape[1:]).astype(
            "float")
        if self.load_into_ram:
            moco_data = moco_data[:]

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

        if self.device == "auto":
            device = torch_select_device()
        display("Running demixing analysis")

        pmd_denoise = masknmf.PMDArray.from_hdf5(self.outpath_compression)
        if self.spatial_highpass_config is None:
            spatial_highpass_config = SpatialHighpassConfig()
        spatial_filt_pmd = masknmf.demixing.filters.spatial_filter_pmd(pmd_denoise,
                                                                       batch_size=self.frame_batch_size,
                                                                       filter_sigma=spatial_highpass_config.filter_sigma,
                                                                       device=device)

        highpass_pmd_demixer = masknmf.demixing.signal_demixer.SignalDemixer(spatial_filt_pmd,
                                                                             device=device,
                                                                             frame_batch_size=self.frame_batch_size)

        if self.filtered_demixing_config is None:
            conf_list = []
            for corr_threshold in [0.8, 0.6]:
                curr_init_conf = SuperpixelInitConfig(mad_correlation_threshold=corr_threshold)
                curr_nmf_conf = NMFConfig(support_threshold=(0.95, corr_threshold),
                                          ring_model_start_pt=None)
                curr_demix_conf = SinglepassDemixingConfig(curr_init_conf, curr_nmf_conf)
                conf_list.append(curr_demix_conf)
            filtered_demixing_config = MultipassDemixingConfig(conf_list)

        if self.unfiltered_demixing_config is None:
            conf_list = []
            for corr_threshold in [0.8, 0.6, 0.4]:
                curr_init_conf = SuperpixelInitConfig(mad_correlation_threshold=corr_threshold)
                curr_nmf_conf = NMFConfig(support_threshold=(0.95, corr_threshold),
                                          ring_model_start_pt=0)
                curr_demix_conf = SinglepassDemixingConfig(curr_init_conf, curr_nmf_conf)
                conf_list.append(curr_demix_conf)
            unfiltered_demixing_config = MultipassDemixingConfig(conf_list)

        # Run the demixing rounds on the filtered data
        for k in range(len(filtered_demixing_config.DemixingConfigs)):
            highpass_pmd_demixer = run_singlepass_demixing(highpass_pmd_demixer,
                                                           filtered_demixing_config.DemixingConfigs[k])

        ## Define the unfiltered demixer object
        ac_arr = highpass_pmd_demixer.results.ac_array
        a_init = ac_arr.export_a()
        c_init = ac_arr.export_c()

        ##Now overwrite the first pass of the UnfilteredDemixingConfig to be "custom" since we're using results from above
        unfiltered_demixing_config.DemixingConfigs[0].InitConfig = CustomInitConfig(a_init, c_init, c_nonneg=True)

        unfiltered_pmd_demixer = masknmf.demixing.signal_demixer.SignalDemixer(
            pmd_denoise,
            device=device,
            frame_batch_size=self.frame_batch_size)

        # Run the demixing rounds on the unfiltered data
        for k in range(len(unfiltered_demixing_config.DemixingConfigs)):
            unfiltered_pmd_demixer = run_singlepass_demixing(unfiltered_pmd_demixer,
                                                             unfiltered_demixing_config.DemixingConfigs[k])

        unfiltered_pmd_demixer.results.export(os.path.abspath(self.outpath_demixing))
        return unfiltered_pmd_demixer.results







