from dataclasses import asdict
import masknmf
from masknmf.compression import CompressStrategy, CompressDenoiseStrategy, CompressionArray
from masknmf.arrays import LazyFrameLoader, ArrayLike
from masknmf.motion_correction import BaseRegistrationArray, DummyMotionCorrector, RigidMotionCorrector, PiecewiseRigidMotionCorrector
from masknmf.utils import display, drop_group
from masknmf.demixing import NoSignalsDetectedError, DemixingError

from masknmf.compression.preprocessing import MaximinSplineDetrend

from masknmf.pipelines._base import BasePipeline
from masknmf.pipelines.configs.motion_correction_configs import RigidMotionCorrectionConfig, PiecewiseRigidMotionCorrectionConfig
from masknmf.pipelines.configs.compression_configs import CompressConfig, CompressDenoiseConfig
from masknmf.pipelines.configs.demixing_configs import NMFConfig, CustomInitConfig, SuperpixelInitConfig, SpatialHighpassConfig, SinglepassDemixingConfig, MultipassDemixingConfig

from typing import *
import numpy as np
from pathlib import Path
import torch


def run_singlepass_demixing(demixing_obj: masknmf.SignalDemixer,
                            singlepass_config: SinglepassDemixingConfig) -> None | masknmf.SignalDemixer:
    init_config = singlepass_config.InitConfig
    nmf_config = singlepass_config.NMFConfig

    try:
        demixing_obj.initialize_signals(**asdict(init_config))
    except NoSignalsDetectedError:
        return None
    else:
        demixing_obj.demix(**asdict(nmf_config))
        return demixing_obj


class TwoPhotonCalciumPipeline(BasePipeline):
    def __init__(self,
                 motion_correct_config: RigidMotionCorrectionConfig | PiecewiseRigidMotionCorrectionConfig | Literal[
                     "skip"] | None = None,
                 compress_config: CompressConfig | CompressDenoiseConfig | Literal["skip"] | None = None,
                 spatial_highpass_config: SpatialHighpassConfig | None = None,
                 filtered_demixing_config: MultipassDemixingConfig | None = None,
                 unfiltered_demixing_config: MultipassDemixingConfig | None = None,
                 output_folder: str | Path | None = None,
                 frame_batch_size: int = 300,
                 device: Literal["auto", "cuda", "cpu"] = "auto"
                 ):
        self._motion_correct_config = motion_correct_config
        self._compress_config = compress_config
        self._spatial_highpass_config = spatial_highpass_config
        self._filtered_demixing_config = filtered_demixing_config
        self._unfiltered_demixing_config = unfiltered_demixing_config
        super().__init__(output_folder, frame_batch_size, device)

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
    def config(self):
        return {'motion_correct_config': self.motion_correct_config,
                'compress_config': self.compress_config,
                'spatial_highpass_config': self.spatial_highpass_config,
                'filtered_demixing_config': self.filtered_demixing_config,
                'unfiltered_demixing_config': self.unfiltered_demixing_config,
                'output_folder': self.output_folder,
                'frame_batch_size': self.frame_batch_size,
                'device': self.device}

    def run(self,
            data: np.ndarray | ArrayLike | None,
            frame_rate: float,
            exclude_border_radius: int = 0,
            remove_intermediates: bool = True):
        """
                Uses the API to run rigid motion correction, compression (with denoising), and demixing.

                The pipeline takes the compressed data and filters to suppress background and identify signal. After demixing
                this filtered data, it returns to the unfiltered data to further demix.
                Args:
                    data (Union[np.ndarray, ArrayLike]): The raw (frames, height, width) data stack
                    motion_correct_config: Config object specifying parameters for motion correcting the data. If None,
                        uses RigidMotionCorrectionConfig defaults. If "skip", skips motion correction entirely.
                    compress_config: Config object specifying parameters for compressing the data.
                        If None is specified, the joint compression + denoising code is run
                    DemixConfig: Config object specifying parameters for demixing the data
                    output_folder: Every stage is written to ``<output_folder>/<timestamp>_two-photon-calcium/results.hdf5``,
                        one hdf5 group per stage. With compress_config "skip", output_folder is instead an existing run
                        folder whose results.hdf5 holds the compression; demixing is written into that same file
                    load_into_ram (bool): Whether or not to load the full dataset into RAM for faster processing
                    remove_intermediates (bool): drop the PMDArray group once demixing is done (the demixing
                        results carry the pmd); the registration shifts stay
                """

        if isinstance(self.compress_config, str):
            if self.compress_config.lower() == "skip":
                results_path = self.results_path(resume=True)
            else:
                raise ValueError(f"If compress_config is a string, it can only be `skip`")
        else:
            results_path = self.results_path()
            ## Decide whether to motion correct data or not
            if data is None:
                raise ValueError("data is None starting from the motion correction step. Specify a dataset")
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
            else: ## If motion correction is meant to be run, this branch must execute
                ##Compute template if one is not provided
                if moco_strategy.template is None:
                    moco_strategy.compute_template(data)
                moco_data = moco_strategy.motion_correct(data)
                moco_data.output_device = moco_data.strategy.device
                moco_data.export(results_path)

            if isinstance(moco_data, BaseRegistrationArray):
                shift_mask = masknmf.motion_correction.moco_preprocessing.construct_moco_template(moco_data.shifts.cpu().numpy(),
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
                compress_strategy = CompressDenoiseStrategy(device=self.device, **curr_config)
            else:
                raise ValueError("Invalid compression config")


            num_frames = data.shape[0]
            recording_seconds = num_frames / frame_rate
            window = int(40 * frame_rate)  # 40s rolling window
            sigma = max(2.0, 0.3 * frame_rate)  # 0.3s smoothing
            num_knots = max(4, int(recording_seconds / 25))  # one knot per ~25s

            detrender_device = self.torch_device

            detrender = MaximinSplineDetrend(
                num_frames=num_frames,
                num_knots=num_knots,
                window=window,
                sigma=sigma,
                device=detrender_device,
            )

            compress_strategy.detrender = detrender

            compressed_results = compress_strategy.compress(moco_data)
            compressed_results.export(results_path)

        device = self.torch_device
        display("Running demixing analysis")

        pmd_denoise = masknmf.CompressionArray.from_hdf5(results_path)
        pmd_denoise.to(device)
        spatial_highpass_config = self.spatial_highpass_config
        if spatial_highpass_config is None:
            spatial_highpass_config = SpatialHighpassConfig()
        spatial_filt_pmd = masknmf.demixing.filters.spatial_filter_compressed_array(pmd_denoise,
                                                                                    batch_size=self.frame_batch_size,
                                                                                    filter_sigma=spatial_highpass_config.filter_sigma)

        torch.cuda.empty_cache()


        ## Define the pixel batch size so that the max number of full pixels loaded matches frame batch size
        highpass_pmd_demixer = masknmf.demixing.signal_demixer.SignalDemixer(spatial_filt_pmd,
                                                                             device=device,
                                                                             frame_batch_size=self.frame_batch_size)

        num_frames = pmd_denoise.shape[0]
        recording_seconds = num_frames / frame_rate
        window = int(20 * frame_rate)  # 20s rolling window
        sigma = max(2.0, 0.3 * frame_rate)  # 0.3s smoothing
        num_knots = max(4, int(recording_seconds / 20))  # one knot per ~20s

        detrender_device = self.torch_device

        detrender = MaximinSplineDetrend(
            num_frames=num_frames,
            num_knots=num_knots,
            window=window,
            sigma=sigma,
            device=detrender_device,
        )

        ## Use spline detrending to more effectively pick out signals. 1 knot point per 20 seconds of data
        if self.filtered_demixing_config is None:
            conf_list = []
            for corr_threshold in [0.8, 0.8]:
                curr_init_conf = SuperpixelInitConfig(mad_correlation_threshold=corr_threshold,
                                                      detrender=detrender,
                                                      sign="positive") #Only prioritize positive deviations for 2p calcium imaging
                curr_nmf_conf = NMFConfig(support_threshold=(0.95, corr_threshold),
                                          ring_model_start_pt=None,
                                          detrender=detrender)
                curr_demix_conf = SinglepassDemixingConfig(curr_init_conf, curr_nmf_conf)
                conf_list.append(curr_demix_conf)
            filtered_demixing_config_used = MultipassDemixingConfig(conf_list)
        else:
            filtered_demixing_config_used = self.filtered_demixing_config

        if self.unfiltered_demixing_config is None:
            conf_list = []
            for corr_threshold, support_threshold in [(0.8, 0.4), (0.8, 0.4), (0.8, 0.4)]:
                curr_init_conf = SuperpixelInitConfig(mad_correlation_threshold=corr_threshold,
                                                      detrender=detrender,
                                                      sign="positive")
                curr_nmf_conf = NMFConfig(support_threshold=(0.95, support_threshold),
                                          ring_model_start_pt=0,
                                          detrender=detrender)
                curr_demix_conf = SinglepassDemixingConfig(curr_init_conf, curr_nmf_conf)
                conf_list.append(curr_demix_conf)
            unfiltered_demixing_config_used = MultipassDemixingConfig(conf_list)
        else:
            unfiltered_demixing_config_used = self.unfiltered_demixing_config

        # Run the demixing rounds on the filtered data
        curr_demix_results = None
        for k in range(len(filtered_demixing_config_used.DemixingConfigs)):
            highpass_pmd_demixer = run_singlepass_demixing(highpass_pmd_demixer,
                                                           filtered_demixing_config_used.DemixingConfigs[k])
            if highpass_pmd_demixer is not None:
                curr_demix_results = highpass_pmd_demixer.results
            if highpass_pmd_demixer is None:
                if curr_demix_results is None:
                    raise ValueError("The demixer did not identify any signals in the highpass filtered movie. Lower thresholds or inspect"
                                     "data to resolve this issue.")
                else:
                    break
            torch.cuda.empty_cache()

        ## Define the unfiltered demixer object
        signals_array = curr_demix_results.signals_array
        a_init = signals_array.export_spatial_demixed()
        c_init = signals_array.export_temporal_demixed()

        ##Now overwrite the first pass of the UnfilteredDemixingConfig to be "custom" since we're using results from above
        # unfiltered_demixing_config_used.DemixingConfigs[0].InitConfig = CustomInitConfig(a_init, c_init, c_nonneg=True)
        custom_unfiltered_conf = SinglepassDemixingConfig(CustomInitConfig(a_init, c_init, c_nonneg=True),
                                                          unfiltered_demixing_config_used.DemixingConfigs[0].NMFConfig)

        unfiltered_pmd_demixer = masknmf.demixing.signal_demixer.SignalDemixer(
            pmd_denoise,
            device=device,
            frame_batch_size=self.frame_batch_size)

        # Run the demixing rounds on the unfiltered data
        latest_demix_results = None
        for k in range(len(unfiltered_demixing_config_used.DemixingConfigs)):
            if k == 0:
                unfiltered_pmd_demixer = run_singlepass_demixing(unfiltered_pmd_demixer,
                                                                 custom_unfiltered_conf)
            else:
                unfiltered_pmd_demixer = run_singlepass_demixing(unfiltered_pmd_demixer,
                                                                 unfiltered_demixing_config_used.DemixingConfigs[k])
            if unfiltered_pmd_demixer is not None:
                latest_demix_results = unfiltered_pmd_demixer.results
            elif unfiltered_pmd_demixer is None:
                if latest_demix_results is None:
                    raise ValueError("The unfiltered pmd demixer did not complete a full round of demixing.")
                else:
                    break

        latest_demix_results.export(results_path)
        if remove_intermediates:
            display("Removing intermediates")
            drop_group(results_path, CompressionArray.__name__)
        return latest_demix_results





