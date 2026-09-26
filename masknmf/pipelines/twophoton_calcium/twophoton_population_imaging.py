import masknmf
from masknmf.arrays import LazyFrameLoader, ArrayLike
from masknmf.utils import display

from masknmf.pipelines._base import BasePipeline
from masknmf.pipelines.configs.motion_correction_configs import RigidMotionCorrectionConfig, MotionCorrectionConfigs
from masknmf.pipelines.configs.compression_configs import CompressDenoiseConfig, CompressionConfigs
from masknmf.pipelines.configs.demixing_configs import NMFConfig, CustomInitConfig, SuperpixelInitConfig, SpatialHighpassConfig, SinglepassDemixingConfig, MultipassDemixingConfig, SpatialHighpassConfigs, MultipassDemixingConfigs

from typing import *
import numpy as np
from pathlib import Path
import torch


class TwoPhotonCalciumPipeline(BasePipeline):
    def __init__(self,
                 motion_correct_config: MotionCorrectionConfigs | Literal["skip"] | None = None,
                 compress_config: CompressionConfigs | Literal["skip"] | None = None,
                 spatial_highpass_config: SpatialHighpassConfigs | None = None,
                 filtered_demixing_config: MultipassDemixingConfigs | Literal["skip"] | None = None,
                 unfiltered_demixing_config: MultipassDemixingConfigs | None = None,
                 output_folder: str | Path | None = None,
                 frame_batch_size: int = 300,
                 device: Literal["auto", "cuda", "cpu"] = "auto"
                 ):
        super().__init__(output_folder=output_folder, frame_batch_size=frame_batch_size, device=device,
                         motion_correct_config=motion_correct_config, compress_config=compress_config,
                         spatial_highpass_config=spatial_highpass_config,
                         filtered_demixing_config=filtered_demixing_config,
                         unfiltered_demixing_config=unfiltered_demixing_config)

    @classmethod
    def default_configs(cls) -> dict:
        """
        Rigid motion correction, compression with denoising, and positive-signed demixing: two passes over the spatially
        highpassed data, then three over the unfiltered data. The demixing passes get the run's spline detrender, built
        from its frame rate.
        """
        filtered_passes = []
        for corr_threshold in [0.8, 0.8]:
            curr_init_conf = SuperpixelInitConfig(mad_correlation_threshold=corr_threshold,
                                                  sign="positive") #Only prioritize positive deviations for 2p calcium imaging
            curr_nmf_conf = NMFConfig(support_threshold=(0.95, corr_threshold),
                                      ring_model_start_pt=None)
            filtered_passes.append(SinglepassDemixingConfig(curr_init_conf, curr_nmf_conf))

        unfiltered_passes = []
        for corr_threshold, support_threshold in [(0.8, 0.4), (0.8, 0.4), (0.8, 0.4)]:
            curr_init_conf = SuperpixelInitConfig(mad_correlation_threshold=corr_threshold,
                                                  sign="positive")
            curr_nmf_conf = NMFConfig(support_threshold=(0.95, support_threshold),
                                      ring_model_start_pt=0)
            unfiltered_passes.append(SinglepassDemixingConfig(curr_init_conf, curr_nmf_conf))

        return {'motion_correct_config': RigidMotionCorrectionConfig(),
                'compress_config': CompressDenoiseConfig(),
                'spatial_highpass_config': SpatialHighpassConfig(),
                'filtered_demixing_config': MultipassDemixingConfig(filtered_passes),
                'unfiltered_demixing_config': MultipassDemixingConfig(unfiltered_passes)}

    def run(self,
            data: np.ndarray | ArrayLike | None,
            frame_rate: float,
            exclude_border_radius: int = 0,
            remove_intermediates: bool = True) -> Path:
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
                    DemixConfig: Config object specifying parameters for demixing the data. With filtered_demixing_config
                        "skip", the run ends after compression, kept for a later run with compress_config "skip"
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
            moco_data, shift_mask = self.motion_correct(data, self.motion_correct_config, results_path,
                                                        exclude_border_radius)

            display("Running Compression")
            compress_strategy = self.compress_strategy(self.compress_config, shift_mask)
            compress_strategy.detrender = self.spline_detrender(data.shape[0], frame_rate, window_seconds=40,
                                                                knot_seconds=25, sigma_seconds=0.3)

            compressed_results = compress_strategy.compress(moco_data)
            compressed_results.export(results_path)

        if isinstance(self.filtered_demixing_config, str):
            if self.filtered_demixing_config.lower() != "skip":
                raise ValueError(f"If filtered_demixing_config is a string, it can only be `skip`")
            return Path(results_path).parent

        device = self.torch_device
        display("Running demixing analysis")

        pmd_denoise = masknmf.CompressionArray.from_hdf5(results_path)
        pmd_denoise.to(device)
        spatial_filt_pmd = masknmf.demixing.filters.spatial_filter_compressed_array(pmd_denoise,
                                                                                    batch_size=self.frame_batch_size,
                                                                                    filter_sigma=self.spatial_highpass_config.filter_sigma)

        torch.cuda.empty_cache()


        ## Define the pixel batch size so that the max number of full pixels loaded matches frame batch size
        highpass_pmd_demixer = masknmf.demixing.signal_demixer.SignalDemixer(spatial_filt_pmd,
                                                                             device=device,
                                                                             frame_batch_size=self.frame_batch_size)

        detrender = self.spline_detrender(pmd_denoise.shape[0], frame_rate, window_seconds=20, knot_seconds=20,
                                          sigma_seconds=0.3)

        ## Use spline detrending to more effectively pick out signals. 1 knot point per 20 seconds of data
        filtered_demixing_config_used = self.with_detrender(self.filtered_demixing_config, detrender)
        unfiltered_demixing_config_used = self.with_detrender(self.unfiltered_demixing_config, detrender)

        curr_demix_results = self.run_multipass(highpass_pmd_demixer, filtered_demixing_config_used)

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

        latest_demix_results = self.run_multipass(
            unfiltered_pmd_demixer,
            MultipassDemixingConfig([custom_unfiltered_conf] + unfiltered_demixing_config_used.DemixingConfigs[1:]))

        latest_demix_results.export(results_path)
        if remove_intermediates:
            self.drop_compression(results_path)
        return Path(results_path).parent





