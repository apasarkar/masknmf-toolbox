import masknmf
from masknmf.arrays import LazyFrameLoader, ArrayLike

from masknmf.pipelines._base import BasePipeline
from masknmf.pipelines.configs.motion_correction_configs import RigidMotionCorrectionConfig, MotionCorrectionConfigs
from masknmf.pipelines.configs.compression_configs import CompressDenoiseConfig, CompressionConfigs
from masknmf.pipelines.configs.demixing_configs import NMFConfig, CustomInitConfig, SuperpixelInitConfig, SpatialHighpassConfig, SinglepassDemixingConfig, MultipassDemixingConfig, SpatialHighpassConfigs, MultipassDemixingConfigs

from typing import *
import numpy as np
from pathlib import Path
import gc
import torch


class TwoPhotonCalciumPipeline(BasePipeline):
    def __init__(self,
                 motion_correct_config: MotionCorrectionConfigs | Literal["skip"] | None = None,
                 compress_config: CompressionConfigs | Literal["skip"] | None = None,
                 spatial_highpass_config: SpatialHighpassConfigs | None = None,
                 filtered_demixing_config: MultipassDemixingConfigs | Literal["skip"] | None = None,
                 unfiltered_demixing_config: MultipassDemixingConfigs | Literal["skip"] | None = None,
                 output_folder: str | Path | None = None,
                 frame_batch_size: int = 300,
                 device: Literal["auto", "cuda", "cpu"] = "auto",
                 log_level: Literal["debug", "info", "warning"] = "info",
                 load_into_ram: bool = False,
                 device_memory_gb: float | None = None,
                 host_memory_gb: float | None = None,
                 ):
        super().__init__(output_folder=output_folder, frame_batch_size=frame_batch_size, device=device,
                         log_level=log_level,
                         load_into_ram=load_into_ram,
                         motion_correct_config=motion_correct_config, compress_config=compress_config,
                         spatial_highpass_config=spatial_highpass_config,
                         filtered_demixing_config=filtered_demixing_config,
                         unfiltered_demixing_config=unfiltered_demixing_config,
                         device_memory_gb=device_memory_gb,
                         host_memory_gb=host_memory_gb)

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
                                      ring_model_start_pt=None,
                                      sign="positive")
            filtered_passes.append(SinglepassDemixingConfig(curr_init_conf, curr_nmf_conf))

        unfiltered_passes = []
        for corr_threshold, support_threshold in [(0.8, 0.4), (0.6, 0.4), (0.6, 0.4)]:
            curr_init_conf = SuperpixelInitConfig(mad_correlation_threshold=corr_threshold,
                                                  sign="positive")
            curr_nmf_conf = NMFConfig(support_threshold=(0.95, support_threshold),
                                      ring_model_start_pt=0,
                                      sign="positive")
            unfiltered_passes.append(SinglepassDemixingConfig(curr_init_conf, curr_nmf_conf))

        return {'motion_correct_config': RigidMotionCorrectionConfig(),
                'compress_config': CompressDenoiseConfig(),
                'spatial_highpass_config': SpatialHighpassConfig(),
                'filtered_demixing_config': MultipassDemixingConfig(filtered_passes),
                'unfiltered_demixing_config': MultipassDemixingConfig(unfiltered_passes)}

    @property
    def demixing_configs(self) -> list[SinglepassDemixingConfig | MultipassDemixingConfig]:
        return [self.filtered_demixing_config, self.unfiltered_demixing_config]

    def run(self,
            data: np.ndarray | ArrayLike | None,
            frame_rate: float,
            exclude_border_radius: int = 0,
            indicator_polarity: Literal["positive", "negative"] = "positive",
            good_frame_indicator: torch.Tensor | np.ndarray | None = None,
            remove_intermediates: bool = False,
            stop_after: Literal["registration", "compression", "demixing"] = "demixing",
            resume: str | Path | None = None) -> Path:
        """
                Uses the API to run rigid motion correction, compression (with denoising), and demixing.

                The pipeline takes the compressed data and filters to suppress background and identify signal. After demixing
                this filtered data, it returns to the unfiltered data to further demix.
                Args:
                    data (Union[np.ndarray, ArrayLike]): The raw (frames, height, width) data stack
                    frame_rate (float): The frame rate in Hz
                    exclude_border_radius (int): The number of pixels in border radius to exclude
                    indicator_polarity (Literal["positive", "negative"]): Whether the indicator is positively or negatively tuned
                    good_frame_indicator (torch.Tensor | np.ndarray | None): An optional 1D vector indicating which frames are good vs. bad
                    remove_intermediates (bool): drop the PMDArray group once demixing is done (the demixing
                        results carry the pmd); the registration shifts stay
                    stop_after: the last stage to run: "registration" writes the shifts and template and ends,
                        "compression" ends once the compression is written, "demixing" runs everything
                    resume: an earlier results file copied into the new run folder: its registration is applied to
                        data instead of estimating one, and with compress_config "skip" its compression is reused too.
                        The settings not given, all but stop_after, are that run's, from the config.json beside it.
                        The earlier file is left as it is
                """

        reuse_compression = isinstance(self.compress_config, str)
        if reuse_compression and self.compress_config.lower() != "skip":
            raise ValueError(f"If compress_config is a string, it can only be `skip`")
        if reuse_compression and stop_after != "demixing":
            raise ValueError('compress_config "skip" reuses a compression for its demixing; stop_after leaves nothing to run')
        if stop_after == "registration" and isinstance(self.motion_correct_config, str) and resume is None:
            raise ValueError('stop_after "registration" with motion_correct_config "skip" has nothing to write')
        if data is None and not reuse_compression:
            raise ValueError("data is None starting from the motion correction step. Specify a dataset")
        resume, base = self.resume_source(resume, reuse_compression)
        self.run_config = {"frame_rate": frame_rate, "exclude_border_radius": exclude_border_radius,
                           "indicator_polarity": indicator_polarity, "remove_intermediates": remove_intermediates, "stop_after": stop_after,
                           "resume": None if resume is None else str(resume)}
        results_path = self.results_path(base)
        stored = None if resume is None else self.resume(resume, results_path, reuse_compression)
        # a resumed run without the raw movie has no registered movie to re-estimate raw traces from
        moco_data = None
        if data is not None:
            if self.load_into_ram:
                data = self.read_into_ram(data)
            data = self._construct_ophys_array(data,
                                               indicator_polarity=indicator_polarity,
                                               good_frame_indicator=good_frame_indicator)
            moco_data, shift_mask = self.motion_correct(data,
                                                        self.motion_correct_config,
                                                        results_path,
                                                        exclude_border_radius=exclude_border_radius,
                                                        stored=stored)
        if stop_after == "registration":
            return self.finish()

        if not reuse_compression:
            compress_strategy = self.compress_strategy(self.compress_config, shift_mask)
            compress_strategy.detrender = self.spline_detrender(data.shape[0], frame_rate, window_seconds=40,
                                                                knot_seconds=25, sigma_seconds=0.3)

            with self.step("compression"):
                compressed_results = compress_strategy.compress(moco_data)
                compressed_results.export(results_path)

        if stop_after == "compression":
            return self.finish()
        if isinstance(self.filtered_demixing_config, str):
            if self.filtered_demixing_config.lower() != "skip":
                raise ValueError(f"If filtered_demixing_config is a string, it can only be `skip`")
            return self.finish()

        device = self.torch_device
        pmd_denoise = masknmf.CompressionArray.from_hdf5(results_path)
        pmd_denoise.to(device)
        spatial_filt_pmd = masknmf.demixing.filters.spatial_filter_compressed_array(pmd_denoise,
                                                                                    batch_size=self.frame_batch_size,
                                                                                    filter_sigma=self.spatial_highpass_config.filter_sigma)
        pmd_denoise.to('cpu')
        spatial_filt_pmd.to('cpu')
        torch.cuda.empty_cache()


        ## Define the pixel batch size so that the max number of full pixels loaded matches frame batch size
        highpass_pmd_demixer = masknmf.demixing.signal_demixer.SignalDemixer(spatial_filt_pmd,
                                                                             device=device,
                                                                             frame_batch_size=self.frame_batch_size)

        detrender = self.spline_detrender(pmd_denoise.shape[0], frame_rate, window_seconds=20, knot_seconds=20,
                                          sigma_seconds=0.3)

        ## Use spline detrending to more effectively pick out signals. 1 knot point per 20 seconds of data
        filtered_demixing_config_used = self.with_detrender(self.filtered_demixing_config, detrender)

        curr_demix_results = self.run_multipass(highpass_pmd_demixer, filtered_demixing_config_used, "filtered demixing")

        if isinstance(self.unfiltered_demixing_config, str):
            # skipped: the filtered passes' results are the run's demixing
            latest_demix_results = curr_demix_results
        else:
            unfiltered_demixing_config_used = self.with_detrender(self.unfiltered_demixing_config, detrender)
            ## Define the unfiltered demixer object
            signals_array = curr_demix_results.signals_array
            a_init = signals_array.export_spatial_demixed()
            c_init = signals_array.export_temporal_demixed()

            del highpass_pmd_demixer, spatial_filt_pmd, curr_demix_results
            gc.collect()
            torch.cuda.empty_cache()

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
                MultipassDemixingConfig([custom_unfiltered_conf] + unfiltered_demixing_config_used.DemixingConfigs[1:]),
                "unfiltered demixing")

        with self.step("raw traces"):
            latest_demix_results.temporal_demixed_raw = masknmf.demixing.estimate_temporal_demixed_raw(
                latest_demix_results, moco_data, device=self.torch_device, nonneg=True, frame_batch_size=self.frame_batch_size)

        if good_frame_indicator is not None:
            latest_demix_results = masknmf.DemixingResults.infill_bad_frames(latest_demix_results,
                                                                  good_frame_indicator=good_frame_indicator)

        latest_demix_results.export(results_path)
        if remove_intermediates:
            self.drop_compression(results_path)
        return self.finish()





