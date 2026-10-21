import masknmf
from masknmf.arrays import LazyFrameLoader, ArrayLike
from masknmf.utils import display

from masknmf.pipelines._base import BasePipeline
from masknmf.pipelines.configs.motion_correction_configs import (PiecewiseRigidMotionCorrectionConfig,
                                                                 MotionCorrectionConfigs, MotionCorrectionFilterConfig, SpatialHighpassMotionCorrectionConfig)
from masknmf.pipelines.configs.compression_configs import CompressDenoiseConfig, CompressionConfigs
from masknmf.pipelines.configs.demixing_configs import (NMFConfig, CustomInitConfig, SuperpixelInitConfig,
                                                        SpatialHighpassConfig, SpatialHighpassConfigs,
                                                        GlobalSignalFilterConfig,
                                                        CompressedFilterConfigs, SinglepassDemixingConfig,
                                                        MultipassDemixingConfig, MultipassDemixingConfigs)

from typing import *
import numpy as np
from pathlib import Path
import torch

class OnePhotonCalciumPipeline(BasePipeline):
    """
    Population calcium imaging pipeline for 1p imaging. Common characteristics: high background contamination,
    photobleaching, blood vessel contamination, etc.).

    Key ideas: motion correction shifts are estimated on highpass filtered frames, per-pixel spline detrending
    helps separate out photobleaching effects from the neural activity. To demix the signal, we subtract
    a global background estimate and/or spatially highpass filter the compressed data. We demix this filtered dataset
    and then use the results to demix signals from the unfiltered compressed movie.
    """
    def __init__(self,
                 motion_correct_config: MotionCorrectionConfigs | Literal["skip"] | None = None,
                 motion_correct_filter_config: MotionCorrectionFilterConfig | None = None,
                 compress_config: CompressionConfigs | Literal["skip"] | None = None,
                 global_filter_config: GlobalSignalFilterConfig | None = None,
                 spatial_highpass_config: SpatialHighpassConfigs | None = None,
                 filtered_demixing_config: MultipassDemixingConfigs | Literal["skip"] | None = None,
                 unfiltered_demixing_config: MultipassDemixingConfigs | None = None,
                 output_folder: str | Path | None = None,
                 frame_batch_size: int = 300,
                 device: Literal["auto", "cuda", "cpu"] = "auto",
                 log_level: Literal["debug", "info", "warning"] = "info",
                 load_into_ram: bool = False
                 ):
        super().__init__(output_folder=output_folder, frame_batch_size=frame_batch_size, device=device,
                         log_level=log_level,
                         load_into_ram=load_into_ram,
                         motion_correct_config=motion_correct_config,
                         motion_correct_filter_config=motion_correct_filter_config,
                         compress_config=compress_config,
                         global_filter_config=global_filter_config,
                         spatial_highpass_config=spatial_highpass_config,
                         filtered_demixing_config=filtered_demixing_config,
                         unfiltered_demixing_config=unfiltered_demixing_config)

    @classmethod
    def default_configs(cls) -> dict:
        """
        Piecewise rigid motion correction with shifts estimated on highpassed frames, compression with denoising,
        global-signal subtraction (rank 20) to make the filtered data, then two demixing passes on the filtered data
        and two on the unfiltered data. The second unfiltered pass is looser (0.5) and carries the background over.
        Demixing passes get the run's spline detrender, built from its frame rate.
        """
        default_sign = "positive" #Otherwise unconstrained or negative
        filtered_passes = []
        for i in range(2):
            init_conf = SuperpixelInitConfig(mad_correlation_threshold=0.8, min_peak_distance=10, mad_threshold=1,
                                               residual_threshold=0.3, patch_size=(50,50), sign=default_sign)
            nmf_conf = NMFConfig(maxiter=40,
                                 support_threshold=(0.95,0.7),
                                 ring_model_start_pt=None,
                                 sign=default_sign)
            filtered_passes.append(SinglepassDemixingConfig(init_conf, nmf_conf))

        # the init of the first unfiltered pass is replaced by the filtered results in run()
        first_unfiltered = SinglepassDemixingConfig(
            SuperpixelInitConfig(mad_correlation_threshold=0.8,
                                 min_peak_distance=10,
                                 patch_size=(50, 50),
                                 sign=default_sign),
            NMFConfig(maxiter=40,
                      support_threshold=(0.95, 0.7),
                      ring_model_start_pt=0,
                      ring_radius=50,
                      background_downsampling_factor=50,
                      sign=default_sign)
        )
        final_unfiltered = SinglepassDemixingConfig(
            SuperpixelInitConfig(mad_correlation_threshold=0.5,
                                 min_peak_distance=10,
                                 mad_threshold=1,
                                 residual_threshold=0.3,
                                 patch_size=(40, 40),
                                 carry_background=True,
                                 sign=default_sign),
            NMFConfig(maxiter=40,
                      support_threshold=(0.95, 0.6),
                      ring_model_start_pt=0,
                      ring_radius=50,
                      background_downsampling_factor=50,
                      sign=default_sign)
        )

        return {'motion_correct_config': PiecewiseRigidMotionCorrectionConfig(minimum_patch_sizes=(100, 100),
                                                                              overlaps=(30, 30),
                                                                              max_rigid_shifts=(30, 30),
                                                                              max_deviation_rigid=(3, 3)),
                'motion_correct_filter_config': SpatialHighpassMotionCorrectionConfig(reference_highpass_sigma=5.0),
                'compress_config': CompressDenoiseConfig(block_sizes=(32, 32)),
                'global_filter_config': GlobalSignalFilterConfig(rank=20),
                'spatial_highpass_config': SpatialHighpassConfig(filter_sigma=5.0),
                'filtered_demixing_config': MultipassDemixingConfig(filtered_passes),
                'unfiltered_demixing_config': MultipassDemixingConfig([first_unfiltered, final_unfiltered])}

    def _filtered_compression(self, compression_array: masknmf.CompressionArray) -> masknmf.CompressionArray:
        """The copy of the compressed data the filtered demixing passes run on.
        Note the composition. We first run the global filter (if the config is not None) and pass that result into the
        spatial filter step (if the highpass filter config is not None)"""
        if isinstance(self.global_filter_config, GlobalSignalFilterConfig):
            compression_array = masknmf.demixing.filters.filter_global_signal_compression_array(compression_array,
                                                                                                  rank=self.global_filter_config.rank,
                                                                                                  num_oversamples=self.global_filter_config.num_oversamples)

            compression_array.to(self.torch_device)
        if isinstance(self.spatial_highpass_config, SpatialHighpassConfigs):
            compression_array = masknmf.demixing.filters.spatial_filter_compressed_array(compression_array,
                                                                                         batch_size=self.frame_batch_size,
                                                                                         filter_sigma=self.spatial_highpass_config.filter_sigma)
        return compression_array

    @property
    def demixing_configs(self) -> list[SinglepassDemixingConfig | MultipassDemixingConfig]:
        return [self.filtered_demixing_config, self.unfiltered_demixing_config]


    def run(self,
            data: np.ndarray | ArrayLike | None,
            frame_rate: float,
            indicator_polarity: Literal["positive", "negative"] = "positive",
            good_frame_indicator: torch.Tensor | np.ndarray | None = None,
            exclude_border_radius: int = 0,
            remove_intermediates: bool = False,
            stop_after: Literal["registration", "compression", "demixing"] = "demixing",
            resume_from: str | Path | None = None,
            min_brightness: float | None = None) -> Path:
        """
        Motion corrects, compresses (with denoising) and demixes a 1p movie
        Args:
            data: The raw (frames, height, width) movie
            frame_rate (float): frames per second; sets the detrenders' windows
            indicator_polarity (Literal["positive", "negative"]): Whether the indicator is positively or negatively tuned
            good_frame_indicator (torch.Tensor | np.ndarray | None): An optional 1D vector indicating which frames are good vs. bad
            exclude_border_radius (int): Pixels at each fov edge that are ignored when compressing
            remove_intermediates (bool): Drop the CompressionArray group once demixing is done
            stop_after: the last stage to run: "registration" writes the shifts and template and ends,
                "compression" ends once the compression is written, "demixing" runs everything
            resume_from: an earlier results file copied into the new run folder: its registration is applied to
                data instead of estimating one, and with compress_config "skip" its compression is reused too.
                The settings not given, all but stop_after, are that run's, from the config.json beside it.
                The earlier file is left as it is
        """
        if min_brightness is not None:
            self._update_min_brightness(min_brightness)
            print(self.demixing_configs)
        reuse_compression = isinstance(self.compress_config, str)
        if reuse_compression and self.compress_config.lower() != "skip":
            raise ValueError(f"If compress_config is a string, it can only be `skip`")
        if reuse_compression and stop_after != "demixing":
            raise ValueError('compress_config "skip" reuses a compression for its demixing; stop_after leaves nothing to run')
        if stop_after == "registration" and isinstance(self.motion_correct_config, str) and resume_from is None:
            raise ValueError('stop_after "registration" with motion_correct_config "skip" has nothing to write')
        if data is None and not reuse_compression:
            raise ValueError("data is None starting from the motion correction step. Specify a dataset")
        resume_from, base = self.resume_source(resume_from, reuse_compression)
        self.run_config = {"frame_rate": frame_rate, "exclude_border_radius": exclude_border_radius,
                           "indicator_polarity": indicator_polarity, "remove_intermediates": remove_intermediates,
                           "stop_after": stop_after, "resume_from": None if resume_from is None else str(resume_from)}
        results_path = self.results_path(base)
        stored = None if resume_from is None else self.resume(resume_from, results_path, reuse_compression)
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
                                                        filter_config=self.motion_correct_filter_config,
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
        display("Running demixing analysis")

        pmd_denoise = masknmf.CompressionArray.from_hdf5(results_path)
        pmd_denoise.to(device)
        filtered_pmd = self._filtered_compression(pmd_denoise)
        torch.cuda.empty_cache()
        pmd_denoise.to('cpu') #Move data off CPU until it needs to be used

        filtered_demixer = masknmf.demixing.signal_demixer.SignalDemixer(filtered_pmd, device=device,
                                                                         frame_batch_size=self.frame_batch_size)

        detrender = self.spline_detrender(pmd_denoise.shape[0], frame_rate, window_seconds=20, knot_seconds=20,
                                          sigma_seconds=0.3)
        filtered_config_used = self.with_detrender(self.filtered_demixing_config, detrender)
        unfiltered_config_used = self.with_detrender(self.unfiltered_demixing_config, detrender)

        filtered_results = self.run_multipass(filtered_demixer, filtered_config_used, "filtered demixing")

        signals_array = filtered_results.signals_array
        a_init = signals_array.export_spatial_demixed()
        c_init = signals_array.export_temporal_demixed()
        del filtered_demixer, filtered_pmd, filtered_results, signals_array
        torch.cuda.empty_cache()

        pmd_denoise.to(device)
        # the first unfiltered pass starts from the filtered results instead of a fresh initialization
        custom_first_pass = SinglepassDemixingConfig(CustomInitConfig(a_init, c_init, c_nonneg=True),
                                                     unfiltered_config_used.DemixingConfigs[0].NMFConfig)
        unfiltered_demixer = masknmf.demixing.signal_demixer.SignalDemixer(pmd_denoise, device=device,
                                                                           frame_batch_size=self.frame_batch_size)
        final_results = self.run_multipass(
            unfiltered_demixer,
            MultipassDemixingConfig([custom_first_pass] + unfiltered_config_used.DemixingConfigs[1:]),
            "unfiltered demixing")
        if moco_data is not None:
            with self.step("raw traces"):
                final_results.temporal_demixed_raw = masknmf.demixing.estimate_temporal_demixed_raw(
                    final_results, moco_data, device=self.torch_device,
                    nonneg=True, frame_batch_size=self.frame_batch_size)

        if good_frame_indicator is not None:
            final_results = masknmf.DemixingResults.infill_bad_frames(final_results,
                                                                  good_frame_indicator=good_frame_indicator)
        final_results.export(results_path)
        if remove_intermediates:
            self.drop_compression(results_path)
        return self.finish()

