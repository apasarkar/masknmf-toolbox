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

class FrameWindow(LazyFrameLoader):
    """A lazy view of frames [start, stop) of a movie. Nothing is read until frames are asked for."""

    def __init__(self, movie: ArrayLike, start: int, stop: int):
        self._movie = movie
        self._start = start
        self._stop = stop

    @property
    def dtype(self):
        return self._movie.dtype

    @property
    def shape(self) -> tuple[int, int, int]:
        return (self._stop - self._start, *self._movie.shape[1:])

    @property
    def ndim(self) -> int:
        return 3

    def _compute_at_indices(self, indices: list | int | slice | np.ndarray) -> np.ndarray:
        num_frames = self.shape[0]
        if isinstance(indices, slice):
            first, last, step = indices.indices(num_frames)
            return self._movie[self._start + first:self._start + last:step]
        if isinstance(indices, (int, np.integer)):
            return self._movie[self._start + (int(indices) % num_frames)]
        return self._movie[np.asarray(indices) % num_frames + self._start]

# frames dropped from each end of the movie
CROP_FRAMES = 300
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

        if isinstance(self.spatial_highpass_config, SpatialHighpassConfigs):
            compression_array = masknmf.demixing.filters.spatial_filter_compressed_array(compression_array,
                                                                                         batch_size=self.frame_batch_size,
                                                                                         filter_sigma=self.spatial_highpass_config.filter_sigma)
        return compression_array

    def run(self,
            data: np.ndarray | ArrayLike | None,
            frame_rate: float,
            exclude_border_radius: int = 0,
            remove_intermediates: bool = True) -> Path:
        """
        Motion corrects, compresses (with denoising) and demixes a 1p movie. The first and last CROP_FRAMES (300) frames
        are dropped
        Args:
            data: The raw (frames, height, width) movie
            frame_rate (float): frames per second; sets the detrenders' windows
            exclude_border_radius (int): Pixels at each fov edge that are ignored when compressing
            remove_intermediates (bool): Drop the CompressionArray group once demixing is done
        """
        self.run_config =  {"frame_rate": frame_rate, "exclude_border_radius": exclude_border_radius,
                           "crop_frames": CROP_FRAMES, "remove_intermediates": remove_intermediates}
        moco_data = None
        if isinstance(self.compress_config, str):
            if self.compress_config.lower() == "skip":
                results_path = self.results_path(resume=True)
            else:
                raise ValueError(f"If compress_config is a string, it can only be `skip`")
        else:
            results_path = self.results_path()
            if data is None:
                raise ValueError("data is None need to specify dataset")
            if data.shape[0] <= 2 * CROP_FRAMES:
                raise ValueError(f"Movie has {data.shape[0]} frames; it needs more than {2 * CROP_FRAMES} "
                                 f"to drop {CROP_FRAMES} from each end")
            # data = FrameWindow(data, CROP_FRAMES, data.shape[0] - CROP_FRAMES)
            data = data

            moco_data, shift_mask = self.motion_correct(data,
                                                        self.motion_correct_config,
                                                        results_path,
                                                        filter_config=self.motion_correct_filter_config,
                                                        exclude_border_radius=exclude_border_radius)

            display("Running Compression")
            compress_strategy = self.compress_strategy(self.compress_config, shift_mask)
            compress_strategy.detrender = self.spline_detrender(data.shape[0], frame_rate, window_seconds=40,
                                                                knot_seconds=25, sigma_seconds=0.3)
            with self.step("compression"):
                compressed_results = compress_strategy.compress(moco_data)
                compressed_results.export(results_path)

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

        filtered_results = self.run_multipass(filtered_demixer, filtered_config_used)

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
            MultipassDemixingConfig([custom_first_pass] + unfiltered_config_used.DemixingConfigs[1:]))
        if moco_data is not None:
            with self.step("raw traces"):
                final_results.temporal_demixed_raw = masknmf.demixing.estimate_temporal_demixed_raw(
                    final_results, moco_data, device=self.torch_device,
                    nonneg=True, frame_batch_size=self.frame_batch_size)
        final_results.export(results_path)
        if remove_intermediates:
            self.drop_compression(results_path)
        return self.finish()

