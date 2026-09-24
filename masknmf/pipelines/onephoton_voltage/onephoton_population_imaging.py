import masknmf
from masknmf.arrays import LazyFrameLoader, ArrayLike
from masknmf.motion_correction import BaseRegistrationArray, DummyMotionCorrector, RigidMotionCorrector, PiecewiseRigidMotionCorrector, GradientMotionCorrector, GradientRegistrationArray
from masknmf.utils import display
from masknmf.utils._serialization import save_dict
from dataclasses import replace
import torch
import math
from tqdm import tqdm

from masknmf.motion_correction.registration_arrays import OphysArray
from masknmf.pipelines._base import BasePipeline
from masknmf.pipelines.configs.motion_correction_configs import GradientMotionCorrectionConfig
from masknmf.pipelines.configs.compression_configs import CompressDenoiseConfig, CompressionConfigs
from masknmf.pipelines.configs.demixing_configs import NMFConfig, CustomInitConfig, SuperpixelInitConfig, SpatialHighpassConfig, SinglepassDemixingConfig, MultipassDemixingConfig, MultipassDemixingConfigs

from typing import *
import numpy as np
from pathlib import Path



def _hals_on_a_trend(blocks: list[torch.Tensor],
                     spatial_trend: torch.Tensor,
                     temporal_trend: torch.Tensor,
                     a: torch.sparse_coo_tensor,
                     c: torch.Tensor,
                     nonneg: bool = False):
    """
    This is a fast routine to do HALS on some frames of raw data. This routine assumes you've subtracted of all the stuff you don't want (background, etc.) from the movie tensor,
    so all that's left to do is run the HALS regression.

    Args:
        blocks (list[torch.Tensor]): A list of tensors. The indices in a single tensor describe neurons that can be updated in parallel
        spatial_trend (torch.Tensor): Shape (num_pixels, trend_rank)
        temporal_trend (torch.Tensor): Shape (trend_rank, num_frames)
        a (torch.sparse_coo_tensor): Shape (num_pixels, num_signals). A sparse tensor where each column describes the spatial footprints of the cells
        c (torch.Tensor): Shape (num_frames, num_signals). A tensor describing the temporal profiles of all signals
    """
    atac = torch.sparse.mm(a.t(), a) @ c.T  # Shape (num_signals, num_frames)
    a_sq_norm = masknmf.demixing.regression_update._fast_a_squared_norm(a)  # (num_signals)

    for block in blocks:
        a_subset = torch.index_select(a, 1, block).t().coalesce()
        projection = torch.sparse.mm(a_subset, spatial_trend) @ temporal_trend - atac[
            block]  # Numerator for the regression
        projection /= a_sq_norm[block][:, None]

        c[:, block] += projection.T
        if nonneg:
            c.clamp_(min=0)
    return c


def hals_multi_iter_trend(block: list[torch.Tensor],
                          spatial_trend_basis: torch.Tensor,
                          temporal_trend_basis: torch.Tensor,
                          a: torch.sparse_coo_tensor,
                          c: torch.Tensor,
                          nonneg: bool = False,
                          num_iters=10):
    for k in range(num_iters):
        c = _hals_on_a_trend(block,
                             spatial_trend_basis,
                             temporal_trend_basis,
                             a,
                             c,
                             nonneg=nonneg)

    return c


def hals_on_trend(a: torch.sparse_coo_tensor,
                  c: torch.Tensor,
                  spatial_trend_basis: torch.Tensor,
                  temporal_trend_basis: torch.Tensor,
                  batch_size: int = 200,
                  num_iters: int = 10,
                  device='cuda'):
    """
    Here we want to run hierarchical alternating least squares (unconstrained) to update the temporal components of each signal w.r.t. the raw data
    """
    blocks = masknmf.demixing.signal_demixer._compute_hals_schedule(a,
                                                                    device,
                                                                    frame_batch_size=batch_size)
    c_new = c.clone()
    c_new = hals_multi_iter_trend(blocks,
                                  spatial_trend_basis,
                                  temporal_trend_basis,
                                  a,
                                  c_new,
                                  nonneg=False)
    return c_new




def hals_multi_iter_fullpmd(u,
                            v,
                            a,
                            c,
                            b,
                            c_nonneg: bool = False,
                            num_iters: int = 10,
                            batch_size: int = 200):
    device = u.device
    batch_size = batch_size
    blocks = masknmf.demixing.signal_demixer._compute_hals_schedule(a,
                                                                    device,
                                                                    frame_batch_size=batch_size)
    c_new = c.clone()
    for k in range(num_iters):
        c_new = masknmf.demixing.regression_update.temporal_update_hals(u,
                                                                        v,
                                                                        a,
                                                                        c_new,
                                                                        b,
                                                                        blocks=blocks,
                                                                        c_nonneg=c_nonneg)

    return c_new


#### Below is code to rescale "a" to match the raw data scale

def rescale_a(a: torch.sparse_coo_tensor,
              var_img: torch.Tensor):
    row, col = a.indices()
    values = a.values()
    var_img_indexed_values = var_img.flatten()[row]
    new_values = values * var_img_indexed_values

    new_a = torch.sparse_coo_tensor(a.indices(), new_values, a.shape)
    return new_a


#### Below is code to run HALS on the raw data:

def hals_on_rawdata(moco_data: np.ndarray,
                    a: torch.sparse_coo_tensor,
                    c: torch.Tensor,
                    batch_size: int = 200,
                    num_iters: int = 10,
                    device='cuda'):
    """
    Here we want to run hierarchical alternating least squares (unconstrained) to update the temporal components of each signal w.r.t. the raw data

    For now assumes there is no background/baseline
    """
    device = a.device
    num_batches = math.ceil(moco_data.shape[0] / batch_size)
    frames, height, width = moco_data.shape
    # mean_image = dmr.mean_image
    # noise_variance_image = dmr.noise_variance_image
    # noise_variance_image[noise_variance_image == 0] = 1.0 #Avoids divide by 0 issues
    # fluctuating_background_array = dmr.fluctuating_background_array
    # baseline = dmr.baseline
    blocks = masknmf.demixing.signal_demixer._compute_hals_schedule(a,
                                                                    device,
                                                                    frame_batch_size=200)  ##TODO: set this in a principled way later
    c_new = c.clone()
    for k in tqdm(range(num_batches)):
        start_pt = batch_size * k
        end_pt = min(moco_data.shape[0], start_pt + batch_size)
        data = torch.as_tensor(moco_data[start_pt:end_pt, :, :], device=device,
                               dtype=torch.float32)  # frames, height, width
        # data -= mean_image[None, ...]
        # data /= noise_variance_image[None, ...]
        # data -= fluctuating_background_array.getitem_tensor(slice(start_pt, end_pt))
        # data -= baseline[None, ...]

        c_new[start_pt:end_pt, :] = hals_multi_iter_raw(blocks,
                                                        data.reshape(data.shape[0], height * width),
                                                        a,
                                                        c_new[start_pt:end_pt, :])

    return c_new


def _hals_on_raw(blocks: list[torch.Tensor],
                 movie: torch.Tensor,
                 a: torch.sparse_coo_tensor,
                 c: torch.Tensor,
                 nonneg: bool = False):
    """
    This is a fast routine to do HALS on some frames of raw data. This routine assumes you've subtracted of all the stuff you don't want (background, etc.) from the movie tensor,
    so all that's left to do is run the HALS regression.

    Args:
        blocks (list[torch.Tensor]): A list of tensors. The indices in a single tensor describe neurons that can be updated in parallel
        movie (torch.Tensor): Shape (num_frames, num_pixels)
        a (torch.sparse_coo_tensor): Shape (num_pixels, num_signals). A sparse tensor where each column describes the spatial footprints of the cells
        c (torch.Tensor): Shape (num_frames, num_signals). A tensor describing the temporal profiles of all signals
    """
    atac = torch.sparse.mm(a.t(), a) @ c.T  # Shape (num_signals, num_frames)
    a_sq_norm = masknmf.demixing.regression_update._fast_a_squared_norm(a)  # (num_signals)

    for block in blocks:
        a_subset = torch.index_select(a, 1, block).t().coalesce()
        projection = torch.sparse.mm(a_subset, movie.T) - atac[block]  # Numerator for the regression
        projection /= a_sq_norm[block][:, None]

        c[:, block] += projection.T
    return c


def hals_multi_iter_raw(block: list[torch.Tensor],
                        movie: torch.Tensor,
                        a: torch.sparse_coo_tensor,
                        c: torch.Tensor,
                        nonneg: bool = False,
                        num_iters=10):
    for k in range(num_iters):
        c = _hals_on_raw(block,
                         movie,
                         a,
                         c,
                         nonneg=nonneg)
    return c


def compute_final_denoised_c_estimates(pmd_arr: masknmf.CompressionArray,
                                       dmr: masknmf.DemixingResults,
                                       c: torch.Tensor):
    """
    The demixer gives us an estimate of "c" (shape num_frames x num_neurons).
    This workflow performs the steps needed to re-incorporate subthreshold trends back into this "c" matrix and rescale
    the estimates back to the raw data space, so that we can revisit the raw data and get any missed signal
    Args:
        pmd_arr (masknmf.CompressionArray)
        dmr (masknmf.DemixingResults)
        c (torch.Tensor): Shape (num_frames, num_neurons). Initial temporal estimates of the spiking activity
    """

    c_spike_estimate = hals_multi_iter_fullpmd(pmd_arr.spatial_compressed,
                                               pmd_arr.temporal_compressed,
                                               dmr.spatial_demixed,
                                               c,
                                               dmr.static_baseline[:, None])

    rescaled_a = rescale_a(dmr.spatial_demixed, pmd_arr.noise_variance_image).coalesce()
    c_trend_estimate = hals_on_trend(rescaled_a,
                                     c_spike_estimate,
                                     pmd_arr.spatial_trend_basis,
                                     pmd_arr.temporal_trend_basis)

    c_total_estimate = c_trend_estimate + c_spike_estimate
    return rescaled_a, c_total_estimate


def expand_traces_to_all_frames(c: torch.Tensor,
                                active_frames: np.ndarray):
    """
    The demixed neural activity traces are estimated on a subset of the frames. The final estimates we want to produce
    should reflect all frames of data, so this procedure just places the existing estimates into an expanded tensor
    describing activity at all frames.
    """

    updated_c = torch.zeros(active_frames.shape[0], c.shape[1], device=c.device)
    frame_subset = active_frames.astype('bool')
    populated_indices = np.arange(active_frames.shape[0])[frame_subset]
    updated_c[populated_indices, :] = c.clone()[:]
    return updated_c


class OnePhotonCulturePipeline(BasePipeline):
    def __init__(self,
                 motion_correct_config: GradientMotionCorrectionConfig | Literal["skip"] | None = None,
                 compress_config: CompressionConfigs | Literal["skip"] | None = None,
                 demixing_config: MultipassDemixingConfigs | None = None,
                 output_folder: str | Path | None = None,
                 load_into_ram: bool = False,
                 frame_batch_size: int = 300,
                 device: Literal["auto", "cuda", "cpu"] = "auto"
                 ):
        """
        Every config left as None takes its value from default_configs().
        """
        defaults = self.default_configs()
        self._motion_correct_config = defaults['motion_correct_config'] if motion_correct_config is None else motion_correct_config
        self._compress_config = defaults['compress_config'] if compress_config is None else compress_config
        self._demixing_config = defaults['demixing_config'] if demixing_config is None else demixing_config
        self._load_into_ram = load_into_ram
        super().__init__(output_folder, frame_batch_size, device)

    @classmethod
    def default_configs(cls) -> dict:
        """
        Gradient motion correction, compression with denoising, and two positive-signed demixing passes without
        detrending.
        """
        conf_list = []
        for corr_threshold, support_threshold in [(0.8, 0.7), (0.8, 0.5)]:
            curr_init_conf = SuperpixelInitConfig(mad_correlation_threshold=corr_threshold,
                                                  detrender=None,  # If we truncate the frames, detrending should be off
                                                  sign="positive",
                                                  residual_threshold=0.1)  # Only prioritize positive deviations to pick spikes
            curr_nmf_conf = NMFConfig(support_threshold=(0.95, support_threshold),
                                      ring_model_start_pt=None,
                                      merge_overlap_threshold=0.9,
                                      detrender=None)
            curr_demix_conf = SinglepassDemixingConfig(curr_init_conf, curr_nmf_conf)
            conf_list.append(curr_demix_conf)

        return {'motion_correct_config': GradientMotionCorrectionConfig(),
                'compress_config': CompressDenoiseConfig(),
                'demixing_config': MultipassDemixingConfig(conf_list)}

    @property
    def motion_correct_config(self) -> GradientMotionCorrectionConfig | Literal["skip"]:
        return self._motion_correct_config

    @property
    def compress_config(self) -> CompressionConfigs | Literal["skip"]:
        return self._compress_config

    @property
    def demixing_config(self) -> MultipassDemixingConfigs:
        return self._demixing_config

    @property
    def load_into_ram(self) -> bool:
        return self._load_into_ram

    @property
    def config(self):
        return {'motion_correct_config': self.motion_correct_config,
                'compress_config': self.compress_config,
                'demixing_config': self.demixing_config,
                'output_folder': self.output_folder,
                'frame_batch_size': self.frame_batch_size,
                'device': self.device}

    def run(self,
            data: np.ndarray | ArrayLike,
            frame_rate: float,
            indicator_sign: Literal["negative", "positive"],
            active_frames: np.ndarray,
            remove_intermediates: bool = True) -> Path:
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
                    output_folder: Every stage is written to ``<output_folder>/<timestamp>_one-photon-culture/results.hdf5``,
                        one hdf5 group per stage. With compress_config "skip", output_folder is instead an existing run
                        folder whose results.hdf5 holds the compression; demixing is written into that same file
                    load_into_ram (bool): Whether or not to load the full dataset into RAM for faster processing
                    remove_intermediates (bool): drop the PMDArray group once demixing is done (the demixing
                        results carry the pmd)

                The raw-scale footprints and the denoised and raw-regressed traces over all frames are written to the
                RawScaleEstimates group as a, c_denoised and c_raw.
                """

        device = self.torch_device

        ## Decide whether to motion correct data or not. You must have access to raw data
        negative_indicator = True if indicator_sign == "negative" else False
        if isinstance(self.motion_correct_config, str):
            if self.motion_correct_config.lower() == "skip":
                moco_array = OphysArray(data,
                                        negative_indicator=negative_indicator,
                                        include_mean=True,
                                        device=device)
            else:
                raise ValueError("Invalid MotionCorrectionConfig input")
        else:
            mov = OphysArray(data,
                             negative_indicator=negative_indicator,
                             include_mean=True,
                             device=device)

            mean_img = torch.mean(mov[:self.motion_correct_config.num_frames_template], dim=0)
            corrector = GradientMotionCorrector(template=mean_img)
            moco_array = corrector.motion_correct(mov)
            moco_array.output_device=device

        if isinstance(self.compress_config, str):
            if self.compress_config.lower() == "skip":
                results_path = self.results_path(resume=True)
            else:
                raise ValueError(f"If compress_config is a string, it can only be `skip`")
        else:
            results_path = self.results_path()

            display("Running Compression")

            ## Add the run-specific frame weighting to a copy of the config, so the pipeline's own is untouched
            if self.compress_config.frame_weighting is not None:
                frame_weighting = self.compress_config.frame_weighting * active_frames.astype(self.compress_config.frame_weighting.dtype)
            else:
                frame_weighting = active_frames

            compress_strategy = self.compress_strategy(replace(self.compress_config, frame_weighting=frame_weighting))

            compress_strategy.detrender = self.spline_detrender(moco_array.shape[0], frame_rate, window_seconds=0.05,
                                                                knot_seconds=0.05, sigma_seconds=0.01)
            compressed_results = compress_strategy.compress(moco_array)
            compressed_results.export(results_path)

        device = self.torch_device
        display("Running demixing analysis")

        pmd_denoise = masknmf.CompressionArray.from_hdf5(results_path)

        v = pmd_denoise.temporal_compressed[:, active_frames.astype('bool')]
        new_shape = (v.shape[1], pmd_denoise.shape[1], pmd_denoise.shape[2])

        pmd_arr_truncated = masknmf.CompressionArray.from_tensors(new_shape,  # fov shape
                                                                  pmd_denoise.spatial_compressed,
                                                                  v,
                                                                  pmd_denoise.mean_image,
                                                                  pmd_denoise.noise_variance_image,
                                                                  pmd_denoise.spatial_compressed_local_projector,
                                                                  pmd_denoise.spatial_trend_basis,
                                                                  pmd_denoise.temporal_trend_basis)



        truncated_pmd_demixer = masknmf.demixing.signal_demixer.SignalDemixer(pmd_arr_truncated,
                                                                              device=device,
                                                                              frame_batch_size=self.frame_batch_size)


        curr_demix_results = self.run_multipass(truncated_pmd_demixer, self.demixing_config)

        """
        The rest of the pipeline involves: 
        - Filling in the time series for frames that were ignored in the demixing procedure
        - Re-scaling the results to match the raw data
        - Re-incorporating any subthreshold trends from the PMD demixing
        """
        pmd_denoise.to(device)
        curr_demix_results.to(device)

        c_all_frames = expand_traces_to_all_frames(curr_demix_results.temporal_demixed,
                                                   active_frames)

        a_rawdata_scale, full_c_estimate_denoised = compute_final_denoised_c_estimates(pmd_denoise,
                                                                                 curr_demix_results,
                                                                                 c_all_frames)

        c_regressed_on_raw = hals_on_rawdata(moco_array,
                                             a_rawdata_scale,
                                             full_c_estimate_denoised)



        curr_demix_results.export(results_path)
        save_dict({"a": a_rawdata_scale, "c_denoised": full_c_estimate_denoised, "c_raw": c_regressed_on_raw},
                  filename=results_path, group="RawScaleEstimates", exists_ok=True)
        if remove_intermediates:
            self.drop_compression(results_path)

        return Path(results_path).parent





