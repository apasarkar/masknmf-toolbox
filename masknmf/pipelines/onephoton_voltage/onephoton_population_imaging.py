from dataclasses import asdict
import masknmf
from masknmf.compression import CompressStrategy, CompressDenoiseStrategy
from masknmf.arrays import LazyFrameLoader, ArrayLike
from masknmf.motion_correction import RegistrationArray, DummyMotionCorrector, RigidMotionCorrector, PiecewiseRigidMotionCorrector
from masknmf.utils import display
from masknmf.demixing import NoSignalsDetectedError, DemixingError
import torch
import math
from tqdm import tqdm

from masknmf.compression.preprocessing import MaximinSplineDetrend

from masknmf.motion_correction.registration_arrays import VoltageArray
from masknmf.pipelines._base import BasePipeline
from masknmf.pipelines.configs.compression_configs import CompressConfig, CompressDenoiseConfig
from masknmf.pipelines.configs.demixing_configs import NMFConfig, CustomInitConfig, SuperpixelInitConfig, SpatialHighpassConfig, SinglepassDemixingConfig, MultipassDemixingConfig

from masknmf.utils import torch_select_device
from typing import *
import numpy as np
import os



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
    # mean_img = dmr.mean_img
    # var_img = dmr.var_img
    # var_img[var_img == 0] = 1.0 #Avoids divide by 0 issues
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
        # data -= mean_img[None, ...]
        # data /= var_img[None, ...]
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


def compute_final_denoised_c_estimates(pmd_arr: masknmf.PMDArray,
                                       dmr: masknmf.DemixingResults,
                                       c: torch.Tensor):
    """
    The demixer gives us an estimate of "c" (shape num_frames x num_neurons).
    This workflow performs the steps needed to re-incorporate subthreshold trends back into this "c" matrix and rescale
    the estimates back to the raw data space, so that we can revisit the raw data and get any missed signal
    Args:
        pmd_arr (masknmf.PMDArray)
        dmr (masknmf.DemixingResults)
        c (torch.Tensor): Shape (num_frames, num_neurons). Initial temporal estimates of the spiking activity
    """

    c_spike_estimate = hals_multi_iter_fullpmd(pmd_arr.u,
                                               pmd_arr.v,
                                               dmr.a,
                                               c,
                                               dmr.b[:, None])

    rescaled_a = rescale_a(dmr.a, pmd_arr.var_img).coalesce()
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


class OnePhotonCulturePipeline(BasePipeline):
    def __init__(self,
                 motion_correct_config: Literal["skip"] | None = None,
                 compress_config: CompressConfig | CompressDenoiseConfig | Literal["skip"] | None = None,
                 demixing_config: MultipassDemixingConfig | None = None,
                 outpath_compression: Optional[str] = "compression.hdf5",
                 outpath_demixing: Optional[str] = "demixing_results.hdf5",
                 load_into_ram: bool = False,
                 frame_batch_size: int = 300,
                 device: Literal["auto", "cuda", "cpu"] = "auto"
                 ):
        self._motion_correct_config = motion_correct_config

        ## Set the compress config
        if compress_config is None:
            curr_config = CompressDenoiseConfig()
        else:
            curr_config = compress_config

        self._compress_config = curr_config
        self._outpath_compression = outpath_compression
        self._outpath_demixing = outpath_demixing
        self._load_into_ram = load_into_ram
        self._frame_batch_size = frame_batch_size
        self._device = device

        if demixing_config is None:
            conf_list = []
            for corr_threshold, support_threshold in [(0.8, 0.7), (0.8, 0.5)]:
                curr_init_conf = SuperpixelInitConfig(mad_correlation_threshold=corr_threshold,
                                                      detrender=None,  # If we truncate the frames, detrending should be off
                                                      sign="positive")  # Only prioritize positive deviations for 2p calcium imaging
                curr_nmf_conf = NMFConfig(support_threshold=(0.95, support_threshold),
                                          ring_model_start_pt=None,
                                          merge_overlap_threshold=0.9,
                                          detrender=None)
                curr_demix_conf = SinglepassDemixingConfig(curr_init_conf, curr_nmf_conf)
                conf_list.append(curr_demix_conf)
            self._demixing_config = MultipassDemixingConfig(conf_list)
        else:
            self._demixing_config = demixing_config

    @property
    def motion_correct_config(self) -> Literal["skip"] | None:
        """
        For now the config here is hard coded -- either you skip it or run the gradient corrector out of the box
        """
        return self._motion_correct_config

    @property
    def compress_config(self) -> CompressConfig | CompressDenoiseConfig | None:
        return self._compress_config

    @property
    def demixing_config(self) -> MultipassDemixingConfig:
        return self._demixing_config

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
                'outpath_compression': self.outpath_compression,
                'outpath_demixing': self.outpath_demixing,
                'load_into_ram': self.load_into_ram,
                'frame_batch_size': self.frame_batch_size,
                'device': self.device}

    def run(self,
            data: np.ndarray | ArrayLike,
            frame_rate: float,
            indicator_sign: Literal["negative", "positive"],
            active_frames: np.ndarray,
            remove_intermediates: bool = True):
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

        device = torch_select_device()
        if isinstance(self.compress_config, str):
            if self.compress_config.lower() == "skip":
                if not os.path.exists(self.outpath_compression):
                    raise ValueError("You specified that compression should be skipped but did not specify a valid location for the "
                                     "compression hdf5 file")
            else:
                raise ValueError(f"If compress_config is a string, it can only be `skip`")
        else:
            ## Decide whether to motion correct data or not
            if isinstance(self.motion_correct_config, str):
                if self.motion_correct_config.lower() == "skip":
                    moco_array = data
                else:
                    raise ValueError("Invalid MotionCorrectionConfig input")
            else:
                negative_indicator = True if indicator_sign == "negative" else False
                print(f"negative indicator is {negative_indicator}")
                mov = masknmf.VoltageArray(data,
                                           negative_indicator=negative_indicator,
                                           include_mean=True,
                                           device=device)

                ## TODO: The template estimation should be something the user can more cleanly specify
                mean_img = torch.mean(mov[:300], dim=0)
                corrector = masknmf.GradientMotionCorrector(template=mean_img)
                moco_array = masknmf.GradientRegistrationArray(mov,
                                                               corrector,
                                                               output_device=device)

            display("Running Compression")

            ## First add in the run-specific parameters to the config object:
            self.compress_config.device=device
            if self.compress_config.frame_weighting is not None:
                self.compress_config.frame_weighting *= active_frames.astype(self.compress_config.frame_weighting.dtype)
            else:
                self.compress_config.frame_weighting = active_frames
            self.compress_config.frame_batch_size = self.frame_batch_size

            ## Make the strategy object
            if isinstance(self.compress_config, CompressConfig):
                compress_strategy = CompressStrategy(**asdict(self.compress_config))
            elif isinstance(self.compress_config, CompressDenoiseConfig):
                compress_strategy = CompressDenoiseStrategy(**asdict(self.compress_config))
            else:
                raise ValueError("Invalid config")

            num_frames = moco_array.shape[0]
            recording_seconds = num_frames / frame_rate
            window = int(0.05 * frame_rate)  # rolling window time interval
            sigma = max(2.0, 0.01 * frame_rate)  # less smoothing
            num_knots = max(4, int(recording_seconds / 0.05))  # one knot per 0.05 seconds

            detrender_device = (
                torch_select_device() if self.device == "auto" else self.device
            )

            detrender = MaximinSplineDetrend(
                num_frames=num_frames,
                num_knots=num_knots,
                window=window,
                sigma=sigma,
                device=detrender_device,
            )

            compress_strategy.detrender = detrender
            compress_strategy.frame_batch_size = self.frame_batch_size

            compressed_results = compress_strategy.compress(moco_array)
            compressed_results.export(self.outpath_compression)

        if self.device == "auto":
            device = torch_select_device()
        else:
            device = self.device
        display("Running demixing analysis")

        pmd_denoise = masknmf.PMDArray.from_hdf5(self.outpath_compression)

        v = pmd_denoise.v[:, active_frames.astype('bool')]
        new_shape = (v.shape[1], pmd_denoise.shape[1], pmd_denoise.shape[2])

        pmd_arr_truncated = masknmf.PMDArray.from_tensors(new_shape,  # fov shape
                                                          pmd_denoise.u,
                                                          v,
                                                          pmd_denoise.mean_img,
                                                          pmd_denoise.var_img,
                                                          pmd_denoise.u_local_projector,
                                                          pmd_denoise.spatial_trend_basis,
                                                          pmd_denoise.temporal_trend_basis)



        truncated_pmd_demixer = masknmf.demixing.signal_demixer.SignalDemixer(pmd_arr_truncated,
                                                                              device=device,
                                                                              frame_batch_size=self.frame_batch_size)


        # Run demixing
        curr_demix_results = None
        for k in range(len(self.demixing_config.DemixingConfigs)):
            truncated_pmd_demixer = run_singlepass_demixing(truncated_pmd_demixer,
                                                           self.demixing_config.DemixingConfigs[k])
            if truncated_pmd_demixer is not None:
                curr_demix_results = truncated_pmd_demixer.results
            if truncated_pmd_demixer is None:
                if curr_demix_results is None:
                    raise ValueError("The demixer did not identify any signals in the highpass filtered movie. Lower thresholds or inspect"
                                     "data to resolve this issue.")
                else:
                    break

        """
        The rest of the pipeline involves: 
        - Filling in the time series for frames that were ignored in the demixing procedure
        - Re-scaling the results to match the raw data
        - Re-incorporating any subthreshold trends from the PMD demixing
        """
        pmd_denoise.to(device)
        curr_demix_results.to(device)

        c_all_frames = expand_traces_to_all_frames(curr_demix_results.c,
                                                   active_frames)

        a_rawdata_scale, full_c_estimate_denoised = compute_final_denoised_c_estimates(pmd_denoise,
                                                                                 curr_demix_results,
                                                                                 c_all_frames)

        c_regressed_on_raw = hals_on_rawdata(moco_array,
                                             a_rawdata_scale,
                                             full_c_estimate_denoised)



        if os.path.exists(os.path.abspath(self.outpath_demixing)):
            os.remove(os.path.abspath(self.outpath_demixing))
        curr_demix_results.export(os.path.abspath(self.outpath_demixing))

        if remove_intermediates:
            display("Removing intermediates")
            os.remove(os.path.abspath(self.outpath_compression))

        return curr_demix_results, a_rawdata_scale, full_c_estimate_denoised, c_regressed_on_raw





