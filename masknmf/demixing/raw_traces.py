import numpy as np
import torch
import math
from masknmf.arrays import ArrayLike
from masknmf.demixing._base_results import BaseResults
from masknmf.demixing.demixing_results import DemixingResults
from masknmf.demixing.local_correlation import local_correlation_image
from masknmf.demixing.regression_update import _fast_a_squared_norm
from masknmf.demixing.signal_demixer import _compute_hals_schedule
from masknmf.demixing.demixing_utils import csr_index_select_rows, csr_index_select_cols
from masknmf.utils import SparseCSRTensor



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
    clip = torch.relu if nonneg else (lambda x: x)
    a_sq_norm = _fast_a_squared_norm(a)  # (num_signals)
    ata = torch.sparse.mm(a.t(), a)

    for block in blocks:
        a_ia = csr_index_select_rows(ata, block)
        a_iac_block = torch.sparse.mm(a_ia, c.t())

        a_subset = csr_index_select_cols(a, block).t()
        projection = torch.sparse.mm(a_subset, movie.T) - a_iac_block  # Numerator for the regression
        projection /= a_sq_norm[block][:, None]

        c[:, block] = clip(c[:, block] + projection.T)
    return c



def estimate_temporal_demixed_raw(results: BaseResults,
                                  movie: np.ndarray | ArrayLike | None,
                                  device: torch.device | str,
                                  nonneg: bool = True,
                                  frame_batch_size: int = 300,
                                  noise: torch.Tensor | None = None,
                                  num_iters: int = 10) -> torch.Tensor | None:
    """
    temporal_demixed re-estimated from the registered raw movie with the spatial footprints, background and baseline
    of results held fixed, so it carries no compression or denoising. Shape (number of frames, number of signals)
    at the data scale of the noise whitened motion corrected movie

    For each batch of frames this solves  min_c || movie / noise - background - a c ||^2  by HALS,
    starting from temporal_demixed. Everything the regression needs from the movie is a^T (movie / noise), a
    (signals, frames) matrix, so each batch costs one sparse product with the movie; the background enters as
    (a^T U) term1 term2, and the HALS sweeps then only touch (signals, frames) matrices:

        c[block] += (a^T movie / noise - a^T background - (a^T a)[block] c) / ||a_block||^2

    Results of another pipeline have no noise image, so noise is 1 and the traces are at the scale of the movie.
    Their background is the frames of fluctuating_background_array, subtracted from each batch of the movie.
    """
    if hasattr(movie, "output_device"):
        original_device = movie.output_device
        movie.output_device = torch.device(device)
    else:
        original_device = None

    results.to(device)
    if movie is None:
        return None
    num_frames, fov_height, fov_width = movie.shape
    clip = torch.relu if nonneg else (lambda x: x)

    a = results.spatial_demixed.coalesce().float()                            # (pixels, signals)
    rows, cols = a.indices()
    if noise is not None:
        noise = noise.to(device=a.device, dtype=torch.float32, copy=True).flatten()
    elif results.noise_variance_image is not None:
        noise = results.noise_variance_image.flatten().to(device=a.device, dtype=torch.float32, copy=True)
    else:
        noise = torch.ones(a.shape[0], device=a.device)
    noise[noise <= 0] = 1.0

    # Per run, not per batch: everything below depends on the footprints and background factors only
    at_div_noise = torch.sparse_coo_tensor(torch.stack([cols, rows]), a.values() / noise[rows],
                                          size=(a.shape[1], a.shape[0])).coalesce().to_sparse_csr()  # (a / noise)^T
    a_csr = a.to_sparse_csr()
    ata = torch.sparse.mm(a_csr.t().to_sparse_csr(), a_csr).to_sparse_csr()  # (signals, signals)
    a_sq_norm = _fast_a_squared_norm(a_csr)
    blocks = _compute_hals_schedule(results.spatial_demixed, device, frame_batch_size=200)
    ata_rows = [csr_index_select_rows(ata, block) for block in blocks]
    norm_rows = [a_sq_norm[block][:, None] for block in blocks]

    at_background, background_array = None, None
    if isinstance(results, DemixingResults):
        term1, term2 = results.factorized_background_term1, results.factorized_background_term2
        if term1 is not None and term2 is not None:
            # a^T (U term1 term2[:, frames]); U and term1 are fixed, so a^T U term1 is (signals, background rank)
            at_u = torch.sparse.mm(a.t().coalesce(), results.spatial_compressed.coalesce().float())
            at_background = (at_u.to_dense() if at_u.is_sparse else at_u) @ term1.float()
    else:
        background_array = results.fluctuating_background_array

    c_new = results.temporal_demixed.clone().float()
    for start_pt in range(0, num_frames, frame_batch_size):
        end_pt = min(start_pt + frame_batch_size, num_frames)
        data = torch.as_tensor(movie[start_pt:end_pt], device=device, dtype=torch.float32)
        if background_array is not None:
            data = data - torch.as_tensor(background_array[start_pt:end_pt], device=device, dtype=torch.float32)
        rhs = torch.sparse.mm(at_div_noise, data.reshape(end_pt - start_pt, -1).T)  # (signals, frames)
        if at_background is not None:
            rhs -= at_background @ term2[:, start_pt:end_pt].float()

        c = c_new[start_pt:end_pt].T.contiguous()                                # (signals, frames)
        for _ in range(num_iters):
            for block, ata_block, norm in zip(blocks, ata_rows, norm_rows):
                c[block] = clip(c[block] + (rhs[block] - torch.sparse.mm(ata_block, c)) / norm)
        c_new[start_pt:end_pt] = c.T

    if hasattr(movie, "output_device"):
        movie.output_device = torch.device(original_device)
    return c_new


def estimate_raw_residual_correlation_image(results: DemixingResults,
                                            movie: np.ndarray | ArrayLike | None) -> torch.Tensor | None:
    """
    The local correlation image of the raw residual: movie minus the raw-scale signals and fluctuating background of
    results. Shape (height, width). movie becomes the motion_corrected_array of results. None when movie is None or
    results has no temporal_demixed_raw.
    """
    if movie is None:
        return None
    results.motion_corrected_array = movie
    if results.raw_residual_array is None:
        return None
    return local_correlation_image(results.raw_residual_array)


def _hals_on_a_trend(blocks: list[torch.Tensor],
                     spatial_trend: torch.Tensor,
                     temporal_trend: torch.Tensor,
                     a: SparseCSRTensor,
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
    a_sq_norm = _fast_a_squared_norm(a)  # (num_signals)

    for block in blocks:
        a_subset = csr_index_select_cols(a, block).t()
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
                          a: SparseCSRTensor,
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


def estimate_temporal_demixed_trend(a: SparseCSRTensor,
                  c: torch.Tensor,
                  pixelwise_normalizer: torch.Tensor,
                  spatial_trend_basis: torch.Tensor,
                  temporal_trend_basis: torch.Tensor,
                  batch_size: int = 200,
                  c_nonneg = False):
    """
    In very limited cases, we might want to specifically regress the per-pixel trends back onto the signals.
    This is a general utility function for demixing overlapping signals on the estimated trend of a movie.

    No background model here
    """
    device = a.device
    blocks = _compute_hals_schedule(a,
                                    device,
                                    frame_batch_size=batch_size)
    c_new = c.clone()
    spatial_trend_basis = torch.nan_to_num(spatial_trend_basis / pixelwise_normalizer.reshape(-1, 1), nan = 0.0, posinf=0.0, neginf=0.0)
    c_new = hals_multi_iter_trend(blocks,
                                  spatial_trend_basis,
                                  temporal_trend_basis,
                                  a,
                                  c_new,
                                  nonneg=c_nonneg)
    return c_new
