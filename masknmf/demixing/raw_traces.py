import numpy as np
import torch
import math
from masknmf.arrays import ArrayLike
from masknmf.demixing.demixing_results import DemixingResults
from masknmf.demixing.regression_update import _fast_a_squared_norm
from masknmf.demixing.signal_demixer import _compute_hals_schedule


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
        a_ia = torch.index_select(ata, 0, block)
        a_iac_block = torch.sparse.mm(a_ia, c.t())

        a_subset = torch.index_select(a, 1, block).t().coalesce()
        projection = torch.sparse.mm(a_subset, movie.T) - a_iac_block  # Numerator for the regression
        projection /= a_sq_norm[block][:, None]

        c[:, block] = clip(c[:, block] + projection.T)
    return c

def estimate_temporal_demixed_raw(results: DemixingResults,
                                  movie: np.ndarray | ArrayLike | None,
                                  device: torch.device | str,
                                  nonneg: bool = True,
                                  frame_batch_size: int = 300) -> torch.Tensor | None:
    """
    temporal_demixed re-estimated from the registered raw movie with the spatial footprints, background and baseline
    of results held fixed, so it carries no compression or denoising. Shape (number of frames, number of signals),
    in temporal_demixed units. None when there is no movie to go back to.
    """
    results.to(device)
    if movie is None:
        return None
    num_iters = math.ceil(movie.shape[0] / frame_batch_size)

    blocks = _compute_hals_schedule(results.spatial_demixed,
                                    device,
                                    frame_batch_size=200)  ##TODO: set this in a principled way later

    num_frames, fov_height, fov_width = movie.shape
    fluctuating_background = results.fluctuating_background_array
    baseline_image = results.baseline_image #(fov_height, fov_width)
    mean_image = results.mean_image
    noise_variance_image = results.noise_variance_image.clone()
    noise_variance_image[noise_variance_image <= 0] = 1.0
    c_new = results.temporal_demixed.clone()
    for k in range(num_iters):
        start_pt = frame_batch_size * k
        end_pt = min(start_pt + frame_batch_size, movie.shape[0])
        data = torch.as_tensor(movie[start_pt:end_pt, :, :], device=device,
                               dtype=torch.float32)
        # as_tensor shares memory with a cpu numpy movie, so the first step must not be in place
        data = data - mean_image[None, :, :]
        data /= noise_variance_image[None, :, :]
        data -= fluctuating_background.getitem_tensor(slice(start_pt, end_pt))
        data -= baseline_image[None, :, :]

        data = data.reshape(-1, fov_height*fov_width) # (num_frames, num_pixels)

        c_new[start_pt:end_pt, :] = hals_multi_iter_raw(blocks,
                                                    data,
                                                    results.spatial_demixed,
                                                    c_new[start_pt:end_pt, :],
                                                    nonneg=nonneg)

    return c_new