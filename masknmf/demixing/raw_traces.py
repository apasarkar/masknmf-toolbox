import numpy as np
import torch

from masknmf.arrays import ArrayLike
from masknmf.demixing.demixing_results import DemixingResults


def estimate_temporal_demixed_raw(results: DemixingResults,
                                  movie: np.ndarray | ArrayLike | None,
                                  frame_batch_size: int = 300) -> torch.Tensor | None:
    """
    temporal_demixed re-estimated from the registered raw movie with the spatial footprints, background and baseline
    of results held fixed, so it carries no compression or denoising. Shape (number of frames, number of signals),
    in temporal_demixed units. None when there is no movie to go back to.

    PLACEHOLDER: returns temporal_demixed plus gaussian noise at a fifth of each signal's standard deviation and does
    not read movie yet.
    """
    if movie is None:
        return None
    c = results.temporal_demixed
    generator = torch.Generator(device=c.device).manual_seed(0)
    noise = torch.randn(c.shape, generator=generator, device=c.device, dtype=c.dtype)
    return c + 0.2 * c.std(dim=0, keepdim=True) * noise
