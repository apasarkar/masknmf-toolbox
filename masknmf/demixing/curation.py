"""Run hand-drawn masks through the same initialization and NMF pass as the demixer's own signals."""

import os
from dataclasses import asdict, is_dataclass

import numpy as np

from masknmf.demixing.demixing_results import DemixingResults
from masknmf.demixing.signal_demixer import SignalDemixer


def add_signals(
    results: DemixingResults,
    masks: np.ndarray,
    nmf_config,
    device: str = "cpu",
    frame_batch_size: int = 5000,
) -> DemixingResults:
    """
    Re-demix with ``masks`` appended to the existing signals.

    Args:
        results (DemixingResults): the results to extend
        masks (np.ndarray): shape (fov dim1, fov dim2, num_masks), the drawn spatial footprints
        nmf_config: an NMFConfig (or its dict) so the new signals get the same NMF pass as the old ones
        device (str): "cpu" or "cuda"
        frame_batch_size (int): frames loaded onto the device at a time

    Returns:
        DemixingResults with the drawn masks as ordinary signals, after the NMF pass
    """
    demixer = SignalDemixer.from_results(results, device=device, frame_batch_size=frame_batch_size)
    demixer.initialize_signals(is_custom=True, spatial_footprints=np.asarray(masks, dtype=np.float32))
    kwargs = asdict(nmf_config) if is_dataclass(nmf_config) else dict(nmf_config)
    demixer.demix(carry_background=True, **kwargs)
    return demixer.results


def replace_results(path, results: DemixingResults) -> str:
    """Overwrite an exported results file: written beside it, then renamed over it."""
    path = str(path)
    tmp = path + ".tmp"
    if os.path.exists(tmp):
        os.remove(tmp)
    results.export(tmp)
    os.replace(tmp, path)
    return path
