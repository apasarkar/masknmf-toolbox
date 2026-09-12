"""Edit saved demixing results: add hand-drawn masks and/or remove signals, then rerun the NMF pass."""

import os
from dataclasses import asdict, is_dataclass
from typing import Sequence

import numpy as np

from masknmf.demixing.demixing_results import DemixingResults
from masknmf.demixing.signal_demixer import SignalDemixer


def update_signals(
    results: DemixingResults,
    masks: np.ndarray | None = None,
    drop: Sequence[int] | None = None,
    nmf_config=None,
    device: str = "cpu",
    frame_batch_size: int = 5000,
) -> DemixingResults:
    """
    Re-demix with the ``drop`` signals removed and ``masks`` appended to the rest.

    Args:
        results (DemixingResults): the results to edit
        masks (np.ndarray | None): shape (fov dim1, fov dim2, num_masks), drawn spatial footprints to add
        drop (Sequence[int] | None): indices of existing signals to remove
        nmf_config: an NMFConfig (or its dict) so the pass matches the one that produced ``results``
        device (str): "cpu" or "cuda"
        frame_batch_size (int): frames loaded onto the device at a time

    Returns:
        DemixingResults after the NMF pass over the remaining and added signals
    """
    drop = sorted({int(i) for i in drop}) if drop else []
    num_masks = 0 if masks is None else masks.shape[-1]
    if num_masks == 0 and not drop:
        raise ValueError("nothing to do: no masks to add and no signals to drop")
    if num_masks == 0 and len(drop) >= results.a.shape[1]:
        raise ValueError("dropping every signal leaves nothing to demix")
    demixer = SignalDemixer.from_results(results, device=device, frame_batch_size=frame_batch_size, drop=drop)
    if num_masks:
        demixer.initialize_signals(is_custom=True, spatial_footprints=np.asarray(masks, dtype=np.float32))
    else:
        demixer.keep_existing_signals()
    kwargs = asdict(nmf_config) if is_dataclass(nmf_config) else dict(nmf_config or {})
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
