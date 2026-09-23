"""Edit saved demixing results: add hand-drawn masks and/or remove signals, then rerun the NMF pass."""

import os
from dataclasses import asdict, is_dataclass
from typing import Sequence

import h5py
import numpy as np

from masknmf.demixing.demixing_results import DemixingResults
from masknmf.utils import get_timestamp
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
    if num_masks == 0 and len(drop) >= results.spatial_demixed.shape[1]:
        raise ValueError("dropping every signal leaves nothing to demix")
    demixer = SignalDemixer.from_results(results, device=device, frame_batch_size=frame_batch_size, drop=drop)
    if num_masks:
        demixer.initialize_signals(is_custom=True, spatial_footprints=np.asarray(masks, dtype=np.float32))
    else:
        demixer.keep_existing_signals()
    kwargs = asdict(nmf_config) if is_dataclass(nmf_config) else dict(nmf_config or {})
    demixer.demix(carry_background=True, **kwargs)
    return demixer.results


def write_curated(path, results: DemixingResults, drop: Sequence[int] = (), num_masks: int = 0) -> str:
    """
    Write ``results`` to a new ``<timestamp>.curated.hdf5`` beside ``path``; the file at ``path`` is never
    changed. The new file's ``description`` attribute says what was done.

    Returns:
        the path written
    """
    out = os.path.join(os.path.dirname(str(path)), f"{get_timestamp()}.curated.hdf5")
    results.export(out)
    with h5py.File(out, "a") as f:
        f.attrs["description"] = (
            f"Demixing results after curation: {len(drop)} signal(s) removed and {int(num_masks)} drawn "
            "roi(s) added, then a full NMF pass."
        )
    return out
