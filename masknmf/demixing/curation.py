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


CURATED_SUFFIX = ".curated.hdf5"


def results_stem(path) -> str:
    """
    The name of the results file ``path`` is or descends from, without its extension:
    ``results.calcium.hdf5`` and ``results.calcium.<timestamp>.curated.hdf5`` both give ``results.calcium``.
    """
    name = os.path.basename(str(path))
    if name.endswith(CURATED_SUFFIX):
        return name[: -len(CURATED_SUFFIX)].rsplit(".", 1)[0]
    return os.path.splitext(name)[0]


def latest_results(paths: Sequence) -> list[str]:
    """
    One file per results file among ``paths``: its newest curated file when one is among them, else the results
    file itself, in the order each first appears. A curated file only replaces a results file in the same folder.
    """
    latest = {}
    for path in map(str, paths):
        key = (os.path.dirname(os.path.abspath(path)), results_stem(path))
        # curated beats uncurated; among curated the timestamp in the name sorts by time
        if key not in latest or (path.endswith(CURATED_SUFFIX), path) > (latest[key].endswith(CURATED_SUFFIX), latest[key]):
            latest[key] = path
    return list(latest.values())


def write_curated(
    path, results: DemixingResults, drop: Sequence[int] = (), num_masks: int = 0, filters: Sequence[dict] = ()
) -> str:
    """
    Write ``results`` to a new ``<stem>.<timestamp>.curated.hdf5`` beside ``path``, the stem being the results
    file it descends from (``results.hdf5 -> results.<timestamp>.curated.hdf5``); the file at ``path`` is never
    changed. The new file's ``description`` attribute says what was done, and which range filter removed
    which signals: each of ``filters`` has a ``column``, a ``range`` (lo, hi), ``outside`` (which side of the
    range it took) and the ``signals`` it marked.

    Returns:
        the path written
    """
    out = os.path.join(os.path.dirname(str(path)), f"{results_stem(path)}.{get_timestamp()}{CURATED_SUFFIX}")
    results.export(out)
    by_filter = "; ".join(
        f"{flt['column']} {'outside' if flt['outside'] else 'inside'} {flt['range'][0]:g} to {flt['range'][1]:g}: "
        + ", ".join(str(k) for k in sorted(flt["signals"]))
        for flt in filters
        if flt["signals"]
    )
    with h5py.File(out, "a") as f:
        f.attrs["description"] = (
            f"Demixing results after curation: {len(drop)} signal(s) removed"
            + (f" ({by_filter})" if by_filter else "")
            + f" and {int(num_masks)} drawn roi(s) added, then a full NMF pass."
        )
    return out
