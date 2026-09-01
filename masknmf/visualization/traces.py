"""
ROI traces for the interactive viewers: extraction from a movie, storage keyed
so a deleted ROI only prunes, and the display transform the plot draws.
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np

__all__ = [
    "TraceSet",
    "baseline",
    "display_trace",
    "make_entry",
    "roi_trace",
    "trace_stats",
]

# percentile of a trace taken as its resting fluorescence
BASELINE_PERCENTILE = 20


@dataclass
class TraceSet:
    """
    Traces from one movie, keyed so that deleting an ROI only ever prunes.

    Args:
        name (str): the movie the traces came from, shown as the source column
        data (dict): key -> entry, where a key is stable across ROI deletions
        visible (bool): whether the plot may draw this set
    """

    name: str
    data: dict = field(default_factory=dict)
    visible: bool = True

    def prune(self, keys) -> None:
        """Drop entries whose key is not in ``keys``."""
        keep = set(keys)
        for key in [k for k in self.data if k not in keep]:
            del self.data[key]


def _bbox(mask: np.ndarray) -> Tuple[int, int, int, int]:
    rows, cols = np.nonzero(mask)
    return int(rows.min()), int(rows.max()) + 1, int(cols.min()), int(cols.max()) + 1


def _factorized(movie) -> bool:
    """True for a masknmf factorized array, which reconstructs only the crop."""
    return callable(getattr(movie, "getitem_tensor", None))


def roi_trace(movie, mask: np.ndarray, weights: Optional[np.ndarray] = None, batch: int = 500) -> np.ndarray:
    """
    Mean over ``mask`` per frame, reading only the mask's bounding box.

    Args:
        movie: ``(T, Y, X)`` array; a masknmf factorized array reconstructs the
            crop alone, so it is read in one call regardless of ``batch``
        mask (np.ndarray): ``(Y, X)`` boolean mask
        weights (Optional[np.ndarray]): full-frame weight image, e.g. from
            ``feather_mask``; makes the trace a weighted mean over the mask
        batch (int): frames per read, so a long recording never lands in RAM

    Returns:
        np.ndarray: ``(num_frames,)`` float32 trace
    """
    mask = np.asarray(mask, bool)
    if not mask.any():
        raise ValueError("mask is empty")
    n_frames = int(movie.shape[0])
    if _factorized(movie):
        batch = n_frames
    y0, y1, x0, x1 = _bbox(mask)
    crop_mask = mask[y0:y1, x0:x1]
    crop_weights = None
    if weights is not None:
        crop_weights = np.asarray(weights, np.float32)[y0:y1, x0:x1][crop_mask]
        crop_weights = crop_weights / (float(crop_weights.sum()) or 1.0)
    out = np.empty(n_frames, np.float32)
    for start in range(0, n_frames, batch):
        stop = min(start + batch, n_frames)
        block = np.asarray(movie[start:stop, y0:y1, x0:x1]).reshape(
            stop - start, y1 - y0, x1 - x0
        )
        picked = block[:, crop_mask]
        out[start:stop] = picked @ crop_weights if crop_weights is not None else picked.mean(axis=1)
    return out


def baseline(trace: np.ndarray) -> Optional[float]:
    """
    Resting fluorescence of a trace, or None when it is not positive.

    A movie whose baseline has already been removed has no meaningful dF/F, so
    the entries built from one display raw instead.
    """
    if not trace.size:
        return None
    f0 = float(np.percentile(trace, BASELINE_PERCENTILE))
    return f0 if f0 > 0 else None


def make_entry(trace: np.ndarray, f0: Optional[float] = None, zeroed: bool = False) -> dict:
    """
    One stored trace.

    Args:
        trace (np.ndarray): the samples
        f0 (Optional[float]): the ROI's resting fluorescence, which scales the
            dF/F display; None turns that display off for this trace
        zeroed (bool): the trace already has its baseline removed, so dF/F
            scales it by ``f0`` without subtracting it again. This is what puts
            a residual trace on the same percent axis as its PMD trace.
    """
    return {
        "trace": np.ascontiguousarray(trace, np.float32),
        "f0": f0,
        "zeroed": bool(zeroed),
    }


def display_trace(entry: dict, dff: bool = True) -> np.ndarray:
    """The stored trace as plotted: percent dF/F over its baseline, or raw."""
    trace = entry["trace"]
    f0 = entry.get("f0")
    if not dff or f0 is None:
        return trace
    return (trace if entry.get("zeroed") else trace - f0) / f0 * 100.0


def trace_stats(trace: np.ndarray) -> Tuple[int, float, float, float]:
    """
    ``(frames, mean, peak, snr)`` of a displayed trace.

    ``snr`` is the peak over the median in robust standard deviations, so an
    outlier-heavy trace does not inflate it.
    """
    if not trace.size:
        return 0, 0.0, 0.0, 0.0
    median = float(np.median(trace))
    mad = float(np.median(np.abs(trace - median)))
    peak = float(trace.max())
    snr = (peak - median) / (1.4826 * mad) if mad > 0 else 0.0
    return int(trace.size), float(trace.mean()), peak, snr
