"""Feathered RGBA overlays of demixed footprints for the interactive viewers."""

import colorsys
from dataclasses import dataclass
from typing import Iterable, Mapping, Optional, Tuple

import matplotlib
import numpy as np
import torch

__all__ = ["FootprintSet", "MASK_CUTOFF", "SELECTED_ALPHA", "feathered_rgba", "roi_color"]

# opacity a selected mask is filled at, whatever the overlay opacity
SELECTED_ALPHA = 0.9
MARKED_COLOR = (1.0, 0.15, 0.15)  # footprints marked for deletion
# fraction of a footprint's own peak weight below which the overlay drops its pixels: the halo the contours drop too
MASK_CUTOFF = 0.1


def _make_roi_colors() -> np.ndarray:
    # one saturated color per footprint, hues shuffled so neighbors contrast
    hues = np.random.default_rng(0).permutation(180)
    return np.array(
        [[int(round(c * 255)) for c in colorsys.hsv_to_rgb(h / 180.0, 1.0, 1.0)] for h in hues],
        dtype=np.uint8,
    )


ROI_COLORS = _make_roi_colors()


def roi_color(index: int) -> Tuple[int, int, int]:
    """uint8 rgb for a footprint, wrapping past the palette end"""
    return tuple(int(v) for v in ROI_COLORS[index % len(ROI_COLORS)])


def _rim(mask: np.ndarray) -> np.ndarray:
    """Boundary pixels of a boolean mask, 4-connected."""
    core = mask.copy()
    core[1:, :] &= mask[:-1, :]
    core[:-1, :] &= mask[1:, :]
    core[:, 1:] &= mask[:, :-1]
    core[:, :-1] &= mask[:, 1:]
    return mask & ~core


def feathered_rgba(shape: Tuple[int, int], comps, selected=(), selected_alpha: float = SELECTED_ALPHA) -> np.ndarray:
    """
    Compose an (ny, nx, 4) uint8 overlay from footprints.

    Args:
        shape (tuple): (ny, nx) of the FOV
        comps: iterable of (ypix, xpix, weight, rgb, fill), weights in 0..1; each pixel takes weight * fill
            as its alpha, and where footprints overlap the higher alpha wins color and coverage
        selected: iterable of (ypix, xpix, weight, rgb), each pixel filled at weight * ``selected_alpha`` with
            a white rim at ``selected_alpha``, drawn over everything else, in order
        selected_alpha (float): opacity the selected footprints peak at
    """
    ny, nx = shape
    rgba = np.zeros((ny, nx, 4), np.uint8)
    best = np.zeros((ny, nx), np.float32)
    for ypix, xpix, weight, rgb, fill in comps:
        color = np.rint(np.asarray(rgb, np.float32) * 255).astype(np.uint8)
        alpha = np.asarray(weight, np.float32) * fill
        win = alpha > best[ypix, xpix]
        yy, xx = ypix[win], xpix[win]
        best[yy, xx] = alpha[win]
        rgba[yy, xx, :3] = color
        rgba[yy, xx, 3] = np.rint(alpha[win] * 255).astype(np.uint8)
    for ypix, xpix, weight, rgb in selected:
        mask = np.zeros((ny, nx), bool)
        mask[ypix, xpix] = True
        rgba[ypix, xpix, :3] = np.rint(np.asarray(rgb, np.float32) * 255).astype(np.uint8)
        rgba[ypix, xpix, 3] = np.rint(np.asarray(weight, np.float32) * selected_alpha * 255).astype(np.uint8)
        rgba[_rim(mask)] = (255, 255, 255, round(selected_alpha * 255))
    return rgba


@dataclass
class FootprintSet:
    """
    Footprints an algorithm produced, as (ypix, xpix, lam) per component; ``colors`` overrides the id palette and
    ``peaks`` (one per footprint, its trace's maximum) scales the overlay so a weak signal draws faint.
    """

    footprints: list
    colors: Optional[np.ndarray] = None
    peaks: Optional[np.ndarray] = None

    @classmethod
    def from_sparse(cls, a: torch.Tensor, shape: Tuple[int, int]) -> "FootprintSet":
        """
        Args:
            a (torch.Tensor): sparse (pixels, components) footprints
            shape (tuple): (ny, nx) the pixel axis unravels to
        """
        a = a.coalesce().cpu()
        rows, cols = a.indices().numpy()
        values = a.values().numpy().astype(np.float32)
        keep = values > 0
        rows, cols, values = rows[keep], cols[keep], values[keep]
        order = np.argsort(cols, kind="stable")
        rows, cols, values = rows[order], cols[order], values[order]
        bounds = np.searchsorted(cols, np.arange(a.shape[1] + 1))
        footprints = []
        for k in range(a.shape[1]):
            lo, hi = bounds[k], bounds[k + 1]
            ypix, xpix = np.divmod(rows[lo:hi], shape[1])
            footprints.append((ypix.astype(np.int32), xpix.astype(np.int32), values[lo:hi]))
        return cls(footprints)

    def __len__(self) -> int:
        return len(self.footprints)

    def color(self, index: int) -> Tuple[float, float, float]:
        if self.colors is not None:
            return tuple(float(v) for v in self.colors[index])
        return tuple(v / 255.0 for v in roi_color(index))

    def recolor(self, values=None, cmap: str = "viridis"):
        """Color every footprint by its rank in ``values`` (no value: gray), or back to the id palette with None."""
        if values is None:
            self.colors = None
            return
        values = np.asarray(values, dtype=np.float64)
        finite = np.isfinite(values)
        ranks = np.zeros(len(values))
        ranks[finite] = np.argsort(np.argsort(values[finite])) / max(finite.sum() - 1, 1)
        colors = np.asarray(matplotlib.colormaps[cmap](ranks))[:, :3]
        colors[~finite] = 0.5
        self.colors = colors

    @property
    def areas(self) -> np.ndarray:
        """Pixel count of every footprint."""
        return np.array([len(ypix) for ypix, _xpix, _lam in self.footprints], dtype=np.int64)

    def rgba(
        self,
        shape: Tuple[int, int],
        opacity: float,
        selected: Optional[int] = None,
        marked: Iterable[int] = (),
        grouped: Optional[Mapping[int, Tuple[float, float, float]]] = None,
        selected_opacity: float = SELECTED_ALPHA,
        by_peak: bool = True,
    ) -> np.ndarray:
        """
        (ny, nx, 4) uint8 overlay. With ``by_peak`` a pixel's alpha is its weight times the footprint's peak
        against one maximum over the field, so the masks read like the signals movie; without, every footprint
        is scaled to its own peak. Pixels under MASK_CUTOFF of a footprint's own peak are dropped as the contours
        drop them. ``grouped`` (index -> rgb) and then ``selected`` are feathered to ``selected_opacity`` at
        their own peak with a white rim, and ``marked`` footprints are drawn in MARKED_COLOR.
        """
        marked = set(marked)
        grouped = dict(grouped or {})
        peaks = np.ones(len(self.footprints), np.float32) if self.peaks is None else np.asarray(self.peaks, np.float32)
        top = max((float(lam.max()) * peaks[k] for k, (_y, _x, lam) in enumerate(self.footprints) if lam.size), default=0.0) or 1.0
        comps, highlighted = [], []
        for k, (ypix, xpix, lam) in enumerate(self.footprints):
            if not lam.size:
                continue
            keep = lam >= MASK_CUTOFF * lam.max()
            scale = peaks[k] / top if by_peak else 1.0 / float(lam.max())
            comps.append((ypix[keep], xpix[keep], lam[keep] * scale, MARKED_COLOR if k in marked else self.color(k), opacity))
        picks = [k for k in grouped if k != selected]
        if selected is not None:
            picks.append(selected)
        for k in picks:
            ypix, xpix, lam = self.footprints[k]
            if not lam.size:
                continue
            keep = lam >= MASK_CUTOFF * lam.max()
            highlighted.append((ypix[keep], xpix[keep], lam[keep] / lam.max(), MARKED_COLOR if k in marked else grouped.get(k, self.color(k))))
        return feathered_rgba(shape, comps, highlighted, selected_opacity)
