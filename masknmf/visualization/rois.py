"""Feathered RGBA overlays of demixed footprints for the interactive viewers."""

import colorsys
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch

__all__ = ["FootprintSet", "SELECTED_ALPHA", "feathered_rgba", "roi_color"]

# opacity a selected mask is filled at, whatever the overlay opacity
SELECTED_ALPHA = 0.9


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


def feathered_rgba(shape: Tuple[int, int], comps, selected=None) -> np.ndarray:
    """
    Compose an (ny, nx, 4) uint8 overlay from footprints.

    Args:
        shape (tuple): (ny, nx) of the FOV
        comps: iterable of (ypix, xpix, lam, rgb, fill); each pixel takes lam / lam.max() * fill
            as its alpha, and where footprints overlap the higher alpha wins color and coverage
        selected: (ypix, xpix, rgb) filled at SELECTED_ALPHA with a white rim, drawn over everything else
    """
    ny, nx = shape
    rgba = np.zeros((ny, nx, 4), np.uint8)
    best = np.zeros((ny, nx), np.float32)
    for ypix, xpix, lam, rgb, fill in comps:
        color = np.rint(np.asarray(rgb, np.float32) * 255).astype(np.uint8)
        lam = np.asarray(lam, np.float32)
        peak = float(lam.max()) if lam.size else 0.0
        alpha = lam / peak * fill if peak > 0 else np.full(lam.shape, fill, np.float32)
        win = alpha > best[ypix, xpix]
        yy, xx = ypix[win], xpix[win]
        best[yy, xx] = alpha[win]
        rgba[yy, xx, :3] = color
        rgba[yy, xx, 3] = np.rint(alpha[win] * 255).astype(np.uint8)
    if selected is not None:
        ypix, xpix, rgb = selected
        mask = np.zeros((ny, nx), bool)
        mask[ypix, xpix] = True
        fill = np.uint8(round(SELECTED_ALPHA * 255))
        rgba[mask, :3] = np.rint(np.asarray(rgb, np.float32) * 255).astype(np.uint8)
        rgba[mask, 3] = fill
        rgba[_rim(mask)] = (255, 255, 255, fill)
    return rgba


@dataclass
class FootprintSet:
    """Footprints an algorithm produced, as (ypix, xpix, lam) per component."""

    footprints: list

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
        return tuple(v / 255.0 for v in roi_color(index))

    def rgba(self, shape: Tuple[int, int], opacity: float, selected: Optional[int] = None) -> np.ndarray:
        """(ny, nx, 4) uint8 overlay; ``selected`` is filled at SELECTED_ALPHA with a white rim."""
        comps = [(ypix, xpix, lam, self.color(k), opacity) for k, (ypix, xpix, lam) in enumerate(self.footprints)]
        pick = None
        if selected is not None:
            ypix, xpix, _lam = self.footprints[selected]
            pick = (ypix, xpix, self.color(selected))
        return feathered_rgba(shape, comps, pick)
