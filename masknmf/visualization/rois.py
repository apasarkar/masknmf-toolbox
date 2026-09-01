"""
ROI masks for the interactive viewers: the label image the user paints into,
the read-only footprint sets an algorithm produced, and the RGBA overlays both
are drawn with.

Nothing here imports a GUI toolkit, so the same model backs a widget, a script
or a batch tool.
"""

import colorsys
from dataclasses import dataclass, field, replace
from typing import Optional, Sequence, Tuple

import numpy as np
import torch

from masknmf.visualization.imgui.labels import UNLABELED, class_color

__all__ = [
    "FootprintSet",
    "ROI_COLORS",
    "RoiLabelStore",
    "RoiRecord",
    "SELECTED_ALPHA",
    "build_pick_map",
    "feather_mask",
    "feathered_rgba",
    "roi_color",
]

# opacity a selected or grouped mask is filled at, whatever the overlay opacity
SELECTED_ALPHA = 0.9


def _make_roi_colors() -> np.ndarray:
    # one saturated color per unlabeled ROI, hues shuffled so consecutive ROIs
    # contrast and every fill stays readable over the movie colormaps
    hues = np.random.default_rng(0).permutation(180)
    return np.array(
        [
            [int(round(c * 255)) for c in colorsys.hsv_to_rgb(h / 180.0, 1.0, 1.0)]
            for h in hues
        ],
        dtype=np.uint8,
    )


# (180, 3) uint8 fill colors for ROIs with no class label
ROI_COLORS = _make_roi_colors()


def roi_color(index: int) -> Tuple[int, int, int]:
    """uint8 rgb for an unlabeled ROI, wrapping past the palette end"""
    return tuple(int(v) for v in ROI_COLORS[index % len(ROI_COLORS)])


@dataclass
class RoiRecord:
    """Per-ROI metadata; the pixels live in the store's label image."""

    area: int
    class_index: int = UNLABELED
    note: str = ""
    uid: int = 0  # persistent id, never reused
    source: str = ""  # "" = drawn by hand, otherwise "<set name>:<row>"
    color: Optional[Tuple[int, int, int]] = None  # explicit group color, uint8 rgb


class RoiLabelStore:
    """
    A ``(Y, X)`` uint16 label image plus one record per ROI.

    0 is background and ROI ``i`` owns value ``i + 1``, so ROIs can never
    overlap: pixels already claimed by another ROI are dropped from a new one.
    Deleting an ROI renumbers the values above it, so they stay contiguous.
    Each ROI also keeps a ``uid`` that is never reused, so traces and other
    per-ROI state key off it and a delete never remaps anyone else.

    Args:
        ny (int): FOV height
        nx (int): FOV width
        min_pixels (int): free pixels a new mask needs before it is stored
    """

    def __init__(self, ny: int, nx: int, min_pixels: int = 1):
        self.labels = np.zeros((int(ny), int(nx)), np.uint16)
        self.rois: list[RoiRecord] = []
        self.min_pixels = int(min_pixels)
        self.next_uid = 1

    @property
    def ny(self) -> int:
        return self.labels.shape[0]

    @property
    def nx(self) -> int:
        return self.labels.shape[1]

    def __len__(self) -> int:
        return len(self.rois)

    def add_roi(self, mask: np.ndarray, source: str = "") -> Optional[int]:
        """
        Claim the free pixels of a boolean ``(Y, X)`` mask.

        Returns the new ROI index, or None when fewer than ``min_pixels`` free
        pixels remain, in which case the label image is untouched.
        """
        rows, cols = np.nonzero(np.asarray(mask, bool) & (self.labels == 0))
        if rows.size < self.min_pixels:
            return None
        self.rois.append(
            RoiRecord(area=int(rows.size), uid=self.next_uid, source=str(source))
        )
        self.next_uid += 1
        self.labels[rows, cols] = len(self.rois)
        return len(self.rois) - 1

    def delete_roi(self, index: int) -> bool:
        """Drop one ROI and renumber the label values above it."""
        if not 0 <= index < len(self.rois):
            return False
        self.labels[self.labels == index + 1] = 0
        self.labels[self.labels > index + 1] -= 1
        self.rois.pop(index)
        return True

    def clear(self):
        self.labels[:] = 0
        self.rois.clear()

    def snapshot(self) -> "RoiLabelStore":
        """Deep copy that later mutations of either store cannot reach."""
        out = RoiLabelStore(self.ny, self.nx, self.min_pixels)
        out.labels = self.labels.copy()
        out.rois = [replace(r) for r in self.rois]
        out.next_uid = self.next_uid
        return out

    def set_class(self, index: int, class_index: int):
        self.rois[index].class_index = int(class_index)

    def set_note(self, index: int, note: str):
        self.rois[index].note = str(note)

    def set_color(self, index: int, rgb: Optional[Tuple[int, int, int]]):
        """Give one ROI an explicit color; None reverts it to class or hue."""
        self.rois[index].color = None if rgb is None else tuple(int(v) for v in rgb)

    def uid_index(self, uid: int) -> Optional[int]:
        """Index currently holding ``uid``, or None when that ROI is gone."""
        for i, record in enumerate(self.rois):
            if record.uid == uid:
                return i
        return None

    def roi_at(self, row: int, col: int) -> int:
        """ROI index under a pixel, or -1 for background or out of range."""
        if not (0 <= row < self.ny and 0 <= col < self.nx):
            return -1
        return int(self.labels[row, col]) - 1

    def mask(self, index: int) -> np.ndarray:
        """Boolean ``(Y, X)`` mask of one ROI."""
        return self.labels == index + 1

    def masks(self) -> np.ndarray:
        """Every ROI as a ``(Y, X, num_rois)`` float32 binary stack."""
        out = np.zeros((self.ny, self.nx, len(self.rois)), np.float32)
        for i in range(len(self.rois)):
            out[:, :, i] = self.labels == i + 1
        return out

    @property
    def areas(self) -> list:
        return [r.area for r in self.rois]

    def rgb(self, index: int) -> Tuple[int, int, int]:
        """
        Display color of one ROI as uint8 rgb: its explicit group color when
        set, else its class color when labeled, else its own hue.
        """
        record = self.rois[index]
        if record.color is not None:
            return tuple(int(v) for v in record.color)
        if record.class_index >= 0:
            return tuple(int(round(c * 255)) for c in class_color(record.class_index))
        return roi_color(index)


def feather_mask(mask: np.ndarray, edge_width: int = 3) -> np.ndarray:
    """
    Soft-edged weights for a binary mask: 1 in the interior, falling off over
    ``edge_width`` px toward the boundary. Weights stay inside the mask, so a
    trace taken with them never reads a neighboring cell.
    """
    from scipy.ndimage import distance_transform_edt

    inside = distance_transform_edt(np.asarray(mask, bool))
    return np.clip(inside / max(int(edge_width), 1), 0.0, 1.0).astype(np.float32)


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
    Compose an ``(ny, nx, 4)`` uint8 overlay from footprints.

    Args:
        shape (tuple): (ny, nx) of the FOV
        comps: iterable of ``(ypix, xpix, lam, rgb, fill)``; each pixel takes
            ``lam / lam.max() * fill`` as its alpha, and where footprints
            overlap the higher alpha wins both color and coverage
        selected: ``(ypix, xpix, rgb)`` filled at ``SELECTED_ALPHA`` with a
            white rim, drawn over everything else
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


def build_pick_map(footprints: Sequence[tuple], shape: Tuple[int, int]) -> np.ndarray:
    """
    ``(ny, nx)`` int32 footprint index per pixel, -1 for background.

    Footprints are painted in ascending peak weight, so the strongest one wins
    a contested pixel and picking matches what the overlay draws on top.
    """
    pick = np.full(shape, -1, np.int32)
    peaks = [float(np.max(lam)) if len(lam) else 0.0 for _y, _x, lam in footprints]
    for k in np.argsort(peaks, kind="stable"):
        ypix, xpix, _lam = footprints[k]
        pick[ypix, xpix] = k
    return pick


@dataclass
class FootprintSet:
    """
    Read-only footprints an algorithm produced, shown beside the drawn ROIs.

    Args:
        name (str): how the set is named in the table and status line
        footprints (list): ``(ypix, xpix, lam)`` per component
        pick_map (np.ndarray): ``(ny, nx)`` int32 component index per pixel
        classes (dict): component index -> class index, for the ones labeled
        colors (dict): component index -> float rgb, overriding class and hue
        visible (bool): whether the overlay draws this set
    """

    name: str
    footprints: list
    pick_map: np.ndarray
    classes: dict = field(default_factory=dict)
    colors: dict = field(default_factory=dict)
    visible: bool = True

    @classmethod
    def from_sparse(cls, name: str, a: torch.Tensor, shape: Tuple[int, int]) -> "FootprintSet":
        """
        Build from a masknmf spatial matrix.

        Args:
            name (str): set name
            a (torch.Tensor): sparse ``(pixels, components)`` footprints
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
            footprints.append(
                (ypix.astype(np.int32), xpix.astype(np.int32), values[lo:hi])
            )
        return cls(name, footprints, build_pick_map(footprints, shape))

    def __len__(self) -> int:
        return len(self.footprints)

    def color(self, index: int) -> Tuple[float, float, float]:
        """
        Float rgb of one component: its explicit group color when set, else its
        class color when labeled, else its own hue.
        """
        rgb = self.colors.get(index)
        if rgb is not None:
            return tuple(rgb)
        class_index = self.classes.get(index)
        if class_index is not None:
            return class_color(class_index)
        return tuple(v / 255.0 for v in roi_color(index))

    def area(self, index: int) -> int:
        return int(len(self.footprints[index][0]))

    def rgba(self, shape: Tuple[int, int], opacity: float, selected=None, grouped=()) -> np.ndarray:
        """
        ``(ny, nx, 4)`` uint8 overlay of this set.

        Args:
            shape (tuple): (ny, nx) of the FOV
            opacity (float): fill of an unselected component
            selected (Optional[int]): component filled at ``SELECTED_ALPHA``
                with a white rim
            grouped: component indices of a multi-selection, also filled at
                ``SELECTED_ALPHA``
        """
        comps = []
        for k, (ypix, xpix, lam) in enumerate(self.footprints):
            fill = SELECTED_ALPHA if k in grouped else opacity
            comps.append((ypix, xpix, lam, self.color(k), fill))
        pick = None
        if selected is not None:
            ypix, xpix, _lam = self.footprints[selected]
            pick = (ypix, xpix, self.color(selected))
        return feathered_rgba(shape, comps, pick)
