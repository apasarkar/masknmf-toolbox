from typing import Optional, Sequence

import numpy as np

from masknmf.visualization.imgui.labels import LABEL_COLORS
from masknmf.visualization.imgui.masks import (
    OUTLINE_PLACEMENT,
    OUTLINE_WIDTH,
    outline_labels,
    selected_rim,
)

SELECTED_ALPHA = 0.9


def footprint_rgba(footprint: np.ndarray, color, peak: Optional[float] = None) -> np.ndarray:
    """One footprint as a flat-coloured RGBA whose alpha is the normalised weight."""
    rgba = np.zeros((*footprint.shape, 4), dtype=np.float32)
    rgba[..., :3] = color
    rgba[..., 3] = footprint / (peak if peak else (footprint.max() or 1.0))
    return rgba


def label_image_rgba(
    labels: np.ndarray,
    colors: Sequence = LABEL_COLORS,
    alpha: float = 0.45,
    selected: int = -1,
    show_masks: bool = True,
    show_outlines: bool = True,
    edges: Optional[np.ndarray] = None,
    outline_width: int = OUTLINE_WIDTH,
    outline_alpha: float = 1.0,
    outline_placement: str = OUTLINE_PLACEMENT,
) -> np.ndarray:
    """
    uint16 label image as uint8 RGBA.

    A blended fill washes out over a bright cell, so outlines carry the mask
    boundaries and the fill only tints. ``alpha`` is the fill, ``outline_alpha``
    the boundary. The selected ROI gets a white rim, which ``show_outlines``
    clears along with every other line.
    """
    ny, nx = labels.shape
    rgba = np.zeros((ny, nx, 4), np.uint8)
    lut = (np.asarray(colors, dtype=np.float32) * 255).astype(np.uint8)
    painted = labels > 0
    chosen = painted & (labels == selected + 1)

    if show_masks:
        rgba[painted, :3] = lut[(labels[painted] - 1) % len(lut)]
        rgba[painted, 3] = int(255 * alpha)
        rgba[chosen, 3] = int(255 * SELECTED_ALPHA)
    if show_outlines:
        if edges is None:
            edges = outline_labels(labels, outline_width, outline_placement)
        edges = np.asarray(edges)
        if edges.dtype == bool:
            edges = np.where(edges, labels, 0)
        drawn = edges > 0
        rgba[drawn, :3] = lut[(edges[drawn] - 1) % len(lut)]
        rgba[drawn, 3] = int(255 * outline_alpha)
        if chosen.any():
            rgba[selected_rim(chosen, outline_width, outline_placement)] = 255
    return rgba


class OverlayPair:
    """A background ImageGraphic plus an RGBA overlay on one subplot."""

    def __init__(self, subplot, shape, bg_cmap: str = "gray", offset: int = 1):
        ny, nx = shape
        self.subplot = subplot
        self.bg = subplot.add_image(
            np.zeros((ny, nx), np.float32), cmap=bg_cmap, name="overlay_bg",
            alpha_mode="blend",
        )
        self.fg = subplot.add_image(
            np.zeros((ny, nx, 4), np.float32), name="overlay_fg",
            alpha_mode="blend", offset=(0, 0, offset),
        )
        self.show_bg = True
        self.show_fg = True
        self.bg_alpha = 0.5
        self.fg_alpha = 1.0

    def set_background(self, image: np.ndarray, vrange=None):
        self.bg.data = image
        lo, hi = vrange if vrange is not None else (float(image.min()), float(image.max()))
        self.bg.vmin = lo
        self.bg.vmax = hi or 1.0

    def set_overlay(self, rgba: np.ndarray):
        self.fg.data = rgba

    def exclude_from_picking(self):
        """Keep the overlay out of hit-testing so tooltips report the image beneath."""
        for graphic in (self.bg, self.fg):
            for tile in graphic.world_object.children:
                tile.material.pick_write = False

    def apply(self):
        self.bg.visible = self.show_bg
        self.bg.alpha = self.bg_alpha
        self.fg.visible = self.show_fg
        self.fg.alpha = self.fg_alpha

    def remove(self):
        for graphic in (self.bg, self.fg):
            if graphic is not None:
                self.subplot.delete_graphic(graphic)
