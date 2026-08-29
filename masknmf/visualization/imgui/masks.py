from __future__ import annotations

import cv2
import numpy as np

# a stroke enclosing fewer unclaimed pixels than this is a misclick
MIN_ROI_PIXELS = 9

# half-width in px of a mask outline. 1 draws a one-pixel line, which is what
# a dense field of small ROIs needs; raise it for a heavier boundary.
OUTLINE_WIDTH = 1


def rim_kernel(width: int = OUTLINE_WIDTH) -> np.ndarray:
    """Square structuring element for an outline ``width`` px thick."""
    n = max(int(width), 1) * 2 + 1
    return np.ones((n, n), np.uint8)


class LabelImage:
    """
    ROIs as one uint16 label image; 0 is background, ROI i is i + 1.

    Labels cannot overlap: pixels already claimed are dropped from a new stroke.
    """

    def __init__(self, shape, labels=None):
        self.ny, self.nx = shape
        self.labels = (
            np.zeros((self.ny, self.nx), np.uint16) if labels is None
            else np.asarray(labels, np.uint16)
        )
        self.counts = [
            int((self.labels == i + 1).sum()) for i in range(int(self.labels.max()))
        ]
        self.last_error: str | None = None

    def __len__(self) -> int:
        return len(self.counts)

    def add(self, stroke) -> int:
        """Fill a closed stroke; returns the new index, or -1 with .last_error set."""
        self.last_error = None
        if len(stroke) < 3:
            self.last_error = "stroke too short"
            return -1
        points = np.round(np.asarray(stroke, np.float32)).astype(np.int32)
        points[:, 0] = points[:, 0].clip(0, self.nx - 1)
        points[:, 1] = points[:, 1].clip(0, self.ny - 1)

        filled = np.zeros((self.ny, self.nx), np.uint8)
        cv2.fillPoly(filled, [points], 1)
        rows, cols = np.nonzero(filled.astype(bool) & (self.labels == 0))
        if rows.size < MIN_ROI_PIXELS:
            self.last_error = f"under {MIN_ROI_PIXELS} free px, not added"
            return -1

        self.counts.append(int(rows.size))
        self.labels[rows, cols] = len(self.counts)
        return len(self.counts) - 1

    def delete(self, index: int) -> bool:
        """Drop one ROI and renumber the labels above it."""
        if not 0 <= index < len(self.counts):
            return False
        self.labels[self.labels == index + 1] = 0
        self.labels[self.labels > index + 1] -= 1
        self.counts.pop(index)
        return True

    def clear(self):
        self.labels[:] = 0
        self.counts.clear()

    def at(self, row: int, col: int) -> int:
        """ROI index under a pixel, or -1."""
        if 0 <= row < self.ny and 0 <= col < self.nx:
            return int(self.labels[row, col]) - 1
        return -1

    def areas(self) -> np.ndarray:
        return np.asarray(self.counts, dtype=np.int64)

    def footprints(self) -> np.ndarray:
        """(n, ny, nx) float32 binary masks, one per ROI."""
        out = np.zeros((len(self.counts), self.ny, self.nx), np.float32)
        for i in range(len(self.counts)):
            out[i][self.labels == i + 1] = 1.0
        return out

    def edges(self, width: int = OUTLINE_WIDTH) -> np.ndarray:
        """Boundary pixels; the label-image gradient keeps seams between touching ROIs."""
        painted = self.labels > 0
        kernel = rim_kernel(width)
        return painted & (cv2.morphologyEx(self.labels, cv2.MORPH_GRADIENT, kernel) > 0)
