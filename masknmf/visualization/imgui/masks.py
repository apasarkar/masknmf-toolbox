import cv2
import numpy as np

# a stroke enclosing fewer unclaimed pixels than this is a misclick
MIN_ROI_PIXELS = 9

_RIM = np.ones((5, 5), np.uint8)


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

    def __len__(self) -> int:
        return len(self.counts)

    def add(self, stroke) -> int:
        """Fill a closed stroke; returns the new index, or -1 if rejected."""
        if len(stroke) < 3:
            return -1
        points = np.round(np.asarray(stroke, np.float32)).astype(np.int32)
        points[:, 0] = points[:, 0].clip(0, self.nx - 1)
        points[:, 1] = points[:, 1].clip(0, self.ny - 1)

        filled = np.zeros((self.ny, self.nx), np.uint8)
        cv2.fillPoly(filled, [points], 1)
        rows, cols = np.nonzero(filled.astype(bool) & (self.labels == 0))
        if rows.size < MIN_ROI_PIXELS:
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

    def edges(self) -> np.ndarray:
        """Boundary pixels; the label-image gradient keeps seams between touching ROIs."""
        painted = self.labels > 0
        return painted & (cv2.morphologyEx(self.labels, cv2.MORPH_GRADIENT, _RIM) > 0)
