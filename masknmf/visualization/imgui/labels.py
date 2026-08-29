from typing import Optional, Sequence

import numpy as np
from imgui_bundle import imgui

LABEL_COLORS = (
    (0.12, 0.47, 0.71), (1.00, 0.50, 0.05), (0.17, 0.63, 0.17),
    (0.84, 0.15, 0.16), (0.58, 0.40, 0.74), (0.55, 0.34, 0.29),
    (0.89, 0.47, 0.76), (0.50, 0.50, 0.50), (0.74, 0.74, 0.13),
    (0.09, 0.75, 0.81),
)

LABEL_KEYS = (
    imgui.Key._1, imgui.Key._2, imgui.Key._3, imgui.Key._4, imgui.Key._5,
    imgui.Key._6, imgui.Key._7, imgui.Key._8, imgui.Key._9,
)

UNLABELED = -1

# returned by draw_label_buttons for its "unlabel all" button; not a label
# index, so callers must branch on it before assigning
UNLABEL_ALL = -2


class LabelSet:
    """Class names plus a per-item label vector; -1 is unlabeled."""

    def __init__(self, n_items: int, names: Sequence[str] = (), labels=None):
        self._names = tuple(names)
        if labels is None:
            labels = np.full((n_items,), UNLABELED, dtype=np.int64)
        else:
            labels = np.asarray(labels).astype(np.int64)
            if labels.shape[0] != n_items:
                raise ValueError(f"labels has {labels.shape[0]} entries, expected {n_items}")
        self._labels = labels
        self._extend_names_to_fit()

    def _extend_names_to_fit(self):
        # labels restored from disk can name classes this set doesn't have yet
        top = int(self._labels.max(initial=UNLABELED))
        if top >= len(self._names):
            self._names = (*self._names, *(f"class{i}" for i in range(len(self._names), top + 1)))

    @property
    def names(self) -> tuple:
        return self._names

    @property
    def labels(self) -> np.ndarray:
        return self._labels

    def __len__(self) -> int:
        return len(self._names)

    def color(self, index: int) -> tuple:
        return LABEL_COLORS[index % len(LABEL_COLORS)]

    def count(self, index: int) -> int:
        return int((self._labels == index).sum())

    def name_of(self, item: int) -> str:
        label = int(self._labels[item])
        return self._names[label] if label >= 0 else "unlabeled"

    def assign(self, items, index: int):
        self._labels[list(items)] = index

    def clear(self):
        """Unlabel every item."""
        self._labels[:] = UNLABELED

    def add(self, name: str) -> bool:
        if not name or name in self._names:
            return False
        self._names = (*self._names, name)
        return True

    def remove(self, index: int) -> bool:
        """Its items become unlabeled; higher labels shift down."""
        if not 0 <= index < len(self._names):
            return False
        self._labels[self._labels == index] = UNLABELED
        self._labels[self._labels > index] -= 1
        self._names = tuple(n for i, n in enumerate(self._names) if i != index)
        return True

    def resize(self, n_items: int):
        """Grow or shrink the label vector, keeping existing assignments."""
        old = self._labels
        self._labels = np.full((n_items,), UNLABELED, dtype=np.int64)
        keep = min(n_items, old.shape[0])
        self._labels[:keep] = old[:keep]

    def progress(self) -> tuple:
        done = int((self._labels >= 0).sum())
        return done, len(self._labels)

    def hotkey_pressed(self) -> Optional[int]:
        """Label index for a pressed 1-9 key, -1 for 0, or None."""
        if imgui.is_key_pressed(imgui.Key._0, False):
            return UNLABELED
        for i, key in enumerate(LABEL_KEYS[: len(self._names)]):
            if imgui.is_key_pressed(key, False):
                return i
        return None
