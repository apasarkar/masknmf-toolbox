"""
ROI labels, classifier predictions and provenance stored under ``DemixingResults/``
in a session hdf5, next to the demixing arrays.

Datasets: ``class_labels`` (num_rois,) int64 with -1 = unlabeled, ``label_names``,
``labels_complete``, ``roi_masks``, ``class_predictions``, ``class_probabilities``,
``classified_with``, ``classifier_path``, ``classifier_history``.
"""

import time
from typing import Optional, Sequence

import h5py
import numpy as np

GROUP = "DemixingResults"
CLASSIFIER_SUFFIX = ".roicat_classifier"


def _write(group, key: str, data):
    data = np.asarray(data)
    if key in group and group[key].shape == data.shape and group[key].dtype == data.dtype:
        group[key][...] = data
    else:
        if key in group:
            del group[key]
        group.create_dataset(key, data=data)


def read_labels(path) -> tuple[Optional[np.ndarray], Optional[list[str]]]:
    """(class_labels, label_names) stored in a session file; None where absent."""
    with h5py.File(path, "r") as f:
        g = f.get(GROUP)
        if g is None:
            return None, None
        labels = g["class_labels"][()] if "class_labels" in g else None
        names = (
            [n.decode() if isinstance(n, bytes) else str(n) for n in g["label_names"][()]]
            if "label_names" in g
            else None
        )
    return labels, names


def write_labels(path, labels, label_names: Sequence[str]):
    labels = np.asarray(labels, dtype=np.int64)
    with h5py.File(path, "r+") as f:
        g = f.require_group(GROUP)
        _write(g, "class_labels", labels)
        _write(g, "label_names", np.array([n.encode() for n in label_names]))
        _write(g, "labels_complete", np.bool_((labels >= 0).all()))


def write_predictions(path, predictions, probabilities, classified_with: str = ""):
    """Per-ROI predicted label index, its confidence, and which classifier produced them."""
    with h5py.File(path, "r+") as f:
        g = f.require_group(GROUP)
        _write(g, "class_predictions", np.asarray(predictions, dtype=np.int64))
        _write(g, "class_probabilities", np.asarray(probabilities, dtype=np.float32))
        _write(g, "classified_with", np.bytes_(classified_with))


def write_masks(path, masks):
    with h5py.File(path, "r+") as f:
        _write(f.require_group(GROUP), "roi_masks", np.asarray(masks, dtype=np.float32))


def record_classifier(path, classifier_path: str):
    """Set ``classifier_path`` and append a dated entry to ``classifier_history``."""
    entry = f"{time.strftime('%Y-%m-%d')} {classifier_path}".encode()
    with h5py.File(path, "r+") as f:
        g = f.require_group(GROUP)
        _write(g, "classifier_path", np.bytes_(classifier_path))
        history = list(g["classifier_history"][()]) if "classifier_history" in g else []
        _write(g, "classifier_history", np.array([*history, entry], dtype="S"))
