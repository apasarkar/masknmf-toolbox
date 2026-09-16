"""
ROI labels, classifier predictions and provenance for a demixing session, kept in a
sidecar ``<results>.labels.hdf5`` next to the results file so ``demixing_results.hdf5``
itself is never modified.

Datasets: ``class_labels`` (num_rois,) int64 with -1 = unlabeled, ``label_names``,
``labels_complete``, ``roi_masks``, ``class_predictions``, ``class_probabilities``,
``classified_with``, ``classifier_path``, ``classifier_history``. Every function takes
either the results path or the sidecar path.
"""

import os
import time
from typing import Optional, Sequence

import h5py
import numpy as np

SIDECAR_SUFFIX = ".labels.hdf5"
CLASSIFIER_SUFFIX = ".roicat_classifier"
_LEGACY_GROUP = "DemixingResults"  # early files carried the labels inside the results


def labels_path(path) -> str:
    """Sidecar path for a results file: ``demixing_results.hdf5 -> demixing_results.labels.hdf5``."""
    path = str(path)
    if path.endswith(SIDECAR_SUFFIX):
        return path
    return os.path.splitext(path)[0] + SIDECAR_SUFFIX


def _open(path, mode: str):
    sidecar = labels_path(path)
    f = h5py.File(sidecar, mode)
    if mode != "r" and "source" not in f.attrs:
        f.attrs["source"] = os.path.basename(str(path))
    return f


def _write(group, key: str, data):
    data = np.asarray(data)
    if key in group and group[key].shape == data.shape and group[key].dtype == data.dtype:
        group[key][...] = data
    else:
        if key in group:
            del group[key]
        group.create_dataset(key, data=data)


def _decode(names) -> list[str]:
    return [n.decode() if isinstance(n, bytes) else str(n) for n in names]


def read_labels(path) -> tuple[Optional[np.ndarray], Optional[list[str]]]:
    """(class_labels, label_names) for a session; None where absent."""
    sidecar = labels_path(path)
    if os.path.isfile(sidecar):
        with h5py.File(sidecar, "r") as f:
            return _read_pair(f)
    if str(path) != sidecar and os.path.isfile(path):
        with h5py.File(path, "r") as f:  # legacy: labels written into the results file
            g = f.get(_LEGACY_GROUP)
            if g is not None:
                return _read_pair(g)
    return None, None


def _read_pair(g):
    labels = g["class_labels"][()] if "class_labels" in g else None
    names = _decode(g["label_names"][()]) if "label_names" in g else None
    return labels, names


def write_labels(path, labels, label_names: Sequence[str]):
    labels = np.asarray(labels, dtype=np.int64)
    with _open(path, "a") as f:
        _write(f, "class_labels", labels)
        _write(f, "label_names", np.array([n.encode() for n in label_names]))
        _write(f, "labels_complete", np.bool_((labels >= 0).all()))


def write_predictions(path, predictions, probabilities, classified_with: str = ""):
    """Per-ROI predicted label index, its confidence, and which classifier produced them."""
    with _open(path, "a") as f:
        _write(f, "class_predictions", np.asarray(predictions, dtype=np.int64))
        _write(f, "class_probabilities", np.asarray(probabilities, dtype=np.float32))
        _write(f, "classified_with", np.bytes_(classified_with))


def write_masks(path, masks):
    with _open(path, "a") as f:
        _write(f, "roi_masks", np.asarray(masks, dtype=np.float32))


def record_classifier(path, classifier_path: str):
    """Set ``classifier_path`` and append a dated entry to ``classifier_history``."""
    entry = f"{time.strftime('%Y-%m-%d')} {classifier_path}".encode()
    with _open(path, "a") as f:
        _write(f, "classifier_path", np.bytes_(classifier_path))
        history = list(f["classifier_history"][()]) if "classifier_history" in f else []
        _write(f, "classifier_history", np.array([*history, entry], dtype="S"))
