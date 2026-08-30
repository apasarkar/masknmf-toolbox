from pathlib import Path
from typing import Optional, Sequence

import numpy as np

HDF5_GROUP = "DemixingResults"


def _write_dataset(group, key: str, data: np.ndarray):
    # overwrite in place when possible so the file doesn't grow with each write
    if key in group and group[key].shape == data.shape and group[key].dtype == data.dtype:
        group[key][...] = data
    else:
        if key in group:
            del group[key]
        group.create_dataset(key, data=data)


class LabelStore:
    """
    Persist labels to an npz and/or into per-session hdf5 files.

    hdf5 writes land in DemixingResults/{class_labels,label_names,labels_complete}
    so the labels travel with the data they describe.
    """

    def __init__(
        self,
        npz_path: Optional[str] = None,
        hdf5_files: Optional[Sequence[str]] = None,
        session_sizes: Optional[Sequence[int]] = None,
    ):
        self.npz_path = str(npz_path) if npz_path is not None else None
        self.hdf5_files = [str(f) for f in hdf5_files] if hdf5_files else None
        self.session_sizes = tuple(session_sizes) if session_sizes else None
        self.error: Optional[str] = None

    @property
    def enabled(self) -> bool:
        return self.npz_path is not None or self.hdf5_files is not None

    def load(self, n_items: int) -> Optional[dict]:
        """Names and labels from the npz, or None when absent or the wrong size."""
        if self.npz_path is None or not Path(self.npz_path).exists():
            return None
        saved = np.load(self.npz_path)
        labels = saved["class_labels"]
        if labels.shape[0] != n_items:
            return {"label_names": [str(n) for n in saved["label_names"]]}
        return {
            "label_names": [str(n) for n in saved["label_names"]],
            "class_labels": labels,
        }

    def split(self, labels: np.ndarray) -> list:
        """Labels split per session, the shape RoicatDataAdapter.set_class_labels takes."""
        if self.session_sizes is None:
            return [labels]
        return list(np.split(labels, np.cumsum(self.session_sizes)[:-1]))

    def save(self, label_names: Sequence[str], labels: np.ndarray, masks=None) -> bool:
        """Write both targets. Returns False and sets .error on failure."""
        try:
            if self.hdf5_files is not None:
                self._save_hdf5(label_names, labels, masks)
            if self.npz_path is not None:
                data = {
                    "label_names": np.array(list(label_names)),
                    "class_labels": labels,
                }
                if self.session_sizes is not None:
                    data["session_sizes"] = np.array(self.session_sizes)
                np.savez(self.npz_path, **data)
            self.error = None
            return True
        except OSError as e:
            self.error = f"save failed: {e}"
            return False

    def _save_hdf5(self, label_names, labels, masks):
        import h5py

        names = np.array([n.encode() for n in label_names])
        start = 0
        for fname, part in zip(self.hdf5_files, self.split(labels)):
            with h5py.File(fname, "r+") as f:
                g = f.require_group(HDF5_GROUP)
                _write_dataset(g, "class_labels", part.astype(np.int64))
                _write_dataset(g, "label_names", names)
                _write_dataset(g, "labels_complete", np.bool_((part >= 0).all()))
                if masks is not None:
                    _write_dataset(g, "roi_masks", masks[start : start + len(part)])
            start += len(part)
