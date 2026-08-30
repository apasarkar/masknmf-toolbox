import json
import os
import tempfile
import time
from collections import Counter
from pathlib import Path
from typing import Literal, Optional, Sequence

import numpy as np
import roicat
from roicat.classification import ClassifierPackage
from roicat.data_importing import Data_roicat

from masknmf.demixing.labels import (
    CLASSIFIER_SUFFIX,
    read_labels,
    write_labels,
    write_predictions,
)
from masknmf.multisession.roicat_tracking import RoicatDataAdapter
from masknmf.utils import torch_select_device, display

TRAINING_SUFFIX = ".training.json"
_ROINET_URL = "https://osf.io/c8m3b/download"
_ROINET_HASH = "357a8d9b630ec79f3e015d0056a4c2d5"


def _um_per_pixel_list(adapter: Data_roicat) -> list[float]:
    um = adapter.um_per_pixel
    if isinstance(um, (list, tuple, np.ndarray)):
        return [float(u) for u in um]
    return [float(um)] * adapter.n_sessions


class RoicatClassifier:
    """
    ROICaT ROI classifier trained on one or more demixing sessions.

    Workflow: ``from_masknmf(files)`` picks up labels the labeling GUI stored in the
    files (or set ``labels`` yourself), ``train()``, ``save(path)``. Later
    ``from_disk(path).classify(other_files, write=True)`` labels new sessions.
    ``save`` writes the network (``.roicat_classifier``) plus a ``.training.json``
    with the labels and the session files they came from.
    """

    def __init__(
        self,
        training_data: Data_roicat | None = None,
        labels: list[list] | None = None,
        classifier: ClassifierPackage | None = None,
        device: Literal["auto", "cpu", "cuda"] = "auto",
        training_files: Sequence[str] | None = None,
    ):
        self._device = torch_select_device(device)
        self._training_data = training_data
        if training_files is None and training_data is not None:
            training_files = getattr(training_data, "session_files", None) or ()
        self._training_files = tuple(training_files or ())
        self._labels = None
        self.labels = labels
        self._classifier = classifier
        self._path: Optional[str] = None

    @classmethod
    def from_masknmf(cls, demixing_result_files: Sequence[str | Path], **adapter_kwargs) -> "RoicatClassifier":
        """
        Build the training data from demixing .hdf5 files (one per session). Labels
        the labeling GUI stored next to them (``<results>.labels.hdf5``) are used when
        every session is completely labeled; otherwise set ``labels`` before ``train``.
        """
        files = [str(f) for f in demixing_result_files]
        adapter = RoicatDataAdapter.from_masknmf(files, **adapter_kwargs)
        labels, incomplete = [], []
        for f in files:
            stored, names = read_labels(f)
            if stored is None or names is None or (stored < 0).any():
                incomplete.append(os.path.basename(f))
            else:
                labels.append([names[i] for i in stored])
        clf = cls(training_data=adapter, training_files=files)
        if incomplete:
            display(f"labels missing or incomplete in {incomplete}: label them in ClassificationVis or set clf.labels")
        else:
            clf.labels = labels
        return clf

    @property
    def device(self) -> Literal["cpu", "cuda"]:
        return self._device

    @property
    def training_data(self) -> Data_roicat | None:
        return self._training_data

    data_adapter = training_data

    @property
    def training_files(self) -> tuple:
        """Session files the training data came from (also restored by ``from_disk``)"""
        return self._training_files

    @property
    def labels(self) -> list[list] | None:
        """One list per session, one label (str or int) per ROI"""
        return self._labels

    @labels.setter
    def labels(self, new_labels: list[list] | None):
        if new_labels is None:
            self._labels = None
            return
        if self.training_data is not None:
            if len(new_labels) != self.training_data.n_sessions:
                raise ValueError(
                    f"training data has {self.training_data.n_sessions} sessions but "
                    f"{len(new_labels)} label lists were provided"
                )
            for k, session_labels in enumerate(new_labels):
                n = self.training_data.n_roi[k]
                if n != len(session_labels):
                    raise ValueError(
                        f"At session {k} the number of neural signals is {n} but the "
                        f"number of labels provided is {len(session_labels)}"
                    )
        self._labels = [list(l) for l in new_labels]

    @property
    def trainable(self) -> bool:
        """Labels are attached to training data; False for a classifier loaded from disk"""
        return self.training_data is not None and self.labels is not None

    @property
    def class_counts(self) -> dict | None:
        """Labeled ROI count per class across all sessions"""
        if self.labels is None:
            return None
        return dict(Counter(l for session_labels in self.labels for l in session_labels))

    @property
    def valid_classes(self) -> list | None:
        counts = self.class_counts
        return None if counts is None else sorted(counts, key=str)

    @property
    def classifier(self) -> ClassifierPackage | None:
        return self._classifier

    @classifier.setter
    def classifier(self, updated_classifier: ClassifierPackage | None):
        self._classifier = updated_classifier

    @property
    def label_names(self) -> list[str] | None:
        """Class names of the trained/loaded classifier, in prediction-index order"""
        return None if self.classifier is None else list(self.classifier.label_names)

    def train(self, roinet_dir: Optional[str | Path] = None, num_workers: int = -1) -> ClassifierPackage:
        """
        Embed every ROI image with ROInet, fit a logistic regression on the
        labels and package the result as a ClassifierPackage (self.classifier).

        Parameters
        ----------
        roinet_dir : path, optional
            Where the pretrained ROInet weights are cached (default: system temp dir).
        num_workers : int
            Dataloader workers for the embedding pass; 0 runs in-process.
        """
        if not self.trainable:
            raise ValueError("training needs both training_data and labels")
        counts = self.class_counts
        if len(counts) < 2:
            raise ValueError(f"training needs at least 2 classes, labels only contain {sorted(counts, key=str)}")
        rare = {c: n for c, n in counts.items() if n < 2}
        if rare:
            raise ValueError(
                "training needs at least 2 labeled ROIs per class for the stratified "
                f"train/test split, too few: {rare}"
            )
        adapter = self.training_data
        adapter.set_class_labels(labels=self.labels)

        roinet = roicat.ROInet.ROInet_embedder(
            device=self.device,
            dir_networkFiles=str(roinet_dir or tempfile.gettempdir()),
            download_method="check_local_first",
            download_url=_ROINET_URL,
            download_hash=_ROINET_HASH,
            forward_pass_version="head",
            verbose=True,
        )
        roinet.generate_dataloader(
            ROI_images=adapter.ROI_images,
            um_per_pixel=adapter.um_per_pixel,
            pref_plot=False,
            numWorkers_dataloader=num_workers,
            persistentWorkers_dataloader=num_workers != 0,
        )
        roinet.generate_latents()

        x = np.asarray(roinet.latents, dtype=np.float32)
        y = np.concatenate(adapter.class_labels_index).astype(np.int64)
        names = [str(n) for n in adapter.unique_class_labels]

        autoclassifier = roicat.classification.classifier.Auto_LogisticRegression(
            X=x,
            y=y,
            params_LogisticRegression={"C": [1e-13, 1e3]},
            label_names=names,
            verbose=True,
        )
        autoclassifier.fit()

        self.classifier = ClassifierPackage(
            classifier=autoclassifier,
            embedder=roinet,
            label_names=[names[int(c)] for c in autoclassifier.model_best.classes_],
            size_images_in=tuple(adapter.ROI_images[0].shape[1:]),
            um_per_pixel_training=_um_per_pixel_list(adapter)[0],
        )
        return self.classifier

    def save(self, outpath: Path | str) -> str:
        """
        Write ``<outpath>.roicat_classifier`` (the network) and ``<outpath>.training.json``
        (labels, label names, training session files). Returns the classifier path.
        """
        if self.classifier is None:
            raise ValueError("nothing to save: call train() or from_disk() first")
        path = os.path.abspath(outpath)
        self.classifier.save(path, overwrite=True)
        if not path.endswith(CLASSIFIER_SUFFIX):
            path += CLASSIFIER_SUFFIX
        meta = {
            "classifier": os.path.basename(path),
            "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "label_names": self.label_names,
            "training_files": [list(f) if isinstance(f, tuple) else f for f in self.training_files],
            "labels": None if self.labels is None else [[str(l) for l in s] for s in self.labels],
        }
        with open(path[: -len(CLASSIFIER_SUFFIX)] + TRAINING_SUFFIX, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        self._path = path
        display(f"Saved classifier to: {path}")
        return path

    @classmethod
    def from_disk(
        cls, filepath: Path | str, device: Literal["auto", "cpu", "cuda"] = "auto"
    ) -> "RoicatClassifier":
        """
        Load a saved classifier for inference (``trainable`` is False). The labels and
        training files are restored from the ``.training.json`` next to it when present.
        The net stays on CPU until predict moves it to ``device``.
        """
        path = os.path.abspath(filepath)
        clf = cls(classifier=ClassifierPackage.load(path), device=device)
        clf._path = path
        meta_path = path[: -len(CLASSIFIER_SUFFIX)] + TRAINING_SUFFIX if path.endswith(CLASSIFIER_SUFFIX) else None
        if meta_path and os.path.isfile(meta_path):
            with open(meta_path, encoding="utf-8") as f:
                meta = json.load(f)
            clf._training_files = tuple(tuple(x) if isinstance(x, list) else x for x in meta.get("training_files", ()))
            clf._labels = meta.get("labels")
        return clf

    from_file = from_disk

    def classify(
        self,
        data: Data_roicat | Sequence[str | Path],
        write: bool = False,
        batch_size: int = 256,
    ) -> tuple[list[np.ndarray], list[list[str]], list[np.ndarray]]:
        """
        Classify every ROI in every session.

        Parameters
        ----------
        data : Data_roicat or sequence of demixing .hdf5 paths
            ROI images to classify; paths are loaded into a RoicatDataAdapter.
        write : bool
            With paths: store ``class_predictions`` / ``class_probabilities`` /
            ``classified_with`` in each session's ``.labels.hdf5`` sidecar and give
            still-unlabeled ROIs the predicted label (existing labels are kept).

        Returns
        -------
        (label_ids, label_names, probabilities), one entry per session: indices into
        ``self.label_names``, the names, and (n_roi, n_classes) probabilities.
        """
        if self.classifier is None:
            raise ValueError("no classifier: call train() or from_disk() first")
        files = None
        if not isinstance(data, Data_roicat):
            files = [str(f) for f in data]
            data = RoicatDataAdapter.from_masknmf(files)
        elif write:
            raise ValueError("write=True needs demixing .hdf5 paths, not an adapter")

        size_expected = tuple(self.classifier.preprocessing["size_images_in"])
        size_given = tuple(data.ROI_images[0].shape[1:])
        if size_expected != size_given:
            display(f"Classifier expects ROI images of {size_expected}, got {size_given}: re-cropping.")
            data.transform_spatialFootprints_to_ROIImages(out_height_width=size_expected)

        label_ids, labels, probabilities = [], [], []
        for images, um in zip(data.ROI_images, _um_per_pixel_list(data)):
            ids, probs = self.classifier.predict(
                roi_images=np.asarray(images), um_per_pixel=um, batch_size=batch_size, device=self.device
            )
            label_ids.append(ids)
            probabilities.append(probs)
            labels.append([self.classifier.label_names[i] for i in ids])
        if write:
            for f, names, probs in zip(files, labels, probabilities):
                self._write_session(f, names, probs)
        return label_ids, labels, probabilities

    def _write_session(self, path: str, predicted: list[str], probs: np.ndarray):
        stored, names = read_labels(path)
        names = list(names or [])
        for n in dict.fromkeys(predicted):
            if n not in names:
                names.append(n)
        pred = np.array([names.index(n) for n in predicted], dtype=np.int64)
        labels = np.full(len(pred), -1, dtype=np.int64) if stored is None or len(stored) != len(pred) else stored.copy()
        fill = labels < 0
        labels[fill] = pred[fill]
        write_labels(path, labels, names)
        write_predictions(path, pred, probs.max(axis=1), self._path or "")
