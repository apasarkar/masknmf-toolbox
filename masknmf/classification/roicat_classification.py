import os
import tempfile
from collections import Counter
from pathlib import Path
from typing import Literal, Optional

import numpy as np
import roicat
from roicat.classification import ClassifierPackage

from masknmf.multisession.roicat_tracking import RoicatDataAdapter
from masknmf.utils import torch_select_device, display

CLASSIFIER_SUFFIX = ".roicat_classifier"
_ROINET_URL = "https://osf.io/c8m3b/download"
_ROINET_HASH = "357a8d9b630ec79f3e015d0056a4c2d5"


def _um_per_pixel_list(adapter: RoicatDataAdapter) -> list[float]:
    um = adapter.um_per_pixel
    if isinstance(um, (list, tuple, np.ndarray)):
        return [float(u) for u in um]
    return [float(um)] * adapter.n_sessions


class RoicatClassifier:
    """
    Train / apply a ROICaT ROI classifier over one or more sessions held in a
    RoicatDataAdapter (built from demixing result .hdf5 files).

    Workflow: set `labels` (one list per session, one entry per ROI, str or int),
    call `train()`, `save(path)`; later `from_file(path).classify(adapter)`.
    """

    def __init__(
        self,
        data_adapter: RoicatDataAdapter | None = None,
        labels: list[list] | None = None,
        classifier: ClassifierPackage | None = None,
        device: Literal["auto", "cpu", "cuda"] = "auto",
    ):
        self._device = torch_select_device(device)
        self._data_adapter = data_adapter
        self._labels = None
        self.labels = labels
        self._classifier = classifier

    @property
    def device(self) -> Literal["cpu", "cuda"]:
        return self._device

    @property
    def data_adapter(self) -> RoicatDataAdapter | None:
        return self._data_adapter

    @property
    def labels(self) -> list[list] | None:
        """One list per session, one label (str or int) per ROI"""
        return self._labels

    @labels.setter
    def labels(self, new_labels: list[list] | None):
        if new_labels is None:
            self._labels = None
            return
        if self.data_adapter is None:
            raise ValueError("set data_adapter before labels: labels are matched against its ROIs")
        if len(new_labels) != self.data_adapter.n_sessions:
            raise ValueError(
                f"data_adapter has {self.data_adapter.n_sessions} sessions but "
                f"{len(new_labels)} label lists were provided"
            )
        for k, session_labels in enumerate(new_labels):
            n = self.data_adapter.n_roi[k]
            if n != len(session_labels):
                raise ValueError(
                    f"At session {k} the number of neural signals is {n} but the "
                    f"number of labels provided is {len(session_labels)}"
                )
        self._labels = [list(l) for l in new_labels]

    @property
    def is_trainable(self) -> bool:
        return self.data_adapter is not None and self.labels is not None

    @property
    def class_counts(self) -> dict | None:
        """Labeled ROI count per class across all sessions"""
        if self.labels is None:
            return None
        return dict(Counter(l for session_labels in self.labels for l in session_labels))

    @property
    def valid_classes(self) -> set | None:
        counts = self.class_counts
        return None if counts is None else set(counts)

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

        Args:
            roinet_dir: where the pretrained ROInet weights are cached (default: system temp dir)
            num_workers: dataloader workers for the embedding pass; 0 runs in-process
        """
        if not self.is_trainable:
            raise ValueError(
                "Training did not occur because at least one of the following required "
                "attributes are missing: data_adapter, labels"
            )
        counts = self.class_counts
        if len(counts) < 2:
            raise ValueError(f"training needs at least 2 classes, labels only contain {sorted(counts)}")
        rare = {c: n for c, n in counts.items() if n < 2}
        if rare:
            raise ValueError(
                "training needs at least 2 labeled ROIs per class for the stratified "
                f"train/test split, too few: {rare}"
            )
        adapter = self.data_adapter
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
        """Write the classifier package; returns the path (suffix .roicat_classifier is appended if missing)"""
        if self.classifier is None:
            raise ValueError("nothing to save: call train() or from_file() first")
        path = os.path.abspath(outpath)
        self.classifier.save(path, overwrite=True)
        if not path.endswith(CLASSIFIER_SUFFIX):
            path += CLASSIFIER_SUFFIX
        display(f"Saved classifier to: {path}")
        return path

    def classify(
        self, data_adapter: RoicatDataAdapter
    ) -> tuple[list[np.ndarray], list[list[str]], list[np.ndarray]]:
        """
        Classify every ROI in every session of data_adapter.

        Returns three lists with one entry per session:
            - label ids, shape (n_roi,), indices into self.label_names
            - label names, one str per ROI
            - probabilities, shape (n_roi, n_classes), columns in self.label_names order
        """
        if self.classifier is None:
            raise ValueError("no classifier: call train() or from_file() first")
        size_expected = tuple(self.classifier.preprocessing["size_images_in"])
        size_given = tuple(data_adapter.ROI_images[0].shape[1:])
        if size_expected != size_given:
            display(
                f"Classifier expects ROI spatial dimensions {size_expected} but the "
                f"data_adapter has {size_given}. Transforming the shapes to match."
            )
            data_adapter.transform_spatialFootprints_to_ROIImages(out_height_width=size_expected)

        label_ids, labels, probabilities = [], [], []
        for images, um in zip(data_adapter.ROI_images, _um_per_pixel_list(data_adapter)):
            ids, probs = self.classifier.predict(
                roi_images=np.asarray(images), um_per_pixel=um, device=self.device
            )
            label_ids.append(ids)
            probabilities.append(probs)
            labels.append([self.classifier.label_names[i] for i in ids])
        return label_ids, labels, probabilities

    @classmethod
    def from_file(
        cls, filepath: Path | str, device: Literal["auto", "cpu", "cuda"] = "auto"
    ) -> "RoicatClassifier":
        """
        Load a saved classifier package. The net is kept on CPU until predict
        moves it to `device` (the roicat API has no device setter).
        """
        return cls(classifier=ClassifierPackage.load(os.path.abspath(filepath)), device=device)
