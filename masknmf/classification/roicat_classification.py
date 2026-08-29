from masknmf.demixing.demixing_results import DemixingResults
import torch
import os
import sys
from typing import *
from pathlib import Path
import scipy
import scipy.sparse
from roicat.classification import ClassifierPackage
from masknmf.multisession.roicat_tracking import RoicatDataAdapter
from roicat.data_importing import Data_roicat
from roicat.pipelines import pipeline_tracking
from roicat.util import get_default_parameters
from roicat import helpers
from roicat.util import RichFile_ROICaT
import json
import datetime

import warnings
from typing import Callable, Optional, Sequence

import numpy as np
import scipy.sparse
from masknmf.utils import torch_select_device, display

class RoicatClassifier:
    """
    Intended use: pass one or more demixing results objects (.hdf5) into a RoicatDataAdapter

    """
    def __init__(self,
                 data_adapter: RoicatDataAdapter | None = None,
                 labels: list[list] | None = None,
                 classifier: ClassifierPackage | None = None,
                 device: Literal["auto", "cpu", "cuda"] = "auto"):

        self._device = torch_select_device(device)
        self._data_adapter = data_adapter
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
        if self.data_adapter is not None:
            return self._labels
        else:
            warnings.warn("Did not set labels, you have not passed in a data_adapter object so there are no images to label")

    @labels.setter
    def labels(self, new_labels: list[list] | None):
        if new_labels is not None:
            for k in range(self.data_adapter.n_sessions):
                num_neurons_sess_k = self.data_adapter.n_roi[k]
                if not num_neurons_sess_k == len(new_labels[k]):
                    raise ValueError(f"At session {k} the number of neural signals "
                                     f"is {num_neurons_sess_k} but the number of labels provided is {len(new_labels[k])}")

        self._labels = new_labels

    @property
    def is_trainable(self) -> bool:
        if self.data_adapter is not None and self.labels is not None:
            return True
        return False

    @property
    def valid_classes(self) -> set | None:
        if self.labels is None:
            return None
        else:
            classes = set()
            for session_labels in self.labels:
                classes.update(session_labels)
            return classes

    @property
    def classifier(self) -> ClassifierPackage | None:
        return self._classifier

    @classifier.setter
    def classifier(self, updated_classifier: ClassifierPackage | None):
        self._classifier = updated_classifier

    def train(self):
        """
        Output of this is to generate a ClassifierPackage object that can can be used for classification
        """
        if not self.is_trainable():
            raise ValueError("Training did not occur because at least one of the following required attributes are missing: data_adapter, labels")

        ## First associate with the data adapter the labels
        self.data_adapter.set_class_labels(labels=self.labels)

        
        device = self.device
        dir_temp = tempfile.gettempdir()

        roinet = roicat.ROInet.ROInet_embedder(
            device=device,  ## Which torch device to use ('cpu', 'cuda', etc.)
            dir_networkFiles=dir_temp,  ## Directory to download the pretrained network to
            download_method='check_local_first',
            ## Check to see if a model has already been downloaded to the location (will skip if hash matches)
            download_url='https://osf.io/c8m3b/download',  ## URL of the model
            download_hash='357a8d9b630ec79f3e015d0056a4c2d5',  ## Hash of the model file
            forward_pass_version='head',  ## How the data is passed through the network
            verbose=True,  ## Whether to print updates
        )

        roinet.generate_dataloader(
            ROI_images=self.data_adapter.ROI_images,  ## Input images of ROIs
            um_per_pixel=self.data_adapter.um_per_pixel,  ## Resolution of FOV
            pref_plot=False,  ## Whether or not to plot the ROI sizes
        )

        roinet.generate_latents()

        x = np.array(roinet.latents).astype(np.float32)
        y = np.concatenate(self.data_adapter.class_labels_index).astype(np.int64)

        autoclassifier = roicat.classification.classifier.Auto_LogisticRegression(
            X=x,
            y=y,
            params_LogisticRegression={
                'C': [1e-13, 1e3],
            },
            verbose=True,
        )
        autoclassifier.fit()

        self.classifier = roicat.classification.ClassifierPackage(
            classifier=autoclassifier,  ## The fitted Auto_LogisticRegression from above
            embedder=roinet,  ## The ROInet_embedder that produced the latents
            label_names=[str(l) for l in autoclassifier.model_best.classes_],  ## Class names, in classes_ order
            size_images_in=roicat_input.ROI_images[0].shape[1:],  ## (height, width) of the RAW ROI images
            um_per_pixel_training=roicat_input.um_per_pixel[0],  ## Recorded as provenance only
        )

    def save(self, outpath: Path | str):
        paths_save = os.path.abspath(outpath)
        self.classifier.save(paths_save, overwrite=True)
        display(f"Saved packet to: {paths_save}")


    def classify(self, data_adapter: RoicatDataAdapter) -> list[list] | None:
        """
        Takes as input a data_adapter, which might contain ROIs from many sessions
        Generates classifications for all ROIs across all sessions

        RoicatDataAdapter takes the ROIs from the individual session FOVs, computes their center of masses
        Args:
            data_adapter (RoicatDataAdapter): RoicatDataAdapter containing one or more sessions of data to be classified
        Returns:
            - list[list] containing label_ids. One list per session (and this list has length = number of rois for that session)
            - list[list] containing labels corresponding to the above label_ids
            - list[list] containing probability estimates for each classification
        """
        if self.classifier is not None:
            ## First verify that the data ROI shape dimensions match what the net expects
            size_expected = tuple(self.classifier.preprocessing['size_images_in'])
            if not size_expected == data_adapter.ROI_images[0].shape[1:]:
                display(f"Classifier expects ROI spatial dimensions {size_expected} "
                        f"but the data_adapter has ROI spatial dimensions {data_adapter.ROI_images[0].shape[1:]}. "
                        f"Transforming the shapes to match.")
                data_adapter.transform_spatialFootprints_to_ROIImages(out_height_width={size_expected})

            if isinstance(data_adapter.um_per_pixel, float):
                um_per_pixel_list = [data_adapter.um_per_pixel for k in range(data_adapter.n_sessions)]
            elif isinstance(data_adapter.um_per_pixel, list):
                um_per_pixel_list = data_adapter.um_per_pixel
            else:
                raise ValueError("data_adapter um per pixel ")

            predicted_label_ids = [] ## Each value here indexes into self.classifier.
            predicted_labels = []
            probabilities = []
            for images, um in zip(data_adapter.ROI_images, um_per_pixel_list):
                curr_predicted_label_ids, curr_probabilities = self.classifier.predict(roi_images=images,
                                                                             um_per_pixel=um)
                curr_predicted_labels = [self.classifier.label_names[i] for i in curr_predicted_label_ids]

                predicted_label_ids.append(curr_predicted_label_ids)
                probabilities.append(curr_probabilities)
                predicted_labels.append(curr_predicted_labels)

            return predicted_label_ids, predicted_labels, predicted_probabilities

        else:
            return None


    @classmethod
    def from_file(cls,
                  filepath: Path | str):
        abs_path = os.path.abspath(filepath)

        """
        For now the net is constructed and resides on CPU. If GPU is available, it will run on GPU when we call "predict"
        While this is slightly inefficient, the roicat API does not have a device setter (and classification does
        not need to be super high throughput in general)
        """
        classifier = roicat.classification.ClassifierPackage.load(abs_path)

        return cls(classifier=classifier)










