from typing import *
import numpy as np
from fastplotlib.widgets import ImageWidget
import fastplotlib as fpl
from imgui_bundle import imgui
from fastplotlib import ui
import pygfx
import torch
from collections import OrderedDict
import masknmf.arrays
from masknmf.utils import display
from functools import partial
from fastplotlib.widgets.nd_widget._index import ReferenceIndex


class MultisessionDemixingVis:
    """
    This is a general viewer for analyzing cross-session tracking data
    """
    def __init__(
            self,
            demixing_results: list[masknmf.DemixingResults],
            labels_per_session: list[list],
            device='cpu'
    ):
        self._device = device
        self._demixing_results = demixing_results
        for elt in self._demixing_results:
            elt.to(device)
        self._labels_per_session = labels_per_session


    @property
    def device(self):
        return self._device