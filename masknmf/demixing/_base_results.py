from abc import ABC, abstractmethod

import torch
import numpy as np
import masknmf
from masknmf.arrays.array_interfaces import ArrayLike
from masknmf.demixing.demixing_arrays import SignalsArray
from masknmf.demixing.demixing_arrays import FluctuatingBackgroundArray
from masknmf.utils import SparseCOOTensor

class BaseResults:

    def __init__(self):
        pass

    @property
    def spatial_demixed(self) -> SparseCOOTensor:
        raise NotImplementedError

    @property
    def temporal_demixed(self) -> torch.Tensor:
        raise NotImplementedError

    @property
    def temporal_demixed_raw(self) -> torch.Tensor:
        raise NotImplementedError

    @property
    def global_residual_correlation_image(self) -> torch.Tensor | None:
        raise NotImplementedError

    @property
    def signals_array(self) -> SignalsArray | None:
        """Across pipelines this term should be a factorized product (spatial components @ temporal components)"""
        raise NotImplementedError

    @property
    def fluctuating_background_array(self) -> masknmf.ArrayLike | None:
        """Different pipelines have different models for representing the background"""
        raise NotImplementedError

    @property
    def compression_array(self) -> masknmf.CompressionArray | None:
        """By default, a pipeline does NOT have a compression array"""
        return None

    @property
    def shifts(self) -> np.ndarray | None:
        """ By default, a pipeline does NOT have registration shifts"""
        return None

    @property
    def raw_array(self) -> ArrayLike | None:
        """By default, a pipeline does NOT have the raw movie"""
        return None

    @property
    def registered_array(self) -> ArrayLike | None:
        """By default, a pipeline does NOT have the registered movie"""
        return None

    @property
    def residual_array(self) -> ArrayLike | None:
        raise NotImplementedError

    @property
    def compression_array_roi_averages(self) -> np.ndarray:
        raise NotImplementedError

    @property
    def fluctuating_background_roi_averages(self) -> np.ndarray:
        raise NotImplementedError

    @property
    def residual_roi_averages(self) -> np.ndarray:
        raise NotImplementedError


