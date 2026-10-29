from abc import ABC, abstractmethod
from pathlib import Path

import torch
import numpy as np
import masknmf
from masknmf.arrays.array_interfaces import ArrayLike
from masknmf.demixing.demixing_arrays import SignalsArray
from masknmf.demixing.demixing_arrays import FluctuatingBackgroundArray
from masknmf.utils import SparseCOOTensor

class BaseResults:

    # an adapter reads another pipeline's results: it sets these four and overrides matches and from_path
    pipeline = "masknmf"
    source = ""  # names what the adapter opens, in the CLI and launcher help
    raw_source = ""  # names what --raw must be, when the adapter cannot open without it
    patterns = ()  # file dialog patterns beyond *.h5 *.hdf5

    def __init__(self):
        pass

    @classmethod
    def matches(cls, path: Path) -> bool:
        return False

    @classmethod
    def expand(cls, path: Path) -> list[Path]:
        return [path]

    @classmethod
    def describe(cls, path: Path) -> str:
        return cls.pipeline

    @classmethod
    def from_path(cls, path: Path, raw: ArrayLike | str | Path | None = None, device: str = "cpu") -> tuple["BaseResults", ArrayLike | None, list[str]]:
        """The results at path, the raw movie to show, and notes on what was found but not used."""
        raise NotImplementedError

    @property
    def frame_rate(self) -> float | None:
        return None

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
    def compression_array_roi_averages(self) -> torch.Tensor | None:
        """By default, a pipeline does NOT have a compression array to average"""
        return None

    @property
    def fluctuating_background_roi_averages(self) -> np.ndarray:
        raise NotImplementedError

    @property
    def residual_roi_averages(self) -> np.ndarray:
        raise NotImplementedError


