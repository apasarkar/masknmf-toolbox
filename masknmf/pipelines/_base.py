from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
import os
import numpy as np
from typing import *

from masknmf.pipelines.scraper import slugify
from masknmf.compression import CompressionArray
from masknmf.utils import display, has_group, torch_select_device

class BasePipeline(ABC):
    def __init__(self,
                 output_folder: str | Path | None = None,
                 frame_batch_size: int = 300,
                 device: Literal["auto", "cuda", "cpu"] = "auto"):
        self._output_folder = output_folder
        self._frame_batch_size = frame_batch_size
        self._device = device

    @property
    @abstractmethod
    def config(self):
        pass

    @property
    def output_folder(self) -> str | Path | None:
        return self._output_folder

    @property
    def frame_batch_size(self) -> int:
        return self._frame_batch_size

    @property
    def device(self) -> Literal["auto", "cuda", "cpu"]:
        return self._device

    @property
    def torch_device(self) -> str:
        """``device`` with "auto" resolved to the device pytorch will use."""
        return torch_select_device(self.device)

    def create_run_folder(self) -> Path:
        """
        Make ``<output_folder>/<YYYYmmdd_HHMMSS>_<pipeline slug>/`` (the working directory when output_folder is None),
        adding a numeric suffix when a run started in the same second.
        """
        base = Path.cwd() if self.output_folder is None else Path(self.output_folder).expanduser().resolve()
        name = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{slugify(name_class=type(self).__name__)}"
        candidate = base / name
        suffix = 0
        while True:
            try:
                candidate.mkdir(parents=True, exist_ok=False)
                return candidate
            except FileExistsError:
                suffix += 1
                candidate = base / f"{name}_{suffix}"

    def results_path(self, resume: bool = False) -> str:
        """
        ``results.hdf5`` in a new run folder. With ``resume``, the one in output_folder itself, an earlier run
        folder whose compression is reused.
        """
        if not resume:
            path = os.path.join(self.create_run_folder(), "results.hdf5")
            display(f"Writing results to {path}")
            return path
        path = os.path.join(Path.cwd() if self.output_folder is None else self.output_folder, "results.hdf5")
        if not has_group(path, CompressionArray.__name__):
            raise ValueError(f"You specified that compression should be skipped but {path} holds no compression")
        return path

    @abstractmethod
    def run(self, data):
        """
        Run the analysis pipeline

        Args:
            data: input dataset
        """
        raise NotImplementedError
