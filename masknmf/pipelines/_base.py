from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
import numpy as np
from typing import *

from masknmf.pipelines.scraper import slugify

class BasePipeline(ABC):

    @property
    @abstractmethod
    def config(self):
        pass

    @property
    def output_folder(self) -> str | Path | None:
        return self._output_folder

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

    @abstractmethod
    def run(self, data):
        """
        Run the analysis pipeline

        Args:
            data: input dataset
        """
        raise NotImplementedError
