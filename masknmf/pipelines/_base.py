from abc import ABC, abstractmethod
import numpy as np
from typing import *

class BasePipeline(ABC):

    @property
    @abstractmethod
    def config(self):
        pass
    @abstractmethod
    def run(self, data):
        """
        Run the analysis pipeline

        Args:
            data: input dataset
        """
        raise NotImplementedError