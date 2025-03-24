from masknmf.arrays.array_interfaces import LazyFrameLoader
from .strategies import MotionCorrectionStrategy
import torch
from typing import *
from .strategies import MotionCorrectionStrategy
import math
import numpy as np

class RegistrationArray:
    def __init__(self,
                 reference_dataset: LazyFrameLoader,
                 strategy: MotionCorrectionStrategy,
                 template: torch.tensor,
                 device: str = "cpu",
                 batch_size: int = 200,
                 target_dataset: Optional[LazyFrameLoader] = None):
        """
        Array-like motion correction representation that support on-the-fly motion correction

        Args:
            reference_dataset (LazyFrameLoder): Image stack that we use to compute motion correction transform relative to template
            strategy (masknmf.MotionCorrectionStrategy): The method used to register each frame to the template
            template (torch.tensor): The template to which we register the data
            device (torch.tensor): The device on which computations are performed (for e.g. 'cuda' or 'cpu')
            batch_size (int): The number of frames we load onto the computation device at a time to do motion correction.
            target_dataset (Optional[LazyFrameLoader]): Once we learn the motion correction transform by aligning reference_dataset
                with template, we actually apply the transform to target_dataset, if it is specified. If None, we apply the
                transform to reference_dataset
        """
        self._reference_dataset = reference_dataset
        self._strategy = strategy
        self._template = template
        self._device = device
        self._batch_size = batch_size
        if target_dataset is None:
            self._target_dataset = reference_dataset
            self._same_data = True
        else:
            self._target_dataset = target_dataset
            self._same_data = False

        self._shape = self.reference_dataset.shape
        self._ndim = self.reference_dataset.ndim


    @property
    def ndim(self) -> int:
        return self._ndim

    @property
    def shape(self) -> Tuple[int, int, int]:
        return self._shape

    @property
    def dtype(self) -> str:
        return "float32"

    @property
    def reference_dataset(self) -> LazyFrameLoader:
        return self._reference_dataset

    @property
    def target_dataset(self) -> LazyFrameLoader:
        return self._target_dataset

    @property
    def strategy(self) -> MotionCorrectionStrategy:
        return self._strategy

    @property
    def template(self) -> torch.tensor:
        return self._template

    @property
    def device(self) -> str:
        return self._device

    @property
    def batch_size(self) -> int:
        return self._batch_size

    def getitem_tensor(self,
                       idx: Union[int, list, np.ndarray, Tuple[Union[int, np.ndarray, slice, range]]]) -> np.ndarray:
        """Retrieve motion-corrected frame at index `idx`."""
        reference_data_indexed = self._reference_dataset[idx]
        if self._same_data is False:
            target_data_indexed = self._target_dataset
        else:
            target_data_indexed = reference_data_indexed

        if reference_data_indexed.ndim == 2:
            reference_data_indexed = reference_data_indexed[None, ...]
            target_data_indexed = target_data_indexed[None, ...]

        if self.batch_size > reference_data_indexed.shape[0]:
            # Directly motion correct the data
            reference_subset = torch.from_numpy(reference_data_indexed).float().to(self.device)
            target_data_subset = torch.from_numpy(reference_data_indexed).float().to(self.device)
            moco_output = self.strategy.correct(reference_subset,
                                                target_frames=target_data_subset)[0].cpu()

        else:
            num_iters = math.ceil(reference_data_indexed.shape[0])
            outputs = []
            for k in range(num_iters):
                start = k * self.batch_size
                end = min(start + self.batch_size, reference_data_indexed.shape[0])

                reference_subset = torch.from_numpy(reference_data_indexed[start:end]).float().to(self.device)
                target_subset = torch.from_numpy(target_data_indexed[start:end]).float().to(self.device)

                subset_output = self.strategy.correct(reference_subset,
                                                      target_frames=target_subset)[0].cpu()
                outputs.append(subset_output)
            moco_output = torch.concatenate(outputs, dim=0)
        return moco_output


    def __getitem__(self,
                    item: Union[int, list, np.ndarray, Tuple[Union[int, np.ndarray, slice, range]]]
                    ) -> np.ndarray:
        moco_output = self.getitem_tensor(item)
        return moco_output.numpy()