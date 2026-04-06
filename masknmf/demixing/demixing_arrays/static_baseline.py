from typing import *
import numpy as np
from masknmf.arrays.array_interfaces import ArrayLike, TensorFlyWeight
import torch
from masknmf.demixing.demixing_arrays.demixing_array_utils import check_spatial_crop_effect

class StaticBackgroundArray(ArrayLike):
    """
    Class for managing static baseline estimates. These are typically 2D tensors, shape (height, width)
    """

    def __init__(
            self,
            flyweight: TensorFlyWeight,
            rescale: bool = False,
    ):
        self._flyweight=flyweight
        self.flyweight.validate_attributes(["baseline"])
        self._shape = self.flyweight.baseline.shape
        self._rescale=rescale

        self._default_normalizer = torch.ones_like(self.baseline, device=self.device).float()
        if hasattr(self.flyweight, "normalizer"):
            if self.flyweight.normalizer.shape[0] != self.shape[0] or self.flyweight.normalizer.shape[1] != self.shape[
                1]:
                raise ValueError("Normalizer from flyweight had dimensions not equal to the fov dimensions")

    @property
    def flyweight(self) -> TensorFlyWeight:
        return self._flyweight

    @classmethod
    def from_tensors(cls,
                     baseline: torch.Tensor,
                     normalizer: Optional[torch.Tensor] = None,
                     rescale: bool=False):
        """
        Constructor for static baseline class
        Args:
            baseline (torch.Tensor): Shape (height, width)
            normalizer (Optional[torch.Tensor]): Shape (height, width)
        """

        flyweight = TensorFlyWeight(baseline=baseline, normalizer=normalizer)
        return cls(flyweight,
                   rescale=rescale)

    @classmethod
    def from_flyweight(cls,
                       flyweight: TensorFlyWeight,
                       rescale: bool = False):
        return cls(flyweight,
                   rescale=rescale)

    @property
    def device(self) -> str:
        return self.flyweight.device

    def to(self, new_device: str):
        if self.flyweight.device != new_device:
            self.flyweight.to(new_device)
        self._move_local_tensors(new_device)

    def _move_local_tensors(self, new_device: str):
        self._default_normalizer = self._default_normalizer.to(new_device)


    @property
    def shape(self) -> tuple[int, int]:
        return self._shape

    @property
    def dtype(self) -> str:
        """
        data type, default np.float32
        """
        return np.float32

    @property
    def normalizer(self) -> torch.Tensor:
        if not hasattr(self.flyweight, "normalizer"):
            return self._default_normalizer
        return self.flyweight.normalizer

    @property
    def rescale(self):
        return self._rescale

    @rescale.setter
    def rescale(self, new_value: bool):
        self._rescale = new_value

    @property
    def baseline(self) -> torch.Tensor:
        return self.flyweight.baseline

    def getitem_tensor(
            self,
            item: Union[int, list, np.ndarray, Tuple[Union[int, np.ndarray, slice, range]]],
    ):
        cropped_baseline = self.baseline[item]
        if self.rescale:
            cropped_normalizer = self.normalizer[item]
            return cropped_baseline * cropped_normalizer

        return cropped_baseline

    def __getitem__(
        self,
        item: Union[int, list, np.ndarray, Tuple[Union[int, np.ndarray, slice, range]]],
    ) -> np.ndarray:
        product = self.getitem_tensor(item)
        product = product.cpu().numpy().astype(self.dtype)
        return product





