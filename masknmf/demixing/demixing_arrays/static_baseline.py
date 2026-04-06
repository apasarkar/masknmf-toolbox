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
        self._shape = self.flyweight.baseline.shape
        self._rescale=rescale

    @property
    def flyweight(self) -> TensorFlyWeight:
        return self._flyweight

    @classmethod
    def from_tensors(cls,
                     baseline: torch.Tensor,
                     normalizer: Optional[torch.Tensor] = None):
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
            cropped_baseline *= cropped_normalize

        return cropped_baseline

    def __getitem__(
        self,
        item: Union[int, list, np.ndarray, Tuple[Union[int, np.ndarray, slice, range]]],
    ) -> np.ndarray:
        product = self.getitem_tensor(item)
        product = product.cpu().numpy().astype(self.dtype)
        return product





