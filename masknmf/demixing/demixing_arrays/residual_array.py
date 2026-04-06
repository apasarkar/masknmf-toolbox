from typing import *
import numpy as np
from masknmf.arrays.array_interfaces import ArrayLike
from masknmf.compression.pmd_array import PMDArray
from masknmf.demixing.demixing_arrays.ac_array import ACArray
from masknmf.demixing.demixing_arrays.fluctuating_background_array import FluctuatingBackgroundArray
from masknmf.demixing.demixing_arrays.static_baseline import StaticBackgroundArray
import torch

class ResidualArray(ArrayLike):
    """
    Factorized video for the spatial and temporal extracted sources from the data
    """

    def __init__(
        self,
        pmd_array: PMDArray,
        ac_array: ACArray,
        fluctuating_background_array: FluctuatingBackgroundArray,
        static_baseline_array: StaticBackgroundArray,
    ):
        """
        Args:
            pmd_array (PMDArray)
            ac_array (ACArray)
            fluctuating_array (FluctuatingBackgroundArray)
            baseline (StaticBackgroundArray): Shape (height, width)
        """

        self._pmd_array = pmd_array
        self._ac_array = ac_array
        self._baseline = static_baseline_array
        self._fluctuating_background_array = fluctuating_background_array

        self._shape = self.pmd_array.shape

    @property
    def device(self) -> str:
        if self.pmd_array.device == self.ac_array.device == self.fluctuating_background_array.device == self.baseline.device:
            return self.pmd_array.device
        else:
            raise ValueError("Not all arrays are on same device")
        
    @property
    def dtype(self) -> str:
        """
        data type, default np.float32
        """
        return self.pmd_array.dtype

    @property
    def pmd_array(self) -> PMDArray:
        return self._pmd_array

    @property
    def ac_array(self) -> ACArray:
        return self._ac_array

    @property
    def fluctuating_background_array(self) -> FluctuatingBackgroundArray:
        return self._fluctuating_background_array

    @property
    def baseline(self) -> StaticBackgroundArray:
        return self._baseline


    @property
    def shape(self) -> Tuple[int, int, int]:
        """
        Array shape (n_frames, dims_x, dims_y)
        """
        return self._shape

    @property
    def ndim(self) -> int:
        """
        Number of dimensions
        """
        return len(self.shape)

    def __getitem__(
        self,
        item: Union[int, list, np.ndarray, Tuple[Union[int, np.ndarray, slice, range]]],
    ):
        # In this case there is spatial cropping
        if isinstance(item, tuple) and len(item) > 1:
            output = (
                self.pmd_array.getitem_tensor(item)
                - self.fluctuating_array.getitem_tensor(item)
                - self.ac_array.getitem_tensor(item)
                - self.baseline.getitem_tensor(item[1:])[None, ...]
            )
        else:
            output = (
                self.pmd_array.getitem_tensor(item)
                - self.fluctuating_array.getitem_tensor(item)
                - self.ac_array.getitem_tensor(item)
                - self.baseline.getitem_tensor((slice(0, self.shape[1]), slice(0, self.shape[2])))[None, :]
            )

        return output.cpu().numpy()
