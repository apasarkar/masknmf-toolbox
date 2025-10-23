from typing import *
import numpy as np
from masknmf.arrays.array_interfaces import FactorizedVideo
from masknmf.compression.pmd_array import PMDArray
from masknmf.demixing.demixing_arrays.ac_array import ACArray
from masknmf.demixing.demixing_arrays.fluctuating_background_array import FluctuatingBackgroundArray
import torch

class ResidualArray(FactorizedVideo):
    """
    Factorized video for the spatial and temporal extracted sources from the data
    """

    def __init__(
        self,
        pmd_arr: PMDArray,
        ac_arr: ACArray,
        fluctuating_arr: FluctuatingBackgroundArray,
        baseline: torch.tensor,
    ):
        """
        Args:
            pmd_arr (PMDArray)
            ac_arr (ACArray)
            fluctuating_arr (FluctuatingBackgroundArray)
            baseline (torch.tensor): Shape (fov dim 1, fov dim 2)
        """
        self.pmd_arr = pmd_arr
        self.ac_arr = ac_arr
        self.baseline = baseline
        self.fluctuating_arr = fluctuating_arr

        if not (
            self.pmd_arr.device
            == self.ac_arr.device
            == self.baseline.device
            == self.fluctuating_arr.device
        ):
            raise ValueError(f"Input arrays not all on same device")
        self._device = self.pmd_arr.device
        self._shape = self.pmd_arr.shape

    @property
    def dtype(self) -> str:
        """
        data type, default np.float32
        """
        return self.pmd_arr.dtype

    @property
    def device(self) -> str:
        """
        Returns the device that all the internal tensors are on at init time
        """
        return self._device

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
                self.pmd_arr.getitem_tensor(item)
                - self.fluctuating_arr.getitem_tensor(item)
                - self.ac_arr.getitem_tensor(item)
                - self.baseline[item[1:]][None, ...]
            )
        else:
            output = (
                self.pmd_arr.getitem_tensor(item)
                - self.fluctuating_arr.getitem_tensor(item)
                - self.ac_arr.getitem_tensor(item)
                - self.baseline[None, :]
            )

        return output.cpu().numpy().squeeze()
