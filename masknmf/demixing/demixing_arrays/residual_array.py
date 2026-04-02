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
        pmd_array: PMDArray,
        ac_array: ACArray,
        fluctuating_background_array: FluctuatingBackgroundArray,
        baseline: torch.Tensor,
    ):
        """
        Args:
            pmd_array (PMDArray)
            ac_array (ACArray)
            fluctuating_array (FluctuatingBackgroundArray)
            baseline (torch.Tensor): Shape (fov dim 1, fov dim 2)
        """

        DATA_ARRAYS = ["pmd_array",
                       "ac_array",
                       "fluctuating_background_array",
                       "baseline"]

        self._pmd_array = pmd_array
        # Demixing is run on the U/V representation, without rescaling, so we set rescale = False here to make sure scales match
        self._pmd_array.rescale = False
        self._ac_array = ac_array
        self._baseline = baseline
        self._fluctuating_background_array = fluctuating_background_array

        #Check that all data is on same device
        self._find_common_device()

        self._shape = self.pmd_array.shape

    def _find_common_device(self):
        """
        Finds the common device that for all data tensors. Throws error if no such device exists
        """
        device=None
        for i, name in enumerate(DATA_ARRAYS):
            arr = getattr(self, name)
            if i == 0:
                device = arr.device
            else:
                if not arr.device == device:
                    raise ValueError("Not all tensors in fluctuating background array are on same device")
        return device

    @property
    def device(self) -> str:
        return self._find_common_device()

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
    def baseline(self) -> torch.Tensor:
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
                - self.baseline[item[1:]][None, ...]
            )
        else:
            output = (
                self.pmd_array.getitem_tensor(item)
                - self.fluctuating_array.getitem_tensor(item)
                - self.ac_array.getitem_tensor(item)
                - self.baseline[None, :]
            )

        return output.cpu().numpy()
