from typing import *
import numpy as np
from masknmf.arrays.array_interfaces import FactorizedVideo
from masknmf.compression.pmd_array import PMDArray
from masknmf.demixing.demixing_arrays.ac_array import ACArray
from masknmf.demixing.demixing_arrays.fluctuating_background_array import FluctuatingBackgroundArray
import torch

def test_slice_effect(my_slice: slice, spatial_dim: int) -> bool:
    """
    Returns True if slice will actually have an effect
    """

    if not (
        (isinstance(my_slice.start, int) and my_slice.start == 0)
        or my_slice.start is None
    ):
        return True
    elif not (
        (isinstance(my_slice.stop, int) and my_slice.stop >= spatial_dim)
        or my_slice.stop is None
    ):
        return True
    elif not (
        my_slice.step is None or (isinstance(my_slice.step, int) and my_slice.step == 1)
    ):
        return True
    return False


def test_range_effect(my_range: range, spatial_dim: int) -> bool:
    """
    Returns True if the range will actually have an effect.

    Parameters:
    my_range (range): The range object to test.
    spatial_dim (int): The size of the dimension that the range is applied to.

    Returns:
    bool: True if the range will affect the selection; False otherwise.
    """
    # Check if the range starts from the beginning
    if my_range.start != 0:
        return True
    # Check if the range stops at the end of the dimension
    elif my_range.stop != spatial_dim:
        return True
    # Check if the range step is not 1
    elif my_range.step != 1:
        return True
    return False


def test_spatial_crop_effect(my_tuple, spatial_dims) -> bool:
    """
    Returns true if the tuple used for spatial cropping actually has an effect on the underlying data. Otherwise
    cropping can be an expensive and avoidable operation.
    """
    for k in range(len(my_tuple)):
        if isinstance(my_tuple[k], np.ndarray):
            if my_tuple[k].shape[0] < spatial_dims[k]:
                return True

        if isinstance(my_tuple[k], np.integer):
            return True

        if isinstance(my_tuple[k], int):
            return True

        if isinstance(my_tuple[k], slice):
            if test_slice_effect(my_tuple[k], spatial_dims[k]):
                return True
        if isinstance(my_tuple[k], range):
            if test_range_effect(my_tuple[k], spatial_dims[k]):
                return True
    return False

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
