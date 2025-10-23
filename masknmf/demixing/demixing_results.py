from typing import *
import numpy as np
from masknmf import display
from masknmf.compression import PMDArray
from masknmf.demixing.demixing_arrays import ACArray, ResidualCorrelationImages, StandardCorrelationImages, ColorfulACArray, FluctuatingBackgroundArray, ResidualArray, ResidCorrMode
import torch
from masknmf.utils import Serializer


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

class DemixingResults(Serializer):
    _serialized = {
        "shape",
        "u",
        "v",
        "a",
        "c",
        "b",
        "factorized_bkgd_term1",
        "factorized_bkgd_term2",
        "global_residual_correlation_image",
        "std_corr_img_mean",
        "std_corr_img_normalizer",
        "resid_corr_img_support_values",
        "resid_corr_img_mean",
        "resid_corr_img_normalizer",
        "bkgd_corr_img_mean",
        "bkgd_corr_img_normalizer"
    }

    def __init__(
            self,
            shape: Tuple[int, int, int] | np.ndarray,
            u: torch.sparse_coo_tensor,
            v: torch.tensor,
            a: torch.sparse_coo_tensor,
            c: torch.tensor,
            factorized_bkgd_term1: Optional[torch.Tensor] = None,
            factorized_bkgd_term2: Optional[torch.Tensor] = None,
            b: Optional[torch.tensor] = None,
            std_corr_img_mean: Optional[torch.Tensor] = None,
            std_corr_img_normalizer: Optional[torch.Tensor] = None,
            resid_corr_img_support_values: Optional[torch.sparse_coo_tensor] = None,
            resid_corr_img_mean: Optional[torch.tensor] = None,
            resid_corr_img_normalizer: Optional[torch.tensor] = None,
            bkgd_corr_img_mean: Optional[torch.Tensor] = None,
            bkgd_corr_img_normalizer: Optional[torch.Tensor] = None,
            global_residual_correlation_image: Optional[torch.Tensor] = None,
            order: str = "C",
            device="cpu",
    ):
        """
        This class provides a convenient way to export all demixing result as array-like objects.
        Args:
            shape (tuple): (number of frames, field of view dimension 1, field of view dimension 2)
            u (torch.sparse_coo_tensor): shape (pixels, rank 1)
            v (torch.tensor): shape (rank 2, num_frames)
            a (torch.sparse_coo_tensor): shape (pixels, number of neural signals)
            c (torch.tensor): shape (number of frames, number of neural signals)
            factorized_bkgd_term1: Optional[torch.Tensor]: tensor used to express low-rank background estimate
            factorized_bkgd_term2: Optioal[torch.Tensor]: tensor used to express low-rank background estimate
            b (torch.tensor): Optional[torch.tensor]. The per-pixel static baseline.
                If not provided, the below code will set it so that the residual movie has mean 0.
                The residual is defined as UV - AC - Fluctuaating background - Static Background
            std_corr_img_mean (Optional[torch.Tensor]): the mean image used to lazily construct the standard correlation image per neuron
            std_corr_img_normalizer (Optional[torch.Tensor]): the normalizer image used to lazily construct the standard correlation image per neuron
            resid_corr_img_support_values (Optional[torch.sparse_coo_tensor]): Shape (num_pixels, num_neurons). A sparse tensor describing the residual correlation
                image only at values where the neuron footprint is nonzero
            resid_corr_img_mean (Optional[torch.Tensor]): Shape (height, width). The mean image used to lazily
                compute the residual correlation image per neural signal.
            resid_corr_img_normalizer (Optional[torch.Tensor]): Shape (height, width). The normalizer image used to lazily
                compute the residual correlation image per neural signal.
            bkgd_corr_img_mean (Optional[torch.Tensor]): The mean image used to compute the correlation between the signal and the background.
            bkgd_corr_img_normalizer (Optional[torch.Tensor]): The mean image used to compute the correlation between the signal and the background.
            global_resid_correlation_image (torch.Tensor): The global correlation image of the residual. Shape (FOV dim 1, FOV dim 2).
            order (str): order used to reshape data from 2D to 1D
            device (str): 'cpu' or 'cuda'. used to manage where the tensors reside
        """
        self._device = device
        self._order = order
        self._shape = tuple(shape)
        self._u_sparse = u.to(self.device).float()
        self._v = v.to(self.device).float()
        self._a = a.to(self.device).float()
        self._c = c.to(self.device).float()

        if factorized_bkgd_term1 is None or factorized_bkgd_term2 is None:
            display("Background term empty")
            self._factorized_bkgd_term1 = torch.zeros(self.u.shape[1], 1, dtype=self.u.dtype, device=self.device)
            self._factorized_bkgd_term2 = torch.zeros((1, self.v.shape[1]), dtype=self.u.dtype, device=self.device)
        else:
            self._factorized_bkgd_term1 = factorized_bkgd_term1.to(self.device)
            self._factorized_bkgd_term2 = factorized_bkgd_term2.to(self.device)

        if global_residual_correlation_image is None:
            self._global_residual_corr_img = torch.zeros(self.shape[1], self.shape[2], device=self.device, dtype=self._u_sparse.dtype)
        else:
            self._global_residual_corr_img = global_residual_correlation_image

        if b is None:
            display("Static term was not provided, constructing baseline to ensure residual is mean 0")
            self._b = (torch.sparse.mm(self.u, torch.mean(self.v, dim=1, keepdim=True)) -
                       torch.sparse.mm(self._a, torch.mean(self._c.T, dim=1, keepdim=True)) -
                       torch.sparse.mm(self.u, (
                                   self.factorized_bkgd_term1 @ torch.mean(self.factorized_bkgd_term2, axis=1,
                                                                           keepdim=True))))
        else:
            self._b = b

        if std_corr_img_mean is None or std_corr_img_normalizer is None:
            self._std_corr_img_mean = None
            self._std_corr_img_normalizer = None
        else:
            self._std_corr_img_mean = std_corr_img_mean  # standard_correlation_image.movie_mean
            self._std_corr_img_normalizer = std_corr_img_normalizer  # standard_correlation_image.movie_normalizer

        if resid_corr_img_mean is None or resid_corr_img_support_values is None or resid_corr_img_normalizer is None:
            self._resid_corr_img_support_values = None
            self._resid_corr_img_mean = None
            self._resid_corr_img_normalizer = None
        else:
            self._resid_corr_img_support_values = resid_corr_img_support_values
            self._resid_corr_img_mean = resid_corr_img_mean
            self._resid_corr_img_normalizer = resid_corr_img_normalizer

        if bkgd_corr_img_mean is None or bkgd_corr_img_normalizer is None:
            self._bkgd_corr_img_mean = None
            self._bkgd_corr_img_normalizer = None
        else:
            self._bkgd_corr_img_mean = bkgd_corr_img_mean
            self._bkgd_corr_img_normalizer = bkgd_corr_img_normalizer

        # Move all tracked tensors to desired location so everything is on one device
        self.to(self.device)


    @property
    def factorized_bkgd_term1(self) -> Union[None, torch.Tensor]:
        return self._factorized_bkgd_term1

    @property
    def factorized_bkgd_term2(self) -> Union[None, torch.Tensor]:
        return self._factorized_bkgd_term2

    @property
    def shape(self):
        return self._shape

    @property
    def order(self):
        return self._order

    @property
    def device(self):
        return self._device

    def to(self, new_device):
        self._device = new_device
        self._u_sparse = self._u_sparse.to(self.device)
        self._factorized_bkgd_term1 = self._factorized_bkgd_term1.to(self.device)
        self._factorized_bkgd_term2 = self._factorized_bkgd_term2.to(self.device)
        self._v = self._v.to(self.device)
        self._a = self._a.to(self.device)
        self._c = self._c.to(self.device)
        self._b = self._b.to(self.device)

        if self._std_corr_img_mean is not None: #This means all the std corr img data is not None from init logic
            self._std_corr_img_mean = self._std_corr_img_mean.to(self.device)
            self._std_corr_img_normalizer = self._std_corr_img_normalizer.to(self.device)

        if self._bkgd_corr_img_mean is not None: #This means all the bkgd corr img data is not None from init logic
            self._bkgd_corr_img_mean = self._bkgd_corr_img_mean.to(self.device)
            self._bkgd_corr_img_normalizer = self._bkgd_corr_img_normalizer.to(self.device)

        if self._resid_corr_img_mean is not None: #This means all the resid corr img data is not None from init logic
            self._resid_corr_img_support_values = self._resid_corr_img_support_values.to(self.device)
            self._resid_corr_img_mean = self._resid_corr_img_mean.to(self.device)
            self._resid_corr_img_normalizer = self._resid_corr_img_normalizer.to(self.device)

        if self._global_residual_corr_img is not None:
            self._global_residual_corr_img = self._global_residual_corr_img.to(self.device)

    @property
    def fov_shape(self) -> Tuple[int, int]:
        return self.shape[1:3]

    @property
    def num_frames(self) -> int:
        return self.shape[0]

    @property
    def u(self) -> torch.sparse_coo_tensor:
        return self._u_sparse

    @property
    def b(self) -> torch.tensor:
        return self._b

    @property
    def v(self) -> torch.tensor:
        return self._v

    @property
    def a(self) -> torch.sparse_coo_tensor:
        return self._a

    @property
    def c(self) -> torch.tensor:
        return self._c

    @property
    def std_corr_img_mean(self) -> Union[None, torch.Tensor]:
        return self._std_corr_img_mean

    @property
    def std_corr_img_normalizer(self) -> Union[None, torch.Tensor]:
        return self._std_corr_img_normalizer

    @property
    def resid_corr_img_support_values(self) -> Union[None, torch.sparse_coo_tensor]:
        return self._resid_corr_img_support_values

    @property
    def resid_corr_img_mean(self) -> Union[None, torch.Tensor]:
        return self._resid_corr_img_mean

    @property
    def resid_corr_img_normalizer(self) -> Union[None, torch.Tensor]:
        return self._resid_corr_img_normalizer


    @property
    def global_residual_correlation_image(self) -> Union[None, torch.Tensor]:
        return self._global_residual_corr_img

    @property
    def bkgd_corr_img_mean(self) -> Union[None, torch.Tensor]:
        return self._bkgd_corr_img_mean

    @property
    def bkgd_corr_img_normalizer(self) -> Union[None, torch.Tensor]:
        return self._bkgd_corr_img_normalizer


    @property
    def standard_correlation_image(self) -> Union[None, StandardCorrelationImages]:
        if self.std_corr_img_mean is not None:
            return StandardCorrelationImages(self._u_sparse,
                                             self._v,
                                             self._c,
                                             self.std_corr_img_mean,
                                             self.std_corr_img_normalizer,
                                             (self._shape[1], self._shape[2]),
                                             order=self.order)
        else:
            return None

    @property
    def background_to_signal_correlation_image(self) -> Union[None, StandardCorrelationImages]:
        if self.bkgd_corr_img_mean is not None:
            return StandardCorrelationImages(self._u_sparse,
                                             self.factorized_bkgd_term1 @ self.factorized_bkgd_term2,
                                             self._c,
                                             self.bkgd_corr_img_mean,
                                             self.bkgd_corr_img_normalizer,
                                             (self._shape[1], self._shape[2]),
                                             order=self.order)
        else:
            return None

    @property
    def residual_correlation_image(self) -> Union[None, ResidualCorrelationImages]:
        if self.resid_corr_img_mean is not None:
            return ResidualCorrelationImages(self.u,
                                             self.v,
                                             (self.factorized_bkgd_term1, self.factorized_bkgd_term2),
                                             self.a,
                                             self.c,
                                             self.resid_corr_img_support_values,
                                             self.resid_corr_img_mean,
                                             self.resid_corr_img_normalizer,
                                             (self.shape[1], self.shape[2]),
                                             mode=ResidCorrMode.RESIDUAL,
                                             order=self._order)
        else:
            return None

    @property
    def ac_array(self) -> ACArray:
        """
        Returns an ACArray using the tensors stored in this object
        """
        return ACArray(self.fov_shape, self.order, self.a, self.c)

    @property
    def pmd_array(self) -> PMDArray:
        """
        Returns a PMDArray using the tensors stored in this object
        """
        mean_img = torch.zeros(self.shape[1], self.shape[2], device=self.device)
        var_img = torch.ones(self.shape[1], self.shape[2], device=self.device)
        return PMDArray(
            self.shape,
            self.u,
            self.v,
            mean_img,
            var_img,
            device=self.device,
            rescale=True,
        )

    @property
    def fluctuating_background_array(self) -> FluctuatingBackgroundArray:
        """
        Returns a PMDArray using the tensors stored in this object
        """
        return FluctuatingBackgroundArray(self.fov_shape,
                                          self.order,
                                          self.u,
                                          self.factorized_bkgd_term1,
                                          self.factorized_bkgd_term2)

    @property
    def residual_array(self) -> ResidualArray:
        return ResidualArray(
            self.pmd_array,
            self.ac_array,
            self.fluctuating_background_array,
            self.b.reshape(self.fov_shape),
        )

    @property
    def colorful_ac_array(self) -> ColorfulACArray:
        return ColorfulACArray(self.fov_shape, self.order, self.a, self.c)


