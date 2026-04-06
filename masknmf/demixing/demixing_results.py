from typing import *
import numpy as np
from masknmf import display
from masknmf.compression import PMDArray
from masknmf.demixing.demixing_arrays import ACArray, ResidualCorrelationImages, StandardCorrelationImages, ColorfulACArray, StaticBackgroundArray, FluctuatingBackgroundArray, ResidualArray, ResidCorrMode
import torch
from masknmf.utils import Serializer
from masknmf.arrays.array_interfaces import TensorFlyWeight


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

class DemixingResults(Serializer, TensorFlyWeight):
    _serialized = {
        "shape",
        "u",
        "v",
        "a",
        "c",
        "b",
        "pmd_mean_img",
        "pmd_var_img",
        "pmd_u_projector",
        "factorized_bkgd_term1",
        "factorized_bkgd_term2",
        "global_residual_correlation_image",
        "std_corr_img_mean",
        "std_corr_img_normalizer",
        "resid_corr_img_support_values",
        "resid_corr_img_mean",
        "resid_corr_img_normalizer",
        "bkgd_corr_img_mean",
        "bkgd_corr_img_normalizer",
        "pmd_roi_averages",
        "fluctuating_background_roi_averages",
        "residual_roi_averages"
    }

    def __init__(
            self,
            shape: Tuple[int, int, int] | np.ndarray,
            u: torch.sparse_coo_tensor,
            v: torch.tensor,
            a: torch.sparse_coo_tensor,
            c: torch.tensor,
            pmd_mean_img: Optional[torch.Tensor] = None,
            pmd_var_img: Optional[torch.Tensor] = None,
            pmd_u_projector: Optional[torch.sparse_coo_tensor] = None,
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
            pmd_roi_averages: Optional[torch.Tensor] = None,
            fluctuating_background_roi_averages: Optional[torch.Tensor] = None,
            residual_roi_averages: Optional[torch.Tensor] = None,
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
            pmd_mean_img (Optional[torch.tensor]): The mean image of the imaging data, used for reconstructing PMD Arrays
            pmd_var_img (Optional[torch.tensor]): The pixelwise noise variance image of the data, used for reconstructing PMD Arrays
            pmd_u_projector (Optional[torch.sparse_coo_tensor]): A projection matrix used to project frames of data onto the PMD U subspace
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
            device (str): 'cpu' or 'cuda'. used to manage where the tensors reside
        """
        self._device = device
        self._shape = tuple(shape)
        self._u_sparse = u.to(self.device).float().coalesce()
        self._v = v.to(self.device).float()
        self._a = a.to(self.device).float().coalesce()
        self._c = c.to(self.device).float()

        if pmd_mean_img is not None:
            self._pmd_mean_img = pmd_mean_img
        else:
            self._pmd_mean_img = torch.zeros(self.shape[1], self.shape[2], device=self.device)
        if pmd_var_img is not None:
            self._pmd_var_img = pmd_var_img
        else:
            self._pmd_var_img= torch.ones(self.shape[1], self.shape[2], device=self.device)

        self._pmd_u_projector = pmd_u_projector.coalesce() if pmd_u_projector is not None else None

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
        self._baseline = self._b.reshape(self.fov_shape)

        if pmd_roi_averages is not None:
            self._pmd_roi_averages = pmd_roi_averages
        if fluctuating_background_roi_averages is not None:
            self._fluctuating_background_roi_averages = fluctuating_background_roi_averages
        if residual_roi_averages is not None:
            self._residual_roi_averages = residual_roi_averages

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
            self._resid_corr_img_support_values = resid_corr_img_support_values.coalesce()
            self._resid_corr_img_mean = resid_corr_img_mean
            self._resid_corr_img_normalizer = resid_corr_img_normalizer

        if bkgd_corr_img_mean is None or bkgd_corr_img_normalizer is None:
            self._bkgd_corr_img_mean = None
            self._bkgd_corr_img_normalizer = None
        else:
            self._bkgd_corr_img_mean = bkgd_corr_img_mean
            self._bkgd_corr_img_normalizer = bkgd_corr_img_normalizer

        self._pmd_roi_averages = None
        self._fluctuating_background_roi_averages = None
        self._residual_roi_averages = None

        self._ac_array = None
        self._colorful_ac_array = None
        self._pmd_array = None
        self._fluctuating_background_array = None
        self._static_background_array = None
        self._residual_array = None
        self._residual_correlation_images = None
        self._standard_correlation_images = None

        self._rescale = False

        # Move all tracked tensors to desired location so everything is on one device
        self.to(self.device)

    @property
    def rescale(self):
        return self._rescale

    @rescale.setter
    def rescale(self, new_value: bool):
        self._rescale = new_value

        self.pmd_array.rescale = new_value
        self.ac_array.rescale = new_value
        self.static_background_array.rescale = new_value
        self.fluctuating_background_array.rescale = new_value

    @property
    def pmd_mean_img(self) -> Union[None, torch.Tensor]:
        return self._pmd_mean_img

    @property
    def mean_img(self) -> torch.Tensor:
        return self._pmd_mean_img

    @property
    def pmd_var_img(self) -> Union[None, torch.Tensor]:
        return self._pmd_var_img

    @property
    def var_img(self) -> torch.Tensor:
        return self._pmd_var_img

    @property
    def normalizer(self) -> torch.Tensor:
        return self._pmd_var_img

    @property
    def pmd_u_projector(self) -> Union[None, torch.Tensor]:
        return self._pmd_u_projector

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
        self._move_managed_tensors(new_device)

        self._move_managed_arrays(new_device)


    def _move_managed_tensors(self, new_device: str):
        self._device = new_device
        self._u_sparse = self._u_sparse.to(self.device)
        self._factorized_bkgd_term1 = self._factorized_bkgd_term1.to(self.device)
        self._factorized_bkgd_term2 = self._factorized_bkgd_term2.to(self.device)
        self._v = self._v.to(self.device)
        self._a = self._a.to(self.device)
        self._c = self._c.to(self.device)
        self._b = self._b.to(self.device)
        self._baseline = self._baseline.to(self.device)

        if self._pmd_mean_img is not None:
            self._pmd_mean_img = self._pmd_mean_img.to(self.device)
        if self._pmd_var_img is not None:
            self._pmd_var_img = self._pmd_var_img.to(self.device)
        if self._pmd_u_projector is not None:
            self._pmd_u_projector.to(self.device)

        if self._std_corr_img_mean is not None:  # This means all the std corr img data is not None from init logic
            self._std_corr_img_mean = self._std_corr_img_mean.to(self.device)
            self._std_corr_img_normalizer = self._std_corr_img_normalizer.to(self.device)

        if self._bkgd_corr_img_mean is not None:  # This means all the bkgd corr img data is not None from init logic
            self._bkgd_corr_img_mean = self._bkgd_corr_img_mean.to(self.device)
            self._bkgd_corr_img_normalizer = self._bkgd_corr_img_normalizer.to(self.device)

        if self._resid_corr_img_mean is not None:  # This means all the resid corr img data is not None from init logic
            self._resid_corr_img_support_values = self._resid_corr_img_support_values.to(self.device)
            self._resid_corr_img_mean = self._resid_corr_img_mean.to(self.device)
            self._resid_corr_img_normalizer = self._resid_corr_img_normalizer.to(self.device)

        if self._global_residual_corr_img is not None:
            self._global_residual_corr_img = self._global_residual_corr_img.to(self.device)

    def _move_managed_arrays(self, new_device: str):

        if self._ac_array is not None:
            self.ac_array.to(self.device)

        if self._colorful_ac_array is not None:
            self.colorful_ac_array.to(self.device)

        if self._pmd_array is not None:
            self.pmd_array.to(self.device)

        if self._fluctuating_background_array is not None:
            self.fluctuating_background_array.to(self.device)

        if self._static_background_array is not None:
            self.static_background_array.to(self.device)

        if self._residual_correlation_images is not None:
            self.residual_correlation_images.to(self.device)

        if self._standard_correlation_images is not None:
            self.standard_correlation_images.to(self.device)

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
    def b(self) -> torch.Tensor:
        return self._b

    @property
    def baseline(self) -> torch.Tensor:
        return self._baseline

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

    def _roi_averages(self) -> tuple[torch.tensor, torch.tensor, torch.tensor]:
        """
        Returns the ROI averages for each spatial footprint of the AC Array in the PMD movie, fluctuating background movie,
        and residual movie.
        """
        if self._residual_roi_averages is None or self._pmd_roi_averages is None or self._fluctuating_background_roi_averages is None:
            residual_roi_averages = torch.zeros_like(self.c)
            pmd_roi_averages = torch.zeros_like(self.c)
            fluctuating_background_roi_averages = torch.zeros_like(self.c)

            ind_select_tensor = torch.arange(self.a.shape[1], device=self.device).long()
            avg_tensor = torch.zeros(self.a.shape[0], device=self.device)
            u_t = self.u.t()
            a_t = self.a.t()
            for k in range(self.a.shape[1]):
                row, col = torch.index_select(self.a, 1, ind_select_tensor[k:k+1]).coalesce().indices()
                avg_tensor[row] = 1.0
                divisor = torch.sum(avg_tensor)
                avg_tensor[row] /= divisor
                u_avg = torch.sparse.mm(u_t, avg_tensor[:, None]).T
                avg_pmd = u_avg @ self.v
                avg_bkgd = (u_avg @ self.factorized_bkgd_term1) @ self.factorized_bkgd_term2
                avg_static_bkgd = avg_tensor[None, :] @ self.b
                a_avg = torch.sparse.mm(a_t, avg_tensor[:, None])
                ac_avg = (self.c @ a_avg).T
                resid = avg_pmd - avg_bkgd - avg_static_bkgd - ac_avg

                pmd_roi_averages[:, k] = avg_pmd.squeeze()
                fluctuating_background_roi_averages[:, k] = avg_bkgd.squeeze()
                residual_roi_averages[:, k] = resid.squeeze()
                avg_tensor *= 0 #Reset this
            self._pmd_roi_averages = pmd_roi_averages
            self._fluctuating_background_roi_averages = fluctuating_background_roi_averages
            self._residual_roi_averages = residual_roi_averages

        return (self._pmd_roi_averages, self._fluctuating_background_roi_averages, self._residual_roi_averages)

    @property
    def pmd_roi_averages(self) -> torch.tensor:
        return self._roi_averages()[0]

    @property
    def fluctuating_background_roi_averages(self) -> torch.tensor:
        return self._roi_averages()[1]

    @property
    def residual_roi_averages(self) -> torch.tensor:
        return self._roi_averages()[2]

    @property
    def standard_correlation_images(self) -> Union[None, StandardCorrelationImages]:
        if self.std_corr_img_mean is not None:
            if self._standard_correlation_images is None:
                self._standard_correlation_images = StandardCorrelationImages.from_flyweight(self,
                                                                                             (self._shape[1], self._shape[2]))
            return self._standard_correlation_images
        else:
            return None

    @property
    def background_to_signal_correlation_image(self) -> Union[None, StandardCorrelationImages]:
        """
        This array will not use the FlyWeight pattern that the other arrays use, since this is primarily an exploratory
        property. If this becomes crucial, can re-organize
        """
        if self.bkgd_corr_img_mean is not None:
            return StandardCorrelationImages.from_tensors(self._u_sparse,
                                             self.factorized_bkgd_term1 @ self.factorized_bkgd_term2,
                                             self._c,
                                             self.bkgd_corr_img_mean,
                                             self.bkgd_corr_img_normalizer,
                                             (self._shape[1], self._shape[2]))
        else:
            return None

    @property
    def residual_correlation_images(self) -> Union[None, ResidualCorrelationImages]:
        if self.resid_corr_img_mean is not None:
            if self._residual_correlation_images is None:
                self._residual_correlation_images = ResidualCorrelationImages.from_flyweight(self,
                                                                                             (self.shape[1], self.shape[2]),
                                                                                             mode=ResidCorrMode.RESIDUAL)
            return self._residual_correlation_images
        else:
            return None

    @property
    def ac_array(self) -> ACArray:
        """
        Returns an ACArray using the tensors stored in this object
        """
        if self._ac_array is None:
            self._ac_array = ACArray.from_flyweight(self.fov_shape, self, rescale=self.rescale)
        return self._ac_array

    @property
    def pmd_array(self) -> PMDArray:
        """
        Returns a PMDArray using the tensors stored in this object
        """
        if self._pmd_array is None:
            self._pmd_array = PMDArray.from_flyweight(
                self.shape,
                self,
                device=self.device,
                rescale=self.rescale,
            )
        return self._pmd_array

    @property
    def fluctuating_background_array(self) -> FluctuatingBackgroundArray:
        """
        Returns a PMDArray using the tensors stored in this object
        """
        if self._fluctuating_background_array is None:
            self._fluctuating_background_array = FluctuatingBackgroundArray.from_flyweight(self.fov_shape,
                                                                            self,
                                                                            rescale=self.rescale)
        return self._fluctuating_background_array

    @property
    def static_background_array(self) -> StaticBackgroundArray:

        if self._static_background_array is None:
            self._static_background_array = StaticBackgroundArray.from_flyweight(self,
                                                                             rescale = self.rescale)
        return self._static_background_array

    @property
    def residual_array(self) -> ResidualArray:
        if self._residual_array is None:
            self._residual_array = ResidualArray(self.pmd_array,
                                                self.ac_array,
                                                self.fluctuating_background_array,
                                                self.static_background_array,
                                            )
        return self._residual_array

    @property
    def colorful_ac_array(self) -> ColorfulACArray:
        if self._colorful_ac_array is None:
            self._colorful_ac_array = ColorfulACArray.from_flyweight(self.fov_shape, self)
        return self._colorful_ac_array

