from typing import *
import numpy as np
from masknmf import display
from masknmf.compression import PMDArray
from masknmf.demixing.demixing_arrays import ACArray, ResidualCorrelationImages, StandardCorrelationImages, ColorfulACArray, StaticBackgroundArray, FluctuatingBackgroundArray, ResidualArray, ResidCorrMode
import torch
from masknmf.utils import Serializer
from masknmf.arrays.array_interfaces import TensorFlyWeight
from masknmf.utils import display


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

    """
    This lists arrays which are explicitly managed by demixing results.
    When you do DemixingResults.to(new_device), this object is responsible for making sure all of these arrays are moved to that device
    """
    _managed_arrays = ["pmd_array",
                       "ac_array",
                       "colorful_ac_array",
                       "fluctuating_background_array",
                       "static_background_array",
                       "standard_correlation_images",
                       "residual_correlation_images",
                       ]
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
            pmd_mean_img (Optional[torch.Tensor]): The mean image of the imaging data, used for reconstructing PMD Arrays
            pmd_var_img (Optional[torch.Tensor]): The pixelwise noise variance image of the data, used for reconstructing PMD Arrays
            pmd_u_projector (Optional[torch.sparse_coo_tensor]): A projection matrix used to project frames of data onto the PMD U subspace
            factorized_bkgd_term1: Optional[torch.Tensor]: tensor used to express low-rank background estimate
            factorized_bkgd_term2: Optional[torch.Tensor]: tensor used to express low-rank background estimate
            b (torch.tensor): Optional[torch.Tensor]. The per-pixel static baseline.
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
        self._flyweight = TensorFlyWeight()
        self.flyweight.u = u.to(self.device).float().coalesce()
        self.flyweight.v = v.to(self.device).float()
        self.flyweight.a = a.to(self.device).float().coalesce()
        self.flyweight.c = c.to(self.device).float()

        self.flyweight.pmd_mean_img = pmd_mean_img if pmd_mean_img is not None else torch.zeros(self.shape[1], self.shape[2], device=self.device)
        self.flyweight.pmd_var_img = pmd_var_img if pmd_var_img is not None else torch.ones(self.shape[1], self.shape[2], device=self.device)
        ## Below two lines are for compatibility with existing results
        self.flyweight.mean_img = self.flyweight.pmd_mean_img
        self.flyweight.var_img = self.flyweight.pmd_var_img

        #This is called
        self.flyweight.normalizer = self.flyweight.pmd_var_img

        self.flyweight.pmd_u_projector = pmd_u_projector.float().coalesce() if pmd_u_projector is not None else None

        if factorized_bkgd_term1 is None or factorized_bkgd_term2 is None:
            display("Background term empty")
            self.flyweight.factorized_bkgd_term1 = torch.zeros(self.u.shape[1], 1, dtype=self.u.dtype, device=self.device)
            self.flyweight.factorized_bkgd_term2 = torch.zeros((1, self.v.shape[1]), dtype=self.u.dtype, device=self.device)
        else:
            self.flyweight.factorized_bkgd_term1 = factorized_bkgd_term1.to(self.device)
            self.flyweight.factorized_bkgd_term2 = factorized_bkgd_term2.to(self.device)

        self.flyweight.global_residual_correlation_image = global_residual_correlation_image if global_residual_correlation_image is not None else torch.zeros(self.shape[1], self.shape[2], device=self.device, dtype=self.u.dtype)


        if b is None:
            display("Static term was not provided, constructing baseline to ensure residual is mean 0")
            self.flyweight.b = (torch.sparse.mm(self.u, torch.mean(self.v, dim=1, keepdim=True)) -
                       torch.sparse.mm(self.a, torch.mean(self.c.T, dim=1, keepdim=True)) -
                       torch.sparse.mm(self.u, (
                                   self.factorized_bkgd_term1 @ torch.mean(self.factorized_bkgd_term2, axis=1,
                                                                           keepdim=True))))
        else:
            self.flyweight.b = b
        self.flyweight.baseline = self.b.reshape(self.fov_shape)

        self.flyweight.pmd_roi_averages = pmd_roi_averages
        self.flyweight.fluctuating_background_roi_averages = fluctuating_background_roi_averages
        self.flyweight.residual_roi_averages = residual_roi_averages

        ## Set the roi averages above that are None
        self._set_roi_averages()



        if std_corr_img_mean is None or std_corr_img_normalizer is None:
            self.flyweight.std_corr_img_mean = None
            self.flyweight.std_corr_img_normalizer = None
        else:
            self.flyweight.std_corr_img_mean = std_corr_img_mean  # standard_correlation_image.movie_mean
            self.flyweight.std_corr_img_normalizer = std_corr_img_normalizer  # standard_correlation_image.movie_normalizer

        if resid_corr_img_mean is None or resid_corr_img_support_values is None or resid_corr_img_normalizer is None:
            self.flyweight.resid_corr_img_support_values = None
            self.flyweight.resid_corr_img_mean = None
            self.flyweight.resid_corr_img_normalizer = None
        else:
            self.flyweight.resid_corr_img_support_values = resid_corr_img_support_values.coalesce()
            self.flyweight.resid_corr_img_mean = resid_corr_img_mean
            self.flyweight.resid_corr_img_normalizer = resid_corr_img_normalizer

        if bkgd_corr_img_mean is None or bkgd_corr_img_normalizer is None:
            self.flyweight.bkgd_corr_img_mean = None
            self.flyweight.bkgd_corr_img_normalizer = None
        else:
            self.flyweight.bkgd_corr_img_mean = bkgd_corr_img_mean
            self.flyweight.bkgd_corr_img_normalizer = bkgd_corr_img_normalizer

        self._ac_array = None
        self._colorful_ac_array = None
        self._pmd_array = None
        self._fluctuating_background_array = None
        self._static_background_array = None
        self._residual_array = None
        self._residual_correlation_images = None
        self._standard_correlation_images = None



        #Manage state of relevant arrays
        self._rescale = False

        # Move all tracked tensors to desired location so everything is on one device
        self.to(self._device)


    @property
    def flyweight(self) -> TensorFlyWeight:
        return self._flyweight

    @property
    def device(self) -> str:
        return self.flyweight.device

    @property
    def rescale(self):
        return self._rescale

    @rescale.setter
    def rescale(self, new_value: bool):
        managed_arrays_rescale = ['pmd_array',
                          'ac_array',
                          'static_background_array',
                          'fluctuating_background_array']

        self._rescale = new_value
        for name in managed_arrays_rescale:
            arr = getattr(self, name)
            arr.rescale = new_value

    @property
    def pmd_mean_img(self) -> torch.Tensor:
        return self.flyweight.pmd_mean_img

    @property
    def mean_img(self) -> torch.Tensor:
        """
        Property added for compatibility purposes -- identical to pmd mean img
        """
        return self.flyweight.mean_img

    @property
    def pmd_var_img(self) -> torch.Tensor:
        return self.flyweight.pmd_var_img

    @property
    def var_img(self) -> torch.Tensor:
        """
        Property added for compatibility purposes -- identical to pmd var img
        """
        return self.flyweight.var_img

    @property
    def normalizer(self) -> torch.Tensor:
        return self.flyweight.normalizer

    @property
    def pmd_u_projector(self) -> None | torch.Tensor:
        return self.flyweight.pmd_u_projector

    @property
    def factorized_bkgd_term1(self) -> None | torch.Tensor:
        return self.flyweight.factorized_bkgd_term1

    @property
    def factorized_bkgd_term2(self) -> None | torch.Tensor:
        return self.flyweight.factorized_bkgd_term2

    @property
    def shape(self):
        return self._shape

    @property
    def device(self):
        return self.flyweight.device

    def to(self, new_device):
        self.flyweight.to(new_device)
        self._move_managed_arrays(new_device)


    def _move_managed_tensors(self, new_device: str):
        self.flyweight.to(new_device)

    def _move_managed_arrays(self, new_device: str):
        for arr_name in self._managed_arrays:
            curr_arr = getattr(self, arr_name)
            curr_arr.to(self.device)

    @property
    def fov_shape(self) -> Tuple[int, int]:
        return self.shape[1:3]

    @property
    def num_frames(self) -> int:
        return self.shape[0]

    @property
    def u(self) -> torch.Tensor:
        return self.flyweight.u

    @property
    def b(self) -> torch.Tensor:
        return self.flyweight.b

    @property
    def baseline(self) -> torch.Tensor:
        """
        Returns a (height, width)-shaped 2D tensor
        """
        return self.flyweight.baseline

    @property
    def v(self) -> torch.Tensor:
        return self.flyweight.v

    @property
    def a(self) -> torch.Tensor:
        return self.flyweight.a

    @property
    def c(self) -> torch.Tensor:
        return self.flyweight.c

    @property
    def std_corr_img_mean(self) -> None | torch.Tensor:
        return self.flyweight.std_corr_img_mean

    @property
    def std_corr_img_normalizer(self) -> None | torch.Tensor:
        return self.flyweight.std_corr_img_normalizer

    @property
    def resid_corr_img_support_values(self) -> None | torch.Tensor:
        return self.flyweight.resid_corr_img_support_values

    @property
    def resid_corr_img_mean(self) ->  None | torch.Tensor:
        return self.flyweight.resid_corr_img_mean

    @property
    def resid_corr_img_normalizer(self) -> None | torch.Tensor:
        return self.flyweight.resid_corr_img_normalizer

    @property
    def global_residual_correlation_image(self) -> None | torch.Tensor:
        return self.flyweight.global_residual_correlation_image

    @property
    def bkgd_corr_img_mean(self) -> None | torch.Tensor:
        return self.flyweight.bkgd_corr_img_mean

    @property
    def bkgd_corr_img_normalizer(self) -> None | torch.Tensor:
        return self.flyweight.bkgd_corr_img_normalizer

    def _set_roi_averages(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns the ROI averages for each spatial footprint of the AC Array in the PMD movie, fluctuating background movie,
        and residual movie.
        """
        if self.flyweight.residual_roi_averages is None or self.flyweight.pmd_roi_averages is None or self.flyweight.fluctuating_background_roi_averages is None:
            device = self.c.device

            ## Compute an "ROI Average" tensor, which is just "a" where each neuron is binarized + normalized by size of support
            values = self.a.values()
            rows, cols = self.a.indices()

            values_keep = values != 0
            values = values[values_keep]
            rows = rows[values_keep]
            cols = cols[values_keep]

            values_bin = torch.ones_like(values)
            counts = torch.zeros(self.a.shape[1], device=device)
            counts.scatter_reduce_(0, cols, values_bin, reduce="sum")
            values_bin /= counts[cols]
            values_bin = torch.nan_to_num(values_bin, nan=0.0)

            #Note we do [cols, rows] instead of [rows, cols] because we want the transposed mat
            roi_avg_operator = torch.sparse_coo_tensor(torch.stack([cols, rows], dim=0),
                                                   values_bin,
                                                   size=(self.a.shape[1], self.a.shape[0])).to(self.a.device).coalesce()

            rU = torch.sparse.mm(roi_avg_operator, self.u)
            rA = torch.sparse.mm(roi_avg_operator, self.a)

            pmd_roi_averages = torch.sparse.mm(rU, self.v)
            ac_roi_averages = torch.sparse.mm(rA, self.c.T)
            static_background_roi_averages = torch.sparse.mm(roi_avg_operator, self.b[..., None])
            fluctuating_background_roi_averages = torch.sparse.mm(rU, self.factorized_bkgd_term1) @ self.factorized_bkgd_term2
            residual_roi_averages = pmd_roi_averages - ac_roi_averages - static_background_roi_averages - fluctuating_background_roi_averages

            self.flyweight.pmd_roi_averages = pmd_roi_averages
            self.flyweight.fluctuating_background_roi_averages = fluctuating_background_roi_averages
            self.flyweight.residual_roi_averages = residual_roi_averages

    @property
    def pmd_roi_averages(self) -> torch.Tensor:
        return self.flyweight.pmd_roi_averages

    @property
    def fluctuating_background_roi_averages(self) -> torch.Tensor:
        return self.flyweight.fluctuating_background_roi_averages

    @property
    def residual_roi_averages(self) -> torch.Tensor:
        return self.flyweight.residual_roi_averages

    @property
    def standard_correlation_images(self) -> None | StandardCorrelationImages:
        if self.std_corr_img_mean is not None:
            if self._standard_correlation_images is None:
                self._standard_correlation_images = StandardCorrelationImages.from_flyweight(self.flyweight,
                                                                                             (self._shape[1], self._shape[2]))
            return self._standard_correlation_images
        else:
            return None

    @property
    def background_to_signal_correlation_image(self) -> None | StandardCorrelationImages:
        """
        This array will not use the FlyWeight pattern that the other arrays use, since this is primarily an exploratory
        property. If this becomes crucial, can re-organize
        """
        if self.bkgd_corr_img_mean is not None:
            return StandardCorrelationImages.from_tensors(self.u,
                                             self.factorized_bkgd_term1 @ self.factorized_bkgd_term2,
                                             self.c,
                                             self.bkgd_corr_img_mean,
                                             self.bkgd_corr_img_normalizer,
                                             (self._shape[1], self._shape[2]))
        else:
            return None

    @property
    def residual_correlation_images(self) -> None | ResidualCorrelationImages:
        if self.resid_corr_img_mean is not None:
            if self._residual_correlation_images is None:
                self._residual_correlation_images = ResidualCorrelationImages.from_flyweight(self.flyweight,
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
            self._ac_array = ACArray.from_flyweight(self.fov_shape, self.flyweight, rescale=self.rescale)
        return self._ac_array

    @property
    def pmd_array(self) -> PMDArray:
        """
        Returns a PMDArray using the tensors stored in this object
        """
        if self._pmd_array is None:
            self._pmd_array = PMDArray.from_flyweight(
                self.shape,
                self.flyweight,
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
                                                                            self.flyweight,
                                                                            rescale=self.rescale)
        return self._fluctuating_background_array

    @property
    def static_background_array(self) -> StaticBackgroundArray:

        if self._static_background_array is None:
            self._static_background_array = StaticBackgroundArray.from_flyweight(self.flyweight,
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
            self._colorful_ac_array = ColorfulACArray.from_flyweight(self.fov_shape, self.flyweight)
        return self._colorful_ac_array

