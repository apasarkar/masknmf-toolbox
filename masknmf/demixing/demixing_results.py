from typing import *
import numpy as np
from masknmf import display
from masknmf.compression import CompressionArray, TrendArray
from masknmf.demixing.demixing_arrays import SignalsArray, ResidualCorrelationImages, StandardCorrelationImages, ColorfulSignalsArray, StaticBackgroundArray, FluctuatingBackgroundArray, ResidualArray, ResidCorrMode, MultiunitBackgroundArray
import torch
from masknmf.utils import Serializer, SparseCOOTensor
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
        "spatial_compressed",
        "temporal_compressed",
        "spatial_demixed",
        "temporal_demixed",
        "b",
        "mean_image",
        "noise_variance_image",
        "spatial_compressed_local_projector",
        "spatial_trend_basis",
        "temporal_trend_basis",
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
        "residual_roi_averages",
        "multiunit_basis_term1",
        "multiunit_basis_term2",
    }

    """
    This lists arrays which are explicitly managed by demixing results.
    When you do DemixingResults.to(new_device), this object is responsible for making sure all of these arrays are moved to that device
    """
    _managed_arrays = ["compression_array",
                       "signals_array",
                       "colorful_ac_array",
                       "fluctuating_background_array",
                       "static_background_array",
                       "standard_correlation_images",
                       "residual_correlation_images",
                       "multiunit_background_array",
                       "trend_array"
                       ]
    def __init__(
            self,
            shape: tuple[int, int, int] | np.ndarray,
            spatial_compressed: SparseCOOTensor,
            temporal_compressed: torch.Tensor,
            spatial_demixed: SparseCOOTensor,
            temporal_demixed: torch.Tensor,
            mean_image: torch.Tensor | None = None,
            noise_variance_image: torch.Tensor | None = None,
            spatial_compressed_local_projector: SparseCOOTensor | None = None,
            spatial_trend_basis: torch.Tensor | None= None,
            temporal_trend_basis: torch.Tensor | None = None,
            factorized_bkgd_term1: torch.Tensor | None = None,
            factorized_bkgd_term2: torch.Tensor | None = None,
            b: torch.Tensor | None = None,
            std_corr_img_mean: torch.Tensor | None = None,
            std_corr_img_normalizer: torch.Tensor | None = None,
            resid_corr_img_support_values: SparseCOOTensor | None = None,
            resid_corr_img_mean: torch.Tensor | None = None,
            resid_corr_img_normalizer: torch.Tensor | None = None,
            bkgd_corr_img_mean: torch.Tensor | None = None,
            bkgd_corr_img_normalizer: torch.Tensor | None = None,
            global_residual_correlation_image: torch.Tensor | None= None,
            pmd_roi_averages: torch.Tensor | None = None,
            fluctuating_background_roi_averages: torch.Tensor | None= None,
            residual_roi_averages: torch.Tensor | None = None,
            multiunit_basis_term1: torch.Tensor | None = None,
            multiunit_basis_term2: torch.Tensor | None = None,
            device: torch.device | str ="cpu",
            **kwargs
    ):
        """
        This class provides a convenient way to export all demixing result as array-like objects.

        All input parameters must be symmetric with the arrays that demixing results manages.
        For example, if PMDArray has spatial_compressed_local_projector as a constructor arg, the same name is used here

        Args:
            shape (tuple): (number of frames, field of view dimension 1, field of view dimension 2)
            spatial_compressed (torch.sparse_coo_tensor): shape (pixels, rank 1)
            temporal_compressed (torch.Tensor): shape (rank 2, num_frames)
            spatial_demixed (torch.sparse_coo_tensor): shape (pixels, number of neural signals)
            temporal_demixed (torch.Tensor): shape (number of frames, number of neural signals)
            mean_image (torch.Tensor | None): The mean image of the imaging data, used for reconstructing PMD Arrays
            noise_variance_image (torch.Tensor | None): The pixelwise noise variance image of the data, used for reconstructing PMD Arrays
            spatial_compressed_local_projector (SparseCOOTensor | None): A projection matrix used to project frames of data onto the PMD spatial_compressed subspace
            spatial_trend_basis (torch.Tensor | None): Shape (num_pixels, basis_rank). The spatial trend basis identified by PMD
            temporal_trend_basis (torch.Tensor | None): Shape (basis_rank, num_frames). The temporal trend basis identified by PMD
            factorized_bkgd_term1 (torch.Tensor | None): tensor used to express low-rank background estimate
            factorized_bkgd_term2 (torch.Tensor | None): tensor used to express low-rank background estimate
            b (torch.Tensor). The per-pixel static baseline.
                If not provided, the below code will set it so that the residual movie has mean 0.
                The residual is defined as UV - AC - Fluctuaating background - Static Background
            std_corr_img_mean (torch.Tensor | None): the mean image used to lazily construct the standard correlation image per neuron
            std_corr_img_normalizer (torch.Tensor | None): the normalizer image used to lazily construct the standard correlation image per neuron
            resid_corr_img_support_values (SparseCOOTensor | None): Shape (num_pixels, num_neurons). A sparse tensor describing the residual correlation
                image only at values where the neuron footprint is nonzero
            resid_corr_img_mean (torch.Tensor | None): Shape (height, width). The mean image used to lazily
                compute the residual correlation image per neural signal.
            resid_corr_img_normalizer (torch.Tensor | None): Shape (height, width). The normalizer image used to lazily
                compute the residual correlation image per neural signal.
            bkgd_corr_img_mean (torch.Tensor | None): The mean image used to compute the correlation between the signal and the background.
            bkgd_corr_img_normalizer (torch.Tensor | None): The mean image used to compute the correlation between the signal and the background.
            global_resid_correlation_image (torch.Tensor): The global correlation image of the residual. Shape (FOV dim 1, FOV dim 2).
            device (str): 'cpu' or 'cuda'. used to manage where the tensors reside
        """
        self._device = device
        self._shape = tuple(shape)
        self._flyweight = TensorFlyWeight()
        self.flyweight.spatial_compressed = spatial_compressed.to(self._device).float().coalesce()
        self.flyweight.temporal_compressed = temporal_compressed.to(self._device).float()
        self.flyweight.spatial_demixed = spatial_demixed.to(self._device).float().coalesce()
        self.flyweight.temporal_demixed = temporal_demixed.to(self._device).float()

        self.flyweight.mean_image = mean_image.to(self._device) if mean_image is not None else torch.zeros(self.shape[1], self.shape[2], device=self._device)
        self.flyweight.noise_variance_image = noise_variance_image.to(self._device) if noise_variance_image is not None else torch.ones(self.shape[1], self.shape[2], device=self._device)

        #This is called
        self.flyweight.normalizer = self.flyweight.noise_variance_image

        self.flyweight.spatial_compressed_local_projector = spatial_compressed_local_projector.float().coalesce().to(self._device) if spatial_compressed_local_projector is not None else None


        if spatial_trend_basis is None or temporal_trend_basis is None:
            self.flyweight.spatial_trend_basis = torch.zeros(self.spatial_compressed.shape[0], 1, dtype=self.spatial_compressed.dtype,
                                                             device=self._device)
            self.flyweight.temporal_trend_basis = torch.zeros((1, self.temporal_compressed.shape[1]), dtype=self.spatial_compressed.dtype,
                                                              device=self._device)
        else:
            self.flyweight.spatial_trend_basis = spatial_trend_basis.to(self._device)
            self.flyweight.temporal_trend_basis = temporal_trend_basis.to(self._device)

        if factorized_bkgd_term1 is None or factorized_bkgd_term2 is None:
            display("Background term empty")
            self.flyweight.factorized_bkgd_term1 = torch.zeros(self.spatial_compressed.shape[1], 1, dtype=self.spatial_compressed.dtype, device=self._device)
            self.flyweight.factorized_bkgd_term2 = torch.zeros((1, self.temporal_compressed.shape[1]), dtype=self.spatial_compressed.dtype, device=self._device)
        else:
            self.flyweight.factorized_bkgd_term1 = factorized_bkgd_term1.to(self._device)
            self.flyweight.factorized_bkgd_term2 = factorized_bkgd_term2.to(self._device)

        self.flyweight.global_residual_correlation_image = global_residual_correlation_image.to(self._device) if global_residual_correlation_image is not None else torch.zeros(self.shape[1], self.shape[2], device=self._device, dtype=self.spatial_compressed.dtype)


        if b is None:
            display("Static term was not provided, constructing baseline to ensure residual is mean 0")
            self.flyweight.b = (torch.sparse.mm(self.spatial_compressed, torch.mean(self.temporal_compressed, dim=1, keepdim=True)) -
                                torch.sparse.mm(self.spatial_demixed, torch.mean(self.temporal_demixed.T, dim=1, keepdim=True)) -
                                torch.sparse.mm(self.spatial_compressed, (
                                   self.factorized_bkgd_term1 @ torch.mean(self.factorized_bkgd_term2, axis=1,
                                                                           keepdim=True)))).to(self._device)
        else:
            self.flyweight.b = b.to(self._device)
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
            self.flyweight.std_corr_img_mean = std_corr_img_mean.to(self._device)  # standard_correlation_image.movie_mean
            self.flyweight.std_corr_img_normalizer = std_corr_img_normalizer.to(self._device)  # standard_correlation_image.movie_normalizer

        if resid_corr_img_mean is None or resid_corr_img_support_values is None or resid_corr_img_normalizer is None:
            self.flyweight.resid_corr_img_support_values = None
            self.flyweight.resid_corr_img_mean = None
            self.flyweight.resid_corr_img_normalizer = None
        else:
            self.flyweight.resid_corr_img_support_values = resid_corr_img_support_values.coalesce().to(self._device)
            self.flyweight.resid_corr_img_mean = resid_corr_img_mean.to(self._device)
            self.flyweight.resid_corr_img_normalizer = resid_corr_img_normalizer.to(self._device)

        if bkgd_corr_img_mean is None or bkgd_corr_img_normalizer is None:
            self.flyweight.bkgd_corr_img_mean = None
            self.flyweight.bkgd_corr_img_normalizer = None
        else:
            self.flyweight.bkgd_corr_img_mean = bkgd_corr_img_mean.to(self._device)
            self.flyweight.bkgd_corr_img_normalizer = bkgd_corr_img_normalizer.to(self._device)

        if multiunit_basis_term1 is None or multiunit_basis_term2 is None:
            self.flyweight.multiunit_basis_term1 = torch.zeros(self.spatial_compressed.shape[1], 1, dtype=self.spatial_compressed.dtype,
                                                               device=self._device)
            self.flyweight.multiunit_basis_term2 = torch.zeros((1, self.temporal_compressed.shape[1]), dtype=self.spatial_compressed.dtype,
                                                               device=self._device)
        else:
            self.flyweight.multiunit_basis_term1 = multiunit_basis_term1.to(self._device)
            self.flyweight.multiunit_basis_term2 = multiunit_basis_term2.to(self._device)

        self._signals_array = None
        self._colorful_ac_array = None
        self._compression_array = None
        self._fluctuating_background_array = None
        self._multiunit_background_array = None
        self._static_background_array = None
        self._residual_array = None
        self._residual_correlation_images = None
        self._standard_correlation_images = None
        self._trend_array = None



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
        managed_arrays_rescale = ['compression_array',
                          'signals_array',
                          'static_background_array',
                          'fluctuating_background_array',
                          'multiunit_background_array']

        self._rescale = new_value
        for name in managed_arrays_rescale:
            arr = getattr(self, name)
            arr.rescale = new_value

    @property
    def mean_image(self) -> torch.Tensor:
        return self.flyweight.mean_image

    @property
    def noise_variance_image(self) -> torch.Tensor:
        """
        This is the PMD Noise variance image
        """
        return self.flyweight.noise_variance_image

    @property
    def spatial_trend_basis(self) -> torch.Tensor | None:
        return self.flyweight.spatial_trend_basis

    @property
    def temporal_trend_basis(self) -> torch.Tensor | None:
        return self.flyweight.temporal_trend_basis

    @property
    def normalizer(self) -> torch.Tensor:
        return self.flyweight.normalizer

    @property
    def spatial_compressed_local_projector(self) -> None | torch.Tensor:
        return self.flyweight.spatial_compressed_local_projector

    @property
    def factorized_bkgd_term1(self) -> None | torch.Tensor:
        return self.flyweight.factorized_bkgd_term1

    @property
    def factorized_bkgd_term2(self) -> None | torch.Tensor:
        return self.flyweight.factorized_bkgd_term2

    @property
    def multiunit_basis_term1(self) -> None | torch.Tensor:
        return self.flyweight.multiunit_basis_term1

    @property
    def multiunit_basis_term2(self) -> None | torch.Tensor:
        return self.flyweight.multiunit_basis_term2


    @property
    def shape(self):
        return self._shape

    @property
    def device(self):
        return self.flyweight.device

    def to(self, new_device):
        self.flyweight.to(new_device)
        self._move_managed_arrays(new_device)


    def _move_managed_tensors(self, new_device: torch.device | str):
        self.flyweight.to(new_device)

    def _move_managed_arrays(self, new_device: str):
        for arr_name in self._managed_arrays:
            curr_arr = getattr(self, arr_name)
            if curr_arr is not None:
                curr_arr.to(self.device)

    @property
    def fov_shape(self) -> tuple[int, int]:
        return self.shape[1:3]

    @property
    def num_frames(self) -> int:
        return self.shape[0]

    @property
    def spatial_compressed(self) -> torch.Tensor:
        return self.flyweight.spatial_compressed

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
    def temporal_compressed(self) -> torch.Tensor:
        return self.flyweight.temporal_compressed

    @property
    def spatial_demixed(self) -> torch.Tensor:
        return self.flyweight.spatial_demixed

    @property
    def temporal_demixed(self) -> torch.Tensor:
        return self.flyweight.temporal_demixed

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
            device = self.temporal_demixed.device

            ## Compute an "ROI Average" tensor, which is just "a" where each neuron is binarized + normalized by size of support
            values = self.spatial_demixed.values()
            rows, cols = self.spatial_demixed.indices()

            values_keep = values != 0
            values = values[values_keep]
            rows = rows[values_keep]
            cols = cols[values_keep]

            values_bin = torch.ones_like(values)
            counts = torch.zeros(self.spatial_demixed.shape[1], device=device)
            counts.scatter_reduce_(0, cols, values_bin, reduce="sum")
            values_bin /= counts[cols]
            values_bin = torch.nan_to_num(values_bin, nan=0.0)

            #Note we do [cols, rows] instead of [rows, cols] because we want the transposed mat
            roi_avg_operator = torch.sparse_coo_tensor(torch.stack([cols, rows], dim=0),
                                                       values_bin,
                                                       size=(self.spatial_demixed.shape[1], self.spatial_demixed.shape[0])).to(self.spatial_demixed.device).coalesce()

            rU = torch.sparse.mm(roi_avg_operator, self.spatial_compressed)
            rA = torch.sparse.mm(roi_avg_operator, self.spatial_demixed)

            pmd_roi_averages = torch.sparse.mm(rU, self.temporal_compressed)
            ac_roi_averages = torch.sparse.mm(rA, self.temporal_demixed.T)
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
            return StandardCorrelationImages.from_tensors(self.spatial_compressed,
                                                          self.factorized_bkgd_term1 @ self.factorized_bkgd_term2,
                                                          self.temporal_demixed,
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
    def signals_array(self) -> SignalsArray:
        """
        Returns an SignalsArray using the tensors stored in this object
        """
        if self._signals_array is None:
            self._signals_array = SignalsArray.from_flyweight(self.fov_shape, self.flyweight, rescale=self.rescale)
        return self._signals_array

    @property
    def compression_array(self) -> CompressionArray:
        """
        Returns a CompressionArray using the tensors stored in this object
        """
        if self._compression_array is None:
            self._compression_array = CompressionArray.from_flyweight(
                self.shape,
                self.flyweight,
                rescale=self.rescale,
            )
        return self._compression_array

    @property
    def trend_array(self) -> TrendArray:
        if self._trend_array is None:
            self._trend_array = TrendArray.from_flyweight(self.shape,
                                          self.flyweight)

        return self._trend_array

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
    def multiunit_background_array(self) -> MultiunitBackgroundArray:
        if self._multiunit_background_array is None:
            self._multiunit_background_array = MultiunitBackgroundArray.from_flyweight(self.fov_shape,
                                                                                       self.flyweight,
                                                                                       rescale=self.rescale)
        return self._multiunit_background_array

    @property
    def static_background_array(self) -> StaticBackgroundArray:

        if self._static_background_array is None:
            self._static_background_array = StaticBackgroundArray.from_flyweight(self.flyweight,
                                                                             rescale = self.rescale)
        return self._static_background_array

    @property
    def residual_array(self) -> ResidualArray:
        if self._residual_array is None:
            self._residual_array = ResidualArray(self.compression_array,
                                                 self.signals_array,
                                                 self.fluctuating_background_array,
                                                 self.static_background_array,
                                                 )
        return self._residual_array

    @property
    def colorful_ac_array(self) -> ColorfulSignalsArray:
        if self._colorful_ac_array is None:
            self._colorful_ac_array = ColorfulSignalsArray.from_flyweight(self.fov_shape, self.flyweight)
        return self._colorful_ac_array