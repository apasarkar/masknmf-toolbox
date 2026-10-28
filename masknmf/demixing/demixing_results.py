from pathlib import Path
from typing import *
import logging
import h5py
import numpy as np
from masknmf import display
from masknmf.arrays.linear_combination_array import LinearCombinationArray
from masknmf.compression import CompressionArray, TrendArray
from masknmf.demixing.demixing_arrays import SignalsArray, ResidualCorrelationImages, StandardCorrelationImages, ColorfulSignalsArray, StaticBackgroundArray, FluctuatingBackgroundArray, ResidCorrMode, MultiunitBackgroundArray
import torch
from masknmf.utils import Serializer, SparseCOOTensor
from masknmf.utils._serialization import save_dict, load_dict
from masknmf.arrays.array_interfaces import TensorFlyWeight
from masknmf.utils import infill_bad_timepoints
from masknmf.demixing._base_results import BaseResults

logger = logging.getLogger(__name__)

class DemixingResults(Serializer, BaseResults):
    _serialized = {
        "shape",
        "spatial_compressed",
        "temporal_compressed",
        "spatial_demixed",
        "temporal_demixed",
        "static_baseline",
        "mean_image",
        "noise_variance_image",
        "spatial_compressed_local_projector",
        "spatial_trend_basis",
        "temporal_trend_basis",
        "factorized_background_term1",
        "factorized_background_term2",
        "global_residual_correlation_image",
        "standard_correlation_image_mean",
        "standard_correlation_image_normalizer",
        "residual_correlation_image_support_values",
        "residual_correlation_image_mean",
        "residual_correlation_image_normalizer",
        "background_correlation_image_mean",
        "background_correlation_image_normalizer",
        "compression_array_roi_averages",
        "fluctuating_background_roi_averages",
        "residual_roi_averages",
        "multiunit_basis_term1",
        "multiunit_basis_term2",
        "temporal_demixed_raw",
        "temporal_demixed_trend",
    }

    """
    This lists arrays which are explicitly managed by demixing results.
    When you do DemixingResults.to(new_device), this object is responsible for making sure all of these arrays are moved to that device
    """
    _managed_arrays = ["compression_array",
                       "signals_array",
                       "colorful_signals_array",
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
            factorized_background_term1: torch.Tensor | None = None,
            factorized_background_term2: torch.Tensor | None = None,
            static_baseline: torch.Tensor | None = None,
            standard_correlation_image_mean: torch.Tensor | None = None,
            standard_correlation_image_normalizer: torch.Tensor | None = None,
            residual_correlation_image_support_values: SparseCOOTensor | None = None,
            residual_correlation_image_mean: torch.Tensor | None = None,
            residual_correlation_image_normalizer: torch.Tensor | None = None,
            background_correlation_image_mean: torch.Tensor | None = None,
            background_correlation_image_normalizer: torch.Tensor | None = None,
            global_residual_correlation_image: torch.Tensor | None= None,
            compression_array_roi_averages: torch.Tensor | None = None,
            fluctuating_background_roi_averages: torch.Tensor | None= None,
            residual_roi_averages: torch.Tensor | None = None,
            multiunit_basis_term1: torch.Tensor | None = None,
            multiunit_basis_term2: torch.Tensor | None = None,
            temporal_demixed_raw: torch.Tensor | None = None,
            temporal_demixed_trend: torch.Tensor | None = None,
            device: torch.device | str ="cpu",
            **kwargs
    ):
        """
        This class provides a convenient way to export all demixing result as array-like objects.

        All input parameters must be symmetric with the arrays that demixing results manages.
        For example, if PMDArray has spatial_compressed_local_projector as a constructor arg, the same name is used here

        Args:
            shape (tuple): (number of frames, fov_height, fov_width)
            spatial_compressed (torch.sparse_coo_tensor): shape (pixels, rank 1)
            temporal_compressed (torch.Tensor): shape (rank 2, num_frames)
            spatial_demixed (torch.sparse_coo_tensor): shape (pixels, number of neural signals)
            temporal_demixed (torch.Tensor): shape (number of frames, number of neural signals)
            mean_image (torch.Tensor | None): The mean image of the imaging data, used for reconstructing PMD Arrays
            noise_variance_image (torch.Tensor | None): The pixelwise noise variance image of the data, used for reconstructing PMD Arrays
            spatial_compressed_local_projector (SparseCOOTensor | None): A projection matrix used to project frames of data onto the PMD spatial_compressed subspace
            spatial_trend_basis (torch.Tensor | None): Shape (num_pixels, basis_rank). The spatial trend basis identified by PMD
            temporal_trend_basis (torch.Tensor | None): Shape (basis_rank, num_frames). The temporal trend basis identified by PMD
            factorized_background_term1 (torch.Tensor | None): tensor used to express low-rank background estimate
            factorized_background_term2 (torch.Tensor | None): tensor used to express low-rank background estimate
            static_baseline (torch.Tensor). The per-pixel static baseline_image.
                If not provided, the below code will set it so that the residual movie has mean 0.
                The residual is defined as UV - AC - Fluctuating background - Static Background
            standard_correlation_image_mean (torch.Tensor | None): the mean image used to lazily construct the standard correlation image per neuron
            standard_correlation_image_normalizer (torch.Tensor | None): the normalizer image used to lazily construct the standard correlation image per neuron
            residual_correlation_image_support_values (SparseCOOTensor | None): Shape (num_pixels, num_neurons). A sparse tensor describing the residual correlation
                image only at values where the neuron footprint is nonzero
            residual_correlation_image_mean (torch.Tensor | None): Shape (height, width). The mean image used to lazily
                compute the residual correlation image per neural signal.
            residual_correlation_image_normalizer (torch.Tensor | None): Shape (height, width). The normalizer image used to lazily
                compute the residual correlation image per neural signal.
            background_correlation_image_mean (torch.Tensor | None): The mean image used to compute the correlation between the signal and the background.
            background_correlation_image_normalizer (torch.Tensor | None): The mean image used to compute the correlation between the signal and the background.
            global_resid_correlation_image (torch.Tensor): The global correlation image of the residual. Shape (FOV dim 1, FOV dim 2).
            temporal_demixed_raw (torch.Tensor | None): shape (number of frames, number of neural signals). temporal_demixed
                re-estimated from the registered raw movie, with no compression or denoising
            temporal_demixed_trend (torch.Tensor | None): shape (number of frames, number of neural signals).
            device (str): 'cpu' or 'cuda'. used to manage where the tensors reside
            good_frame_indicator (torch.Tensor | None): A 1D boolean indicator, describing which timepoints from the raw data
                were processed (good frames) and which were ignored.
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

        if factorized_background_term1 is None or factorized_background_term2 is None:
            logger.debug("Background term empty")
            self.flyweight.factorized_background_term1 = torch.zeros(self.spatial_compressed.shape[1], 1, dtype=self.spatial_compressed.dtype, device=self._device)
            self.flyweight.factorized_background_term2 = torch.zeros((1, self.temporal_compressed.shape[1]), dtype=self.spatial_compressed.dtype, device=self._device)
        else:
            self.flyweight.factorized_background_term1 = factorized_background_term1.to(self._device)
            self.flyweight.factorized_background_term2 = factorized_background_term2.to(self._device)

        self.flyweight.global_residual_correlation_image = global_residual_correlation_image.to(self._device) if global_residual_correlation_image is not None else torch.zeros(self.shape[1], self.shape[2], device=self._device, dtype=self.spatial_compressed.dtype)


        if static_baseline is None:
            logger.debug("Static term was not provided, constructing baseline_image to ensure residual is mean 0")
            self.flyweight.static_baseline = (torch.sparse.mm(self.spatial_compressed, torch.mean(self.temporal_compressed, dim=1, keepdim=True)) -
                                torch.sparse.mm(self.spatial_demixed, torch.mean(self.temporal_demixed.T, dim=1, keepdim=True)) -
                                torch.sparse.mm(self.spatial_compressed, (
                                   self.factorized_background_term1 @ torch.mean(self.factorized_background_term2, axis=1,
                                                                           keepdim=True)))).to(self._device)
        else:
            self.flyweight.static_baseline = static_baseline.to(self._device)
        self.flyweight.baseline_image = self.static_baseline.reshape(self.fov_shape)

        self.flyweight.compression_array_roi_averages = compression_array_roi_averages
        self.flyweight.fluctuating_background_roi_averages = fluctuating_background_roi_averages
        self.flyweight.residual_roi_averages = residual_roi_averages

        ## Set the roi averages above that are None
        self._set_roi_averages()



        if standard_correlation_image_mean is None or standard_correlation_image_normalizer is None:
            self.flyweight.standard_correlation_image_mean = None
            self.flyweight.standard_correlation_image_normalizer = None
        else:
            self.flyweight.standard_correlation_image_mean = standard_correlation_image_mean.to(self._device)  # standard_correlation_image.movie_mean
            self.flyweight.standard_correlation_image_normalizer = standard_correlation_image_normalizer.to(self._device)  # standard_correlation_image.movie_normalizer

        if residual_correlation_image_mean is None or residual_correlation_image_support_values is None or residual_correlation_image_normalizer is None:
            self.flyweight.residual_correlation_image_support_values = None
            self.flyweight.residual_correlation_image_mean = None
            self.flyweight.residual_correlation_image_normalizer = None
        else:
            self.flyweight.residual_correlation_image_support_values = residual_correlation_image_support_values.coalesce().to(self._device)
            self.flyweight.residual_correlation_image_mean = residual_correlation_image_mean.to(self._device)
            self.flyweight.residual_correlation_image_normalizer = residual_correlation_image_normalizer.to(self._device)

        if background_correlation_image_mean is None or background_correlation_image_normalizer is None:
            self.flyweight.background_correlation_image_mean = None
            self.flyweight.background_correlation_image_normalizer = None
        else:
            self.flyweight.background_correlation_image_mean = background_correlation_image_mean.to(self._device)
            self.flyweight.background_correlation_image_normalizer = background_correlation_image_normalizer.to(self._device)

        if multiunit_basis_term1 is None or multiunit_basis_term2 is None:
            self.flyweight.multiunit_basis_term1 = None
            self.flyweight.multiunit_basis_term2 = None
        else:
            self.flyweight.multiunit_basis_term1 = multiunit_basis_term1.to(self._device)
            self.flyweight.multiunit_basis_term2 = multiunit_basis_term2.to(self._device)

        self.temporal_demixed_raw = temporal_demixed_raw
        self.temporal_demixed_trend = temporal_demixed_trend

        self._signals_array = None
        self._colorful_signals_array = None
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

    def export(self, path: str | Path, prefix: str = "", overwrite: bool = False):
        """
        Write to ``path`` as the group "DemixingResults", or "<prefix>/DemixingResults"; the file's other groups are
        kept.

        Raises FileExistsError when the group already exists, unless ``overwrite``. Tracking manifests and label files
        refer to results by file path, so edited results are written to a new file (see ``write_curated``).
        """
        group = f"{prefix}/{type(self).__name__}" if prefix else type(self).__name__
        if not overwrite and Path(path).is_file():
            with h5py.File(path, "r") as f:
                if group in f:
                    raise FileExistsError(f"{path} already holds {group}; write the new results to another file")
        save_dict(self._to_dict(), filename=path, group=group, exists_ok=True)

    @classmethod
    def from_hdf5(cls, path, prefix: str = "", **kwargs):
        """Load results written by :meth:`export`; kwargs are passed to the constructor."""
        d = load_dict(path, f"{prefix}/{cls.__name__}" if prefix else cls.__name__)
        return cls(**d, **kwargs)

    @classmethod
    def infill_bad_frames(cls,
                          demix_results: "DemixingResults",
                          good_frame_indicator: torch.Tensor | np.ndarray | list,
                          device: torch.device | str = "cpu") -> "DemixingResults":
        """
        If demixing was run on a subset of the frames ("good frames"), this routine produces a new demixing results
        object where all time series are expanded to include zero values at the bad frames. Useful to ensure
        that the output data matches the dimensions of the input data
        """
        ## First validate number of good frames equals number of frames that were actually demixed
        good_frame_indicator = torch.as_tensor(good_frame_indicator).to(device).to(torch.bool)
        indicator_num_good_frames = int(torch.count_nonzero(good_frame_indicator))
        demix_results_num_frames = int(demix_results.shape[0])
        if indicator_num_good_frames != int(demix_results.shape[0]):
            raise ValueError(f"Number of good frames specified by the frame labelling, {indicator_num_good_frames}, "
                             f"does not match the number of frames in the data {demix_results_num_frames}")
        new_shape = (int(good_frame_indicator.numel()), int(demix_results.shape[1]), int(demix_results.shape[2]))

        new_demixing_results = cls(shape=new_shape,
                                   spatial_compressed = demix_results.spatial_compressed,
                                   temporal_compressed = infill_bad_timepoints(demix_results.temporal_compressed, good_frame_indicator, device=device),
                                   spatial_demixed = demix_results.spatial_demixed,
                                   temporal_demixed = None if demix_results.temporal_demixed is None else infill_bad_timepoints(demix_results.temporal_demixed.T, good_frame_indicator, device=device).T,
                                   mean_image = demix_results.mean_image,
                                   noise_variance_image = demix_results.noise_variance_image,
                                   spatial_compressed_local_projector = demix_results.spatial_compressed_local_projector,
                                   spatial_trend_basis = demix_results.spatial_trend_basis,
                                   temporal_trend_basis = infill_bad_timepoints(demix_results.temporal_trend_basis, good_frame_indicator, device=device),
                                   factorized_background_term1 = demix_results.factorized_background_term1,
                                   factorized_background_term2 = infill_bad_timepoints(demix_results.factorized_background_term2, good_frame_indicator, device=device),
                                   static_baseline = demix_results.static_baseline,
                                   standard_correlation_image_mean = demix_results.standard_correlation_image_mean,
                                   standard_correlation_image_normalizer = demix_results.standard_correlation_image_normalizer,
                                   residual_correlation_image_support_values = demix_results.residual_correlation_image_support_values,
                                   residual_correlation_image_mean = demix_results.residual_correlation_image_mean,
                                   residual_correlation_image_normalizer = demix_results.residual_correlation_image_normalizer,
                                   background_correlation_image_mean = demix_results.background_correlation_image_mean,
                                   background_correlation_image_normalizer = demix_results.background_correlation_image_normalizer,
                                   global_residual_correlation_image = demix_results.global_residual_correlation_image,
                                   compression_array_roi_averages = infill_bad_timepoints(demix_results.compression_array_roi_averages, good_frame_indicator, device=device),
                                   fluctuating_background_roi_averages = infill_bad_timepoints(demix_results.fluctuating_background_roi_averages, good_frame_indicator, device=device),
                                   residual_roi_averages = infill_bad_timepoints(demix_results.residual_roi_averages , good_frame_indicator, device=device),
                                   multiunit_basis_term1 = demix_results.multiunit_basis_term1,
                                   multiunit_basis_term2 = infill_bad_timepoints(demix_results.multiunit_basis_term2, good_frame_indicator, device=device),
                                   temporal_demixed_raw = None if demix_results.temporal_demixed_raw is None else infill_bad_timepoints(demix_results.temporal_demixed_raw.T, good_frame_indicator, device=device).T,
                                   temporal_demixed_trend = None if demix_results.temporal_demixed_trend is None else infill_bad_timepoints(demix_results.temporal_demixed_trend.T, good_frame_indicator, device=device).T,
                                   device=device)
        return new_demixing_results


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
    def factorized_background_term1(self) -> None | torch.Tensor:
        return self.flyweight.factorized_background_term1

    @property
    def factorized_background_term2(self) -> None | torch.Tensor:
        return self.flyweight.factorized_background_term2

    @property
    def multiunit_basis_term1(self) -> None | torch.Tensor:
        return self.flyweight.multiunit_basis_term1

    @property
    def multiunit_basis_term2(self) -> None | torch.Tensor:
        return self.flyweight.multiunit_basis_term2

    @property
    def temporal_demixed_raw(self) -> None | torch.Tensor:
        return self.flyweight.temporal_demixed_raw

    @temporal_demixed_raw.setter
    def temporal_demixed_raw(self, value: torch.Tensor | None):
        if value is not None and tuple(value.shape) != tuple(self.temporal_demixed.shape):
            raise ValueError(f"temporal_demixed_raw has shape {tuple(value.shape)}, temporal_demixed {tuple(self.temporal_demixed.shape)}")
        self.flyweight.temporal_demixed_raw = None if value is None else value.to(self._device).float()

    @property
    def temporal_demixed_trend(self) -> None | torch.Tensor:
        return self.flyweight.temporal_demixed_trend

    @temporal_demixed_trend.setter
    def temporal_demixed_trend(self, value: torch.Tensor | None):
        if value is not None and tuple(value.shape) != tuple(self.temporal_demixed.shape):
            raise ValueError(
                f"temporal_demixed_raw has shape {tuple(value.shape)}, temporal_demixed {tuple(self.temporal_demixed.shape)}")
        self.flyweight.temporal_demixed_trend = None if value is None else value.to(self._device).float()


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
    def static_baseline(self) -> torch.Tensor:
        return self.flyweight.static_baseline

    @property
    def baseline_image(self) -> torch.Tensor:
        """
        Returns a (height, width)-shaped 2D tensor. Derived from static_baseline, which is a 1D vectorized version of this
        """
        return self.flyweight.baseline_image

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
    def standard_correlation_image_mean(self) -> None | torch.Tensor:
        return self.flyweight.standard_correlation_image_mean

    @property
    def standard_correlation_image_normalizer(self) -> None | torch.Tensor:
        return self.flyweight.standard_correlation_image_normalizer

    @property
    def residual_correlation_image_support_values(self) -> None | torch.Tensor:
        return self.flyweight.residual_correlation_image_support_values

    @property
    def residual_correlation_image_mean(self) -> None | torch.Tensor:
        return self.flyweight.residual_correlation_image_mean

    @property
    def residual_correlation_image_normalizer(self) -> None | torch.Tensor:
        return self.flyweight.residual_correlation_image_normalizer

    @property
    def global_residual_correlation_image(self) -> None | torch.Tensor:
        return self.flyweight.global_residual_correlation_image

    @property
    def background_correlation_image_mean(self) -> None | torch.Tensor:
        return self.flyweight.background_correlation_image_mean

    @property
    def background_correlation_image_normalizer(self) -> None | torch.Tensor:
        return self.flyweight.background_correlation_image_normalizer

    def _set_roi_averages(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns the ROI averages for each spatial footprint of the AC Array in the PMD movie, fluctuating background movie,
        and residual movie.
        """
        if self.flyweight.residual_roi_averages is None or self.flyweight.compression_array_roi_averages is None or self.flyweight.fluctuating_background_roi_averages is None:
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

            compression_array_roi_averages = torch.sparse.mm(rU, self.temporal_compressed)
            ac_roi_averages = torch.sparse.mm(rA, self.temporal_demixed.T)
            static_background_roi_averages = torch.sparse.mm(roi_avg_operator, self.static_baseline[..., None])
            fluctuating_background_roi_averages = torch.sparse.mm(rU, self.factorized_background_term1) @ self.factorized_background_term2
            residual_roi_averages = compression_array_roi_averages - ac_roi_averages - static_background_roi_averages - fluctuating_background_roi_averages

            self.flyweight.compression_array_roi_averages = compression_array_roi_averages
            self.flyweight.fluctuating_background_roi_averages = fluctuating_background_roi_averages
            self.flyweight.residual_roi_averages = residual_roi_averages

    def compute_multiunit_residual_factorization(
            self,
            test_rank: int = 200,
            num_oversamples: int = 5,
            explained_variance_cutoff: float = 0.99,
    ) -> None:
        """
        Compute a low-rank factorization of the residual (UV - AC - static - fluctuating background)
        and store it as multiunit_basis_term1 / multiunit_basis_term2.

        Runs on the current device; call .to("cuda") first for large datasets.
        """
        from masknmf.demixing.demixing_utils import extract_multiunit_factorization

        term1, term2 = extract_multiunit_factorization(
            self.compression_array,
            self.spatial_compressed,
            self.temporal_compressed,
            self.spatial_demixed,
            self.temporal_demixed,
            self.static_baseline,
            factorized_ring_term=(self.factorized_background_term1, self.factorized_background_term2),
            test_rank=test_rank,
            num_oversamples=num_oversamples,
            explained_variance_cutoff=explained_variance_cutoff,
        )
        self.flyweight.multiunit_basis_term1 = term1.to(self.device)
        self.flyweight.multiunit_basis_term2 = term2.to(self.device)
        self._multiunit_background_array = None  # rebuild from the new terms on next access

    @property
    def has_multiunit_factorization(self) -> bool:
        return self.multiunit_basis_term1 is not None and self.multiunit_basis_term2 is not None

    @property
    def multiunit_background_array(self) -> MultiunitBackgroundArray | None:
        if not self.has_multiunit_factorization:
            return None
        if self._multiunit_background_array is None:
            self._multiunit_background_array = MultiunitBackgroundArray.from_flyweight(self.fov_shape,
                                                                                       self.flyweight,
                                                                                       rescale=self.rescale)
        return self._multiunit_background_array

    @property
    def compression_array_roi_averages(self) -> torch.Tensor:
        return self.flyweight.compression_array_roi_averages

    @property
    def fluctuating_background_roi_averages(self) -> torch.Tensor:
        return self.flyweight.fluctuating_background_roi_averages

    @property
    def residual_roi_averages(self) -> torch.Tensor:
        return self.flyweight.residual_roi_averages

    @property
    def standard_correlation_images(self) -> None | StandardCorrelationImages:
        if self.standard_correlation_image_mean is not None:
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
        if self.background_correlation_image_mean is not None:
            return StandardCorrelationImages.from_tensors(self.spatial_compressed,
                                                          self.factorized_background_term1 @ self.factorized_background_term2,
                                                          self.temporal_demixed,
                                                          self.background_correlation_image_mean,
                                                          self.background_correlation_image_normalizer,
                                                          (self._shape[1], self._shape[2]))
        else:
            return None

    @property
    def residual_correlation_images(self) -> None | ResidualCorrelationImages:
        if self.residual_correlation_image_mean is not None:
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
    def static_background_array(self) -> StaticBackgroundArray:

        if self._static_background_array is None:
            self._static_background_array = StaticBackgroundArray.from_flyweight(self.flyweight,
                                                                                 self.num_frames,
                                                                                 rescale = self.rescale)
        return self._static_background_array

    @property
    def residual_array(self) -> LinearCombinationArray:
        if self._residual_array is None:
            data_list = [self.compression_array,
                         self.fluctuating_background_array,
                         self.signals_array,
                         self.static_background_array]
            self._residual_array = LinearCombinationArray(data_list,
                                                          alphas=[1, -1, -1, -1])
        return self._residual_array

    @property
    def colorful_signals_array(self) -> ColorfulSignalsArray:
        if self._colorful_signals_array is None:
            self._colorful_signals_array = ColorfulSignalsArray.from_flyweight(self.fov_shape, self.flyweight)
        return self._colorful_signals_array