from typing import Optional, Literal
import masknmf
from masknmf.compression import PMDArray
from masknmf.compression.denoising import train_total_variance_denoiser
from masknmf.compression.decomposition import pmd_decomposition
from masknmf import ArrayLike
import numpy as np

class CompressStrategy:

    def __init__(self,
                 dataset: ArrayLike | None,
                 block_sizes: tuple[int, int] = (32, 32),
                 frame_range: int | None  = None,
                 max_components: int = 20,
                 sim_conf: int = 5,
                 frame_batch_size: int = 10000,
                 max_consecutive_failures: int=1,
                 spatial_avg_factor:int=1,
                 temporal_avg_factor:int=1,
                 compute_normalizer: Optional[bool] = True,
                 pixel_weighting: Optional[np.ndarray] = None,
                 device: Literal["auto", "cpu", "cuda"] = "auto",
                 ):

        self._dataset = dataset

        ##User-settable parameters
        self._block_sizes = block_sizes
        self._frame_range = frame_range
        self._max_components = max_components
        self._frame_batch_size = frame_batch_size
        self._spatial_avg_factor = spatial_avg_factor
        self._temporal_avg_factor = temporal_avg_factor
        self._device=device

        ##Non user-settable parameters
        self._sim_conf = sim_conf
        self._max_consecutive_failures=max_consecutive_failures
        self._compute_normalizer = compute_normalizer
        self._pixel_weighting = pixel_weighting

        self._results = None

    @property
    def dataset(self) -> ArrayLike | None:
        return self._dataset

    @dataset.setter
    def dataset(self, new_dataset: ArrayLike):
        self._dataset = new_dataset

    @property
    def block_sizes(self) -> tuple[int, int]:
        return self._block_sizes

    @block_sizes.setter
    def block_sizes(self, new_sizes: tuple[int, int]):
        self._block_sizes = new_sizes

    @property
    def frame_range(self) -> int | None:
        return self._frame_range

    @frame_range.setter
    def frame_range(self, new_frame_range: int):
        self._frame_range = new_frame_range

    @property
    def max_consecutive_failures(self) -> int:
        return self._max_consecutive_failures

    @max_consecutive_failures.setter
    def max_consecutive_failures(self, new_num: int):
        self._max_consecutive_failures = new_num

    @property
    def sim_conf(self) -> int:
        return self._sim_conf

    @sim_conf.setter
    def sim_conf(self, new_sim_conf: int):
        self._sim_conf = new_sim_conf

    @property
    def max_components(self):
        return self._max_components

    @max_components.setter
    def max_components(self, num_comps: int):
        self._max_components = num_comps

    @property
    def frame_batch_size(self) -> int:
        return self._frame_batch_size

    @frame_batch_size.setter
    def frame_batch_size(self, new_batch_size:int):
        self._frame_batch_size = new_batch_size

    @property
    def spatial_avg_factor(self) -> int:
        return self._spatial_avg_factor

    @spatial_avg_factor.setter
    def spatial_avg_factor(self, new_spatial_avg_factor: int):
        self._spatial_avg_factor = new_spatial_avg_factor

    @property
    def temporal_avg_factor(self) -> int:
        return self._temporal_avg_factor

    @temporal_avg_factor.setter
    def temporal_avg_factor(self, new_temporal_avg_factor: int):
        self._temporal_avg_factor = new_temporal_avg_factor

    @property
    def device(self) ->str:
        return self._device

    @device.setter
    def device(self, new_device: str):
        self._device = new_device

    def results(self) -> PMDArray | None:
        return self._results

    def compress(self) -> PMDArray:
        if self.dataset is None:
            raise ValueError("No dataset provided. "
                             "Provide a dataset at construction time of the"
                             " strategy object or set the dataset property")
        self._results = pmd_decomposition(self.dataset,
                                          self.block_sizes,
                                          frame_range=self.frame_range,
                                          max_components=self.max_components,
                                          sim_conf=self._sim_conf,
                                          frame_batch_size=self.frame_batch_size,
                                          max_consecutive_failures=self._max_consecutive_failures,
                                          spatial_avg_factor=self.spatial_avg_factor,
                                          temporal_avg_factor=self.temporal_avg_factor,
                                          compute_normalizer=self._compute_normalizer,
                                          pixel_weighting=self._pixel_weighting,
                                          device=self.device)

        return self._results

class CompressDenoiseStrategy(CompressStrategy):


    def __init__(self,
                 dataset: ArrayLike | None,
                 block_sizes: tuple[int, int] = (32, 32),
                 frame_range: int | None  = None,
                 max_components: int = 20,
                 sim_conf: int = 5,
                 frame_batch_size: int = 10000,
                 max_consecutive_failures: int=1,
                 spatial_avg_factor:int=1,
                 temporal_avg_factor:int=1,
                 compute_normalizer: Optional[bool] = True,
                 pixel_weighting: Optional[np.ndarray] = None,
                 device: Literal["auto", "cpu", "cuda"] = "auto",
                 noise_variance_quantile: float = 0.7,
                 num_epochs: int = 5
                 ):

        super().__init__(dataset,
                         block_sizes,
                         frame_range,
                         max_components,
                         sim_conf,
                         frame_batch_size,
                         max_consecutive_failures,
                         spatial_avg_factor,
                         temporal_avg_factor,
                         compute_normalizer,
                         pixel_weighting,
                         device)
        self._num_epochs = num_epochs
        self._noise_variance_quantile = noise_variance_quantile

    @property
    def num_epochs(self) -> int:
        return self._num_epochs

    @num_epochs.setter
    def num_epochs(self, new_num_epochs: int):
        self._num_epochs = new_num_epochs

    @property
    def noise_variance_quantile(self) -> float:
        return self._noise_variance_quantile

    @noise_variance_quantile.setter
    def noise_variance_quantile(self, new_noise_variance_quantile: float):
        self._noise_variance_quantile = new_noise_variance_quantile

    def compress(self):

        if self.dataset is None:
            raise ValueError("No dataset provided. "
                             "Provide a dataset at construction time of the"
                             " strategy object or set the dataset property")
        pmd_no_denoiser = pmd_decomposition(self.dataset,
                                            self.block_sizes,
                                            frame_range=self.frame_range,
                                            max_components=self.max_components,
                                            sim_conf=self._sim_conf,
                                            frame_batch_size=self.frame_batch_size,
                                            max_consecutive_failures=self._max_consecutive_failures,
                                            spatial_avg_factor=self.spatial_avg_factor,
                                            temporal_avg_factor=self.temporal_avg_factor,
                                            compute_normalizer=self._compute_normalizer,
                                            pixel_weighting=self._pixel_weighting,
                                            device=self.device)

        v = pmd_no_denoiser.v.cpu()
        trained_model, _ = masknmf.compression.denoising.train_total_variance_denoiser(v,
                                                                                       max_epochs=self.num_epochs,
                                                                                       batch_size=128,
                                                                                       learning_rate=1e-4)

        curr_temporal_denoiser = masknmf.compression.PMDTemporalDenoiser(trained_model, self.noise_variance_quantile)

        self._results = masknmf.compression.pmd_decomposition(self.dataset,
                                                             self.block_sizes,
                                                             frame_range=self.frame_range,
                                                             max_components=self.max_components,
                                                             sim_conf=self._sim_conf,
                                                             frame_batch_size=self.frame_batch_size,
                                                             max_consecutive_failures=self._max_consecutive_failures,
                                                             spatial_avg_factor=self.spatial_avg_factor,
                                                             temporal_avg_factor=self.temporal_avg_factor,
                                                             compute_normalizer=self._compute_normalizer,
                                                             pixel_weighting=self._pixel_weighting,
                                                             device=self.device,
                                                             temporal_denoiser=curr_temporal_denoiser)

        return self._results