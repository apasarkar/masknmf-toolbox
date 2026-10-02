import torch
import numpy as np
from typing import Optional
from masknmf.utils import SparseCOOTensor
class InitializationResults:

    def __init__(self,
                 spatial_demixed: SparseCOOTensor,
                 spatial_demixed_masks: SparseCOOTensor,
                 temporal_demixed: torch.Tensor,
                 b: torch.Tensor,
                 corr_image: np.ndarray | None = None,
                 nmf_seed_map: np.ndarray | None = None,
                 pure_nmf_seed_map: np.ndarray | None = None):
        """
        Args:
            spatial_demixed (torch.sparse_coo_tensor): Shape (num_pixels, num_neurons)
            spatial_demixed_masks (torch.sparse_coo_tensor): Shape (num_pixels, num_neurons)
            temporal_demixed (torch.Tensor): Shape (num_frames, num_neurons). The demixed temporal traces,
            b (torch.Tensor): Shape (num_pixels, 1)
            corr_image (np.ndarray | None): (fov_height, fov_width)-shaped grayscale correlation image
            nmf_seed_map (Optional[np.ndarray]): (fov_height,fov_width)-shaped binary map describing NMF seed locations
            pure_nmf_seed_map (Optional[np.ndarray]): (fov_height, fov_width)-shaped binary map describing "pure" NMF seed locations
        """
        self._spatial_demixed = spatial_demixed
        self._spatial_demixed_masks = spatial_demixed_masks
        self._temporal_demixed = temporal_demixed
        self._b = b
        self._corr_image = corr_image
        self._nmf_seed_map = nmf_seed_map
        self._pure_nmf_seed_map = pure_nmf_seed_map


    @property
    def spatial_demixed(self) -> SparseCOOTensor:
        return self._spatial_demixed

    @property
    def spatial_demixed_masks(self) -> SparseCOOTensor:
        return self._spatial_demixed_masks

    @property
    def temporal_demixed(self) -> torch.tensor:
        return self._temporal_demixed

    @property
    def b(self) -> torch.Tensor:
        return self._b

    @property
    def corr_image(self) -> np.ndarray | None:
        return self._corr_image

    @property
    def nmf_seed_map(self) -> np.ndarray | None:
        return self._nmf_seed_map

    @property
    def pure_nmf_seed_map(self) -> np.ndarray | None:
        return self._pure_nmf_seed_map



