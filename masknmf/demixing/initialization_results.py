import torch
import numpy as np
from typing import Optional


class InitializationResults:

    def __init__(self,
                 a: torch.sparse_coo_tensor,
                 mask_a: torch.sparse_coo_tensor,
                 c: torch.tensor,
                 b: torch.tensor,
                 correlation_img: Optional[np.ndarray] = None,
                 nmf_seed_map: Optional[np.ndarray] = None,
                 pure_nmf_seed_map: Optional[np.ndarray] = None):
        """
        Args:
            a (torch.sparse_coo_tensor): Shape (num_pixels, num_neurons)
            mask_a (torch.sparse_coo_tensor): Shape (num_pixels, num_neurons)
            c (torch.tensor): Shape (num_frames, num_neurons)
            b (torch.tensor): Shape (num_pixels, 1)
            correlation_img (Optional[np.ndarray]): (height, width)-shaped grayscale correlation image
            nmf_seed_map (Optional[np.ndarray]): (height,width)-shaped binary map describing NMF seed locations
            pure_nmf_seed_map (Optional[np.ndarray]): (height, width)-shaped binary map describing "pure" NMF seed locations
        """
        self._a = a
        self._mask_a = mask_a
        self._c = c
        self._b = b
        self._correlation_img = correlation_img
        self._nmf_seed_map = nmf_seed_map
        self._pure_nmf_seed_map = pure_nmf_seed_map


    @property
    def a(self) -> torch.sparse_coo_tensor:
        return self._a

    @property
    def mask_a(self) -> torch.sparse_coo_tensor:
        return self._mask_a

    @property
    def c(self) -> torch.tensor:
        return self._c

    @property
    def b(self) -> torch.tensor:
        return self._b

    @property
    def correlation_img(self) -> np.ndarray | None:
        return self._correlation_img

    @property
    def nmf_seed_map(self) -> np.ndarray | None:
        return self._nmf_seed_map

    @property
    def pure_nmf_seed_map(self) -> np.ndarray | None:
        return self._pure_nmf_seed_map



