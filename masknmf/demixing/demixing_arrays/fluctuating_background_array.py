from typing import *

import numpy as np
import torch

from masknmf.arrays.array_interfaces import FactorizedVideo
from masknmf.demixing.demixing_arrays.demixing_array_utils import check_spatial_crop_effect
from masknmf.utils.linalg import sparse_mm_async


class FluctuatingBackgroundArray(FactorizedVideo):
    """
    Factorized video for the spatial and temporal extracted sources from the data

    "a" is the matrix of spatial profiles
    "c" is the matrix of temporal profiles
    """

    def __init__(
        self,
        fov_shape: Tuple[int, int],
        order: str,
        u: torch.sparse_coo_tensor,
        a: torch.tensor,
        b: torch.tensor,
    ):
        """
        The background movie can be factorized as the matrix product Uab,
        where u, and v are the standard matrices from the pmd decomposition,
        Args:
            fov_shape (tuple): (fov_dim1, fov_dim2)
            order (str): Order to reshape arrays from 1D to 2D
            u (torch.sparse_coo_tensor): shape (pixels, rank1)
            a (torch.tensor): shape (PMD rank, background_rank)
            b (torch.tensor): shape (background_rank, num_frames)
        """
        t = b.shape[1]
        self._shape = (t,) + fov_shape

        self._u = u.coalesce()
        self._b = b
        self._a= a

        if not (self.u.device == self.a.device == self.b.device):
            raise ValueError(f"Some input tensors are not on the same device")
        self._device = self.u.device
        self.pixel_mat = np.arange(np.prod(self.shape[1:])).reshape(
            [self.shape[1], self.shape[2]], order=order
        )
        self.pixel_mat = torch.from_numpy(self.pixel_mat).long().to(self.device)
        self._order = order

    @property
    def device(self) -> str:
        return self._device

    @property
    def u(self) -> torch.sparse_coo_tensor:
        return self._u

    @property
    def a(self) -> torch.tensor:
        return self._a

    @property
    def b(self) -> torch.tensor:
        return self._b

    @property
    def dtype(self) -> str:
        """
        data type, default np.float32
        """
        return np.float32

    @property
    def shape(self) -> Tuple[int, int, int]:
        """
        Array shape (n_frames, dims_x, dims_y)
        """
        return self._shape

    @property
    def order(self) -> str:
        """
        The spatial data is "flattened" from 2D into 1D. This specifies the order ("F" for column-major or "C" for row-major) in which reshaping happened.
        """
        return self._order

    @property
    def ndim(self) -> int:
        """
        Number of dimensions
        """
        return len(self.shape)

    def getitem_tensor(
        self,
        item: Union[int, list, np.ndarray, Tuple[Union[int, np.ndarray, slice, range]]],
    ):
        frame_indexer, item = self._parse_indices(item)

        # Step 3: Now slice the data with frame_indexer (careful: if the ndims has shrunk, add a dim)
        b_crop = self.b[:, frame_indexer]
        if b_crop.ndim < self.b.ndim:
            b_crop = b_crop.unsqueeze(1)

        # Step 4: Deal with remaining indices after lazy computing the frame(s)
        if isinstance(item, tuple) and check_spatial_crop_effect(
            item[1:], self.shape[1:]
        ):
            if isinstance(item[1], np.ndarray) and len(item[1]) == 1:
                term_1 = slice(int(item[1]), int(item[1]) + 1)
            elif isinstance(item[1], np.integer):
                term_1 = slice(int(item[1]), int(item[1]) + 1)
            elif isinstance(item[1], int):
                term_1 = slice(item[1], item[1] + 1)
            else:
                term_1 = item[1]

            if isinstance(item[2], np.ndarray) and len(item[2]) == 1:
                term_2 = slice(int(item[2]), int(item[2]) + 1)
            elif isinstance(item[2], np.integer):
                term_2 = slice(int(item[2]), int(item[2]) + 1)
            elif isinstance(item[2], int):
                term_2 = slice(item[2], item[2] + 1)
            else:
                term_2 = item[2]

            spatial_crop_terms = (term_1, term_2)

            pixel_space_crop = self.pixel_mat[spatial_crop_terms]
            u_indices = pixel_space_crop.flatten()
            u_crop = torch.index_select(self._u, 0, u_indices)
            implied_fov = pixel_space_crop.shape
            used_order = "C"  # Torch order here by default is C

        else:
            u_crop = self._u
            implied_fov = self.shape[1], self.shape[2]
            used_order = self.order

        # Temporal term is guaranteed to have nonzero "T" dimension below
        if np.prod(implied_fov) <= b_crop.shape[1]:
            product = sparse_mm_async(u_crop, self.a)
            product = torch.matmul(product, b_crop)

        else:
            product = torch.matmul(self.a, b_crop)
            product = sparse_mm_async(u_crop, product)

        if used_order == "F":
            product = product.T.reshape((-1, implied_fov[1], implied_fov[0]))
            product = product.permute((0, 2, 1))
        else:  # order is "C"
            product = product.reshape((implied_fov[0], implied_fov[1], -1))
            product = product.permute(-1, *range(product.ndim - 1))

        return product

    def __getitem__(
        self,
        item: Union[int, list, np.ndarray, Tuple[Union[int, np.ndarray, slice, range]]],
    ) -> np.ndarray:
        # product = self.getitem_tensor(item)
        # product = product.cpu().numpy().astype(self.dtype)
        return self.getitem_tensor(item)
