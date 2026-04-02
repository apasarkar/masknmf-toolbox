from typing import *
import numpy as np
from masknmf.arrays.array_interfaces import FactorizedVideo
import torch
from masknmf.demixing.demixing_arrays.demixing_array_utils import check_spatial_crop_effect

class FluctuatingBackgroundArray(FactorizedVideo):
    """
    Factorized video for the spatial and temporal extracted sources from the data

    "a" is the matrix of spatial profiles
    "c" is the matrix of temporal profiles
    """
    DATA_ARRAYS = ["u", "a", "b"]

    def __init__(
        self,
        fov_shape: Tuple[int, int],
        u: torch.sparse_coo_tensor,
        a: torch.Tensor,
        b: torch.Tensor,
    ):
        """
        The background movie can be factorized as the matrix product Uab,
        where u, and v are the standard matrices from the pmd decomposition,
        Args:
            fov_shape (tuple): (fov_dim1, fov_dim2)
            u (torch.sparse_coo_tensor): shape (pixels, rank1)
            a (torch.Tensor): shape (PMD rank, background_rank)
            b (torch.Tensor): shape (background_rank, num_frames)
        """
        t = b.shape[1]
        self._shape = (t,) + fov_shape

        self._u = u
        self._b = b
        self._a = a

        self.pixel_mat = torch.arange(np.prod(self.shape[1:]), device=self.device, dtype=torch.long).reshape(self.shape[1], self.shape[2])


    def _find_common_device(self):
        """
        Finds the common device that for all data tensors. Throws error if no such device exists
        """
        device=None
        for i, name in enumerate(DATA_ARRAYS):
            arr = getattr(self, name)
            if i == 0:
                device = arr.device
            else:
                if not arr.device == device:
                    raise ValueError("Not all tensors in fluctuating background array are on same device")
        return device

    @property
    def device(self) -> str:
        return self._find_common_device()

    @property
    def u(self) -> torch.sparse_coo_tensor:
        return self._u

    @property
    def a(self) -> torch.Tensor:
        return self._a

    @property
    def b(self) -> torch.Tensor:
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

        else:
            u_crop = self._u
            implied_fov = self.shape[1], self.shape[2]

        # Temporal term is guaranteed to have nonzero "T" dimension below
        if np.prod(implied_fov) <= b_crop.shape[1]:
            product = torch.sparse.mm(u_crop, self.a)
            product = torch.matmul(product, b_crop)

        else:
            product = torch.matmul(self.a, b_crop)
            product = torch.sparse.mm(u_crop, product)


        product = product.reshape((implied_fov[0], implied_fov[1], -1))
        product = product.permute(-1, *range(product.ndim - 1))

        return product

    def __getitem__(
        self,
        item: Union[int, list, np.ndarray, Tuple[Union[int, np.ndarray, slice, range]]],
    ) -> np.ndarray:
        product = self.getitem_tensor(item)
        product = product.cpu().numpy().astype(self.dtype)
        return product
