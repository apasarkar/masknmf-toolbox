from typing import *
import numpy as np
from masknmf.arrays.array_interfaces import FactorizedVideo
import torch
from masknmf.demixing.demixing_arrays.demixing_array_utils import check_spatial_crop_effect


class StandardCorrelationImages(FactorizedVideo):
    def __init__(
        self,
        u_sparse: torch.sparse_coo_tensor,
        v: torch.Tensor,
        c: torch.Tensor,
        movie_mean: torch.Tensor,
        movie_normalizer: torch.Tensor,
        fov_dims: Tuple[int, int],
    ):
        """
        Generates all the standard correlation images for the demixed data. It is more convenient to keep the
        correlation images in a factorized form and

        Args:
            u_sparse (torch.sparse_coo_tensor): shape (pixels, rank)
            v (torch.Tensor): shape (rank, frames)
            c (torch.Tensor): shape (frames, number of neural signals). This is the temporal traces matrix, where every
                column has mean 0 and Frobenius norm 1.
            movie_mean (torch.tensor): shape (pixels), the mean of u_sparse times v
            movie_normalizer (torch.Tensor): shape (pixels), the pixelwise l2 norm of (u_sparse times v) - movie_mean
        """

        if not (u_sparse.device == v.device == c.device):
            raise ValueError("Not all tensors are on same device")

        self._device = u_sparse.device
        self._u = u_sparse
        self._v = v
        self._c = None
        self.c = c # All temporal data goes through the setter to get normalized
        self._movie_mean = movie_mean
        self._movie_normalizer = movie_normalizer
        self._fov_dims = (fov_dims[0], fov_dims[1])

        self.pixel_mat = torch.arange(np.prod(self.shape[1:3]), device=self.device, dtype=torch.long).reshape(
            self.shape[1], self.shape[2])

        self._ones_frames = torch.ones(
            (1, self._v.shape[1]), device=self.device, dtype=torch.float
        )


    @property
    def device(self) -> str:
        """
        This specifies what device the internal tensors used for the lazy computations are located.
        """
        return self._device

    @property
    def c(self) -> torch.tensor:
        return self._c

    @c.setter
    def c(self, new_tensor):
        if new_tensor.shape[0] != self._v.shape[1]:
            raise ValueError(
                f"Input temporal trace matrix has {new_tensor.shape[0]} frames"
                f"which is incompatible with the movie, which has {self._v.shape[1]} frames"
            )
        mean_zero = new_tensor - torch.mean(new_tensor, dim=0, keepdim=True)
        mean_zero /= torch.linalg.norm(mean_zero, dim=0, keepdim=True)
        mean_zero = torch.nan_to_num(mean_zero, nan=0.0, posinf=0.0, neginf=0.0)
        self._c = mean_zero

    @property
    def shape(self) -> Tuple[int, int, int]:
        return self.c.shape[1], self._fov_dims[0], self._fov_dims[1]

    @property
    def movie_mean(self) -> torch.Tensor:
        return self._movie_mean

    @property
    def movie_normalizer(self) -> torch.Tensor:
        return self._movie_normalizer

    @property
    def ndim(self) -> int:
        return len(self.shape)

    @property
    def dtype(self):
        return np.float32

    def getitem_tensor(
        self,
        item: Union[int, list, np.ndarray, Tuple[Union[int, np.ndarray, slice, range]]],
    ) -> torch.tensor:
        frame_indexer, item = self._parse_indices(item)

        # Step 3: Now slice the data with frame_indexer (careful: if the ndims has shrunk, add a dim)
        c_crop = self._c[:, frame_indexer]
        if c_crop.ndim < self._c.ndim:
            c_crop = c_crop.unsqueeze(1)

        v_crop = self._v @ c_crop
        ones_crop = self._ones_frames @ c_crop

        # Step 4: Deal with remaining indices after lazy computing the frame(s)
        if isinstance(item, tuple) and check_spatial_crop_effect(
            item[1:], self.shape[1:]
        ):
            pixel_space_crop = self.pixel_mat[item[1:]]
            u_indices = pixel_space_crop.flatten()
            u_crop = torch.index_select(self._u, 0, u_indices)
            mean_crop = torch.index_select(self._movie_mean, 0, u_indices)
            movie_normalizer_crop = torch.index_select(
                self._movie_normalizer, 0, u_indices
            )
            implied_fov = pixel_space_crop.shape
        else:
            u_crop = self._u
            mean_crop = self._movie_mean
            movie_normalizer_crop = self._movie_normalizer
            implied_fov = self.shape[1], self.shape[2]

        product = (
            torch.sparse.mm(u_crop, v_crop) - mean_crop.unsqueeze(1) @ ones_crop
        ) / movie_normalizer_crop.unsqueeze(1)


        product = product.reshape((implied_fov[0], implied_fov[1], -1))
        product = product.permute(2, 0, 1)

        return torch.nan_to_num(product, nan=0.0, posinf=0.0, neginf=0.0)

    def __getitem__(
        self,
        item: Union[int, list, np.ndarray, Tuple[Union[int, np.ndarray, slice, range]]],
    ) -> np.ndarray:
        product = self.getitem_tensor(item)
        product = product.cpu().numpy().astype(self.dtype)
        return product