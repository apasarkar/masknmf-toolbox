from typing import *
import numpy as np

from masknmf import TensorFlyWeight
from masknmf.arrays.array_interfaces import ArrayLike, TensorFlyWeight
import torch
from masknmf.demixing.demixing_arrays.demixing_array_utils import check_spatial_crop_effect


class StandardCorrelationImages(ArrayLike):

    def __init__(self,
                 flyweight: TensorFlyWeight,
                 fov_dims: tuple[int, int]):
        """
        see from_tensor docs for detailed parameter information
        """

        self._flyweight = flyweight
        self.flyweight.validate_attributes(['u', 'v', 'c', 'movie_mean', 'movie_normalizer'])

        ## Caution: This tensor is "settable" (when you update self._c the correlation image dynamically changes). So the getter for c should just call flyweight.c
        self._c = None
        self.c = flyweight.c
        self._pixel_mat = torch.arange(np.prod(self.shape[1:3]), device=self.device, dtype=torch.long).reshape(
            self.shape[1], self.shape[2])
        self._fov_dims = fov_dims
        self._ones_frames = torch.ones(
            (1, self.v.shape[1]), device=self.device, dtype=torch.float
        )



    @classmethod
    def from_tensors(
        cls,
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
            fov_dims (tuple[int, int]): A (height, width) tuple describing field of view (fov) dimensions
        """
        flyweight = TensorFlyWeight(u=u_sparse,
                                    v=v,
                                    c=c,
                                    movie_mean=movie_mean,
                                    movie_normalizer=movie_normalizer)
        return cls(flyweight,
                   fov_dims)

    @classmethod
    def from_flyweight(cls,
                       flyweight: TensorFlyWeight,
                       fov_dims: tuple[int, int]):
        return cls(flyweight,
            fov_dims)

    @property
    def flyweight(self) -> TensorFlyWeight:
        return self._flyweight

    @property
    def device(self) -> str:
        """
        This specifies what device the internal tensors used for the lazy computations are located.
        """
        return self._flyweight.device

    def to(self, new_device: str):
        self.flyweight.to(new_device)
        self._ones_frames = self._ones_frames.to(new_device)
        self._pixel_mat = self._pixel_mat.to(new_device)
        self._c = self._c.to(new_device)

    @property
    def c(self) -> torch.Tensor:
        return self._c

    @c.setter
    def c(self, new_tensor: torch.Tensor):
        if new_tensor.shape[0] != self.v.shape[1]:
            raise ValueError(
                f"Input temporal trace matrix has {new_tensor.shape[0]} frames"
                f"which is incompatible with the movie, which has {self._v.shape[1]} frames"
            )
        mean_zero = new_tensor - torch.mean(new_tensor, dim=0, keepdim=True)
        mean_zero /= torch.linalg.norm(mean_zero, dim=0, keepdim=True)
        mean_zero = torch.nan_to_num(mean_zero, nan=0.0, posinf=0.0, neginf=0.0)
        self._c = mean_zero

    @property
    def u(self) -> torch.sprase_coo_tensor:
        return self.flyweight.u

    @property
    def v(self) -> torch.Tensor:
        return self.flyweight.v


    @property
    def movie_mean(self) -> torch.Tensor:
        return self.flyweight.movie_mean

    @property
    def movie_normalizer(self) -> torch.Tensor:
        return self.flyweight.movie_normalizer

    @property
    def shape(self) -> Tuple[int, int, int]:
        return self.c.shape[1], self._fov_dims[0], self._fov_dims[1]


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

        v_crop = self.v @ c_crop
        ones_crop = self._ones_frames @ c_crop

        # Step 4: Deal with remaining indices after lazy computing the frame(s)
        if isinstance(item, tuple) and check_spatial_crop_effect(
            item[1:], self.shape[1:]
        ):
            pixel_space_crop = self._pixel_mat[item[1:]]
            u_indices = pixel_space_crop.flatten()
            u_crop = torch.index_select(self.u, 0, u_indices)
            mean_crop = torch.index_select(self.movie_mean, 0, u_indices)
            movie_normalizer_crop = torch.index_select(
                self.movie_normalizer, 0, u_indices
            )
            implied_fov = pixel_space_crop.shape
        else:
            u_crop = self.u
            mean_crop = self.movie_mean
            movie_normalizer_crop = self.movie_normalizer
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