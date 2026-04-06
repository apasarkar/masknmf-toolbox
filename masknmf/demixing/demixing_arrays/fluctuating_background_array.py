from typing import *
import numpy as np
from masknmf.arrays.array_interfaces import ArrayLike, TensorFlyWeight
import torch
from masknmf.demixing.demixing_arrays.demixing_array_utils import check_spatial_crop_effect

class FluctuatingBackgroundArray(ArrayLike):
    """
    Factorized video for the spatial and temporal extracted sources from the data

    """

    def __init__(
        self,
        fov_shape: Tuple[int, int],
        flyweight: TensorFlyWeight,
        rescale: bool = False,
    ):
        """
        See from_tensors for parameter documentation
        """
        self._flyweight = flyweight
        self.flyweight.validate_attributes(['u', 'factorized_bkgd_term1', 'factorized_bkgd_term2'])
        t = self.factorized_bkgd_term2.shape[1]
        self._shape = (t,) + fov_shape
        self._pixel_mat = torch.arange(self.shape[1] * self.shape[2], device=self.device, dtype=torch.long).reshape(self.shape[1], self.shape[2])

        self._default_normalizer = torch.ones(self.shape[1], self.shape[2], device=self.device).float()
        if hasattr(self.flyweight, "normalizer"):
            if self.flyweight.normalizer.shape[0] != self.shape[1] or self.flyweight.normalizer.shape[1] != self.shape[2]:
                raise ValueError("Normalizer from flyweight had dimensions not equal to the fov dimensions")

        self._rescale = rescale

    @classmethod
    def from_tensors(cls,
                     fov_shape: tuple[int, int],
                     u: torch.sparse_coo_tensor,
                     factorized_bkgd_term1: torch.Tensor,
                     factorized_bkgd_term2: torch.Tensor,
                     normalizer: Optional[torch.Tensor],
                     rescale: bool = False
                     ):
        """
        The background movie can be factorized as the matrix product Uab,
        where u, and v are the standard matrices from the pmd decomposition,
        Args:
            fov_shape (tuple): (fov_dim1, fov_dim2)
            u (torch.sparse_coo_tensor): shape (pixels, rank1)
            factorized_bkgd_term1 (torch.Tensor): shape (PMD rank, background_rank)
            factorized_bkgd_term2 (torch.Tensor): shape (background_rank, num_frames)
            normalizer (Optional[torch.Tensor]): Demixing is performed in a normalized space; this tensor of shape (height, width) specifies pixel-wise normalization
            rescale (bool): Whether or not to rescale the data to the original data space (i.e. multiply pixelwise by the normalizer)
        """
        flyweight = TensorFlyWeight(u=u,
                                    factorized_bkgd_term1=factorized_bkgd_term1,
                                    factorized_bkgd_term2=factorized_bkgd_term2,
                                    normalizer=normalizer)
        return cls(fov_shape,
                   flyweight,
                   rescale=rescale)

    @classmethod
    def from_flyweight(cls,
                       fov_shape: tuple[int, int],
                       flyweight: TensorFlyWeight,
                       rescale: bool = False
                       ):
        return cls(fov_shape,
                   flyweight,
                   rescale=rescale)

    @property
    def flyweight(self) -> TensorFlyWeight:
        return self._flyweight

    @property
    def u(self) -> torch.sparse_coo_tensor:
        return self.flyweight.u

    @property
    def factorized_bkgd_term1(self) -> torch.Tensor:
        return self.flyweight.factorized_bkgd_term1

    @property
    def factorized_bkgd_term2(self) -> torch.Tensor:
        return self.flyweight.factorized_bkgd_term2

    @property
    def normalizer(self) -> torch.Tensor:
        if not hasattr(self.flyweight, "normalizer"):
            return self._default_normalizer

    @property
    def device(self):
        return self.flyweight.device

    def to(self, new_device: str):
        self._flyweight.to(new_device)
        self._pixel_mat.to(new_device)
        self._default_normalizer.to(new_device)

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


    @property
    def rescale(self):
        return self._rescale

    @rescale.setter
    def rescale(self, new_value: bool):
        self._rescale = new_value

    def getitem_tensor(
        self,
        item: Union[int, list, np.ndarray, Tuple[Union[int, np.ndarray, slice, range]]],
    ):
        frame_indexer, item = self._parse_indices(item)

        # Step 3: Now slice the data with frame_indexer (careful: if the ndims has shrunk, add a dim)
        factorized_bkgd_term2_crop = self.factorized_bkgd_term2[:, frame_indexer]
        if factorized_bkgd_term2_crop.ndim < self.factorized_bkgd_term2.ndim:
            factorized_bkgd_term2_crop = factorized_bkgd_term2_crop.unsqueeze(1)

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

            pixel_space_crop = self._pixel_mat[spatial_crop_terms]
            u_indices = pixel_space_crop.flatten()
            u_crop = torch.index_select(self.u, 0, u_indices)
            implied_fov = pixel_space_crop.shape

        else:
            u_crop = self.u
            implied_fov = self.shape[1], self.shape[2]

        # Temporal term is guaranteed to have nonzero "T" dimension below
        if np.prod(implied_fov) <= factorized_bkgd_term2_crop.shape[1]:
            product = torch.sparse.mm(u_crop, self.factorized_bkgd_term1)
            product = torch.matmul(product, factorized_bkgd_term2_crop)

        else:
            product = torch.matmul(self.factorized_bkgd_term1, factorized_bkgd_term2_crop)
            product = torch.sparse.mm(u_crop, product)


        product = product.reshape((implied_fov[0], implied_fov[1], -1))
        product = product.permute(-1, *range(product.ndim - 1))

        if self.rescale:
            product *= self.normalizer[None, :, :]

        return product

    def __getitem__(
        self,
        item: Union[int, list, np.ndarray, Tuple[Union[int, np.ndarray, slice, range]]],
    ) -> np.ndarray:
        product = self.getitem_tensor(item)
        product = product.cpu().numpy().astype(self.dtype)
        return product
