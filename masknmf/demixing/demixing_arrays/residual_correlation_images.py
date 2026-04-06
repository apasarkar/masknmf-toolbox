from enum import Enum
from typing import *
import numpy as np
from masknmf.arrays.array_interfaces import ArrayLike, TensorFlyWeight
import torch
from masknmf.demixing.demixing_arrays.demixing_array_utils import check_spatial_crop_effect

class ResidCorrMode(Enum):
    DEFAULT = 0
    MASKED = 1
    RESIDUAL = 2


class ResidualCorrelationImages(ArrayLike):

    def __init__(self,
                 flyweight: TensorFlyWeight,
                 fov_dims: tuple[int, int],
                 mode: ResidCorrMode = ResidCorrMode.DEFAULT):
        """
        See from_tensors for parameter documentation
        """

        self._flyweight = flyweight
        self.flyweight.validate_attributes(["u",
                                            "v",
                                            "factorized_bkgd_term1",
                                            "factorized_bkgd_term2",
                                            "a",
                                            "c",
                                            "resid_corr_img_support_values",
                                            "resid_corr_img_mean",
                                            "resid_corr_img_normalizer"])
        self._c_norm = self.c - torch.mean(self.c, dim=0, keepdim=True)
        self._c_norm = self._c_norm / torch.linalg.norm(
            self._c_norm, dim=0, keepdim=True
        )
        self._c_norm = torch.nan_to_num(self._c_norm, nan=0.0)
        self._fov_dims = (fov_dims[0], fov_dims[1])
        self._index_values = torch.arange(self.c.shape[1], device=self.device).long()

        self._mode = mode

        self._ones_basis = (
            torch.ones([1, self.v.shape[1]], device=self.device) @ self.v.T
        )
        self._pixel_mat = torch.arange(self.shape[1] * self.shape[2], device=self.device, dtype=torch.long).reshape(
            self.shape[1], self.shape[2])

    @classmethod
    def from_tensors(
        cls,
        u_sparse: torch.sparse_coo_tensor,
        v: torch.Tensor,
        factorized_bkgd_term1: torch.Tensor,
        factorized_bkgd_term2: torch.Tensor,
        a: torch.sparse_coo_tensor,
        c: torch.Tensor,
        resid_corr_img_support_values: torch.sparse_coo_tensor,
        resid_corr_img_mean: torch.Tensor,
        resid_corr_img_normalizer: torch.Tensor,
        fov_dims: tuple[int, int],
        mode: ResidCorrMode = ResidCorrMode.DEFAULT,
    ):
        """
        Array interface for interacting with the residual correlation image data. Data is kept in a memory
        efficient factorized form and efficiently expanded on the fly (on GPU or CPU).

        Each neuron has a spatial support (pixels on which its spatial footprint is nonzero). Its residual correlation
        -- for those pixels ONLY -- is stored in support_correlation_values. That has the same level of sparsity as
        "a". For all other pixels in the residual correlation image data are given by the correlation image between
        (URs - AX)V and c.T. This gives us a very memory efficient way to generate corr images without storing the full
        pixels x number of neural signals data.

        Args:
            u_sparse (torch.sparse_coo_tensor): shape (pixels, rank 1)
            v (torch.Tensor): shape (rank 2, frames)
            factorized_bkgd_term1 (torch.Tensor):
            factorized_bkgd_term2 (torch.Tensor):
            a (torch.sparse_coo_tensor): shape (pixels, number of neural signals). Spatial components
            c (torch.tensor): shape (frames, number of neural signals). This is the temporal traces matrix
            resid_corr_img_support_values (torch.sparse_coo_tensor): Shape (pixels, number of neural signals). The i-th
                gives the residual correlation image for neural signal "i" on its spatial support.
            resid_corr_img_mean (torch.Tensor): shape (pixels)
            resid_corr_img_normalizer (torch.Tensor): shape (pixels)
            fov_dims (tuple): A tuple of two values describing the field height/width of the field of view.
            mode (ResidCorrMode): The mode of the residual correlation image
        """
        flyweight = TensorFlyWeight(u=u_sparse,
                                    v=v,
                                    factorized_bkgd_term1=factorized_bkgd_term1,
                                    factorized_bkgd_term2=factorized_bkgd_term2,
                                    a=a,
                                    c=c,
                                    resid_corr_img_support_values=resid_corr_img_support_values,
                                    resid_corr_img_mean=resid_corr_img_mean,
                                    resid_corr_img_normalizer=resid_corr_img_normalizer,
                                    )

        return cls(flyweight,
                   fov_dims,
                   mode=mode)

    @classmethod
    def from_flyweight(cls,
                       flyweight: TensorFlyWeight,
                       fov_dims: tuple[int, int],
                       mode: ResidCorrMode = ResidCorrMode.DEFAULT):
        return cls(flyweight,
                   fov_dims,
                   mode=mode)


    @property
    def flyweight(self) -> TensorFlyWeight:
        return self._flyweight

    @property
    def device(self) -> str:
        return self.flyweight.device

    def to(self, new_device: str):
        if self._flyweight.device != new_device:
            self._flyweight.to(new_device)
        self._move_local_tensors(new_device)

    def _move_local_tensors(self, new_device: str):
        self._index_values = self._index_values.to(new_device)
        self._pixel_mat = self._pixel_mat.to(new_device)
        self._ones_basis = self._ones_basis.to(new_device)
        self._c_norm = self._c_norm.to(new_device)

    @property
    def mode(self) -> ResidCorrMode:
        """
        Sometimes we want to view slightly modified versions of this correlation image. Some examples:
            - We want to zero out pixels belonging to the support of each neuron (ResidCorrMode.MASKED)
            - We want to view the correlation between the i-th temporal component and the full resid movie (
                as opposed to the i-th correlation image). In this case we use ResidCorrMode.RESIDUAL
            - We want the i-th residual correlation image; we use ResidCorrMode.DEFAULT
        """
        return self._mode

    @mode.setter
    def mode(self, new_mode: ResidCorrMode):
        self._mode = new_mode

    @property
    def device(self) -> str:
        """
        This specifies what device the internal tensors used for the lazy computations are located.
        """
        return self.flyweight.device

    @property
    def shape(self) -> tuple[int, int, int]:
        return self.c.shape[1], self._fov_dims[0], self._fov_dims[1]

    @property
    def u(self) -> torch.sparse_coo_tensor:
        return self.flyweight.u

    @property
    def v(self) -> torch.Tensor:
        return self.flyweight.v


    @property
    def a(self) -> torch.sparse_coo_tensor:
        return self.flyweight.a

    @property
    def c(self) -> torch.Tensor:
        return self.flyweight.c

    @property
    def resid_corr_img_support_values(self) -> torch.sparse_coo_tensor:
        return self.flyweight.resid_corr_img_support_values

    @property
    def resid_corr_img_mean(self) -> torch.Tensor:
        return self.flyweight.resid_corr_img_mean

    @property
    def resid_corr_img_normalizer(self) -> torch.Tensor:
        return self.flyweight.resid_corr_img_normalizer

    @property
    def factorized_bkgd_term1(self) -> torch.Tensor:
        return self.flyweight.factorized_bkgd_term1

    @property
    def factorized_bkgd_term2(self) -> torch.Tensor:
        return self.flyweight.factorized_bkgd_term2

    @property
    def ndim(self) -> int:
        return len(self.shape)

    @property
    def dtype(self):
        return np.float32

    def getitem_tensor(
        self,
        item: Union[int, list, np.ndarray, Tuple[Union[int, np.ndarray, slice, range]]],
    ) -> torch.Tensor:
        frame_indexer, item = self._parse_indices(item)

        # Step 3: Now slice the data with frame_indexer (careful: if the ndims has shrunk, add a dim)
        c_crop = self._c_norm[:, frame_indexer]
        if c_crop.ndim < self._c_norm.ndim:
            c_crop = c_crop.unsqueeze(1)

        v_crop = self.v @ c_crop - (self.factorized_bkgd_term1 @ (self.factorized_bkgd_term2 @ c_crop))
        cc_crop = self.c.T @ c_crop
        selected_neurons = self._index_values[frame_indexer]
        if selected_neurons.ndim < 1:
            selected_neurons = selected_neurons.unsqueeze(0)
        support_values_crop = torch.index_select(
            self.resid_corr_img_support_values, 1, selected_neurons
        ).coalesce()

        # Step 4: Deal with remaining indices after lazy computing the frame(s)
        if isinstance(item, tuple) and check_spatial_crop_effect(
            item[1:], self.shape[1:]
        ):
            pixel_space_crop = self._pixel_mat[item[1:]]
            u_indices = pixel_space_crop.flatten()
            u_crop = torch.index_select(self.u, 0, u_indices)
            a_crop = torch.index_select(self.a, 0, u_indices)
            support_values_crop = torch.index_select(
                support_values_crop, 0, u_indices
            ).coalesce()
            mean_crop = torch.index_select(self.resid_corr_img_mean, 0, u_indices)
            movie_normalizer_crop = torch.index_select(
                self.resid_corr_img_normalizer, 0, u_indices
            )
            implied_fov = pixel_space_crop.shape
        else:
            u_crop = self.u
            a_crop = self.a
            mean_crop = self.resid_corr_img_mean
            movie_normalizer_crop = self.resid_corr_img_normalizer
            implied_fov = self.shape[1], self.shape[2]

        # Temporal term is guaranteed to have nonzero "T" dimension below
        ## TODO: If you only had 2 matrices in the factorization, this if/else is useless. But eventually background term will be its own factorization. So keep this for now.
        if np.prod(implied_fov) <= v_crop.shape[1]:
            product = torch.sparse.mm(u_crop, v_crop)
            product -= mean_crop.unsqueeze(1) @ torch.sum(c_crop, dim=0, keepdim=True)
            product -= torch.sparse.mm(a_crop, cc_crop)
            product /= movie_normalizer_crop.unsqueeze(1)

        else:
            product = torch.sparse.mm(u_crop, v_crop)
            product -= torch.sparse.mm(a_crop, cc_crop)
            product -= mean_crop.unsqueeze(1) @ torch.sum(c_crop, dim=0, keepdim=True)

            product /= movie_normalizer_crop.unsqueeze(1)

        rows, cols = support_values_crop.indices()
        values = support_values_crop.values()
        if self.mode == ResidCorrMode.DEFAULT:
            product[(rows, cols)] = values
        elif self.mode == ResidCorrMode.MASKED:
            product[(rows, cols)] = 0
        elif self.mode == ResidCorrMode.RESIDUAL:
            pass

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
