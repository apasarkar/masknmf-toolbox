from enum import Enum
import numpy as np
from masknmf.utils import SparseCOOTensor
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
        self.flyweight.validate_attributes(["spatial_compressed",
                                            "temporal_compressed",
                                            "factorized_background_term1",
                                            "factorized_background_term2",
                                            "spatial_demixed",
                                            "temporal_demixed",
                                            "resid_corr_img_support_values",
                                            "resid_corr_img_mean",
                                            "resid_corr_img_normalizer"])
        self._temporal_demixed_norm = self.temporal_demixed - torch.mean(self.temporal_demixed, dim=0, keepdim=True)
        self._temporal_demixed_norm = self._temporal_demixed_norm / torch.linalg.norm(
            self._temporal_demixed_norm, dim=0, keepdim=True
        )
        self._temporal_demixed_norm = torch.nan_to_num(self._temporal_demixed_norm, nan=0.0)
        self._fov_dims = (fov_dims[0], fov_dims[1])
        self._index_values = torch.arange(self.temporal_demixed.shape[1], device=self.device).long()

        self._mode = mode

        self._ones_basis = (
                torch.ones([1, self.temporal_compressed.shape[1]], device=self.device) @ self.temporal_compressed.T
        )
        self._pixel_mat = torch.arange(self.shape[1] * self.shape[2], device=self.device, dtype=torch.long).reshape(
            self.shape[1], self.shape[2])

    @classmethod
    def from_tensors(
        cls,
        spatial_compressed: SparseCOOTensor,
        temporal_compressed: torch.Tensor,
        factorized_background_term1: torch.Tensor,
        factorized_background_term2: torch.Tensor,
        spatial_demixed: SparseCOOTensor,
        temporal_demixed: torch.Tensor,
        resid_corr_img_support_values: SparseCOOTensor,
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
        (URs - AX)temporal_compressed and c.T. This gives us a very memory efficient way to generate corr images without storing the full
        pixels x number of neural signals data.

        Args:
            spatial_compressed (torch.sparse_coo_tensor): shape (pixels, rank 1)
            temporal_compressed (torch.Tensor): shape (rank 2, frames)
            factorized_background_term1 (torch.Tensor):
            factorized_background_term2 (torch.Tensor):
            spatial_demixed (torch.sparse_coo_tensor): shape (pixels, number of neural signals). Spatial components
            temporal_demixed (torch.Tensor): shape (frames, number of neural signals). This is the temporal traces matrix
            resid_corr_img_support_values (torch.sparse_coo_tensor): Shape (pixels, number of neural signals). The i-th
                gives the residual correlation image for neural signal "i" on its spatial support.
            resid_corr_img_mean (torch.Tensor): shape (pixels)
            resid_corr_img_normalizer (torch.Tensor): shape (pixels)
            fov_dims (tuple): A tuple of two values describing the field height/width of the field of view.
            mode (ResidCorrMode): The mode of the residual correlation image
        """
        flyweight = TensorFlyWeight(spatial_compressed=spatial_compressed,
                                    temporal_compressed=temporal_compressed,
                                    factorized_background_term1=factorized_background_term1,
                                    factorized_background_term2=factorized_background_term2,
                                    spatial_demixed=spatial_demixed,
                                    temporal_demixed=temporal_demixed,
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
        self._temporal_demixed_norm = self._temporal_demixed_norm.to(new_device)

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
        return self.temporal_demixed.shape[1], self._fov_dims[0], self._fov_dims[1]

    @property
    def spatial_compressed(self) -> SparseCOOTensor:
        return self.flyweight.spatial_compressed

    @property
    def temporal_compressed(self) -> torch.Tensor:
        return self.flyweight.temporal_compressed

    @property
    def spatial_demixed(self) -> SparseCOOTensor:
        return self.flyweight.spatial_demixed

    @property
    def temporal_demixed(self) -> torch.Tensor:
        return self.flyweight.temporal_demixed

    @property
    def resid_corr_img_support_values(self) -> SparseCOOTensor:
        return self.flyweight.resid_corr_img_support_values

    @property
    def resid_corr_img_mean(self) -> torch.Tensor:
        return self.flyweight.resid_corr_img_mean

    @property
    def resid_corr_img_normalizer(self) -> torch.Tensor:
        return self.flyweight.resid_corr_img_normalizer

    @property
    def factorized_background_term1(self) -> torch.Tensor:
        return self.flyweight.factorized_background_term1

    @property
    def factorized_background_term2(self) -> torch.Tensor:
        return self.flyweight.factorized_background_term2

    @property
    def ndim(self) -> int:
        return len(self.shape)

    @property
    def dtype(self) -> type:
        return np.float32

    def getitem_tensor(
        self,
        item: int | list | np.ndarray | tuple[int | np.ndarray | slice | range],
    ) -> torch.Tensor:
        frame_indexer, item = self._parse_indices(item)

        # Step 3: Now slice the data with frame_indexer (careful: if the ndims has shrunk, add a dim)
        temporal_demixed_crop = self._temporal_demixed_norm[:, frame_indexer]
        if temporal_demixed_crop.ndim < self._temporal_demixed_norm.ndim:
            temporal_demixed_crop = temporal_demixed_crop.unsqueeze(1)

        temporal_compressed_crop = self.temporal_compressed @ temporal_demixed_crop - (self.factorized_background_term1 @ (self.factorized_background_term2 @ temporal_demixed_crop))
        cc_crop = self.temporal_demixed.T @ temporal_demixed_crop
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
            u_crop = torch.index_select(self.spatial_compressed, 0, u_indices)
            a_crop = torch.index_select(self.spatial_demixed, 0, u_indices)
            support_values_crop = torch.index_select(
                support_values_crop, 0, u_indices
            ).coalesce()
            mean_crop = torch.index_select(self.resid_corr_img_mean, 0, u_indices)
            movie_normalizer_crop = torch.index_select(
                self.resid_corr_img_normalizer, 0, u_indices
            )
            implied_fov = pixel_space_crop.shape
        else:
            u_crop = self.spatial_compressed
            a_crop = self.spatial_demixed
            mean_crop = self.resid_corr_img_mean
            movie_normalizer_crop = self.resid_corr_img_normalizer
            implied_fov = self.shape[1], self.shape[2]

        # Temporal term is guaranteed to have nonzero "T" dimension below
        ## TODO: If you only had 2 matrices in the factorization, this if/else is useless. But eventually background term will be its own factorization. So keep this for now.
        if np.prod(implied_fov) <= temporal_compressed_crop.shape[1]:
            product = torch.sparse.mm(u_crop, temporal_compressed_crop)
            product -= mean_crop.unsqueeze(1) @ torch.sum(temporal_demixed_crop, dim=0, keepdim=True)
            product -= torch.sparse.mm(a_crop, cc_crop)
            product /= movie_normalizer_crop.unsqueeze(1)

        else:
            product = torch.sparse.mm(u_crop, temporal_compressed_crop)
            product -= torch.sparse.mm(a_crop, cc_crop)
            product -= mean_crop.unsqueeze(1) @ torch.sum(temporal_demixed_crop, dim=0, keepdim=True)

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
        item: int | list | np.ndarray | tuple[int | np.ndarray | slice | range],
    ) -> np.ndarray:
        product = self.getitem_tensor(item)
        product = product.cpu().numpy().astype(self.dtype)
        return product
