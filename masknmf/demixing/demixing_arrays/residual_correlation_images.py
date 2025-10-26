from enum import Enum
from typing import *
import numpy as np
from masknmf.arrays.array_interfaces import FactorizedVideo
import torch
from masknmf.demixing.demixing_arrays.demixing_array_utils import check_spatial_crop_effect

class ResidCorrMode(Enum):
    DEFAULT = 0
    MASKED = 1
    RESIDUAL = 2


class ResidualCorrelationImages(FactorizedVideo):
    def __init__(
        self,
        u_sparse: torch.sparse_coo_tensor,
        v: torch.tensor,
        factorized_ring_term: Tuple[torch.tensor, torch.tensor],
        a: torch.sparse_coo_tensor,
        c: torch.tensor,
        support_correlation_values: torch.sparse_coo_tensor,
        residual_movie_mean: torch.tensor,
        residual_movie_normalizer: torch.tensor,
        fov_dims: Tuple[int, int],
        mode: ResidCorrMode = ResidCorrMode.DEFAULT,
        order: str = "F",
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
            v (torch.tensor): shape (rank 2, frames)
            factorized_ring_term (Tuple[torch.tensor, torch.tensor]): A factorized representation of the data background
            a (torch.sparse_coo_tensor): shape (pixels, number of neural signals). Spatial components
            c (torch.tensor): shape (frames, number of neural signals). This is the temporal traces matrix
            support_correlation_values (torch.sparse_coo_tensor): Shape (pixels, number of neural signals). The i-th
                gives the residual correlation image for neural signal "i" on its spatial support.
            residual_movie_mean (torch.tensor): shape (pixels)
            residual_movie_normalizer (torch.tensor): shape (pixels)
            fov_dims (tuple): A tuple of two values describing the field height/width of the field of view.
            zero_support Optional[bool[: If true, for each neuron, i, the support of neuron i is set to 0 in the i-th
                correlation image
        """

        if not (
            u_sparse.device
            == v.device
            == c.device
            == a.device
            == factorized_ring_term[0].device
            == factorized_ring_term[1].device
            == support_correlation_values.device
            == residual_movie_mean.device
            == residual_movie_normalizer.device
        ):
            raise ValueError("Not all tensors are on same device")

        self._device = u_sparse.device
        self._u = u_sparse
        self._v = v
        self._background_term = factorized_ring_term
        self._c = c
        self._c_norm = self._c - torch.mean(self._c, dim=0, keepdim=True)
        self._c_norm = self._c_norm / torch.linalg.norm(
            self._c_norm, dim=0, keepdim=True
        )
        self._c_norm = torch.nan_to_num(self._c_norm, nan=0.0)

        self._a = a
        self._residual_movie_mean = residual_movie_mean
        self._support_correlation_values = support_correlation_values
        self._residual_movie_normalizer = residual_movie_normalizer
        self._fov_dims = (fov_dims[0], fov_dims[1])
        self._index_values = torch.arange(self._c.shape[1], device=self.device).long()
        self._order = order

        self._mode = mode

        self._ones_basis = (
            torch.ones([1, self._v.shape[1]], device=self.device) @ self._v.T
        )

        self.pixel_mat = np.arange(np.prod(self.shape[1:])).reshape(
            [self.shape[1], self.shape[2]], order=order
        )
        self.pixel_mat = torch.from_numpy(self.pixel_mat).long().to(self.device)

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
        return self._device

    @property
    def shape(self) -> Tuple[int, int, int]:
        return self._c.shape[1], self._fov_dims[0], self._fov_dims[1]

    @property
    def support_correlation_values(self) -> torch.sparse_coo_tensor:
        return self._support_correlation_values

    @property
    def residual_movie_mean(self) -> torch.tensor:
        return self._residual_movie_mean

    @property
    def residual_movie_normalizer(self) -> torch.tensor:
        return self._residual_movie_normalizer

    @property
    def order(self) -> str:
        return self._order

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
        # Step 1: index the frames (dimension 0)

        if isinstance(item, tuple):
            if len(item) > len(self.shape):
                raise IndexError(
                    f"Cannot index more dimensions than exist in the array. "
                    f"You have tried to index with <{len(item)}> dimensions, "
                    f"only <{len(self.shape)}> dimensions exist in the array"
                )
            frame_indexer = item[0]
        else:
            frame_indexer = item

        # Step 2: Do some basic error handling for frame_indexer before using it to slice

        if isinstance(frame_indexer, np.ndarray):
            pass

        elif isinstance(frame_indexer, list):
            pass

        elif isinstance(frame_indexer, int):
            pass

        # numpy int scaler
        elif isinstance(frame_indexer, np.integer):
            frame_indexer = frame_indexer.item()

        # treat slice and range the same
        elif isinstance(frame_indexer, (slice, range)):
            start = frame_indexer.start
            stop = frame_indexer.stop
            step = frame_indexer.step

            if start is not None:
                if start > self.shape[0]:
                    raise IndexError(
                        f"Cannot index beyond `n_frames`.\n"
                        f"Desired frame start index of <{start}> "
                        f"lies beyond `n_frames` <{self.shape[0]}>"
                    )
            if stop is not None:
                if stop > self.shape[0]:
                    raise IndexError(
                        f"Cannot index beyond `n_frames`.\n"
                        f"Desired frame stop index of <{stop}> "
                        f"lies beyond `n_frames` <{self.shape[0]}>"
                    )

            if step is None:
                step = 1

            # convert indexer to slice if it was a range, allows things like decord.VideoReader slicing
            frame_indexer = slice(start, stop, step)  # in case it was a range object

        else:
            raise IndexError(
                f"Invalid indexing method, " f"you have passed a: <{type(item)}>"
            )

        # Step 3: Now slice the data with frame_indexer (careful: if the ndims has shrunk, add a dim)
        c_crop = self._c_norm[:, frame_indexer]
        if c_crop.ndim < self._c_norm.ndim:
            c_crop = c_crop.unsqueeze(1)

        v_crop = self._v @ c_crop - (self._background_term[0] @ (self._background_term[1] @ c_crop))
        cc_crop = self._c.T @ c_crop
        selected_neurons = self._index_values[frame_indexer]
        if selected_neurons.ndim < 1:
            selected_neurons = selected_neurons.unsqueeze(0)
        support_values_crop = torch.index_select(
            self._support_correlation_values, 1, selected_neurons
        ).coalesce()

        # Step 4: Deal with remaining indices after lazy computing the frame(s)
        if isinstance(item, tuple) and check_spatial_crop_effect(
            item[1:], self.shape[1:]
        ):
            pixel_space_crop = self.pixel_mat[item[1:]]
            u_indices = pixel_space_crop.flatten()
            u_crop = torch.index_select(self._u, 0, u_indices)
            a_crop = torch.index_select(self._a, 0, u_indices)
            support_values_crop = torch.index_select(
                support_values_crop, 0, u_indices
            ).coalesce()
            mean_crop = torch.index_select(self._residual_movie_mean, 0, u_indices)
            movie_normalizer_crop = torch.index_select(
                self._residual_movie_normalizer, 0, u_indices
            )
            implied_fov = pixel_space_crop.shape
            used_order = "C"  # The crop from pixel mat and flattening means we are now using default torch order
        else:
            u_crop = self._u
            a_crop = self._a
            mean_crop = self._residual_movie_mean
            movie_normalizer_crop = self._residual_movie_normalizer
            implied_fov = self.shape[1], self.shape[2]
            used_order = self.order

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

        if used_order == "F":
            product = product.T.reshape((-1, implied_fov[1], implied_fov[0]))
            product = product.permute((0, 2, 1))
        else:  # order is "C"
            product = product.reshape((implied_fov[0], implied_fov[1], -1))
            product = product.permute(2, 0, 1)

        return torch.nan_to_num(product, nan=0.0, posinf=0.0, neginf=0.0)

    def __getitem__(
        self,
        item: Union[int, list, np.ndarray, Tuple[Union[int, np.ndarray, slice, range]]],
    ) -> np.ndarray:
        product = self.getitem_tensor(item)
        product = product.cpu().numpy().astype(self.dtype).squeeze()
        return product