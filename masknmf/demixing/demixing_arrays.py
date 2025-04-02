from typing import *
import numpy as np
from enum import Enum
from masknmf.arrays.array_interfaces import FactorizedVideo
from masknmf.compression import PMDArray
import torch


def test_slice_effect(my_slice: slice, spatial_dim: int) -> bool:
    """
    Returns True if slice will actually have an effect
    """

    if not (
        (isinstance(my_slice.start, int) and my_slice.start == 0)
        or my_slice.start is None
    ):
        return True
    elif not (
        (isinstance(my_slice.stop, int) and my_slice.stop >= spatial_dim)
        or my_slice.stop is None
    ):
        return True
    elif not (
        my_slice.step is None or (isinstance(my_slice.step, int) and my_slice.step == 1)
    ):
        return True
    return False


def test_range_effect(my_range: range, spatial_dim: int) -> bool:
    """
    Returns True if the range will actually have an effect.

    Parameters:
    my_range (range): The range object to test.
    spatial_dim (int): The size of the dimension that the range is applied to.

    Returns:
    bool: True if the range will affect the selection; False otherwise.
    """
    # Check if the range starts from the beginning
    if my_range.start != 0:
        return True
    # Check if the range stops at the end of the dimension
    elif my_range.stop != spatial_dim:
        return True
    # Check if the range step is not 1
    elif my_range.step != 1:
        return True
    return False


def test_spatial_crop_effect(my_tuple, spatial_dims) -> bool:
    """
    Returns true if the tuple used for spatial cropping actually has an effect on the underlying data. Otherwise
    cropping can be an expensive and avoidable operation.
    """
    for k in range(len(my_tuple)):
        if isinstance(my_tuple[k], slice):
            if test_slice_effect(my_tuple[k], spatial_dims[k]):
                return True
        if isinstance(my_tuple[k], range):
            if test_range_effect(my_tuple[k], spatial_dims[k]):
                return True
    return False


class ACArray(FactorizedVideo):
    """
    Factorized video for the spatial and temporal extracted sources from the data
    Computations happen transparently on GPU, if device = 'cuda' is specified
    """

    def __init__(
        self,
        fov_shape: tuple[int, int],
        order: str,
        a: torch.sparse_coo_tensor,
        c: torch.tensor,
    ):
        """
        Args:
            fov_shape (tuple): (fov_dim1, fov_dim2)
            order (str): Order to reshape arrays from 1D to 2D
            a (torch.sparse_coo_tensor): Shape (pixels, components)
            c (torch.tensor). Shape (frames, components)
        """
        self._a = a
        self._c = c
        # Check that both objects are on same device
        if self._a.device != self._c.device:
            raise ValueError(f"Spatial and Temporal matrices are not on same device")
        self._device = self._a.device
        t = c.shape[0]
        self._shape = (t,) + fov_shape
        self.pixel_mat = np.arange(np.prod(self.shape[1:])).reshape(
            [self.shape[1], self.shape[2]], order=order
        )
        self.pixel_mat = torch.from_numpy(self.pixel_mat).long().to(self.device)
        self._order = order

    @property
    def device(self) -> str:
        return self._device

    @property
    def c(self) -> torch.tensor:
        """
        return temporal time courses of all signals, shape (frames, components)
        """
        return self._c

    @property
    def a(self) -> torch.sparse_coo_tensor:
        """
        return spatial profiles of all signals as sparse matrix, shape (pixels, components)
        """
        return self._a

    def export_a(self) -> np.ndarray:
        """
        returns the spatial components, where each component is a 2D image. output shape (fov dim1, fov dim 2, n_frames)
        """
        output = self.a.cpu().to_dense().numpy()
        output = output.reshape((self.shape[1], self.shape[2], -1), order=self.order)
        return output

    def export_c(self) -> np.ndarray:
        """
        returns the temporal traces, where each trace is a n_frames-shaped time series. output shape (n_frames, n_components)
        """
        return self.c.cpu().numpy()

    @property
    def order(self) -> str:
        """
        The spatial data is "flattened" from 2D into 1D. This specifies the order ("F" for column-major or "C" for row-major) in which reshaping happened.
        """
        return self._order

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

    # @functools.lru_cache(maxsize=global_lru_cache_maxsize)
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
        c_crop = self._c[frame_indexer, :]
        if c_crop.ndim < self._c.ndim:
            c_crop = c_crop.unsqueeze(0)

        # Step 4: First do spatial subselection before multiplying by c
        if isinstance(item, tuple) and test_spatial_crop_effect(
            item[1:], self.shape[1:]
        ):
            pixel_space_crop = self.pixel_mat[item[1:]]
            a_indices = pixel_space_crop.flatten()
            implied_fov = pixel_space_crop.shape
            a_crop = torch.index_select(self._a, 0, a_indices)
            product = torch.sparse.mm(a_crop, c_crop.T)
            product = product.reshape(implied_fov + (-1,))
            product = product.permute(-1, *range(product.ndim - 1))
        else:
            a_crop = self._a
            implied_fov = self.shape[1], self.shape[2]
            product = torch.sparse.mm(a_crop, c_crop.T)
            if self.order == "F":
                product = product.T.reshape((-1, implied_fov[1], implied_fov[0]))
                product = product.permute((0, 2, 1))
            else:  # order is "C"
                product = product.reshape((implied_fov[0], implied_fov[1], -1))
                product = product.permute(2, 0, 1)

        return product

    def __getitem__(
        self,
        item: Union[int, list, np.ndarray, Tuple[Union[int, np.ndarray, slice, range]]],
    ) -> np.ndarray:
        product = self.getitem_tensor(item)
        product = product.cpu().numpy().astype(self.dtype)
        return product.squeeze()


class FluctuatingBackgroundArray(FactorizedVideo):
    """
    Factorized video for the spatial and temporal extracted sources from the data

    "a" is the matrix of spatial profiles
    "c" is the matrix of temporal profiles
    """

    def __init__(
        self,
        fov_shape: tuple[int, int],
        order: str,
        u: torch.sparse_coo_tensor,
        r: torch.tensor,
        q: torch.tensor,
        v: torch.tensor,
    ):
        """
        The background movie can be factorized as the matrix product (u)(r)(q)(v),
        where u, r, and v are the standard matrices from the pmd decomposition,
        Args:
            fov_shape (tuple): (fov_dim1, fov_dim2)
            order (str): Order to reshape arrays from 1D to 2D
            u (torch.sparse_coo_tensor): shape (pixels, rank1)
            r (torch.tensor): shape (rank1, rank2)
            q (torch.tensor): shape (rank 2, rank 2)
            v (torch.tensor): shape (rank2, frames)
        """
        t = v.shape[1]
        self._shape = (t,) + fov_shape

        self._u = u
        self._v = v
        self._r = r
        self._q = q

        if not (self.u.device == self.v.device == self.r.device == self.q.device):
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
    def r(self) -> torch.tensor:
        return self._r

    @property
    def q(self) -> torch.tensor:
        return self._q

    @property
    def v(self) -> torch.tensor:
        return self._v

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

    # @functools.lru_cache(maxsize=global_lru_cache_maxsize)
    def getitem_tensor(
        self,
        item: Union[int, list, np.ndarray, Tuple[Union[int, np.ndarray, slice, range]]],
    ):
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
            frame_indexer = frame_indexer

        elif isinstance(frame_indexer, list):
            pass

        elif isinstance(frame_indexer, int):
            pass

        # numpy int scalar
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
        v_crop = self._v[:, frame_indexer]
        if v_crop.ndim < self._v.ndim:
            v_crop = v_crop.unsqueeze(1)

        # Step 4: Deal with remaining indices after lazy computing the frame(s)
        if isinstance(item, tuple) and test_spatial_crop_effect(
            item[1:], self.shape[1:]
        ):
            pixel_space_crop = self.pixel_mat[item[1:]]
            u_indices = pixel_space_crop.flatten()
            u_crop = torch.index_select(self._u, 0, u_indices)
            implied_fov = pixel_space_crop.shape
            used_order = "C"  # Torch order here by default is C
        else:
            u_crop = self._u
            implied_fov = self.shape[1], self.shape[2]
            used_order = "F"

        # Temporal term is guaranteed to have nonzero "T" dimension below
        if np.prod(implied_fov) <= v_crop.shape[1]:
            product = torch.sparse.mm(u_crop, self._r)
            product = torch.matmul(product, self._q)
            product = torch.matmul(product, v_crop)

        else:
            product = torch.matmul(self._q, v_crop)
            product = torch.matmul(self._r, product)
            product = torch.sparse.mm(u_crop, product)

        if used_order == "F":
            product = product.T.reshape((-1, implied_fov[1], implied_fov[0]))
            product = product.permute((0, 2, 1))
        else:  # order is "C"
            product = product.reshape((implied_fov[0], implied_fov[1], -1))
            product = product.permute(2, 0, 1)

        return product

    def __getitem__(
        self,
        item: Union[int, list, np.ndarray, Tuple[Union[int, np.ndarray, slice, range]]],
    ) -> np.ndarray:
        product = self.getitem_tensor(item)
        product = product.cpu().numpy().astype(self.dtype)
        return product.squeeze()


class ResidualArray(FactorizedVideo):
    """
    Factorized video for the spatial and temporal extracted sources from the data
    """

    def __init__(
        self,
        pmd_arr: PMDArray,
        ac_arr: ACArray,
        fluctuating_arr: FluctuatingBackgroundArray,
        baseline: torch.tensor,
    ):
        """
        Args:
            pmd_arr (PMDArray)
            ac_arr (ACArray)
            fluctuating_arr (FluctuatingBackgroundArray)
            baseline (torch.tensor): Shape (fov dim 1, fov dim 2)
        """
        self.pmd_arr = pmd_arr
        self.ac_arr = ac_arr
        self.baseline = baseline
        self.fluctuating_arr = fluctuating_arr

        if not (
            self.pmd_arr.device
            == self.ac_arr.device
            == self.baseline.device
            == self.fluctuating_arr.device
        ):
            raise ValueError(f"Input arrays not all on same device")
        self._device = self.pmd_arr.device
        self._shape = self.pmd_arr.shape

    @property
    def dtype(self) -> str:
        """
        data type, default np.float32
        """
        return self.pmd_arr.dtype

    @property
    def device(self) -> str:
        """
        Returns the device that all the internal tensors are on at init time
        """
        return self._device

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

    def __getitem__(
        self,
        item: Union[int, list, np.ndarray, Tuple[Union[int, np.ndarray, slice, range]]],
    ):
        # In this case there is spatial cropping
        if isinstance(item, tuple) and len(item) > 1:
            output = (
                self.pmd_arr.getitem_tensor(item)
                - self.fluctuating_arr.getitem_tensor(item)
                - self.ac_arr.getitem_tensor(item)
                - self.baseline[item[1:]][None, :]
            )
        else:
            output = (
                self.pmd_arr.getitem_tensor(item)
                - self.fluctuating_arr.getitem_tensor(item)
                - self.ac_arr.getitem_tensor(item)
                - self.baseline[None, :]
            )

        return output.cpu().numpy().squeeze()


class ColorfulACArray(FactorizedVideo):
    """
    Factorized video for the spatial and temporal extracted sources from the data
    """

    def __init__(
        self,
        fov_shape: tuple[int, int],
        order: str,
        a: torch.sparse_coo_tensor,
        c: torch.tensor,
        min_color: int = 30,
        max_color: int = 255,
    ):
        """
        Args:
            fov_shape (tuple): (fov_dim1, fov_dim2)
            order (str): Order to reshape arrays from 1D to 2D
            a (torch.sparse_coo_tensor): Shape (pixels, components)
            c (torch.tensor). Shape (frames, components)
            min_color (int): Minimum RGB value (from 0 to 255)
            max_color (int): Maximum RGB value (from 0 to 255)
        """
        t = c.shape[0]
        self._a = a
        self._c = c - torch.amin(c, dim=0, keepdim=True)
        if not (self.a.device == self.c.device):
            raise ValueError(f"Input tensors not on same device")
        self._device = self.a.device
        self._shape = (t,) + fov_shape + (3,)
        self.pixel_mat = np.arange(np.prod(self.shape[1:3])).reshape(
            [self.shape[1], self.shape[2]], order=order
        )
        self.pixel_mat = torch.from_numpy(self.pixel_mat).long().to(self.device)

        ## Establish the coloring scheme
        num_neurons = c.shape[1]
        colors = np.random.uniform(low=min_color, high=max_color, size=num_neurons * 3)
        colors = colors.reshape((num_neurons, 3))
        color_sum = np.sum(colors, axis=1, keepdims=True)
        self._colors = torch.from_numpy(colors / color_sum).to(self.device).float()

        if order == "F" or order == "C":
            self._order = order
        else:
            raise ValueError(f"order can only be F or C")

    @property
    def a(self) -> torch.sparse_coo_tensor:
        return self._a

    @property
    def c(self) -> torch.tensor:
        return self._c

    @property
    def device(self) -> str:
        return self._device

    @property
    def colors(self) -> torch.tensor:
        """
        Colors used for each neuron

        Returns:
            colors (np.ndarray): Shape (number_of_neurons, 3). RGB colors of each neuron
        """
        return self._colors

    @colors.setter
    def colors(self, new_colors: torch.tensor):
        """
        Updates the colors used here
        Args:
            new_colors (torch.tensor): Shape (num_neurons, 3)
        """
        self._colors = new_colors.to(self.device)

    @property
    def dtype(self) -> str:
        """
        data type, default np.float32
        """
        return np.float32

    @property
    def shape(self) -> Tuple[int, int, int, int]:
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
    def order(self):
        return self._order

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
        c_crop = self.c[frame_indexer, :]
        if c_crop.ndim < self.c.ndim:
            c_crop = c_crop[None, :]

        c_crop = c_crop.T

        # Step 4: Deal with remaining indices after lazy computing the frame(s)
        if isinstance(item, tuple) and test_spatial_crop_effect(
            item[1:3], self.shape[1:3]
        ):

            pixel_space_crop = self.pixel_mat[item[1:3]]
            a_indices = pixel_space_crop.flatten()
            a_crop = torch.index_select(self._a, 0, a_indices)
            implied_fov = pixel_space_crop.shape
            product_list = []
            for k in range(3):
                product_list.append(
                    torch.sparse.mm(a_crop, c_crop * self.colors[:, [k]])
                )
            product = torch.stack(product_list, dim=2)
            product = product.reshape(implied_fov + (c_crop.shape[1],) + (3,))
            product = product.permute(product.ndim - 2, *range(product.ndim - 2), 3)
        else:
            a_crop = self._a
            implied_fov = self.shape[1], self.shape[2]

            product_list = []
            for k in range(3):
                curr_product = torch.sparse.mm(a_crop, c_crop * self.colors[:, [k]])
                if self.order == "F":
                    curr_product = curr_product.T.reshape(
                        (-1, implied_fov[1], implied_fov[0])
                    )
                    curr_product = curr_product.permute((0, 2, 1))
                elif self.order == "C":  # order is "C"
                    curr_product = curr_product.reshape(
                        (implied_fov[0], implied_fov[1], -1)
                    )
                    curr_product = curr_product.permute(2, 0, 1)
                product_list.append(curr_product)

            product = torch.stack(product_list, dim=3)

        return product

    def __getitem__(
        self,
        item: Union[int, list, np.ndarray, Tuple[Union[int, np.ndarray, slice, range]]],
    ) -> np.ndarray:

        product = self.getitem_tensor(item)
        product = product.cpu().numpy().squeeze()
        return product


class ResidCorrMode(Enum):
    DEFAULT = 0
    MASKED = 1
    RESIDUAL = 2


class ResidualCorrelationImages(FactorizedVideo):

    def __init__(
        self,
        u_sparse: torch.sparse_coo_tensor,
        r: torch.tensor,
        s: torch.tensor,
        v: torch.tensor,
        factorized_ring_term: torch.tensor,
        a: torch.sparse_coo_tensor,
        c: torch.tensor,
        support_correlation_values: torch.sparse_coo_tensor,
        residual_movie_mean: torch.tensor,
        residual_movie_normalizer: torch.tensor,
        fov_dims: tuple[int, int],
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
            r (torch.tensor): shape (rank 1, rank 2)
            s (torch.tensor): shape (rank 2)
            v (torch.tensor): shape (rank 2, frames)
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
            == r.device
            == s.device
            == v.device
            == c.device
            == a.device
            == factorized_ring_term.device
            == support_correlation_values.device
            == residual_movie_mean.device
            == residual_movie_normalizer.device
        ):
            raise ValueError("Not all tensors are on same device")

        self._device = u_sparse.device
        self._u = u_sparse
        self._r = r
        self._s = s
        self._v = v
        self._factorized_ring_term = factorized_ring_term
        self._background_subtracted_term = (
            torch.diag(self._s) - self._factorized_ring_term
        )
        self._c = c
        self._c_norm = self._c - torch.mean(self._c, dim=0, keepdim=True)
        self._c_norm = self._c_norm / torch.linalg.norm(
            self._c_norm, dim=0, keepdim=True
        )
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
    def shape(self) -> tuple[int, int, int]:
        return (self._c.shape[1], self._fov_dims[0], self._fov_dims[1])

    @property
    def support_correlation_values(self) -> torch.sparse_coo_tensor:
        return self._support_correlation_values

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

        v_crop = self._v @ c_crop
        cc_crop = self._c.T @ c_crop
        selected_neurons = self._index_values[frame_indexer]
        if selected_neurons.ndim < 1:
            selected_neurons = selected_neurons.unsqueeze(0)
        support_values_crop = torch.index_select(
            self._support_correlation_values, 1, selected_neurons
        ).coalesce()

        # Step 4: Deal with remaining indices after lazy computing the frame(s)
        if isinstance(item, tuple) and test_spatial_crop_effect(
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
        if np.prod(implied_fov) <= v_crop.shape[1]:
            product = (
                torch.sparse.mm(u_crop, self._r) @ self._background_subtracted_term
            )
            product = torch.matmul(product, v_crop)
            product -= (mean_crop.unsqueeze(1) @ self._ones_basis) @ v_crop
            product -= torch.sparse.mm(a_crop, cc_crop)
            product /= movie_normalizer_crop.unsqueeze(1)

        else:
            product = self._background_subtracted_term @ v_crop
            product = torch.matmul(self._r, product)
            product = torch.sparse.mm(u_crop, product)
            product -= torch.sparse.mm(a_crop, cc_crop)
            product -= mean_crop.unsqueeze(1) @ (self._ones_basis @ v_crop)

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


class StandardCorrelationImages(FactorizedVideo):

    def __init__(
        self,
        u_sparse: torch.sparse_coo_tensor,
        r: torch.tensor,
        s: torch.tensor,
        v: torch.tensor,
        c: torch.tensor,
        movie_mean: torch.tensor,
        movie_normalizer: torch.tensor,
        fov_dims: tuple[int, int],
        order: str = "F",
    ):
        """
        Generates all the standard correlation images for the demixed data. It is more convenient to keep the
        correlation images in a factorized form and

        Args:
            u_sparse (torch.sparse_coo_tensor): shape (pixels, rank 1)
            r (torch.tensor): shape (rank 1, rank 2)
            s (torch.tensor): shape (rank 2)
            v (torch.tensor): shape (rank 2, frames)
            c (torch.tensor): shape (frames, number of neural signals). This is the temporal traces matrix, where every
                column has mean 0 and Frobenius norm 1.
            movie_mean (torch.tensor): shape (pixels)
            movie_normalizer (torch.tensor): shape (pixels)
        """

        if not (u_sparse.device == r.device == s.device == v.device == c.device):
            raise ValueError("Not all tensors are on same device")

        self._device = u_sparse.device
        self._u = u_sparse
        self._r = r
        self._s = s
        self._v = v
        self._c = c
        self._movie_mean = movie_mean
        self._movie_normalizer = movie_normalizer
        self._fov_dims = (fov_dims[0], fov_dims[1])
        self._order = order

        self._ones_basis = (
            torch.ones([1, self._v.shape[1]], device=self.device) @ self._v.T
        )

        self.pixel_mat = np.arange(np.prod(self.shape[1:])).reshape(
            [self.shape[1], self.shape[2]], order=order
        )
        self.pixel_mat = torch.from_numpy(self.pixel_mat).long().to(self.device)

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
    def shape(self) -> tuple[int, int, int]:
        return (self.c.shape[1], self._fov_dims[0], self._fov_dims[1])

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
        c_crop = self._c[:, frame_indexer]
        if c_crop.ndim < self._c.ndim:
            c_crop = c_crop.unsqueeze(1)

        v_crop = self._v @ c_crop

        # Step 4: Deal with remaining indices after lazy computing the frame(s)
        if isinstance(item, tuple) and test_spatial_crop_effect(
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
            used_order = "C"  # The crop from pixel mat and flattening means we are now using default torch order
        else:
            u_crop = self._u
            mean_crop = self._movie_mean
            movie_normalizer_crop = self._movie_normalizer
            implied_fov = self.shape[1], self.shape[2]
            used_order = self.order

        # Temporal term is guaranteed to have nonzero "T" dimension below
        if np.prod(implied_fov) <= v_crop.shape[1]:
            product = torch.sparse.mm(u_crop, self._r)
            product *= self._s.unsqueeze(0)
            product = torch.matmul(product, v_crop)
            product -= (mean_crop.unsqueeze(1) @ self._ones_basis) @ v_crop
            product /= movie_normalizer_crop.unsqueeze(1)

        else:
            product = self._s.unsqueeze(1) * v_crop
            product = torch.matmul(self._r, product)
            product = torch.sparse.mm(u_crop, product)

            product -= mean_crop.unsqueeze(1) @ (self._ones_basis @ v_crop)

            product /= movie_normalizer_crop.unsqueeze(1)

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


class DemixingResults:

    def __init__(
        self,
        u_sparse: torch.sparse_coo_tensor,
        r: torch.tensor,
        s: torch.tensor,
        q: torch.tensor,
        v: torch.tensor,
        a: torch.sparse_coo_tensor,
        c: torch.tensor,
        b: torch.tensor,
        residual_correlation_image: ResidualCorrelationImages,
        standard_correlation_image: StandardCorrelationImages,
        order: str,
        data_shape: tuple[int, int, int],
        device="cpu",
    ):
        """
        Args:
            u_sparse (torch.sparse_coo_tensor): shape (pixels, rank 1)
            r (torch.tensor): shape (rank 1, rank 2)
            s (torch.tensor): shape (rank 2)
            q (torch.tensor): shape (rank 2, rank 2)
            v (torch.tensor): shape (rank 2, num_frames)
            a (torch.sparse_coo_tensor): shape (pixels, number of neural signals)
            c (torch.tensor): shape (number of frames, number of neural signals)
            b (torch.tensor): shape (pixels)
            residual_correlation_image (ResidualCorrelationImages): Shape (number of neural signals, FOV dim 1, FOV dim 2)
            standard_correlation_image (StandardCorrelationImages): Shape (number of neural signals,
                FOV dim 1, FOV dim 2)
            order (str): order used to reshape data from 2D to 1D
            data_shape (tuple): (number of frames, field of view dimension 1, field of view dimension 2)
            device (str): 'cpu' or 'cuda'. used to manage where the tensors reside
        """
        self._device = device
        self._order = order
        self._shape = data_shape
        self._u_sparse = u_sparse.to(device)
        self._r = r.to(device)
        self._s = s.to(device)
        self._q = q.to(device)
        self._v = v.to(device)
        self._a = a.to(device)
        self._c = c.to(device)
        self._residual_correlation_image = residual_correlation_image
        self._standard_correlation_image = standard_correlation_image
        if self.order == "C":
            self._baseline = b.reshape((self.shape[1], self.shape[2])).to(self.device)
        elif self.order == "F":
            # Note we swap 1 and 2 here
            self._baseline = b.reshape((self.shape[2], self.shape[1])).T.to(self.device)

    @property
    def standard_correlation_image(self) -> StandardCorrelationImages:
        return self._standard_correlation_image

    @property
    def residual_correlation_image(self) -> ResidualCorrelationImages:
        return self._residual_correlation_image

    @property
    def shape(self):
        return self._shape

    @property
    def order(self):
        return self._order

    @property
    def device(self):
        return self._device

    def to(self, new_device):
        self._device = new_device
        self._u_sparse = self._u_sparse.to(self.device)
        self._r = self._r.to(self.device)
        self._s = self._s.to(self.device)
        self._q = self._q.to(self.device)
        self._v = self._v.to(self.device)
        self._a = self._a.to(self.device)
        self._c = self._c.to(self.device)
        self._baseline = self.baseline.to(self.device)

    @property
    def fov_shape(self) -> Tuple[int, int]:
        return self.shape[1:3]

    @property
    def num_frames(self) -> int:
        return self.shape[0]

    @property
    def u(self) -> torch.sparse_coo_tensor:
        return self._u_sparse

    @property
    def r(self) -> torch.tensor:
        return self._r

    @property
    def baseline(self) -> torch.tensor:
        return self._baseline

    @property
    def s(self) -> torch.tensor:
        return self._s

    @property
    def q(self) -> torch.tensor:
        return self._q

    @property
    def v(self) -> torch.tensor:
        return self._v

    @property
    def a(self) -> torch.sparse_coo_tensor:
        return self._a

    @property
    def c(self) -> torch.tensor:
        return self._c

    @property
    def ac_array(self) -> ACArray:
        """
        Returns an ACArray using the tensors stored in this object
        """
        return ACArray(self.fov_shape, self.order, self.a, self.c)

    @property
    def pmd_array(self) -> PMDArray:
        """
        Returns a PMDArray using the tensors stored in this object
        """
        mean_img = torch.zeros(self.shape[1], self.shape[2], device = self.device)
        var_img = torch.ones(self.shape[1], self.shape[2], device = self.device)
        return PMDArray(self.shape,
                           self.u,
                           self.r,
                           self.s,
                           self.v,
                           mean_img,
                           var_img,
                           device=self.device,
                           rescale=True)

    @property
    def fluctuating_background_array(self) -> FluctuatingBackgroundArray:
        """
        Returns a FluctuatingBackgroundArray using the tensors stored in this object
        """
        return FluctuatingBackgroundArray(
            self.fov_shape, self.order, self.u, self.r, self.q, self.v
        )

    @property
    def residual_array(self) -> ResidualArray:
        return ResidualArray(
            self.pmd_array,
            self.ac_array,
            self.fluctuating_background_array,
            self.baseline,
        )

    @property
    def colorful_ac_array(self) -> ColorfulACArray:
        return ColorfulACArray(self.fov_shape, self.order, self.a, self.c)
