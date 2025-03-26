from masknmf.arrays.array_interfaces import LazyFrameLoader, FactorizedVideo
import torch
from typing import *
import numpy as np




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

class PMDArray(FactorizedVideo):
    """
    Factorized demixing array for PMD movie
    """

    def __init__(
        self,
        fov_shape: Tuple[int, int, int],
        u: torch.sparse_coo_tensor,
        r: torch.tensor,
        s: torch.tensor,
        v: torch.tensor,
    ):
        """
        The background movie can be factorized as the matrix product (u)(r)(s)(v),
        where u, r, s, v are the standard matrices from the pmd decomposition
        Args:
            fov_shape (tuple): (num_frames, fov_dim1, fov_dim2)
            u (torch.sparse_coo_tensor): shape (pixels, rank1)
            r (torch.tensor): shape (rank1, rank2)
            s (torch.tensor): shape (rank 2)
            v (torch.tensor): shape (rank2, frames)
        """
        self._u = u
        self._r = r
        self._s = s
        self._v = v
        if not (self.u.device == self.r.device == self.s.device == self.v.device):
            raise ValueError(f"Input tensors are not on the same device")
        self._device = self.u.device
        self._shape = fov_shape
        self.pixel_mat = torch.arange(torch.prod(self.shape[1:]),
                                      device=self.device, dtype=torch.float32).reshape(self.shape[1], self.shape[2])
        self._order = "C"
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
    def s(self) -> torch.tensor:
        return self._s

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
        The spatial data is "flattened" from 2D into 1D.
        This is not user-modifiable; "F" ordering is undesirable in PyTorch
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
        else:
            u_crop = self._u
            implied_fov = self.shape[1], self.shape[2]

        # Temporal term is guaranteed to have nonzero "T" dimension below
        if np.prod(implied_fov) <= v_crop.shape[1]:
            product = torch.sparse.mm(u_crop, self._r)
            product *= self._s.unsqueeze(0)
            product = torch.matmul(product, v_crop)

        else:
            product = self._s.unsqueeze(1) * v_crop
            product = torch.matmul(self._r, product)
            product = torch.sparse.mm(u_crop, product)

        product = product.reshape((implied_fov[0], implied_fov[1], -1))
        product = product.permute(2, 0, 1)

        return product

    def __getitem__(
        self,
        item: Union[int, list, np.ndarray, Tuple[Union[int, np.ndarray, slice, range]]],
    ) -> np.ndarray:
        product = self.getitem_tensor(item)
        product = product.cpu().numpy().astype(self.dtype).squeeze()
        return product