from typing import *
import numpy as np
from masknmf.arrays.array_interfaces import FactorizedVideo
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
        if isinstance(my_tuple[k], np.ndarray):
            if my_tuple[k].shape[0] < spatial_dims[k]:
                return True

        if isinstance(my_tuple[k], np.integer):
            return True

        if isinstance(my_tuple[k], int):
            return True

        if isinstance(my_tuple[k], slice):
            if test_slice_effect(my_tuple[k], spatial_dims[k]):
                return True
        if isinstance(my_tuple[k], range):
            if test_range_effect(my_tuple[k], spatial_dims[k]):
                return True
    return False

class StandardCorrelationImages(FactorizedVideo):
    def __init__(
        self,
        u_sparse: torch.sparse_coo_tensor,
        v: torch.tensor,
        c: torch.tensor,
        movie_mean: torch.tensor,
        movie_normalizer: torch.tensor,
        fov_dims: Tuple[int, int],
        order: str = "F",
    ):
        """
        Generates all the standard correlation images for the demixed data. It is more convenient to keep the
        correlation images in a factorized form and

        Args:
            u_sparse (torch.sparse_coo_tensor): shape (pixels, rank)
            v (torch.tensor): shape (rank, frames)
            c (torch.tensor): shape (frames, number of neural signals). This is the temporal traces matrix, where every
                column has mean 0 and Frobenius norm 1.
            movie_mean (torch.tensor): shape (pixels), the mean of u_sparse times v
            movie_normalizer (torch.tensor): shape (pixels), the pixelwise l2 norm of (u_sparse times v) - movie_mean
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
        self._order = order

        self.pixel_mat = np.arange(np.prod(self.shape[1:])).reshape(
            [self.shape[1], self.shape[2]], order=order
        )

        self._ones_frames = torch.ones(
            (1, self._v.shape[1]), device=self.device, dtype=torch.float
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
    def shape(self) -> Tuple[int, int, int]:
        return self.c.shape[1], self._fov_dims[0], self._fov_dims[1]

    @property
    def movie_mean(self) -> torch.tensor:
        return self._movie_mean

    @property
    def movie_normalizer(self) -> torch.tensor:
        return self._movie_normalizer

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
        ones_crop = self._ones_frames @ c_crop

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

        product = (
            torch.sparse.mm(u_crop, v_crop) - mean_crop.unsqueeze(1) @ ones_crop
        ) / movie_normalizer_crop.unsqueeze(1)

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