from typing import *
import numpy as np
from masknmf.arrays.array_interfaces import FactorizedVideo
import torch
from masknmf.demixing.demixing_arrays.demixing_array_utils import check_spatial_crop_effect

class ACArray(FactorizedVideo):
    """
    Factorized video for the spatial and temporal extracted sources from the data
    Computations happen transparently on GPU, if device = 'cuda' is specified
    """

    def __init__(
        self,
        fov_shape: tuple[int, int],
        a: torch.sparse_coo_tensor,
        c: torch.tensor,
    ):
        """
        Args:
            fov_shape (tuple): (fov_dim1, fov_dim2)
            a (torch.sparse_coo_tensor): Shape (pixels, components)
            c (torch.tensor). Shape (frames, components)
            mask (torch.tensor). Shape (num_components). A mask of 1s and 0s indicating which neurons are actively displayed
                (and which are effectively zerod out). Can be toggled
        """

        self._a = a
        self._c = c
        # Check that both objects are on same device
        if self._a.device != self._c.device:
            raise ValueError(f"Spatial and Temporal matrices are not on same device")
        self._device = self._a.device
        num_frames = c.shape[0]
        self._shape = (num_frames, *fov_shape)
        self.pixel_mat = np.arange(np.prod(self.shape[-2:])).reshape([self.shape[-2], self.shape[-1]])
        self.pixel_mat = torch.from_numpy(self.pixel_mat).long().to(self.device)
        self._mask = torch.ones(self.a.shape[1], device=self.device, dtype=self.c.dtype)

    @property
    def device(self) -> str:
        return self._device

    @property
    def mask(self) -> torch.tensor:
        return self._mask

    @mask.setter
    def mask(self, new_mask: torch.tensor):
        self._mask = new_mask.to(self.device).bool().to(self.c.dtype) #Ensures it's all 1s and 0s

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
        output = output.reshape((self.shape[-2], self.shape[-1], -1))
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

        c_crop = c_crop * self._mask[None, :]

        # Step 4: First do spatial subselection before multiplying by c
        if isinstance(item, tuple) and check_spatial_crop_effect(item[1:3], self.shape[1:3]):

            pixel_space_crop = self.pixel_mat[item[1:3]]
            a_indices = pixel_space_crop.flatten()
            a_crop = torch.index_select(self._a, 0, a_indices)
            implied_fov = pixel_space_crop.shape
            product = torch.sparse.mm(a_crop, c_crop.T)
            product = product.reshape(implied_fov + (-1,))
            product = product.permute(-1, *range(product.ndim - 1))

        else:
            a_crop = self._a
            implied_fov = self.shape[-2], self.shape[-1]
            product = torch.sparse.mm(a_crop, c_crop.T)
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