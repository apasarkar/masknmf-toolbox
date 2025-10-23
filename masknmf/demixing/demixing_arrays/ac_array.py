from typing import *
import numpy as np
from masknmf.arrays.array_interfaces import FactorizedVideo
import torch
from masknmf.demixing.demixing_arrays.demixing_array_utils import test_spatial_crop_effect

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
        self._shape = (t, *fov_shape)
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

            pixel_space_crop = self.pixel_mat[spatial_crop_terms]
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
