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

        self._a = a.coalesce()
        self._c = c
        # Check that both objects are on same device
        if self._a.device != self._c.device:
            raise ValueError(f"Spatial and Temporal matrices are not on same device")
        self._device = self._a.device
        num_frames = c.shape[0]
        self._shape = tuple(map(int, (num_frames, *fov_shape)))
        self.pixel_mat = np.arange(np.prod(self.shape[-2:])).reshape([self.shape[-2], self.shape[-1]])
        self.pixel_mat = torch.from_numpy(self.pixel_mat).long().to(self.device)
        self._mask = torch.ones(self.a.shape[1], device=self.device, dtype=self.c.dtype)
        self._centers = None
        self._bbox = None
        self._contours = None

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
    def contours(self) -> torch.sparse_coo_tensor:
        """
        Returns Each column describes a binary contour mask
        Since we are not computing statistics or doing any other analyses on this contour mask, this is computing via a simple 1-step dilation
        procedure which allows us to compute everything using the sparse "a" matrix.
        Returns:
            -  torch.sparse_coo_tensor of shape (num_pixels, num_signals).
        """
        if self._contours is None:
            height, width = self.shape[1:]
            num_signals = self.a.shape[1]
            row, col = self.a.indices()
            values = self.a.values()

            # First get rid of values that are nonzero
            row = row[values != 0]
            col = col[values != 0]
            values = values[values != 0]

            dim0_indices = row // width
            dim1_indices = row % width

            ## Compute a 8-pixel neighbor dilation
            dim0_mods = torch.tensor([-1, 0, 1, 1, 1, 0, -1, -1], device=self.device).long()
            dim1_mods = torch.tensor([-1, -1, -1, 0, 1, 1, 1, 0], device=self.device).long()
            col_dilated = col[:, None] + torch.zeros_like(dim0_mods)[None, :].flatten()

            dim0_dilated = dim0_indices[:, None] + dim0_mods[None, :].flatten() #Shape (dim0_indices.shape[0], 8)
            dim1_dilated = dim1_indices[:, None] + dim1_mods[None, :].flatten() #Shape (dim1_indices.shape[0], 8)

            good_indices = torch.logical_and(dim0_dilated > 0, dim1_dilated > 0)
            good_indices = torch.logical_and(good_indices, dim0_dilated < height)
            good_indices = torch.logical_and(good_indices, dim1_dilated < width)

            dim0_dilated = dim0_dilated[good_indices]
            dim1_dilated = dim1_dilated[good_indices]
            col_dilated = col_dilated[good_indices]

            ##Re-vectorize the dilated indices
            dilated_indices = dim0_dilated*width + dim1_dilated #C order vectorization

            #Now construct and coalesce a sparse tensor to get rid of duplicates
            dilated_tensor = torch.sparse_coo_tensor(torch.stack([dilated_indices, col_dilated], dim = 0),
                                                     torch.ones_like(dilated_indices),
                                                     size= self.a.shape).coalesce()

            row_dilated, col_dilated = dilated_tensor.indices()

            #Combine the data frm the original sparse tensor with this to see which pixels are duplicates
            row_aggregated = torch.concatenate([row_dilated, row], dim = 0)
            col_aggregated = torch.concatenate([col_dilated, col], dim = 0)
            ## The values for pixels in the original are all -40. So when we combine into one array, all non-boundary pixels have negative value
            val_aggregated = torch.concatenate([torch.ones_like(row_dilated).long(), (torch.ones_like(col) * -40).long()], dim = 0)


            aggregate_tensor = torch.sparse_coo_tensor(torch.stack([row_aggregated, col_aggregated], dim = 0),
                                                       val_aggregated,
                                                       size= self.a.shape).coalesce()

            row_dilated, col_dilated = aggregate_tensor.indices()
            val_dilated = aggregate_tensor.values()

            good_indices = val_dilated > 0
            row_dilated = row_dilated[good_indices]
            col_dilated = col_dilated[good_indices]

            val_dilated = torch.ones_like(row_dilated)

            final_tensor = torch.sparse_coo_tensor(torch.stack([row_dilated, col_dilated], dim=0),
                                                   val_dilated,
                                                   size=self.a.shape).coalesce()

            self._contours = final_tensor
        return self._contours

    @property
    def centers(self) -> torch.tensor:
        """
        Returns a (num_signals, 2) shaped tensor describing the height, width dimensions of each signals spatial center
            of mass. The center of mass might not be on an active pixel, but this is ok
        """
        if self._centers is None:
            height, width = self.shape[1:]
            num_signals = self.a.shape[1]
            row, col = self.a.indices()
            values = self.a.values().float()

            # First get rid of values that are nonzero
            row = row[values != 0]
            col = col[values != 0]

            # Need to unvectorize row
            dim0_coords = row // width
            dim1_coords = row % width

            dim0_com_numerator = torch.zeros(num_signals, device=self.device).float()
            dim0_com_denominator = torch.zeros(num_signals, device=self.device).float()

            dim1_com_numerator = torch.zeros(num_signals, device=self.device).float()
            dim1_com_denominator = torch.zeros(num_signals, device=self.device).float()

            dim0_com_numerator.scatter_reduce_(0, col, (dim0_coords * values).float(), reduce="sum")
            dim0_com_denominator.scatter_reduce_(0, col, values, reduce="sum")

            dim1_com_numerator.scatter_reduce_(0, col, (dim1_coords * values).float(), reduce="sum")
            dim1_com_denominator.scatter_reduce_(0, col, values, reduce="sum")

            dim0_com = torch.nan_to_num(dim0_com_numerator / dim0_com_denominator, nan=0.0)
            dim1_com = torch.nan_to_num(dim1_com_numerator / dim1_com_denominator, nan=0.0)

            self._centers = torch.stack([dim0_com, dim1_com], dim=1)

        return self._centers

    @property
    def bbox(self) -> torch.tensor:
        """
        Returns a torch tensor of shape (num_signals, 4). Each row contains 4 elements a1, a2, b1, b2 which define a bounding box
            of a neuron image like img[a1:a2, b1:b2]

        """
        if self._bbox is None:
            height, width = self.shape[1:]
            num_signals = self.a.shape[1]
            row, col = self.a.indices()
            values = self.a.values()

            #First get rid of values that are nonzero
            row = row[values != 0]
            col = col[values != 0]

            #Need to unvectorize row
            dim0_values = (row // width).long()
            dim1_values = (row % width).long()

            min_dim0 = torch.zeros(num_signals, device=self.device).long()
            max_dim0 = torch.zeros(num_signals, device=self.device).long()
            min_dim1 = torch.zeros(num_signals, device=self.device).long()
            max_dim1 = torch.zeros(num_signals, device=self.device).long()

            min_dim0.scatter_reduce_(0, col, dim0_values,  reduce="amin", include_self=False)
            max_dim0.scatter_reduce_(0, col, dim0_values, reduce="amax", include_self=False)
            min_dim1.scatter_reduce_(0, col, dim1_values, reduce="amin", include_self=False)
            max_dim1.scatter_reduce_(0, col, dim1_values, reduce="amax", include_self=False)

            self._bbox = torch.stack([min_dim0, max_dim0, min_dim1, max_dim1], dim=1)
        return self._bbox




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