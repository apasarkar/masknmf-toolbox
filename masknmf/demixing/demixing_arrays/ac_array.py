from typing import *
import numpy as np
from masknmf.arrays.array_interfaces import ArrayLike, TensorFlyWeight
import torch
from masknmf.demixing.demixing_arrays.demixing_array_utils import check_spatial_crop_effect

class ACArray(ArrayLike):
    """
    Factorized video for the spatial and temporal extracted sources from the data
    Computations happen transparently on GPU, if device = 'cuda' is specified
    """

    def __init__(
        self,
        fov_shape: tuple[int, int],
        flyweight: TensorFlyWeight,
        rescale: bool = False
    ):


        self._flyweight = flyweight
        self._validate_attributes(["a", "c"])
        num_frames = self.c.shape[0]
        self._shape = tuple(map(int, (num_frames, *fov_shape)))

        self._pixel_mat = torch.arange(np.prod(self.shape[1:]), device=self.device, dtype=torch.long).reshape(
            self.shape[1], self.shape[2])
        self._mask = torch.ones(self.a.shape[1], device=self.device, dtype=self.c.dtype)
        self._centers = None
        self._bbox = None
        self._contours = None

        self._rescale = rescale
        self._default_normalizer = torch.ones(self.shape[1], self.shape[2], device=self.device).float()
        if hasattr(self.flyweight, "normalizer"):
            if self.flyweight.normalizer.shape[0] != self.shape[1] or self.flyweight.normalizer.shape[1] != self.shape[2]:
                raise ValueError("Normalizer from flyweight had dimensions not equal to the fov dimensions")

    @classmethod
    def from_tensors(cls,
                     fov_shape: tuple[int, int],
                     a: torch.sparse_coo_tensor,
                     c: torch.Tensor,
                     normalizer: Optional[torch.Tensor],
                     rescale: bool = False
                     ):
        """
        Args:
            fov_shape (tuple): (fov_dim1, fov_dim2)
            a (torch.sparse_coo_tensor): Shape (pixels, components)
            c (torch.Tensor). Shape (frames, components)
            normalizer (Optional[torch.Tensor]): A (height, width)-shaped tensor. Multiply the array pixelwise by this
                value to obtain results in the "raw data" space.
            rescale (bool): Whether or not to rescale the data to the raw data space. This determines the output of getitem
        """
        flyweight = TensorFlyWeight(a=a, c=c, normalizer=normalizer)
        return cls(fov_shape,
                   flyweight,
                   rescale=rescale)

    @classmethod
    def from_flyweight(cls,
                       fov_shape: tuple[int, int],
                       flyweight: TensorFlyWeight,
                       rescale: bool = False):
        return cls(fov_shape,
                   flyweight,
                   rescale=rescale)


    def _validate_attributes(self, attr_list):
        for name in attr_list:
            if not hasattr(self.flyweight, name):
                raise ValueError(f"Required attribute: {name} missing from constructor")

    @property
    def flyweight(self) -> TensorFlyWeight:
        return self._flyweight

    @property
    def device(self):
        return self.flyweight.device

    def to(self, new_device):
        self._flyweight.to(new_device)
        self._pixel_mat = self._pixel_mat.to(new_device)
        self._mask = self._mask.to(new_device)


    @property
    def mask(self) -> torch.Tensor:
        return self._mask

    @mask.setter
    def mask(self, new_mask: torch.Tensor):
        self._mask = new_mask.to(self.device).bool().to(self.c.dtype) #Ensures it's all 1s and 0s

    @property
    def normalizer(self) -> torch.Tensor:
        if not hasattr(self.flyweight, "normalizer"):
            return self._default_normalizer

    @property
    def rescale(self):
        return self._rescale

    @rescale.setter
    def rescale(self, new_value: bool):
        self._rescale = new_value

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
    def centers(self) -> torch.Tensor:
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
    def bbox(self) -> torch.Tensor:
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
    def c(self) -> torch.Tensor:
        """
        return temporal time courses of all signals, shape (frames, components)
        """
        return self.flyweight.c

    @property
    def a(self) -> torch.sparse_coo_tensor:
        """
        return spatial profiles of all signals as sparse matrix, shape (pixels, components)
        """
        return self.flyweight.a

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
    ) -> torch.Tensor:
        # Step 1: index the frames (dimension 0)
        frame_indexer, item = self._parse_indices(item)

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
