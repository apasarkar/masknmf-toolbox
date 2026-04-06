from typing import *
import numpy as np
from masknmf.arrays.array_interfaces import ArrayLike, TensorFlyWeight
import torch
from masknmf.demixing.demixing_arrays.demixing_array_utils import check_spatial_crop_effect

class ColorfulACArray(ArrayLike):
    """
    Factorized video for the spatial and temporal extracted sources from the data
    """

    def __init__(
        self,
        fov_shape: Tuple[int, int],
        flyweight: TensorFlyWeight,
        min_color: int = 30,
        max_color: int = 255,
    ):
        """
        See from_tensors class method for documentation
        """

        self._flyweight = flyweight
        self.flyweight.validate_attributes(["a", "c"])
        t = self.c.shape[0]
        self._c_minsub = self.c - torch.amin(self.c, dim=0, keepdim=True)
        fov_shape = tuple(map(int, fov_shape))
        self._shape = (t, *fov_shape, 3)
        self._pixel_mat = torch.arange(self.shape[1] * self.shape[2], device=self.device, dtype=torch.long).reshape(
            self.shape[1], self.shape[2])
        self._mask = torch.ones(self.a.shape[1], device=self.device, dtype=self.c.dtype)

        ## Establish the coloring scheme
        num_neurons = self.c.shape[1]
        colors = np.random.uniform(low=min_color, high=max_color, size=num_neurons * 3)
        colors = colors.reshape((num_neurons, 3))
        color_sum = np.sum(colors, axis=1, keepdims=True)
        self._colors = torch.from_numpy(colors / color_sum).to(self.device).float()

    @property
    def flyweight(self) -> TensorFlyWeight:
        return self._flyweight

    @classmethod
    def from_tensors(cls,
                     fov_shape: tuple[int, int],
                     a: torch.sparse_coo_tensor,
                     c: torch.Tensor,
                     min_color: int = 30,
                     max_color: int = 255,
                     ):
        """
        Args:
            fov_shape (tuple): (fov_dim1, fov_dim2)
            a (torch.sparse_coo_tensor): Shape (pixels, components)
            c (torch.tensor). Shape (frames, components)
            min_color (int): Minimum RGB value (from 0 to 255)
            max_color (int): Maximum RGB value (from 0 to 255)
        """
        flyweight = TensorFlyWeight(a=a, c=c)
        return cls(fov_shape,
                   flyweight,
                   min_color=min_color,
                   max_color=max_color)

    @classmethod
    def from_flyweight(cls,
                       fov_shape,
                       flyweight: TensorFlyWeight,
                       min_color: int = 30,
                       max_color: int = 255,
                       ):
        return cls(fov_shape,
                   flyweight,
                   min_color=min_color,
                   max_color=max_color)


    def _validate_attributes(self, attr_list):
        for name in attr_list:
            if not hasattr(self.flyweight, name):
                raise ValueError(f"Required attribute: {name} missing from constructor")

    @property
    def device(self) -> str:
        return self.flyweight.device

    @property
    def a(self) -> torch.sparse_coo_tensor:
        return self.flyweight.a

    @property
    def c(self) -> torch.Tensor:
        return self.flyweight.c

    def to(self, new_device):
        self._flyweight.to(new_device)
        self._pixel_mat = self._pixel_mat.to(new_device)
        self._c_minsub = self._c_minsub.to(new_device)
        self._mask = self._mask.to(new_device)
        self._colors = self._colors.to(new_device)

    @property
    def mask(self) -> torch.Tensor:
        return self._mask

    @mask.setter
    def mask(self, new_mask: torch.Tensor):
        self._mask = new_mask.to(self.device).bool().to(self.c.dtype) #Ensures it's all 1s and 0s

    @property
    def colors(self) -> torch.Tensor:
        """
        Colors used for each neuron

        Returns:
            colors (np.ndarray): Shape (number_of_neurons, 3). RGB colors of each neuron
        """
        return self._colors

    @colors.setter
    def colors(self, new_colors: torch.Tensor):
        """
        Updates the colors used here
        Args:
            new_colors (torch.Tensor): Shape (num_neurons, 3)
        """
        self._colors = new_colors.to(self.device).to(self.c.dtype)

    @property
    def dtype(self) -> str:
        """
        data type, default np.float32
        """
        return np.float32

    @property
    def shape(self) -> Tuple[int, int, int, int]:
        """
        Array shape (n_frames, dims_x, dims_y, 3)
        """
        return self._shape

    @property
    def ndim(self) -> int:
        """
        Number of dimensions
        """
        return len(self.shape)

    def compute_mip(self) -> torch.Tensor:
        updated_coloring = self.colors * torch.amax(self.c, dim=0, keepdims = True).T #(num_neurons, 3)
        updated_coloring = updated_coloring * self.mask[:, None].float()
        mip_image = torch.sparse.mm(self.a, updated_coloring)
        mip_image = mip_image.reshape(self.shape[1], self.shape[2], -1)
        mip_image /= torch.amax(mip_image, dim=2, keepdim=True)
        mip_image = torch.nan_to_num(mip_image, nan=0.0)
        return mip_image


    def getitem_tensor(
        self,
        item: Union[int, list, np.ndarray, Tuple[Union[int, np.ndarray, slice, range]]],
    ) -> torch.tensor:
        # Step 1: index the frames (dimension 0)

        frame_indexer, item = self._parse_indices(item)

        # Step 3: Now slice the data with frame_indexer (careful: if the ndims has shrunk, add a dim)
        c_crop = self._c_minsub[frame_indexer, :]
        if c_crop.ndim < self.c.ndim:
            c_crop = c_crop[None, :]

        c_crop = c_crop * self._mask[None, :]
        c_crop = c_crop.T

        # Step 4: Deal with remaining indices after lazy computing the frame(s)
        if isinstance(item, tuple) and check_spatial_crop_effect(
            item[1:3], self.shape[1:3]
        ):
            pixel_space_crop = self._pixel_mat[item[1:3]]
            a_indices = pixel_space_crop.flatten()
            a_crop = torch.index_select(self.a, 0, a_indices)
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
            a_crop = self.a
            implied_fov = self.shape[1], self.shape[2]

            product_list = []
            for k in range(3):
                curr_product = torch.sparse.mm(a_crop, c_crop * self.colors[:, [k]])

                curr_product = curr_product.reshape(
                    (implied_fov[0], implied_fov[1], -1)
                )
                curr_product = curr_product.permute(2, 0, 1)
                product_list.append(curr_product)

            product = torch.stack(product_list, dim=3)

        if isinstance(item, tuple) and len(item) == 4:
            product = product[..., item[3]] ##Apply the last crop
        return product

    def __getitem__(
        self,
        item: Union[int, list, np.ndarray, Tuple[Union[int, np.ndarray, slice, range]]],
    ) -> np.ndarray:
        product = self.getitem_tensor(item)
        product = product.cpu().numpy()
        return product
