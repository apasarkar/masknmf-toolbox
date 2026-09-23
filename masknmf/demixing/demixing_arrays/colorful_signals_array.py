import numpy as np
from masknmf.arrays.array_interfaces import ArrayLike, TensorFlyWeight
import torch
from masknmf.utils import SparseCOOTensor
from masknmf.demixing.demixing_arrays.demixing_array_utils import check_spatial_crop_effect

class ColorfulSignalsArray(ArrayLike):
    """
    Factorized video for the spatial and temporal extracted sources from the data
    """

    def __init__(
        self,
        fov_shape: tuple[int, int],
        flyweight: TensorFlyWeight,
        min_color: int = 30,
        max_color: int = 255,
    ):
        """
        See from_tensors class method for documentation
        """

        self._flyweight = flyweight
        self.flyweight.validate_attributes(["spatial_demixed", "temporal_demixed"])
        num_frames = self.temporal_demixed.shape[0]
        self._temporal_demixed_minsub = self.temporal_demixed - torch.amin(self.temporal_demixed, dim=0, keepdim=True)
        fov_shape = tuple(map(int, fov_shape))
        self._shape = (num_frames, *fov_shape, 3)
        self._pixel_mat = torch.arange(self.shape[1] * self.shape[2], device=self.device, dtype=torch.long).reshape(
            self.shape[1], self.shape[2])
        self._mask = torch.ones(self.spatial_demixed.shape[1], device=self.device, dtype=self.temporal_demixed.dtype)

        ## Establish the coloring scheme
        num_neurons = self.temporal_demixed.shape[1]
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
                     spatial_demixed: SparseCOOTensor,
                     temporal_demixed: torch.Tensor,
                     min_color: int = 30,
                     max_color: int = 255,
                     ):
        """
        Args:
            fov_shape (tuple): (fov_height, fov_width)
            spatial_demixed (torch.sparse_coo_tensor): Shape (pixels, components)
            temporal_demixed (torch.Tensor). Shape (frames, components)
            min_color (int): Minimum RGB value (from 0 to 255)
            max_color (int): Maximum RGB value (from 0 to 255)
        """
        flyweight = TensorFlyWeight(spatial_demixed=spatial_demixed, temporal_demixed=temporal_demixed)
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

    @property
    def device(self) -> str:
        return self.flyweight.device

    @property
    def spatial_demixed(self) -> SparseCOOTensor:
        return self.flyweight.spatial_demixed

    @property
    def temporal_demixed(self) -> torch.Tensor:
        return self.flyweight.temporal_demixed

    def to(self, new_device):
        if self._flyweight.device != new_device:
            self._flyweight.to(new_device)
        self._move_local_tensors(new_device)

    def _move_local_tensors(self, new_device):
        self._pixel_mat = self._pixel_mat.to(new_device)
        self._temporal_demixed_minsub = self._temporal_demixed_minsub.to(new_device)
        self._mask = self._mask.to(new_device)
        self._colors = self._colors.to(new_device)

    @property
    def mask(self) -> torch.Tensor:
        return self._mask

    @mask.setter
    def mask(self, new_mask: torch.Tensor):
        self._mask = new_mask.to(self.device).bool().to(self.temporal_demixed.dtype) #Ensures it's all 1s and 0s

    @property
    def colors(self) -> torch.Tensor:
        """
        Colors used for each neuron

        Returns:
            colors (np.ndarray): Shape (num_neurons, 3). RGB colors of each neuron
        """
        return self._colors

    @colors.setter
    def colors(self, new_colors: torch.Tensor):
        """
        Updates the colors used here
        Args:
            new_colors (torch.Tensor): Shape (num_neurons, 3)
        """
        self._colors = new_colors.to(self.device).to(self.temporal_demixed.dtype)

    @property
    def dtype(self) -> str:
        """
        data type, default np.float32
        """
        return np.float32

    @property
    def shape(self) -> tuple[int, int, int, int]:
        """
        Array shape (num_frames, fov_height, fov_width, 3)
        """
        return self._shape

    @property
    def ndim(self) -> int:
        """
        Number of dimensions
        """
        return len(self.shape)

    def compute_mip(self) -> torch.Tensor:
        updated_coloring = self.colors * torch.amax(self.temporal_demixed, dim=0, keepdims = True).T #(num_neurons, 3)
        updated_coloring = updated_coloring * self.mask[:, None].float()
        mip_image = torch.sparse.mm(self.spatial_demixed, updated_coloring)
        mip_image = mip_image.reshape(self.shape[1], self.shape[2], -1)
        mip_image /= torch.amax(mip_image, dim=2, keepdim=True)
        mip_image = torch.nan_to_num(mip_image, nan=0.0)
        return mip_image


    def getitem_tensor(
        self,
        item: int | list | np.ndarray | tuple[int | np.ndarray | slice | range, ...],
    ) -> torch.tensor:
        # Step 1: index the frames (dimension 0)

        frame_indexer, item = self._parse_indices(item)

        # Step 3: Now slice the data with frame_indexer (careful: if the ndims has shrunk, add a dim)
        temporal_demixed_crop = self._temporal_demixed_minsub[frame_indexer, :]
        if temporal_demixed_crop.ndim < self.temporal_demixed.ndim:
            temporal_demixed_crop = temporal_demixed_crop[None, :]

        temporal_demixed_crop = temporal_demixed_crop * self.mask[None, :]
        temporal_demixed_crop = temporal_demixed_crop.T

        # Step 4: Deal with remaining indices after lazy computing the frame(s)
        if isinstance(item, tuple) and check_spatial_crop_effect(
            item[1:3], self.shape[1:3]
        ):
            pixel_space_crop = self._pixel_mat[item[1:3]]
            spatial_demixed_indices = pixel_space_crop.flatten()
            spatial_demixed_crop = torch.index_select(self.spatial_demixed, 0, spatial_demixed_indices)
            implied_fov = pixel_space_crop.shape
            product_list = []
            for k in range(3):
                product_list.append(
                    torch.sparse.mm(spatial_demixed_crop, temporal_demixed_crop * self.colors[:, [k]])
                )
            product = torch.stack(product_list, dim=2)
            product = product.reshape(implied_fov + (temporal_demixed_crop.shape[1],) + (3,))
            product = product.permute(product.ndim - 2, *range(product.ndim - 2), 3)
        else:
            spatial_demixed_crop = self.spatial_demixed
            implied_fov = self.shape[1], self.shape[2]

            product_list = []
            for k in range(3):
                curr_product = torch.sparse.mm(spatial_demixed_crop, temporal_demixed_crop * self.colors[:, [k]])

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
        item: int | list | np.ndarray | tuple[int | np.ndarray | slice | range, ...],
    ) -> np.ndarray:
        product = self.getitem_tensor(item)
        product = product.cpu().numpy()
        return product
