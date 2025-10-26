from typing import *
import numpy as np
from masknmf.arrays.array_interfaces import FactorizedVideo
import torch
from masknmf.demixing.demixing_arrays.demixing_array_utils import check_spatial_crop_effect

class ColorfulACArray(FactorizedVideo):
    """
    Factorized video for the spatial and temporal extracted sources from the data
    """

    def __init__(
        self,
        fov_shape: Tuple[int, int],
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
        if isinstance(item, tuple) and check_spatial_crop_effect(
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
