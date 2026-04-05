from masknmf.arrays.array_interfaces import LazyFrameLoader, ArrayLike, TensorFlyWeight
from masknmf.utils._serialization import load_dict
from masknmf.utils import Serializer
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


def _construct_identity_torch_sparse_tensor(dimsize: int, device: str = "cpu"):
    """
    Constructs an identity torch.sparse_coo_tensor on the specified device.

    Args:
        dimsize (int): The number of rows (or equivalently columns) of the torch.sparse_coo_tensor.
        device (str): 'cpu' or 'cuda'. The device on which the sparse tensor is constructed

    Returns:
        - (torch.sparse_coo_tensor): A (dimsize, dimsize) torch.sparse_coo_tensor.
    """
    # Indices for diagonal elements (rows and cols are the same for diagonal)
    row_col = torch.arange(dimsize, device=device)
    indices = torch.stack([row_col, row_col], dim=0)

    # Values (all ones)
    values = torch.ones(dimsize, device=device)

    sparse_tensor = torch.sparse_coo_tensor(indices, values, (dimsize, dimsize))
    return sparse_tensor

class PMDArray(ArrayLike, Serializer):
    """
    Factorized demixing array for PMD movie
    """
    _serialized = {
        "shape",
        "u",
        "v",
        "u_local_projector",
        "mean_img",
        "var_img"
    }

    def __init__(
        self,
        shape: Tuple[int, int, int] | np.ndarray,
        flyweight: TensorFlyWeight,
        device: str = "cpu",
        rescale: bool = True,
    ):
        """
        See from_tensors class method for documentation
        """
        self._shape = tuple(shape)
        self._rescale = rescale

        ##Set up the flyweight and all other tensors
        self._flyweight = flyweight
        self._flyweight.to(device)

        self._pixel_mat = torch.arange(
            self.shape[1] * self.shape[2], device=self.flyweight.device,
        ).reshape(self.shape[1], self.shape[2])



    @property
    def flyweight(self) -> TensorFlyWeight:
        return self._flyweight
    @classmethod
    def from_tensors(cls,
                   shape: Tuple[int, int, int] | np.ndarray,
                   u: torch.sparse_coo_tensor,
                   v: torch.tensor,
                   mean_img: torch.tensor,
                   var_img: torch.tensor,
                   u_local_projector: Optional[torch.sparse_coo_tensor] = None,
                   device: str = "cpu",
                   rescale: bool = True):

        """
            Key assumption: the spatial basis matrix U has n + k columns; the first n columns is blocksparse (this serves
            as a local spatial basis for the data) and the last k columns can have unconstrained spatial support (these serve
            as a global spatial basis for the data).

            Args:
                shape (tuple): (num_frames, fov_dim1, fov_dim2)
                u (torch.sparse_coo_tensor): shape (pixels, rank)
                v (torch.tensor): shape (rank, frames)
                mean_img (torch.tensor): shape (fov_dim1, fov_dim2). The pixelwise mean of the data
                var_img (torch.tensor): shape (fov_dim1, fov_dim2). A pixelwise noise normalizer for the data
                u_local_projector (Optional[torch.sparse_coo_tensor]): shape (pixels, rank)
                resid_std (torch.tensor): The residual standard deviation, shape (fov_dim1, fov_dim2)
                device (str): The device on which computations occur/data is stored
                rescale (bool): True if we rescale the PMD data (i.e. multiply by the pixelwise normalizer
                    and add back the mean) in __getitem__
        """
        flyweight = TensorFlyWeight(u=u.float(),
                                    v=v.float(),
                                    mean_img=mean_img.float(),
                                    var_img=var_img.float(),
                                    u_local_projector=u_local_projector.float() if u_local_projector is not None else None)
        return cls(shape,
                   flyweight,
                   device=device,
                   rescale = rescale)


    @classmethod
    def from_flyweight(cls,
                       shape: Tuple[int, int, int] | np.ndarray,
                       flyweight: TensorFlyWeight,
                       device: str = "cpu",
                       rescale: bool = True
                       ):
        """
        Memory efficient way to construct PMD Array from a flyweight tensor manager. See from_array for parameter documentation
        """
        return cls(shape,
                   flyweight,
                   device=device,
                   rescale=rescale)

    @classmethod
    def from_hdf5(cls, path, **kwargs):
        d = load_dict(path, Serializer.__name__)
        return cls.from_tensors(**d, **kwargs)

    @property
    def rescale(self) -> bool:
        return self._rescale

    @rescale.setter
    def rescale(self, new_state: bool):
        self._rescale = new_state

    @property
    def mean_img(self) -> torch.Tensor:
        return self.flyweight.mean_img

    @property
    def var_img(self) -> torch.Tensor:
        return self.flyweight.var_img

    def to(self, device: str):
        self._flyweight.to(device)
        self._pixel_mat = self._pixel_mat.to(device)

    @property
    def device(self) -> str:
        return self.flyweight.device

    @property
    def u(self) -> torch.sparse_coo_tensor:
        return self.flyweight.u

    @property
    def u_local_projector(self) -> Optional[torch.sparse_coo_tensor]:
        if hasattr(self.flyweight, "u_local_projector"):
            return self.flyweight.u_local_projector
        return None

    @property
    def pmd_rank(self) -> int:
        return self.u.shape[1]

    @property
    def v(self) -> torch.Tensor:
        return self.flyweight.v

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
    
    def calculate_rank_heatmap(self) -> torch.Tensor:
        """
        Generates rank heatmap image based on U. Equal to row summation of binarized U matrix.
        Returns:
            rank_heatmap (torch.Tensor). Shape (fov_dim1, fov_dim2).
        """
        binarized_u = torch.sparse_coo_tensor(
            self.u.indices(), 
            torch.ones_like(self.u.values()), 
            self.u.size()
            )
        row_sum_u = torch.sparse.sum(binarized_u, dim=1)
        return torch.reshape(row_sum_u.to_dense(), 
                             (self.shape[1],self.shape[2]))

    def project_frames(
        self, frames: torch.Tensor, standardize: Optional[bool] = True
    ) -> torch.Tensor:
        """
        Projects frames onto the spatial basis, using the u_projector property. u_projector must be defined.
        Args:
            frames (torch.Tensor). Shape (fov_dim1, fov_dim2, num_frames) or (fov_dim1*fov_dim2, num_frames).
                Frames which we want to project onto the spatial basis.
            standardize (Optional[bool]): Indicates whether the frames of data are standardized before projection is performed
        Returns:
            projected_frames (torch.Tensor). Shape (fov_dim1 * fov_dim2, num_frames).
        """
        if self.u_local_projector is None:
            raise ValueError(
                "u_projector must be defined to project frames onto spatial basis"
            )
        orig_device = frames.device
        frames = frames.to(self.device).float()
        if len(frames.shape) == 3:
            if standardize:
                frames = (frames - self.mean_img[..., None]) / self.var_img[
                    ..., None
                ]  # Normalize the frames
                frames = torch.nan_to_num(frames, nan=0.0)
            frames = frames.reshape(self.shape[1] * self.shape[2], -1)
        else:
            if standardize:
                frames = (
                    frames - self.mean_img.flatten()[..., None]
                ) / self.var_img.flatten()[..., None]
                frames = torch.nan_to_num(frames, nan=0.0)

        projection = torch.sparse.mm(self.u_local_projector.T, frames)
        return projection.to(orig_device)

    def getitem_tensor(
        self,
        item: Union[int, list, np.ndarray, Tuple[Union[int, np.ndarray, slice, range]]],
    ) -> torch.Tensor:
        frame_indexer, item = self._parse_indices(item)

        # Step 3: Now slice the data with frame_indexer (careful: if the ndims has shrunk, add a dim)
        v_crop = self.v[:, frame_indexer]
        if v_crop.ndim < self.v.ndim:
            v_crop = v_crop.unsqueeze(1)


        # Step 4: Deal with remaining indices after lazy computing the frame(s)
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

            pixel_space_crop = self._pixel_mat[spatial_crop_terms]
            mean_img_crop = self.mean_img[spatial_crop_terms].flatten()
            var_img_crop = self.var_img[spatial_crop_terms].flatten()
            u_indices = pixel_space_crop.flatten()
            u_crop = torch.index_select(self.u, 0, u_indices)
            implied_fov = pixel_space_crop.shape
        else:
            u_crop = self.u
            mean_img_crop = self.mean_img.flatten()
            var_img_crop = self.var_img.flatten()
            implied_fov = self.shape[1], self.shape[2]

        product = torch.sparse.mm(u_crop, v_crop)
        if self.rescale:
            product *= var_img_crop.unsqueeze(1)
            product += mean_img_crop.unsqueeze(1)

        product = product.reshape((implied_fov[0], implied_fov[1], -1))
        product = product.permute(2, 0, 1)

        return product

    def __getitem__(
        self,
        item: Union[int, list, np.ndarray, Tuple[Union[int, np.ndarray, slice, range]]],
    ) -> np.ndarray:
        product = self.getitem_tensor(item)
        product = product.cpu().numpy().astype(self.dtype)
        return product


class PMDResidualArray(ArrayLike):
    """
    Factorized video for the spatial and temporal extracted sources from the data
    """

    def __init__(
        self,
        raw_arr: Union[ArrayLike],
        pmd_arr: PMDArray,
    ):
        """
        Args:
            raw_arr (LazyFrameLoader): Any object that supports LazyFrameLoder functionality
            pmd_arr (PMDArray)
        """
        self.pmd_arr = pmd_arr
        self.raw_arr = raw_arr
        self._shape = self.pmd_arr.shape

        if self.pmd_arr.shape != self.raw_arr.shape:
            raise ValueError("Two image stacks do not have the same shape")


    @property
    def dtype(self) -> str:
        """
        data type, default np.float32
        """
        return self.pmd_arr.dtype

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
        if self.pmd_arr.rescale is False:
            self.pmd_arr.rescale = True
            switch = True
        else:
            switch = False

        output = self.raw_arr[item].astype(self.dtype) - self.pmd_arr[item].astype(self.dtype)
        
        if switch:
            self.pmd_arr.rescale = False
        return output
