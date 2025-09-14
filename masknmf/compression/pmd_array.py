from masknmf.arrays.array_interfaces import LazyFrameLoader, FactorizedVideo, ArrayLike
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

class PMDArray(FactorizedVideo):
    """
    Factorized demixing array for PMD movie
    """

    def __init__(
        self,
        fov_shape: Tuple[int, int, int],
        u: torch.sparse_coo_tensor,
        v: torch.tensor,
        mean_img: torch.tensor,
        var_img: torch.tensor,
        resid_std: torch.tensor,
        u_local_projector: Optional[torch.sparse_coo_tensor] = None,
        device: str = "cpu",
        rescale: bool = True,
    ):
        """
        Key assumption: the spatial basis matrix U has n + k columns; the first n columns is blocksparse (this serves
        as a local spatial basis for the data) and the last k columns can have unconstrained spatial support (these serve
        as a global spatial basis for the data).

        Args:
            fov_shape (tuple): (num_frames, fov_dim1, fov_dim2)
            u (torch.sparse_coo_tensor): shape (pixels, rank)
            v (torch.tensor): shape (rank, frames)
            mean_img (torch.tensor): shape (fov_dim1, fov_dim2). The pixelwise mean of the data
            var_img (torch.tensor): shape (fov_dim1, fov_dim2). A pixelwise noise normalizer for the data
            resid_std (torch.tensor): The residual standard deviation, shape (fov_dim1, fov_dim2)
            u_local_projector (Optional[torch.sparse_coo_tensor]): shape (pixels, rank)
            device (str): The device on which computations occur/data is stored
            rescale (bool): True if we rescale the PMD data (i.e. multiply by the pixelwise normalizer
                and add back the mean) in __getitem__
        """
        self._u = u.to(device).coalesce()
        self._device = self._u.device
        self._v = v.to(device)
        if u_local_projector is not None:
            self._u_local_projector = u_local_projector.to(device).coalesce()
        else:
            self._u_local_projector = None
        # if u_global_projector is not None:
            # self._u_global_projector = u_global_projector.to(device).coalesce()
            # self._u_global_basis = self._compute_global_spatial_basis()
        # else:
        #     self._u_global_projector = None
        #     self._u_global_basis = None
        self._device = self._u.device
        self._shape = fov_shape

        self._resid_std = resid_std
        self.pixel_mat = torch.arange(
            self.shape[1] * self.shape[2], device=self.device
        ).reshape(self.shape[1], self.shape[2])
        self._order = "C"
        self._mean_img = mean_img.to(self.device).float()
        self._var_img = var_img.to(self.device).float()
        self._rescale = rescale

    @property
    def rescale(self) -> bool:
        return self._rescale

    @rescale.setter
    def rescale(self, new_state: bool):
        self._rescale = new_state

    @property
    def mean_img(self) -> torch.tensor:
        return self._mean_img

    @property
    def var_img(self) -> torch.tensor:
        return self._var_img

    @property
    def device(self) -> torch.device:
        return self._device

    def to(self, device: str):
        self._u = self._u.to(device)
        self._v = self._v.to(device)
        self._mean_img = self._mean_img.to(device)
        self._var_img = self._var_img.to(device)
        self.pixel_mat = self.pixel_mat.to(device)
        self._device = self._u.device
        if self.u_local_projector is not None:
            self._u_local_projector = self.u_local_projector.to(device)
        self._resid_std = self._resid_std.to(device)
        # if self.u_global_projector is not None:
        #     self._u_global_projector = self.u_global_projector.to(device)
        #     self._u_global_basis = self.u_global_basis.to(device)

    @property
    def u(self) -> torch.sparse_coo_tensor:
        return self._u

    @property
    def u_local_projector(self) -> Optional[torch.sparse_coo_tensor]:
        return self._u_local_projector
    #
    # @property
    # def u_local_basis(self) -> torch.sparse_coo_tensor:
    #     indices = torch.arange(
    #         self.local_basis_rank, device=self.device, dtype=torch.long
    #     )
    #     cropped_mat = torch.index_select(self.u, 1, indices).coalesce()
    #     return cropped_mat
    #
    # @property
    # def u_global_projector(self) -> Optional[torch.sparse_coo_tensor]:
    #     return self._u_global_projector
    #
    # @property
    # def u_global_basis(self) -> Optional[torch.sparse_coo_tensor]:
    #     return self._u_global_basis
    #
    # @property
    # def global_basis_rank(self) -> int:
    #     if self.u_global_projector is None:
    #         return 0
    #     else:
    #         return int(self.u_global_projector.shape[1])

    @property
    def pmd_rank(self) -> int:
        return self.u.shape[1]
    #
    # @property
    # def v_local_basis(self) -> torch.tensor:
    #     return self.v[: self.local_basis_rank]
    #
    # @property
    # def v_global_basis(self) -> torch.tensor:
    #     return self.v[self.local_basis_rank :]
    #
    # def _compute_global_spatial_basis(self) -> Optional[torch.sparse_coo_tensor]:
    #     if self.global_basis_rank > 0:
    #         indices = torch.arange(
    #             self.local_basis_rank,
    #             self.u.shape[1],
    #             device=self.device,
    #             dtype=torch.long,
    #         )
    #         cropped_mat = torch.index_select(self.u, 1, indices).coalesce()
    #         return cropped_mat
    #     else:
    #         return None

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
    def residual_std(self) -> np.ndarray:
        """
        The standard deviation of the residual. A simple shot-noise estimator
        """
        return self._residual_std

    @property
    def ndim(self) -> int:
        """
        Number of dimensions
        """
        return len(self.shape)
    
    def calculate_rank_heatmap(self) -> torch.tensor:
        """
        Generates rank heatmap image based on U. Equal to row summation of binarized U matrix.
        Returns:
            rank_heatmap (torch.tensor). Shape (fov_dim1, fov_dim2).
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
        self, frames: torch.tensor, standardize: Optional[bool] = True
    ) -> torch.tensor:
        """
        Projects frames onto the spatial basis, using the u_projector property. u_projector must be defined.
        Args:
            frames (torch.tensor). Shape (fov_dim1, fov_dim2, num_frames) or (fov_dim1*fov_dim2, num_frames).
                Frames which we want to project onto the spatial basis.
            standardize (Optional[bool]): Indicates whether the frames of data are standardized before projection is performed
        Returns:
            projected_frames (torch.tensor). Shape (fov_dim1 * fov_dim2, num_frames).
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
            mean_img_crop = self.mean_img[spatial_crop_terms].flatten()
            var_img_crop = self.var_img[spatial_crop_terms].flatten()
            u_indices = pixel_space_crop.flatten()
            u_crop = torch.index_select(self._u, 0, u_indices)
            implied_fov = pixel_space_crop.shape
        else:
            u_crop = self._u
            mean_img_crop = self.mean_img.flatten()
            var_img_crop = self.var_img.flatten()
            implied_fov = self.shape[1], self.shape[2]

        product = torch.sparse.mm(u_crop, v_crop)
        if self.rescale:
            product = (product * var_img_crop.unsqueeze(1)) + mean_img_crop.unsqueeze(1)

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


def convert_dense_image_stack_to_pmd_format(img_stack: Union[torch.tensor, np.ndarray]) -> PMDArray:
    """
    Adapter for converting a dense np.ndarray image stack into a pmd_array. Note that this does not
    run PMD compression; it simply reformats the data into the SVD format needed to construct a PMDArray object.
    The resulting PMDArray should contain identical data to img_stack (up to numerical precision errors).
    All computations are done in numpy on CPU here because that is the only approach that produces an SVD of the
    raw data that is exactly equal to img_stack.

    Args:
        img_stack (Union[np.ndarray, torch.tensor]): A (frames, fov_dim1, fov_dim2) shaped image stack
    Returns:
        pmd_array (masknmf.compression.PMDArray): img_stack expressed in the pmd_array format. pmd_array contains the
            same data as img_stack.
    """

    if isinstance(img_stack, np.ndarray):
        img_stack = torch.from_numpy(img_stack)

    if isinstance(img_stack, torch.Tensor):
        num_frames, fov_dim1, fov_dim2 = img_stack.shape
        img_stack_t = img_stack.permute(1, 2, 0).reshape(
            (fov_dim1 * fov_dim2, num_frames)
        )

        u = _construct_identity_torch_sparse_tensor(fov_dim1 * fov_dim2, device="cpu")
        mean_img = torch.zeros(fov_dim1, fov_dim2, device="cpu", dtype=torch.float32)
        var_img = torch.ones(fov_dim1, fov_dim2, device="cpu", dtype=torch.float32)

        return PMDArray(img_stack.shape,
                        u,
                        img_stack_t,
                        mean_img,
                        var_img,
                        u_local_projector=None,
                        u_global_projector=None,
                        device="cpu")

    else:
        raise ValueError(f"{type(img_stack)} not a supported type")




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
