from masknmf.arrays.array_interfaces import ArrayLike, TensorFlyWeight
from masknmf.utils._serialization import load_dict
from masknmf.utils import Serializer
from masknmf.utils import SparseCOOTensor
import torch
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

class CompressionArray(ArrayLike, Serializer):
    """
    Factorized demixing array interface for the compressed representation of the video
    """
    _serialized = {
        "shape",
        "spatial_compressed",
        "temporal_compressed",
        "spatial_compressed_local_projector",
        "mean_image",
        "noise_variance_image",
        "spatial_trend_basis",
        "temporal_trend_basis",
    }

    def __init__(
        self,
        shape: tuple[int, int, int] | np.ndarray,
        flyweight: TensorFlyWeight,
        device: str = "cpu",
        rescale: bool = True,
        include_trend: bool = False
    ):
        """
        See from_tensors class method for documentation
        """
        self._shape = tuple(shape)
        self.rescale = rescale
        self.include_trend = include_trend

        ##Set up the flyweight and all other tensors
        self._flyweight = flyweight
        self._flyweight.to(device)

        self._pixel_index_image = torch.arange(
            self.shape[1] * self.shape[2], device=self.flyweight.device,
        ).reshape(self.shape[1], self.shape[2])



    @property
    def flyweight(self) -> TensorFlyWeight:
        return self._flyweight
    @classmethod
    def from_tensors(cls,
                     shape: tuple[int, int, int] | np.ndarray,
                     spatial_compressed: SparseCOOTensor,
                     temporal_compressed: torch.Tensor,
                     mean_image: torch.Tensor,
                     noise_variance_image: torch.Tensor,
                     spatial_compressed_local_projector: SparseCOOTensor | None = None,
                     spatial_trend_basis: torch.Tensor | None = None,
                     temporal_trend_basis: torch.Tensor | None = None,
                     device: str = "cpu",
                     rescale: bool = True,
                     include_trend: bool = False):

        """
            Key assumption: the spatial basis matrix spatial_compressed has n + k columns; the first n columns is blocksparse (this serves
            as a local spatial basis for the data) and the last k columns can have unconstrained spatial support (these serve
            as a global spatial basis for the data).

            Args:
                shape (tuple): (num_frames, fov_height, fov_width)
                spatial_compressed (SparseCOOTensor): shape (pixels, rank)
                temporal_compressed (torch.tensor): shape (rank, frames)
                mean_image (torch.tensor): shape (fov_height, fov_width). The pixelwise mean of the data
                noise_variance_image (torch.tensor): shape (fov_height, fov_width). A pixelwise noise normalizer for the data
                spatial_compressed_local_projector (SparseCOOTensor | None): shape (pixels, rank)
                spatial_trend_basis (torch.Tensor | None): Shape (pixels, trend_rank)
                temporal_trend_basis (torch.Tensor | None): Shape (trend_rank, num_frames)
                    spatial_trend_basis @ temporal_trend_basis gives the trend estimate across the full movie
                device (str): The device on which computations occur/data is stored
                rescale (bool): True if we rescale the PMD data (i.e. multiply by the pixelwise normalizer
                    and add back the mean) in __getitem__
                include_trend (bool): Whether or not to include the trend for this data
        """
        flyweight = TensorFlyWeight(spatial_compressed=spatial_compressed.float(),
                                    temporal_compressed=temporal_compressed.float(),
                                    mean_image=mean_image.float(),
                                    noise_variance_image=noise_variance_image.float(),
                                    spatial_compressed_local_projector=spatial_compressed_local_projector.float() if spatial_compressed_local_projector is not None else None,
                                    spatial_trend_basis=spatial_trend_basis.float() if spatial_trend_basis is not None else None,
                                    temporal_trend_basis=temporal_trend_basis.float() if temporal_trend_basis is not None else None)
        return cls(shape,
                   flyweight,
                   device=device,
                   rescale = rescale,
                   include_trend=include_trend)


    @classmethod
    def from_flyweight(cls,
                       shape: tuple[int, int, int] | np.ndarray,
                       flyweight: TensorFlyWeight,
                       rescale: bool = True,
                       include_trend: bool = False
                       ):
        """
        Memory efficient way to construct CompressionArray from a flyweight tensor manager. See from_tensors for parameter documentation
        """
        return cls(shape,
                   flyweight,
                   device=flyweight.device,
                   rescale=rescale,
                   include_trend=include_trend)

    @classmethod
    def from_hdf5(cls, path, prefix: str = "", **kwargs):
        d = load_dict(path, f"{prefix}/{cls.__name__}" if prefix else cls.__name__)
        return cls.from_tensors(**d, **kwargs)

    @property
    def rescale(self) -> bool:
        return self._rescale

    @rescale.setter
    def rescale(self, new_state: bool):
        """
        Setting rescale to False will also set include_trend to False. This is done because the pixelwise trends
        are at the raw data scale, so it does not make sense to ever show them with the rest of the compression is at a different scale
        """
        self._rescale = new_state
        if not new_state:
            self._include_trend = False

    @property
    def include_trend(self) -> bool:
        return self._include_trend

    @include_trend.setter
    def include_trend(self, new_val: bool):
        if not self.rescale and new_val:
            raise ValueError("Cannot add back trend if the data is not being scaled back to the raw data space. self.rescale must be True"
                             "If using the constructor, pass rescale = True.")
        else:
            self._include_trend = new_val

    @property
    def mean_image(self) -> torch.Tensor:
        return self.flyweight.mean_image

    @property
    def noise_variance_image(self) -> torch.Tensor:
        return self.flyweight.noise_variance_image

    def to(self, new_device: torch.device | str):
        if self._flyweight.device != new_device:
            self._flyweight.to(new_device)
        self._move_local_tensors(new_device)

    def _move_local_tensors(self, new_device: torch.device | str):
        self._pixel_index_image = self._pixel_index_image.to(new_device)

    @property
    def device(self) -> torch.device | str:
        return self.flyweight.device

    @property
    def spatial_compressed(self) -> SparseCOOTensor:
        return self.flyweight.spatial_compressed

    @property
    def spatial_compressed_local_projector(self) -> SparseCOOTensor | None:
        if hasattr(self.flyweight, "spatial_compressed_local_projector"):
            return self.flyweight.spatial_compressed_local_projector
        return None

    @property
    def temporal_compressed(self) -> torch.Tensor:
        return self.flyweight.temporal_compressed

    @property
    def spatial_trend_basis(self) -> torch.Tensor | None:
        if hasattr(self.flyweight, "spatial_trend_basis"):
            return self.flyweight.spatial_trend_basis
        return None

    @property
    def temporal_trend_basis(self) -> torch.Tensor | None:
        if hasattr(self.flyweight, "temporal_trend_basis"):
            return self.flyweight.temporal_trend_basis
        return None

    @property
    def compression_rank(self) -> int:
        return self.spatial_compressed.shape[1]

    @property
    def dtype(self) -> type:
        """
        data type, default np.float32
        """
        return np.float32

    @property
    def pixel_index_image(self) -> torch.Tensor:
        """
        Shape (fov_height, fov_width). Gives the row of spatial_compressed that each pixel of the
        field of view corresponds to, so that a spatial crop can be turned into row indices.

        Always returned on the same device as the compressed tensors: the flyweight is shared between
        CompressionArray objects, so another object may have moved it since this one was constructed.
        """
        self._pixel_index_image = self._pixel_index_image.to(self.flyweight.device)  # no-op if already there
        return self._pixel_index_image

    @property
    def shape(self) -> tuple[int, int, int]:
        """
        Array shape (num_frames, fov_height, fov_width)
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
        Generates rank heatmap image based on spatial_compressed. Equal to row summation of binarized spatial_compressed matrix.
        Returns:
            rank_heatmap (torch.Tensor). Shape (fov_height, fov_width).
        """
        binarized_spatial_compressed = torch.sparse_coo_tensor(
            self.spatial_compressed.indices(),
            torch.ones_like(self.spatial_compressed.values()),
            self.spatial_compressed.size()
            )
        row_sum_spatial_compressed = torch.sparse.sum(binarized_spatial_compressed, dim=1)
        return torch.reshape(row_sum_spatial_compressed.to_dense(),
                             (self.shape[1],self.shape[2]))

    def project_frames(
        self, frames: torch.Tensor, standardize: bool = True
    ) -> torch.Tensor:
        """
        Projects frames onto the spatial basis, using the u_projector property. u_projector must be defined.
        Args:
            frames (torch.Tensor). Shape (fov_height, fov_width, num_frames) or (fov_height*fov_width, num_frames).
                Frames which we want to project onto the spatial basis.
            standardize (bool): Indicates whether the frames of data are standardized before projection is performed
        Returns:
            projected_frames (torch.Tensor). Shape (fov_height * fov_width, num_frames).
        """
        if self.spatial_compressed_local_projector is None:
            raise ValueError(
                "spatial_compressed_local_projector must be defined to project frames onto spatial basis"
            )
        orig_device = frames.device
        frames = frames.to(self.device).float()
        if len(frames.shape) == 3:
            if standardize:
                frames = (frames - self.mean_image[..., None]) / self.noise_variance_image[
                    ..., None
                ]  # Normalize the frames
                frames = torch.nan_to_num(frames, nan=0.0)
            frames = frames.reshape(self.shape[1] * self.shape[2], -1)
        else:
            if standardize:
                frames = (
                    frames - self.mean_image.flatten()[..., None]
                ) / self.noise_variance_image.flatten()[..., None]
                frames = torch.nan_to_num(frames, nan=0.0)

        projection = torch.sparse.mm(self.spatial_compressed_local_projector.T, frames)
        return projection.to(orig_device)

    def getitem_tensor(
        self,
        item: int | list | np.ndarray | tuple[int | np.ndarray | slice | range, ...],
    ) -> torch.Tensor:
        frame_indexer, item = self._parse_indices(item)

        # Step 3: Now slice the data with frame_indexer (careful: if the ndims has shrunk, add a dim)
        temporal_compressed_crop = self.temporal_compressed[:, frame_indexer]
        if temporal_compressed_crop.ndim < self.temporal_compressed.ndim:
            temporal_compressed_crop = temporal_compressed_crop.unsqueeze(1)


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

            pixel_space_crop = self.pixel_index_image[spatial_crop_terms]
            mean_image_crop = self.mean_image[spatial_crop_terms].flatten()
            noise_variance_image_crop = self.noise_variance_image[spatial_crop_terms].flatten()
            spatial_compressed_indices = pixel_space_crop.flatten()
            spatial_compressed_crop = torch.index_select(self.spatial_compressed, 0, spatial_compressed_indices)
            implied_fov = pixel_space_crop.shape
        else:
            spatial_crop_terms = None
            spatial_compressed_crop = self.spatial_compressed
            mean_image_crop = self.mean_image.flatten()
            noise_variance_image_crop = self.noise_variance_image.flatten()
            implied_fov = self.shape[1], self.shape[2]

        product = torch.sparse.mm(spatial_compressed_crop, temporal_compressed_crop)
        if self.rescale:
            product *= noise_variance_image_crop.unsqueeze(1)
            product += mean_image_crop.unsqueeze(1)

            if self.include_trend and self.spatial_trend_basis is not None and self.temporal_trend_basis is not None:
                if spatial_crop_terms is not None:
                    pixel_space_crop = self.pixel_index_image[spatial_crop_terms].flatten()
                    spatial_trend_crop = self.spatial_trend_basis[pixel_space_crop]
                else:
                    spatial_trend_crop = self.spatial_trend_basis
                temporal_trend_crop = self.temporal_trend_basis[:, frame_indexer]
                if temporal_trend_crop.ndim < self.temporal_trend_basis.ndim:
                     temporal_trend_crop = temporal_trend_crop.unsqueeze(1)
                product += spatial_trend_crop @ temporal_trend_crop

        product = product.reshape((implied_fov[0], implied_fov[1], -1))
        product = product.permute(2, 0, 1)

        return product

    def __getitem__(
        self,
        item: int | list | np.ndarray | tuple[int | np.ndarray | slice | range, ...],
    ) -> np.ndarray:
        product = self.getitem_tensor(item)
        product = product.cpu().numpy().astype(self.dtype)
        return product


class CompressionResidualArray(ArrayLike):
    """
    An array-like interface for examining the residual movie associated with running compression:
    Raw_Movie - Compressed_Movie
    """

    def __init__(
        self,
        raw_array: ArrayLike,
        compression_array: CompressionArray,
    ):
        """
        Args:
            raw_array (LazyFrameLoader): Any object that supports LazyFrameLoder functionality
            compression_array (CompressionArray)
        """

        ## This object has its own CompressionArray, so we can set its state without affecting any other workflows
        self.compression_array = CompressionArray.from_flyweight(compression_array.shape,
                                                                 compression_array.flyweight)

        ## We want the compression to be rescaled pixelwise to the raw data scale and also include all trends
        self.compression_array.rescale = True
        self.compression_array.include_trend = True
        self.raw_array = raw_array
        self._shape = self.compression_array.shape

        if self.compression_array.shape != self.raw_array.shape:
            raise ValueError("Two image stacks do not have the same shape")


    @property
    def dtype(self) -> type:
        """
        data type, default np.float32
        """
        return self.compression_array.dtype

    @property
    def shape(self) -> tuple[int, int, int]:
        """
        Array shape (num_frames, fov_height, fov_width)
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
            item: int | list | np.ndarray | tuple[int | np.ndarray | slice | range, ...],
    ):
        output = self.raw_array[item].astype(self.dtype) - self.compression_array[item].astype(self.dtype)
        return output

class TrendArray(ArrayLike):
    """
    Utility class that exposes the trend estimates in CompressionArray as an array-like object.
    We don't support serialization or any other things here to keep it simple -- all of that is in the CompressionArray class
    """
    def __init__(self,
                 shape: tuple[int, int, int] | np.ndarray,
                 flyweight: TensorFlyWeight,
                 device: str = "cpu"):
        self._shape = tuple(shape)

        ##Set up the flyweight and all other tensors
        self._flyweight = flyweight
        self._flyweight.to(device)

        self._pixel_index_image = torch.arange(
            self.shape[1] * self.shape[2], device=self.flyweight.device,
        ).reshape(self.shape[1], self.shape[2])

    @classmethod
    def from_flyweight(cls,
                       shape: tuple[int, int, int] | np.ndarray,
                       flyweight: TensorFlyWeight,
                       ):
        """
        Memory efficient way to construct PMD Array from a flyweight tensor manager. See from_tensors for parameter documentation
        """
        return cls(shape,
                   flyweight,
                   device=flyweight.device)

    @classmethod
    def from_tensors(cls,
                     shape: tuple[int, int, int] | np.ndarray,
                     spatial_trend_basis: torch.Tensor,
                     temporal_trend_basis: torch.Tensor,
                     device: str = "cpu"):
        """
        The trend estimate is a pixels x frames estimate given by the product spatial_trend_basis x temporal_trend_basis.
        This class provides array-like access to this trend estimate across the full movie

        Args:
            shape (tuple): (num_frames, fov_height, fov_width)
            spatial_trend_basis (torch.Tensor): The spatial basis for the trend estimate, shape (num_pixels, rank)
            temporal_trend_basis (torch.Tensor): The temporal basis for the trend estimate, shape (rank, num_frames)
            device (str): The device on which computations occur/data is stored
        """
        flyweight = TensorFlyWeight(spatial_trend_basis=spatial_trend_basis.float(),
                                    temporal_trend_basis=temporal_trend_basis.float())
        return cls(shape,
                   flyweight,
                   device=device)

    @property
    def flyweight(self):
        return self._flyweight

    @property
    def dtype(self) -> type:
        """
        data type, default np.float32
        """
        return np.float32

    @property
    def spatial_trend_basis(self) -> torch.Tensor:
        """
        Return a spatial trend basis of shape (num_pixels, trend_rank)
        """
        return self.flyweight.spatial_trend_basis

    @property
    def temporal_trend_basis(self) -> torch.Tensor:
        """
        Returns a temporal trend basis of shape (trend_rank, num_frames)
        """
        return self.flyweight.temporal_trend_basis

    @property
    def pixel_index_image(self) -> torch.Tensor:
        """
        Shape (fov_height, fov_width). Gives the row of spatial_trend_basis that each pixel of the
        field of view corresponds to, so that a spatial crop can be turned into row indices.

        Always returned on the same device as the flyweight tensors
        """
        self._pixel_index_image = self._pixel_index_image.to(self.flyweight.device)  # no-op if already there
        return self._pixel_index_image

    def to(self, new_device: torch.device | str):
        if self._flyweight.device != new_device:
            self._flyweight.to(new_device)
        self._move_local_tensors(new_device)

    def _move_local_tensors(self, new_device: torch.device | str):
        self._pixel_index_image = self._pixel_index_image.to(new_device)

    @property
    def device(self) -> torch.device | str:
        return self.flyweight.device

    @property
    def shape(self) -> tuple[int, int, int]:
        """(num_frames, fov_height, fov_width)"""
        return self._shape

    def getitem_tensor(
            self,
            item: int | list | np.ndarray | tuple[int | np.ndarray | slice | range, ...],
    ) -> torch.Tensor:
        frame_indexer, item = self._parse_indices(item)

        # Step 3: Now slice the data with frame_indexer (careful: if the ndims has shrunk, add a dim)
        temporal_crop = self.temporal_trend_basis[:, frame_indexer]
        if temporal_crop.ndim < self.temporal_trend_basis.ndim:
            temporal_crop = temporal_crop.unsqueeze(1)

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

            pixel_space_crop = self.pixel_index_image[spatial_crop_terms]
            spatial_indices = pixel_space_crop.flatten()
            spatial_crop = torch.index_select(self.spatial_trend_basis, 0, spatial_indices)
            implied_fov = pixel_space_crop.shape
        else:

            spatial_crop = self.spatial_trend_basis
            implied_fov = self.shape[1], self.shape[2]

        product = spatial_crop @ temporal_crop

        product = product.reshape((implied_fov[0], implied_fov[1], -1))
        product = product.permute(2, 0, 1)

        return product

    def __getitem__(
            self,
            item: int | list | np.ndarray | tuple[int | np.ndarray | slice | range, ...],
    ) -> np.ndarray:
        product = self.getitem_tensor(item)
        product = product.cpu().numpy().astype(self.dtype)
        return product