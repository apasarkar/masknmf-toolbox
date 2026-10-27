import numpy as np
from masknmf.utils import SparseCOOTensor, is_cpu
from masknmf.arrays.array_interfaces import ArrayLike, TensorFlyWeight
import torch
from masknmf.arrays.array_utils import test_spatial_crop_effect, normalize_spatial_indexer


class StandardCorrelationImages(ArrayLike):

    def __init__(self,
                 flyweight: TensorFlyWeight,
                 fov_dims: tuple[int, int]):
        """
        see from_tensor docs for detailed parameter information
        """
        self._fov_dims = fov_dims
        self._flyweight = flyweight
        self.flyweight.validate_attributes(['spatial_compressed',
                                            'temporal_compressed',
                                            'temporal_demixed',
                                            'standard_correlation_image_mean',
                                            'standard_correlation_image_normalizer'])

        ## Caution: This tensor is "settable" (when you update self._temporal_demixed the correlation image dynamically changes).
        ## So the getter for should not just call flyweight.temporal_demixed, it should actually store its own field for this
        self._temporal_demixed = None
        self._temporal_demixed_mean = None
        self._temporal_demixed_centered_norm = None
        self.temporal_demixed = flyweight.temporal_demixed
        self._pixel_mat = torch.arange(self.shape[1]*self.shape[2], device=self.device, dtype=torch.long).reshape(
            self.shape[1], self.shape[2])
        self._ones_frames = torch.ones(
            (1, self.temporal_compressed.shape[1]), device=self.device, dtype=torch.float
        )

        self._spatial_compressed_csr = self.spatial_compressed.cpu().to_sparse_csr()


    @classmethod
    def from_tensors(
        cls,
        spatial_compressed: SparseCOOTensor,
        temporal_compressed: torch.Tensor,
        temporal_demixed: torch.Tensor,
        standard_correlation_image_mean: torch.Tensor,
        standard_correlation_image_normalizer: torch.Tensor,
        fov_dims: tuple[int, int],
    ):
        """
        Generates all the standard correlation images for the demixed data. It is more convenient to keep the
        correlation images in a factorized form and

        Args:
            spatial_compressed (torch.sparse_coo_tensor): shape (pixels, rank)
            temporal_compressed (torch.Tensor): shape (rank, frames)
            temporal_demixed (torch.Tensor): shape (frames, number of neural signals). This is the temporal traces matrix
            standard_correlation_image_mean (torch.Tensor): shape (pixels), the mean of spatial_compressed times temporal_compressed
            standard_correlation_image_normalizer (torch.Tensor): shape (pixels), the pixelwise l2 norm of (spatial_compressed times temporal_compressed) - movie_mean
            fov_dims (tuple[int, int]): A (fov_height, fov_width) tuple describing spatial imaging dimensions
        """
        flyweight = TensorFlyWeight(spatial_compressed=spatial_compressed,
                                    temporal_compressed=temporal_compressed,
                                    temporal_demixed=temporal_demixed,
                                    standard_correlation_image_mean=standard_correlation_image_mean,
                                    standard_correlation_image_normalizer=standard_correlation_image_normalizer)
        return cls(flyweight,
                   fov_dims)

    @classmethod
    def from_flyweight(cls,
                       flyweight: TensorFlyWeight,
                       fov_dims: tuple[int, int]):
        return cls(flyweight,
            fov_dims)

    @property
    def flyweight(self) -> TensorFlyWeight:
        return self._flyweight

    @property
    def device(self) -> torch.device | str:
        """
        This specifies what device the internal tensors used for the lazy computations are located.
        """
        return self._flyweight.device

    def to(self, new_device: str):
        if self._flyweight.device != new_device:
            self._flyweight.to(new_device)
        self._move_local_tensors(new_device)

    def _move_local_tensors(self, new_device: str):
        self._ones_frames = self._ones_frames.to(new_device)
        self._pixel_mat = self._pixel_mat.to(new_device)
        self._temporal_demixed = self._temporal_demixed.to(new_device)
        self._temporal_demixed_mean = self._temporal_demixed_mean.to(new_device)
        self._temporal_demixed_centered_norm = self._temporal_demixed_centered_norm.to(new_device)

    @property
    def temporal_demixed(self) -> torch.Tensor:
        """
        Because this tensor is settable, we do not want to default to the flyweight (in other words,
        this attribute is an 'extrinsic' property that is not really shared
        """
        return self._temporal_demixed

    @temporal_demixed.setter
    def temporal_demixed(self, new_tensor: torch.Tensor):
        if new_tensor.shape[0] != self.temporal_compressed.shape[1]:
            raise ValueError(
                f"Input temporal trace matrix has {new_tensor.shape[0]} frames"
                f"which is incompatible with the movie, which has {self.temporal_compressed.shape[1]} frames"
            )
        new_tensor = new_tensor.to(self.device)
        _temporal_demixed_var, self._temporal_demixed_mean = torch.var_mean(new_tensor, dim=0, correction=0, keepdim=True)
        self._temporal_demixed_centered_norm = torch.sqrt((_temporal_demixed_var * new_tensor.shape[0]))
        self._temporal_demixed = new_tensor

    @property
    def spatial_compressed(self) -> torch.sparse_coo_tensor:
        return self.flyweight.spatial_compressed

    @property
    def temporal_compressed(self) -> torch.Tensor:
        return self.flyweight.temporal_compressed

    @property
    def standard_correlation_image_mean(self) -> torch.Tensor:
        return self.flyweight.standard_correlation_image_mean

    @property
    def standard_correlation_image_normalizer(self) -> torch.Tensor:
        return self.flyweight.standard_correlation_image_normalizer

    @property
    def shape(self) -> tuple[int, int, int]:
        """(num_frames, fov_height, fov_width)"""
        return self.temporal_demixed.shape[1], self._fov_dims[0], self._fov_dims[1]

    @property
    def ndim(self) -> int:
        return len(self.shape)

    @property
    def dtype(self) -> torch.dtype:
        return torch.float32

    def __getitem__(
        self,
        item: int | list | np.ndarray | tuple[int | np.ndarray | slice | range],
    ) -> torch.Tensor:
        frame_indexer, item = self._parse_indices(item)

        # Step 3: Now slice the data with frame_indexer (careful: if the ndims has shrunk, add a dim)
        # Note: "frames" here are images for each neuron. These are NOT the timepoints of the signals
        c_crop = self.temporal_demixed[:, frame_indexer]
        if c_crop.ndim < self.temporal_demixed.ndim:
            c_crop = c_crop.unsqueeze(1)
        c_crop = c_crop - self._temporal_demixed_mean[:, frame_indexer]
        c_crop /= self._temporal_demixed_centered_norm[:, frame_indexer]
        c_crop.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0) #In-place

        vc_crop = self.temporal_compressed @ c_crop

        spatial = normalize_spatial_indexer(item, self.shape[1:])
        if test_spatial_crop_effect(spatial, self.shape[1:]):
            pixel_space_crop = self._pixel_mat[spatial]
            u_indices = pixel_space_crop.flatten()
            u_crop = torch.index_select(self.spatial_compressed, 0, u_indices)
            # mean_crop = torch.index_select(self.standard_correlation_image_mean, 0, u_indices)
            movie_normalizer_crop = torch.index_select(
                self.standard_correlation_image_normalizer, 0, u_indices
            )
            implied_fov = pixel_space_crop.shape
        else:
            if is_cpu(self.device):
                u_crop = self._spatial_compressed_csr
            else:
                u_crop = self.spatial_compressed
            movie_normalizer_crop = self.standard_correlation_image_normalizer
            implied_fov = self.shape[1], self.shape[2]

        """
        Now we do ((UV - Mean)/Normalizer) @ (c_centered / c_normalizer) 
        Here division is elementwise division
        Note that since c is centered, (mean / normalizer) @ (c_centered/c_normalizer) is 0, so we can omit this
        """

        product = torch.sparse.mm(u_crop, vc_crop)
        product /= movie_normalizer_crop.unsqueeze(1)
        product.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)

        product = product.reshape((implied_fov[0], implied_fov[1], -1))
        return product.permute(2, 0, 1).to(self.dtype)