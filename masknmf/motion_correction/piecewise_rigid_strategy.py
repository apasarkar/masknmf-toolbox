from masknmf.arrays.array_interfaces import LazyFrameLoader, ArrayLike
from masknmf.utils import Serializer
import torch
from masknmf.utils import torch_select_device
from masknmf.motion_correction.strategies import MotionCorrectionStrategy
from masknmf.motion_correction.rigid_strategy import RigidMotionCorrector
from typing import *
import numpy as np
from masknmf.motion_correction.registration_methods import register_frames_pwrigid, pwrigid_shift_estimation_routine, apply_pwrigid_shifts
import math
from tqdm import tqdm
import copy


def _axis_indices(indexer, dim: int) -> tuple[np.ndarray, bool]:
    """
    Normalize a single-axis indexer into an explicit array of positive indices.

    Returns:
        (indices, drop_axis). ``drop_axis`` is True for integer indexers, which remove the
        axis from the result (matching numpy semantics).
    """
    if isinstance(indexer, (bool, np.bool_)):
        raise IndexError("Scalar boolean indices are not supported")

    if isinstance(indexer, (int, np.integer)):
        i = int(indexer)
        if i < 0:
            i += dim
        if not (0 <= i < dim):
            raise IndexError(f"Index {indexer} is out of bounds for axis of size {dim}")
        return np.array([i], dtype=np.int64), True

    if isinstance(indexer, slice):
        return np.arange(*indexer.indices(dim), dtype=np.int64), False

    if isinstance(indexer, range):
        return np.asarray(list(indexer), dtype=np.int64), False

    arr = np.asarray(indexer)
    if arr.dtype == np.bool_:
        if arr.shape != (dim,):
            raise IndexError(
                f"Boolean mask of shape {arr.shape} does not match axis of size {dim}"
            )
        return np.nonzero(arr)[0].astype(np.int64), False

    arr = arr.astype(np.int64, copy=True).ravel()
    arr[arr < 0] += dim
    if arr.size and (arr.min() < 0 or arr.max() >= dim):
        raise IndexError(f"Index out of bounds for axis of size {dim}")
    return arr, False

def _is_contiguous_run(idx: np.ndarray) -> bool:
    """True if ``idx`` is an ascending run with step 1 (so it can be read as a slice)."""
    return idx.size > 0 and bool(np.all(np.diff(idx) == 1))


class PiecewiseRigidMotionCorrector(MotionCorrectionStrategy, Serializer):
    _serialized = {
        "num_blocks",
        "overlaps",
        "max_rigid_shifts",
        "max_deviation_rigid",
        "template",
        "pixel_weighting",
        "batch_size"
    }

    def __init__(
            self,
            num_blocks: tuple[int, int] = (12, 12),
            overlaps: tuple[int, int] = (5, 5),
            max_rigid_shifts: tuple[int, int] = (15, 15),
            max_deviation_rigid: tuple[int, int] = (2, 2),
            template: Optional[np.ndarray] = None,
            pixel_weighting: Optional[np.ndarray] = None,
            batch_size: int = 200,
            device: str = "auto",
    ):
        super().__init__(template, batch_size=batch_size, device=device)
        self._num_blocks = num_blocks
        self._overlaps = overlaps
        self._max_rigid_shifts = max_rigid_shifts
        self._max_deviation_rigid = max_deviation_rigid
        self._pixel_weighting = pixel_weighting.astype('float') if pixel_weighting is not None else None

    @property
    def num_blocks(self) -> tuple[int, int]:
        """
        Number of blocks that the image plane is split into, [rows, cols].
        Motion is estimated in each block and then interpolated in 2D space across the entire image plane.
        """
        return self._num_blocks

    @num_blocks.setter
    def num_blocks(self, value):
        value = self._validate_tuple_int_int("num_blocks", value)

        self._template = None
        self._num_blocks = value


    @property
    def dtype(self) -> torch.dtype:
        return torch.float32

    @property
    def pixel_weighting(self) -> None | np.ndarray:
        return self._pixel_weighting

    def _pixel_weighting_tensor(self) -> Optional[torch.Tensor]:
        if self.pixel_weighting is None:
            return None
        return torch.as_tensor(
            self.pixel_weighting, device=self.device, dtype=self.dtype
        )

    @property
    def overlaps(self) -> tuple[int, int]:
        """Number of pixels that overlap between adjacent blocks"""
        return self._overlaps

    @overlaps.setter
    def overlaps(self, value):
        value = self._validate_tuple_int_int("overlaps", value)

        self._template = None
        self._overlaps = value

    @property
    def max_rigid_shifts(self) -> tuple[int, int]:
        """maximum shift for rigid iterations, [rows, cols] before piece-wise rigid registration"""
        return self._max_rigid_shifts

    @max_rigid_shifts.setter
    def max_rigid_shifts(self, value: tuple[int, int]):
        value = self._validate_tuple_int_int("max_rigid_shifts", value)

        self._template = None
        self._max_rigid_shifts = value

    @property
    def max_deviation_rigid(self) -> tuple[int, int]:
        """maximum shift allowed in each piece-wise block relative to the rigid registered frame"""
        return self._max_deviation_rigid

    @max_deviation_rigid.setter
    def max_deviation_rigid(self, value):
        value = self._validate_tuple_int_int("max_deviation_rigid", value)

        self._template = None
        self._max_deviation_rigid = value

    def _correct_singlebatch(
            self,
            reference_frames: np.ndarray | torch.Tensor,
            target_frames: np.ndarray | torch.Tensor | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:

        if self.template is None:
            raise ValueError(
                "Template is uninitialized"
            )

        if target_frames is not None:
            target_frames = torch.as_tensor(target_frames, device=self.device, dtype=self.dtype)

        reference_frames = torch.as_tensor(reference_frames, device=self.device, dtype=self.dtype)

        template = torch.as_tensor(self.template, device=self.device, dtype=self.dtype)
        if self.pixel_weighting is not None:
            pixel_weighting = torch.as_tensor(self.pixel_weighting, device=self.device, dtype=self.dtype)
        else:
            pixel_weighting = None

        outputs = register_frames_pwrigid(
            reference_frames.to(self.device),
            template,
            self.num_blocks,
            self.overlaps,
            self.max_rigid_shifts,
            self.max_deviation_rigid,
            target_frames=target_frames,
            pixel_weighting=pixel_weighting
        )

        return outputs[0].cpu().numpy(), outputs[1].cpu().numpy()

    def compute_template(
            self,
            frames: ArrayLike | LazyFrameLoader,
            num_splits_per_iteration: int = 10,
            num_frames_per_split: int = 200,
            num_iterations: int = 1,
    ):
        rigid_strategy = RigidMotionCorrector(
            self.max_rigid_shifts,
            template=self.template if self.template is not None else None,
            pixel_weighting=self.pixel_weighting,
            batch_size=self.batch_size,
            device=self.device,
        )

        rigid_strategy.compute_template(
            frames,
        )

        self._template = rigid_strategy.template

        super().compute_template(
            frames,
            num_splits_per_iteration=num_splits_per_iteration,
            num_frames_per_split=num_frames_per_split,
            num_iterations=num_iterations,
        )
        torch.cuda.empty_cache()


    def _compute_all_shifts(self, reference_movie: ArrayLike) -> np.ndarray:
        """
        Estimate the (n_frames, K1, K2, 2) shift field in a single streaming pass, where K1 and K2 are the number of
        patches in the height and width dimensions of the FOV, respectively.
        """
        ## Maybe add a warning
        if self.template is None:
            self.compute_template(reference_movie)

        num_frames = reference_movie.shape[0]
        num_batches = math.ceil(num_frames / self.batch_size)
        shifts_total = []
        template = torch.as_tensor(self.template, device=self.device, dtype=self.dtype)
        pixel_weighting = self._pixel_weighting_tensor()

        display("extracting all shifts from dataset")
        for k in tqdm(range(num_batches)):
            start = k * self.batch_size
            end = min(start + self.batch_size, num_frames)
            reference_subset = torch.as_tensor(
                reference_movie[start:end], device=self.device, dtype=self.dtype
            )
            if reference_subset.ndim == 2:
                reference_subset = reference_subset[None, ...]

            current_shifts = pwrigid_shift_estimation_routine(reference_subset,
                                                     template,
                                                     self.num_blocks,
                                                     self.overlaps,
                                                     self.max_rigid_shifts,
                                                     self.max_deviation_rigid,
                                                     pixel_weighting=pixel_weighting)
            shifts_total.append(current_shifts.cpu()) ## Move to CPU to avoid overloading

        shifts_total = torch.concatenate(shifts_total, dim=0)
        return shifts_total.cpu()

    def motion_correct(
            self,
            reference_movie: ArrayLike,
            target_movie: ArrayLike | None = None,
    ) -> ArrayLike:

        shift_data = self._compute_all_shifts(reference_movie)
        return PiecewiseRigidRegistrationArray(reference_movie,
                                               shift_data,
                                               copy.deepcopy(self),
                                               target_movie=target_movie)

class PiecewiseRigidRegistrationArray(ArrayLike, Serializer):

    def __init__(self,
                 reference_movie: ArrayLike,
                 shifts: torch.Tensor,
                 strategy: PiecewiseRigidMotionCorrector,
                 output_device: str = 'cpu',
                 target_movie: ArrayLike | None = None):
        self._reference_movie = reference_movie
        self._shifts = shifts
        self._strategy = strategy
        self._device = self.strategy.device
        self._output_device = output_device
        if target_movie is None:
            self._target_movie = self.reference_movie
        else:
            self._target_movie = target_movie

    @property
    def ndim(self) -> int:
        return len(self.shape)

    @property
    def shape(self) -> tuple[int, int, int]:
        return self._target_movie.shape

    @property
    def dtype(self) -> torch.dtype:
        return torch.float32

    @property
    def nbytes(self) -> int:
        return math.prod(self.shape) * self.dtype.itemsize

    @property
    def reference_movie(self) -> ArrayLike:
        return self._reference_movie

    @property
    def target_movie(self) -> ArrayLike:
        return self._target_movie

    @property
    def shifts(self) -> torch.Tensor:
        return self._shifts

    @property
    def strategy(self) -> PiecewiseRigidMotionCorrector:
        return self._strategy

    @property
    def output_device(self):
        return self._output_device

    @output_device.setter
    def output_device(self, new_device:str):
        self._output_device = new_device

    def _corrected_roi(self,
                       frames,
                       row_slice,
                       col_slice):
        num_batches = math.ceil(frames.shape[0] / self.strategy.batch_size)

        outputs = []
        for k in range(num_batches):
            start = k * self.strategy.batch_size
            end = start + self.strategy.batch_size

            ## Below routine intelligently slices the smallest subset of data possible
            block = apply_pwrigid_shifts(self.target_movie,
                                         self.shifts,
                                         row_slice,
                                         col_slice,
                                         temporal_indices=frames[start:end],
                                         device=self._device).to(self.output_device)
            outputs.append(block)

        return torch.concatenate(outputs, dim=0)

    def __getitem__(self, idx):
        frame_indexer, item = self._parse_indices(idx)

        frames, _ = _axis_indices(frame_indexer, self.shape[0])

        spatial = item[1:] if isinstance(item, tuple) else ()
        spatial = tuple(spatial) + (slice(None),) * (2 - len(spatial))

        rows, drop_rows = _axis_indices(spatial[0], self.shape[1])
        cols, drop_cols = _axis_indices(spatial[1], self.shape[2])

        if frames.size == 0 or rows.size == 0 or cols.size == 0:
            empty = torch.empty(
                (frames.size, rows.size, cols.size),
                dtype=self.dtype,
                device=self._output_device,
            )
            return self._drop_axes(empty, drop_rows, drop_cols)

        row_lo, row_hi = int(rows.min()), int(rows.max())
        col_lo, col_hi = int(cols.min()), int(cols.max())

        row_slice = slice(row_lo, row_hi + 1, 1)
        col_slice = slice(col_lo, col_hi + 1, 1)

        block = self._corrected_roi(frames,
                                    row_slice,
                                    col_slice)

        # Residual selection inside the bounding box. Only the (small) materialized block is
        # touched here, so steps / fancy indices / boolean masks cost essentially nothing.
        row_local = rows - row_lo
        col_local = cols - col_lo
        if not (_is_contiguous_run(row_local) and row_local.size == block.shape[1]):
            block = block[:, torch.as_tensor(row_local, device=block.device), :]
        if not (_is_contiguous_run(col_local) and col_local.size == block.shape[2]):
            block = block[:, :, torch.as_tensor(col_local, device=block.device)]

        block = block.to(self._output_device)
        return self._drop_axes(block, drop_rows, drop_cols)

    @staticmethod
    def _drop_axes(block: torch.Tensor, drop_rows: bool, drop_cols: bool) -> torch.Tensor:
        """Collapse axes selected by an integer index, meant to match numpy"""
        if drop_cols:
            block = block[:, :, 0]
        if drop_rows:
            block = block[:, 0]
        return block


