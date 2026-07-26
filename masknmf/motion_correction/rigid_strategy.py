import copy
import math
from pathlib import Path
from typing import Optional, Union, Sequence

import numpy as np
import torch

from masknmf import ArrayLike, LazyFrameLoader
from masknmf.utils import Serializer, display
from masknmf.motion_correction.strategies import MotionCorrectionStrategy
from masknmf.motion_correction.registration_methods import (
    register_frames_rigid,
    estimate_rigid_shifts,
    apply_rigid_shifts,
    interpolate_to_border,
)
from tqdm import tqdm

from masknmf.utils._serialization import save_dict, load_dict

_FAST_FFT_FACTORS = (2, 3, 5, 7)

def next_fast_len(n: int, factors: Sequence[int] = _FAST_FFT_FACTORS) -> int:
    """
    Smallest integer >= n whose prime factorization contains only ``factors``.

    cuFFT/FFTW use radix kernels for small prime factors and fall back to Bluestein's
    algorithm for large prime factors, which is dramatically slower. A 32-pixel ROI with a
    15-pixel guard band on each side gives a 62-length transform (62 = 2 * 31), which hits
    the slow path; rounding up to 64 is both faster and gives a wider guard band.
    """
    if n <= 1:
        return 1
    while True:
        m = n
        for f in factors:
            while m % f == 0:
                m //= f
        if m == 1:
            return n
        n += 1


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


def _read_key(idx: np.ndarray):
    """Prefer a slice over a fancy-index array so backends can use hyperslab reads."""
    if _is_contiguous_run(idx):
        return slice(int(idx[0]), int(idx[-1]) + 1)
    return idx


class RigidMotionCorrector(MotionCorrectionStrategy, Serializer):
    _serialized = {"max_shifts", "template", "pixel_weighting", "batch_size"}

    def __init__(
        self,
        max_shifts: tuple[int, int] = (15, 15),
        template: Optional[np.ndarray] = None,
        pixel_weighting: Optional[np.ndarray] = None,
        batch_size: int = 200,
        device: str = "auto",
    ):
        super().__init__(template, batch_size=batch_size, device=device)
        self._max_shifts = max_shifts
        self._pixel_weighting = pixel_weighting

    @property
    def max_shifts(self) -> tuple[int, int]:
        """Max allowed shift in (row, col)."""
        return self._max_shifts

    @max_shifts.setter
    def max_shifts(self, value: tuple[int, int]):
        value = self._validate_tuple_int_int("max_shifts", value)
        self._template = None
        self._max_shifts = value

    @property
    def dtype(self) -> torch.dtype:
        return torch.float32

    @property
    def pixel_weighting(self) -> Optional[np.ndarray]:
        return self._pixel_weighting

    def _pixel_weighting_tensor(self) -> Optional[torch.Tensor]:
        if self.pixel_weighting is None:
            return None
        return torch.as_tensor(
            self.pixel_weighting, device=self.device, dtype=self.dtype
        )

    def _correct_singlebatch(
        self,
        reference_frames: np.ndarray,
        target_frames: Optional[np.ndarray],
    ) -> tuple[np.ndarray, np.ndarray]:
        if self.template is None:
            raise ValueError("Template is uninitialized")

        if target_frames is not None:
            target_frames = torch.as_tensor(
                target_frames, device=self.device, dtype=self.dtype
            )
        reference_frames = torch.as_tensor(
            reference_frames, device=self.device, dtype=self.dtype
        )
        template = torch.as_tensor(self.template, device=self.device, dtype=self.dtype)

        outputs = register_frames_rigid(
            reference_frames,
            template,
            self.max_shifts,
            target_frames=target_frames,
            pixel_weighting=self._pixel_weighting_tensor(),
        )
        return outputs[0].cpu().numpy(), outputs[1].cpu().numpy()

    def _compute_all_shifts(self, reference_movie: ArrayLike) -> np.ndarray:
        """
        Estimate the (n_frames, 2) shift field in a single streaming pass.

        Estimation is inherently whole-frame (phase correlation against the template), so it
        cannot be restricted to an ROI. Doing it once up front is what makes subsequent
        spatially-cropped reads cheap.
        """
        if self.template is None:
            self.compute_template(reference_movie)

        num_frames = reference_movie.shape[0]
        num_batches = math.ceil(num_frames / self.batch_size)
        shifts_total = torch.zeros(
            num_frames, 2, device=self.device, dtype=self.dtype
        )
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

            shifts_total[start:end, :] = estimate_rigid_shifts(
                reference_subset,
                template,
                self.max_shifts,
                pixel_weighting=pixel_weighting,
            )

        return shifts_total.cpu()

    def motion_correct(
        self,
        reference_movie: ArrayLike,
        target_movie: Optional[ArrayLike] = None,
    ) -> ArrayLike:
        all_shifts = self._compute_all_shifts(reference_movie)
        return RigidRegistrationArray(
            reference_movie,
            all_shifts,
            copy.deepcopy(self),
            target_movie,
        )

class RigidRegistrationArray(ArrayLike, Serializer):
    """
    Lazy rigid-motion-corrected array supporting fast slicing in all dimensions.

    The per-frame shift field is estimated once (eagerly, by
    :meth:`RigidMotionCorrector.motion_correct`) and stored. ``__getitem__`` therefore only
    ever applies a known shift, never re-estimates one, which is what makes a small spatial
    ROI cheap.

    Accuracy note
    -------------
    A Fourier shift is exact band-limited (sinc) interpolation, which is a global operator:
    strictly, every output pixel depends on every input pixel. Applying it to a padded window
    is therefore an approximation:

    * the integer part of the shift is exact, provided the guard band is >= ceil(max|shift|)
      (the circular wrap lands entirely inside the discarded guard band);
    * the subpixel part is approximate, because the sinc tails are truncated at the window
      boundary. The error decays into the interior of the ROI with the tail of the kernel.

    ``sinc_margin`` controls that truncation error by guaranteeing a sufficiently large margin of pixels is present before applying
    Fourier interpolation. Use :func:`roi_consistency_error` to verify the margin is adequate for your data.
    """

    _motion_export_name = "motion_corrected"
    _shifts_export_name = "shifts"
    _serialized = {"shifts", "sinc_margin"}

    def __init__(
        self,
        reference_movie: ArrayLike,
        shifts: torch.Tensor,
        strategy: RigidMotionCorrector,
        target_movie: ArrayLike | None = None,
        sinc_margin: int = 12,
        output_device: Optional[str] = None,
    ):
        """
        Args:
            reference_movie: Stack the shifts were estimated from.
            shifts (torch.Tensor): Shape (n_frames, 2). Precomputed per-frame shifts.
            strategy (RigidMotionCorrector): Snapshot of the RigidMotionCorrector strategy that produced ``shifts`` (params + template).
            target_movie (LazyFrameLoader | ArrayLike): The stack to which shifts are applied. If ``None``, it is assumed that
                the reference_movie is the target_movie. Must match ``reference_movie``
                in frame count (but not necessarily FOV)
            sinc_margin: Extra guard band, in pixels, beyond ceil(max|shift|), giving the
                truncated sinc kernel room to decay. Added to, not maxed with, the wrap
                requirement.
            output_device: Device of the returned tensor. Defaults to the strategy device.
        """
        self._reference_movie = reference_movie
        self._target_movie = target_movie if target_movie is not None else self.reference_movie
        self._strategy = strategy

        self._shape = self.target_movie.shape
        self._device = strategy.device
        self._output_device = self._device if output_device is None else output_device

        if shifts.ndim != 2 or shifts.shape[1] != 2:
            raise ValueError(
                f"shifts must have shape (n_frames, 2); got {shifts.shape}"
            )
        if shifts.shape[0] != self._shape[0]:
            raise ValueError(
                f"Got {shifts.shape[0]} shifts for {self._shape[0]} frames"
            )
        self._shifts = torch.as_tensor(shifts, device=self._device, dtype=self.dtype)

        if target_movie is not None:
            if tuple(reference_movie.shape) != tuple(target_movie.shape):
                raise ValueError(
                    f"reference_movie shape {tuple(reference_movie.shape)} does not match "
                    f"target_movie shape {tuple(target_movie.shape)}"
                )



        # Guard band = circular-wrap requirement + sinc-truncation margin.
        wrap = torch.ceil(torch.amax(torch.abs(self.shifts), dim = 0)).to(torch.int64)
        self._sinc_margin = int(sinc_margin)
        self._padding = (
            int(wrap[0]) + self._sinc_margin,
            int(wrap[1]) + self._sinc_margin,
        )

    @property
    def ndim(self) -> int:
        return len(self.shape)

    @property
    def shape(self) -> tuple[int, int, int]:
        return self._shape

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
        """Shape (n_frames, 2). Precomputed per-frame (row, col) shifts."""
        return self._shifts

    @property
    def padding(self) -> tuple[int, int]:
        """Guard band (rows, cols) used when reading a windowed ROI."""
        return self._padding

    @property
    def sinc_margin(self) -> int:
        return self._sinc_margin

    @property
    def output_device(self) -> str:
        return self._output_device

    @output_device.setter
    def output_device(self, new_device: str):
        self._output_device = new_device

    @property
    def strategy(self) -> MotionCorrectionStrategy:
        return self._strategy

    # -- windowing ---------------------------------------------------------------------

    def _window_bounds(
        self, lo: int, hi: int, guard: int, dim: int
    ) -> tuple[int, int, int, int]:
        """
        Plan a padded, FFT-friendly window around the inclusive ROI extent ``[lo, hi]``.

        Returns:
            (start, stop, offset, length) where ``[start, stop)`` is the window in image
            coordinates (may extend outside ``[0, dim)``; those samples are edge-replicated),
            ``offset`` is where the ROI begins inside the window, and ``length = stop - start``
            is a smooth-factored FFT length.
        """
        extent = hi - lo + 1
        length = next_fast_len(extent + 2 * guard)
        slack = length - extent
        offset = slack // 2
        start = lo - offset
        stop = start + length
        return start, stop, offset, length

    def _gather_window(
        self,
        frame_key,
        row_start: int,
        row_stop: int,
        col_start: int,
        col_stop: int,
    ) -> torch.Tensor:
        """
        Read ``[row_start, row_stop) x [col_start, col_stop)``, edge-replicating any part of
        the window that falls outside the FOV.

        Replication is done by clamping the sample coordinates rather than by
        ``torch.nn.functional.pad``
        """
        height, width = self.shape[1], self.shape[2]

        read_r0, read_r1 = max(0, row_start), min(height, row_stop)
        read_c0, read_c1 = max(0, col_start), min(width, col_stop)

        block = self.target_movie[frame_key, read_r0:read_r1, read_c0:read_c1]
        block = torch.as_tensor(block, device=self._device, dtype=self.dtype)
        if block.ndim == 2:
            block = block[None, ...]

        rows = np.clip(np.arange(row_start, row_stop), 0, height - 1) - read_r0
        cols = np.clip(np.arange(col_start, col_stop), 0, width - 1) - read_c0

        if not (_is_contiguous_run(rows) and rows.size == block.shape[1]):
            block = block[:, torch.as_tensor(rows, device=self._device), :]
        if not (_is_contiguous_run(cols) and cols.size == block.shape[2]):
            block = block[:, :, torch.as_tensor(cols, device=self._device)]
        return block

    def _corrected_roi(
        self,
        frames: np.ndarray,
        row_lo: int,
        row_hi: int,
        col_lo: int,
        col_hi: int,
    ) -> torch.Tensor:
        """
        Motion-corrected block over ``frames`` restricted to the inclusive ROI bounding box.

        Batched over the frame axis so that a large temporal request cannot exhaust device
        memory: a full-FOV read of 4000 frames at 512x512 would otherwise allocate several GB
        for the intermediate complex spectrum alone.
        """
        row_start, row_stop, row_off, _ = self._window_bounds(
            row_lo, row_hi, self._padding[0], self.shape[1]
        )
        col_start, col_stop, col_off, _ = self._window_bounds(
            col_lo, col_hi, self._padding[1], self.shape[2]
        )
        row_extent = row_hi - row_lo + 1
        col_extent = col_hi - col_lo + 1

        batch = max(1, int(self._strategy.batch_size))
        chunks = []
        for start in range(0, frames.size, batch):
            frame_chunk = frames[start:min(frames.size, start + batch)]
            window = self._gather_window(
                _read_key(frame_chunk), row_start, row_stop, col_start, col_stop
            )

            ## If output_device is cpu, this should immediately move things off the GPU
            window = apply_rigid_shifts(
                window,
                self.shifts[torch.as_tensor(frame_chunk, device=self._device)],
            ).to(self.output_device)

            chunks.append(
                window[
                    :,
                    row_off : row_off + row_extent,
                    col_off : col_off + col_extent,
                ]
            )

        if len(chunks) == 1:
            return chunks[0]
        return torch.cat(chunks, dim=0)

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

        block = self._corrected_roi(frames, row_lo, row_hi, col_lo, col_hi)

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

    def export(self, path: str | Path):
        d_array = self._to_dict()
        d_strategy = self.strategy._to_dict()
        save_dict(d_array, filename=path, group=__class__.__name__)
        save_dict(d_strategy, filename=path, exists_ok=True, group=RigidMotionCorrector.__name__)

    @classmethod
    def from_hdf5(cls,
                  path,
                  reference_movie: ArrayLike,
                  target_movie: ArrayLike | None = None,
                  **kwargs):
        strat = RigidMotionCorrector(**load_dict(path, RigidMotionCorrector.__name__))
        reg_arr_dict = load_dict(path, __class__.__name__)
        return cls(reference_movie=reference_movie, target_movie=target_movie, strategy=strat, **reg_arr_dict)

# --------------------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------------------


def roi_consistency_error(
    array: RigidRegistrationArray,
    frames: slice = slice(0, 32),
    rows: slice = slice(None),
    cols: slice = slice(None),
) -> float:
    """
    Max absolute difference between the windowed ROI path and a full-FOV reference.

    The windowed path truncates the sinc interpolation kernel at the window boundary, so the
    two agree only up to that truncation error. Run this on representative data to choose
    ``sinc_margin``; if the error is larger than your noise floor, increase the margin.

    ROIs touching the FOV edge are expected to differ more, because the reference path
    resolves wrap-around with ``interpolate_to_border`` while the windowed path
    edge-replicates.
    """
    frame_idx = np.arange(*frames.indices(array.shape[0]), dtype=np.int64)

    full = torch.as_tensor(
        array.target_movie[_read_key(frame_idx)],
        device=array.strategy.device,
        dtype=array.dtype,
    )
    if full.ndim == 2:
        full = full[None, ...]
    reference = apply_rigid_shifts(
        full, array.shifts[torch.as_tensor(frame_idx, device=array.strategy.device)]
    )
    reference = reference[:, rows, cols]

    windowed = array[frames, rows, cols].to(reference.device)
    return float(torch.amax(torch.abs(reference - windowed)).item())