from typing import *
import torch
import torch.nn as nn
import numpy as np
from scipy.interpolate import BSpline
from abc import ABC, abstractmethod
from typing import Tuple
import torch

class SplineDetrenderBase(torch.nn.Module, ABC):
    """
    Abstract base class for spline-based temporal detrenders.

    A spline detrender takes time-series data and returns:
        1. The detrended data (input minus the fitted spline baseline).
        2. The spline coefficients that define the baseline, so the baseline
           can be stored in a compressed form and reconstructed on demand.

    Subclasses must:
        - Register a buffer named `basis` of shape (num_frames, spline_rank)
          containing the temporal B-spline basis. Registering it as a buffer
          (via `self.register_buffer("basis", ...)`) ensures `.to(device)`
          moves it correctly and that `state_dict` round-trips work.
        - Implement `forward(data) -> (detrended, coeffs)` with the contract
          described below.

    The basis convention is:
        baseline = self.basis @ coeffs           # (num_frames, num_pixels)

    so that `self.basis` has shape (num_frames, spline_rank) and `coeffs` has
    shape (spline_rank, num_pixels).

    Key functionality
    - `self.basis` is a (num_frames, spline_rank) buffer on the right
      device.
    - `self.spline_rank` reports the number of basis functions.
    - `forward(data)` returns the (detrended, coeffs) pair with the
      shapes documented above.

    Subclasses are free to choose how the fit is performed (least squares,
    quantile regression, maximin envelope + spline fit, etc.) and how the
    basis is constructed (uniform knots, custom knot placement, P-splines,
    etc.) as long as this interface is honored.
    """

    @property
    def spline_rank(self) -> int:
        """Number of basis functions (columns of `self.basis`)."""
        if not hasattr(self, "basis"):
            raise AttributeError(
                f"{type(self).__name__} has not registered a `basis` buffer. "
                "Subclasses must call `self.register_buffer('basis', ...)` "
                "during __init__."
            )
        return self.basis.shape[1]

    @property
    def num_frames(self) -> int:
        """Length of the time axis the detrender was constructed for."""
        if not hasattr(self, "basis"):
            raise AttributeError(
                f"{type(self).__name__} has not registered a `basis` buffer."
            )
        return self.basis.shape[0]

    @abstractmethod
    def forward(
        self, data: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Detrend the input data and return the spline coefficients.

        Args:
            data: Tensor of shape (num_frames, num_pixels). The time axis must
                match `self.num_frames`.

        Returns:
            detrended: Tensor of shape (num_frames, num_pixels), equal to
                `data - self.basis @ coeffs`.
            coeffs:    Tensor of shape (spline_rank, num_pixels), the fitted
                spline coefficients. The baseline can be reconstructed at any
                time as `self.basis @ coeffs`.
        """
        ...

    def reconstruct_baseline(self, coeffs: torch.Tensor) -> torch.Tensor:
        """
        Convenience: reconstruct the full baseline from stored coefficients.

        Args:
            coeffs: (spline_rank, num_pixels) tensor of spline coefficients,
                typically the second return value of `forward`.

        Returns:
            baseline: (num_frames, num_pixels) tensor.
        """
        if coeffs.shape[0] != self.spline_rank:
            raise ValueError(
                f"Expected coeffs with leading dim {self.spline_rank} "
                f"(spline_rank), got {coeffs.shape[0]}."
            )
        return self.basis @ coeffs


class SplineDetrend(SplineDetrenderBase):
    """
    Least-squares B-spline detrender.

    Removes slow trends by projecting time-series data onto a B-spline temporal
    basis and subtracting the fit. Stores only the basis B (num_frames,
    spline_rank) and a precomputed solve matrix so the per-call work is two
    matmuls.

    Knot placement:
        - `num_knots` (required): number of interior knots.
        - `knot_positions` (optional): explicit interior knot positions as
          fractions in (0, 1), strictly increasing. If provided, its length
          must equal `num_knots`. If omitted, knots are placed uniformly.

    Usage:
        # Uniform knots:
        detrend = SplineDetrend(num_frames=100_000, num_knots=10, device='cuda')

        # Custom knot positions (must match num_knots in length):
        detrend = SplineDetrend(
            num_frames=100_000,
            num_knots=6,
            knot_positions=[0.05, 0.1, 0.2, 0.35, 0.55, 0.8],
            device='cuda',
        )

        detrended, coeffs = detrend(data)  # (T, N) -> (T, N), (spline_rank, N)
    """

    def __init__(
        self,
        num_frames: int,
        num_knots: int,
        knot_positions: Optional[Sequence[float]] = None,
        degree: int = 3,
        device: str = "cpu",
    ):
        """
        Args:
            num_frames: Length of the time axis.
            num_knots: Number of interior knot points. Total basis dimension
                will be `num_knots + degree - 1`.
            knot_positions: Optional explicit interior knot positions as
                fractions in (0, 1), strictly increasing. If provided, length
                must equal `num_knots`. If omitted, knots are placed uniformly.
            degree: B-spline degree (3 = cubic, the standard choice).
            device: Device for the basis and solve buffers.
        """
        super().__init__()

        if num_knots < 1:
            raise ValueError(f"num_knots must be >= 1, got {num_knots}")

        interior_knots = self._resolve_interior_knots(num_knots, knot_positions)

        basis = self._build_basis(num_frames, interior_knots, degree)  # (T, d)
        basis_torch = torch.tensor(basis, dtype=torch.float32, device=device)

        # Precompute (B^T B)^{-1} B^T so the fit is a single matmul.
        # `linalg.solve` is more numerically stable than explicit `inv`.
        solve_matrix = torch.linalg.solve(basis_torch.T @ basis_torch, basis_torch.T)

        self.register_buffer("basis", basis_torch)              # (T, d)
        self.register_buffer("solve_matrix", solve_matrix)      # (d, T)

    @staticmethod
    def _resolve_interior_knots(
        num_knots: int,
        knot_positions: Optional[Sequence[float]],
    ) -> np.ndarray:
        """
        Validate the knot specification and return interior knots as a 1D
        numpy array of values in (0, 1).
        """
        if knot_positions is None:
            # Uniformly spaced strictly inside (0, 1)
            return np.linspace(0, 1, num_knots + 2)[1:-1]

        knots = np.asarray(knot_positions, dtype=np.float64)
        if knots.ndim != 1:
            raise ValueError("knot_positions must be 1D.")
        if knots.size != num_knots:
            raise ValueError(
                f"knot_positions has length {knots.size} but num_knots is "
                f"{num_knots}. They must match."
            )
        if not np.all((knots > 0.0) & (knots < 1.0)):
            raise ValueError(
                "knot_positions must lie strictly inside (0, 1). "
                f"Got min={knots.min()}, max={knots.max()}."
            )
        if not np.all(np.diff(knots) > 0):
            raise ValueError("knot_positions must be strictly increasing.")
        return knots

    @staticmethod
    def _build_basis(
        num_frames: int,
        interior_knots: np.ndarray,
        degree: int,
    ) -> np.ndarray:
        """
        Construct B-spline basis matrix of shape (num_frames, num_basis).
        """
        t = np.linspace(0, 1, num_frames)
        knots = np.concatenate(
            [
                np.zeros(degree + 1),
                interior_knots,
                np.ones(degree + 1),
            ]
        )
        num_basis = len(knots) - degree - 1
        basis = np.zeros((num_frames, num_basis))
        for i in range(num_basis):
            coeffs = np.zeros(num_basis)
            coeffs[i] = 1.0
            spline = BSpline(knots, coeffs, degree, extrapolate=False)
            basis[:, i] = spline(t)
        return basis

    def forward(self, data: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            data: (num_frames, num_pixels) tensor.

        Returns:
            detrended: (num_frames, num_pixels), data minus the spline fit.
            coeffs:    (spline_rank, num_pixels), the fitted spline coefficients.
        """
        coeffs = self.solve_matrix @ data       # (d, num_pixels)
        trend = self.basis @ coeffs             # (num_frames, num_pixels)
        return data - trend, coeffs



class MaximinSplineDetrend(SplineDetrenderBase):
    """
    Two-stage baseline detrender for calcium imaging traces:

        1. Estimate a smooth lower envelope of each trace via maximin filtering
           (Gaussian smooth -> rolling min -> rolling max). This robustly
           ignores positive calcium transients.
        2. Fit a B-spline through the envelope via least squares. Store only
           the spline coefficients as the compressed baseline.

    Stores only the basis B (num_frames, spline_rank), a precomputed solve
    matrix (B^T B)^{-1} B^T (spline_rank, num_frames), and the smoothing kernel.

    Knot placement:
        - `num_knots` (required): number of interior knots.
        - `knot_positions` (optional): explicit interior knot positions as
          fractions in (0, 1), strictly increasing. If provided, its length
          must equal `num_knots`. If omitted, knots are placed uniformly.

    Suggested defaults for 2P GCaMP at frame rate `fs`:
        window = int(60 * fs)                          # 60 seconds
        sigma  = max(2.0, 0.3 * fs)                    # 0.3 seconds, min 2 frames
        num_knots = max(4, int(num_frames / fs / 45))  # one per ~45s

    Usage:
        # Uniform knots:
        detrend = MaximinSplineDetrend(
            num_frames=100_000,
            num_knots=20,
            window=1800,
            sigma=9.0,
            device='cuda',
        )

        # Custom knot positions (must match num_knots in length):
        detrend = MaximinSplineDetrend(
            num_frames=100_000,
            num_knots=6,
            knot_positions=[0.05, 0.1, 0.2, 0.35, 0.55, 0.8],
            window=1800,
            sigma=9.0,
            device='cuda',
        )

        detrended, coeffs = detrend(data)  # (T, N) -> (T, N), (spline_rank, N)
    """

    def __init__(
        self,
        num_frames: int,
        num_knots: int,
        knot_positions: Optional[Sequence[float]] = None,
        degree: int = 3,
        window: int = 600,
        sigma: float = 10.0,
        device: str = "cpu",
    ):
        """
        Args:
            num_frames: Length of the time axis.
            num_knots: Number of interior knot points. Total basis dimension
                will be `num_knots + degree - 1`.
            knot_positions: Optional explicit interior knot positions as
                fractions in (0, 1), strictly increasing. If provided, length
                must equal `num_knots`. If omitted, knots are placed uniformly.
            degree: B-spline degree (3 = cubic).
            window: Rolling min/max window length in frames. Will be made odd
                if even.
            sigma: Gaussian pre-smoothing width in frames (suppresses per-frame
                noise so the rolling min doesn't latch onto single noisy frames).
            device: Device for buffers.
        """
        super().__init__()

        if num_knots < 1:
            raise ValueError(f"num_knots must be >= 1, got {num_knots}")
        if window < 3:
            raise ValueError(f"window must be >= 3, got {window}")
        if window % 2 == 0:
            window += 1
        if sigma <= 0:
            raise ValueError(f"sigma must be > 0, got {sigma}")

        self.window = window
        self.pad = window // 2

        # Gaussian smoothing kernel
        radius = max(1, int(3 * sigma))
        self.gauss_pad = radius
        t = torch.arange(2 * radius + 1, dtype=torch.float32) - radius
        gauss = torch.exp(-(t ** 2) / (2 * sigma ** 2))
        gauss = gauss / gauss.sum()
        gauss = gauss.to(device).view(1, 1, -1)  # (1, 1, K) for conv1d

        # B-spline basis
        interior_knots = self._resolve_interior_knots(num_knots, knot_positions)
        basis = self._build_basis(num_frames, interior_knots, degree)
        basis_torch = torch.tensor(basis, dtype=torch.float32, device=device)

        # Precomputed solve matrix: (B^T B)^{-1} B^T  of shape (d, T)
        solve_matrix = torch.linalg.solve(basis_torch.T @ basis_torch, basis_torch.T)

        self.register_buffer("basis", basis_torch)              # (T, d)
        self.register_buffer("solve_matrix", solve_matrix)      # (d, T)
        self.register_buffer("gauss_kernel", gauss)             # (1, 1, K)

    @staticmethod
    def _resolve_interior_knots(
        num_knots: int,
        knot_positions: Optional[Sequence[float]],
    ) -> np.ndarray:
        """
        Validate the knot specification and return interior knots as a 1D
        numpy array of values in (0, 1).
        """
        if knot_positions is None:
            # Uniformly spaced strictly inside (0, 1)
            return np.linspace(0, 1, num_knots + 2)[1:-1]

        knots = np.asarray(knot_positions, dtype=np.float64)
        if knots.ndim != 1:
            raise ValueError("knot_positions must be 1D.")
        if knots.size != num_knots:
            raise ValueError(
                f"knot_positions has length {knots.size} but num_knots is "
                f"{num_knots}. They must match."
            )
        if not np.all((knots > 0.0) & (knots < 1.0)):
            raise ValueError(
                "knot_positions must lie strictly inside (0, 1). "
                f"Got min={knots.min()}, max={knots.max()}."
            )
        if not np.all(np.diff(knots) > 0):
            raise ValueError("knot_positions must be strictly increasing.")
        return knots

    @staticmethod
    def _build_basis(
        num_frames: int,
        interior_knots: np.ndarray,
        degree: int,
    ) -> np.ndarray:
        """B-spline basis matrix of shape (num_frames, num_basis)."""
        t = np.linspace(0, 1, num_frames)
        knots = np.concatenate(
            [
                np.zeros(degree + 1),
                interior_knots,
                np.ones(degree + 1),
            ]
        )
        num_basis = len(knots) - degree - 1
        basis = np.zeros((num_frames, num_basis))
        for i in range(num_basis):
            coeffs = np.zeros(num_basis)
            coeffs[i] = 1.0
            spline = BSpline(knots, coeffs, degree, extrapolate=False)
            basis[:, i] = spline(t)
        return basis

    def _maximin(self, data: torch.Tensor) -> torch.Tensor:
        """
        Maximin lower envelope.
        data: (T, N) -> (T, N).
        """
        x = data.t().unsqueeze(1)  # (N, 1, T)

        # Gaussian smooth along time
        x_smooth = torch.nn.functional.conv1d(
            torch.nn.functional.pad(x, (self.gauss_pad, self.gauss_pad), mode="reflect"),
            self.gauss_kernel,
        )

        # Rolling min: negate, max-pool, negate
        x_min = -torch.nn.functional.max_pool1d(
            torch.nn.functional.pad(-x_smooth, (self.pad, self.pad), mode="reflect"),
            kernel_size=self.window,
            stride=1,
        )
        # Rolling max of the rolling min
        x_max = torch.nn.functional.max_pool1d(
            torch.nn.functional.pad(x_min, (self.pad, self.pad), mode="reflect"),
            kernel_size=self.window,
            stride=1,
        )

        return x_max.squeeze(1).t()  # (T, N)

    def forward(self, data: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            data: (num_frames, num_pixels) tensor.

        Returns:
            detrended: (num_frames, num_pixels), data minus the fitted baseline.
            coeffs:    (spline_rank, num_pixels), spline coefficients defining
                the baseline. Reconstruct the baseline as `self.basis @ coeffs`.
        """
        baseline_est = self._maximin(data)          # (T, N)
        coeffs = self.solve_matrix @ baseline_est   # (d, N)
        trend = self.basis @ coeffs                 # (T, N)
        return data - trend, coeffs