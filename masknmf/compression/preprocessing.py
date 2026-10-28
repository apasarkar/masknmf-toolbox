from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Literal
from warnings import warn
import numpy as np
import torch
from scipy.interpolate import BSpline

def _point_reflect_pad(x: torch.Tensor, padding: int, anchor_frames: int = 1) -> torch.Tensor:
    """
    Pad the last axis by point reflection: left pad x[-k] = 2 * a0 - x[k], right pad likewise about a1. Unlike a
    mirror ("reflect"), this continues the local slope, so a monotone trend stays monotone past the ends.

    a0 and a1 are the frame-0 and last-frame values of a least-squares line fit to the first and last anchor_frames
    frames. The reflection doubles any error in the anchor across the whole pad, so the anchor must be far less noisy
    than one frame: a line over many frames averages the noise down and still follows a sloped trend (a median over
    a sloped stretch would not). With anchor_frames=1 this is plain point reflection through the end frames.
    Needs padding <= x.shape[-1] - 1.
    """
    if padding == 0:
        return x
    n = max(1, min(anchor_frames, x.shape[-1]))
    if n == 1:
        a0, a1 = x[..., :1], x[..., -1:]
    else:
        s = torch.arange(n, dtype=x.dtype, device=x.device)
        design = torch.stack([torch.ones_like(s), s], dim=1)              # (n, 2)
        fit = torch.linalg.pinv(design)[:1]                                 # (1, n): intercept at s = 0
        a0 = x[..., :n] @ fit.T                                             # line through the first n frames, at frame 0
        a1 = x[..., -n:].flip(-1) @ fit.T                                   # line through the last n frames, at the last frame
    left = 2 * a0 - x[..., 1:padding + 1].flip(-1)
    right = 2 * a1 - x[..., -padding - 1:-1].flip(-1)
    return torch.cat([left, x, right], dim=-1)


def geometric_knot_positions(num_frames: int,
                             frame_rate: float,
                             first_seconds: float,
                             growth: float,
                             max_spacing_seconds: float) -> list[float]:
    """
    Interior spline knots as fractions of the recording in (0, 1), dense at the start and sparse later: the first
    knot is first_seconds in, each gap is growth times the previous one, and no gap exceeds max_spacing_seconds.
    Suits photobleaching, which is fastest at the start of a recording.
    """
    if first_seconds <= 0 or max_spacing_seconds <= 0:
        raise ValueError("first_seconds and max_spacing_seconds must be > 0")
    if growth < 1:
        raise ValueError(f"growth must be >= 1, got {growth}")
    # knots closer than a few frames make the spline basis ill-conditioned
    min_gap_seconds = 4 / frame_rate
    duration = num_frames / frame_rate
    gap = max(first_seconds, min_gap_seconds)
    t = gap
    positions = []
    while t < duration - min_gap_seconds:  # keep the last knot clear of the end
        positions.append(t / duration)
        gap = min(max(gap * growth, min_gap_seconds), max_spacing_seconds)
        t += gap
    return positions


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
    ) -> tuple[torch.Tensor, torch.Tensor]:
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

        detrended, coeffs = detrend(data)  # (num_frames, num_pixels) -> (num_frames, num_pixels), (spline_rank, num_pixels)
    """

    def __init__(
        self,
        num_frames: int,
        num_knots: int,
        knot_positions: Sequence[float] | None = None,
        degree: int = 3,
        device: str | torch.device = "cpu",
    ):
        """
        Args:
            num_frames: Length of the time axis.
            num_knots: Number of interior knot points. Total basis dimension
                will be `num_knots + degree + 1`.
            knot_positions: Optional explicit interior knot positions as
                fractions in (0, 1), strictly increasing. If provided, length
                must equal `num_knots`. If omitted, knots are placed uniformly.
            degree: B-spline degree (3 = cubic, the standard choice).
            device (str | torch.device): Device for the basis and solve buffers.
        """
        super().__init__()

        if num_knots < 1:
            raise ValueError(f"num_knots must be >= 1, got {num_knots}")

        interior_knots = self._resolve_interior_knots(num_knots, knot_positions)

        basis = self._build_basis(num_frames, interior_knots, degree)  # (num_frames, spline_rank)
        basis_torch = torch.tensor(basis, dtype=torch.float32, device=device)

        # Precompute (B^T B)^{-1} B^T so the fit is a single matmul.
        # `linalg.solve` is more numerically stable than explicit `inv`.
        solve_matrix = torch.linalg.solve(basis_torch.T @ basis_torch, basis_torch.T)

        self.register_buffer("basis", basis_torch)              # (num_frames, spline_rank)
        self.register_buffer("solve_matrix", solve_matrix)      # (spline_rank, num_frames)

    @staticmethod
    def _resolve_interior_knots(
        num_knots: int,
        knot_positions: Sequence[float] | None,
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
        Construct B-spline basis matrix of shape (num_frames, spline_rank).
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

    def forward(self, data: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            data: (num_frames, num_pixels) tensor.

        Returns:
            detrended: (num_frames, num_pixels), data minus the spline fit.
            coeffs:    (spline_rank, num_pixels), the fitted spline coefficients.
        """
        coeffs = self.solve_matrix @ data       # (spline_rank, num_pixels)
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

    Edge handling (`edge_mode`): near the first and last half-window the rolling
    min/max need frames that don't exist, so each trace is padded.
        - "mirror" (default, the original behavior): mirror the trace at each
          end. Unaffected by activity near the edges and exact for flat
          baselines, but a trend that is still falling (bleaching) at an end
          becomes a peak in the mirror, so the baseline over that half-window
          comes out too low.
        - "line": point-reflect each trace about a straight line fit to its
          first / last half-window. Follows a sloped (bleaching) trend to the
          ends; cells active near an edge pull their own line, so per-pixel
          edge error is somewhat larger than "mirror" on stable baselines.
          Cannot follow a fast, curved decay within the first half-window.

    Knot placement:
        - `num_knots` (required): number of interior knots.
        - `knot_positions` (optional): explicit interior knot positions as
          fractions in (0, 1), strictly increasing. If provided, its length
          must equal `num_knots`. If omitted, knots are placed uniformly.

    Suggested defaults for 2P GCaMP at frame rate `frame_rate`:
        window = int(60 * frame_rate)                          # 60 seconds
        sigma  = max(2.0, 0.3 * frame_rate)                    # 0.3 seconds, min 2 frames
        num_knots = max(4, int(num_frames / frame_rate / 45))  # one per ~45s

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

        detrended, coeffs = detrend(data)  # (num_frames, num_pixels) -> (num_frames, num_pixels), (spline_rank, num_pixels)
    """

    def __init__(
        self,
        num_frames: int,
        num_knots: int,
        knot_positions: Sequence[float] | None = None,
        degree: int = 3,
        window: int = 600,
        sigma: float = 10.0,
        device: str | torch.device = "cpu",
        edge_mode: Literal["mirror", "line"] = "mirror",
    ):
        """
        Args:
            num_frames: Length of the time axis.
            num_knots: Number of interior knot points. Total basis dimension
                will be `num_knots + degree + 1`.
            knot_positions: Optional explicit interior knot positions as
                fractions in (0, 1), strictly increasing. If provided, length
                must equal `num_knots`. If omitted, knots are placed uniformly.
            degree: B-spline degree (3 = cubic).
            window: Rolling min/max window length in frames. Will be made odd
                if even.
            sigma: Gaussian pre-smoothing width in frames (suppresses per-frame
                noise so the rolling min doesn't latch onto single noisy frames).
            device: Device for buffers.
            edge_mode: How traces are padded at the ends, "mirror" (default,
                the original behavior) or "line"; see the class docstring.
                Recordings too short for the window or smoothing get a
                smaller one, with a warning.
        """
        super().__init__()

        if num_knots < 1:
            raise ValueError(f"num_knots must be >= 1, got {num_knots}")
        if window < 3:
            raise ValueError(f"window must be >= 3, got {window}")
        if sigma <= 0:
            raise ValueError(f"sigma must be > 0, got {sigma}")
        if edge_mode not in ("mirror", "line"):
            raise ValueError(f'edge_mode must be "mirror" or "line", got {edge_mode!r}')
        if num_frames < 4:
            raise ValueError(f"MaximinSplineDetrend needs at least 4 frames, got {num_frames}")
        self.edge_mode = edge_mode
        radius = max(1, int(3 * sigma))

        if edge_mode == "mirror":
            # The original behavior: a recording shorter than the window is one window. Each of the three mirror
            # pads must be shorter than the trace; only recordings the original code could not run at all are
            # changed below.
            window = min(window, num_frames)
            if window % 2 == 0:
                window += 1
            if radius > num_frames - 1:
                warn(f"MaximinSplineDetrend: {num_frames} frames is too short for sigma={sigma:g}; "
                     f"using sigma={(num_frames - 1) / 3:g} (smoothing radius {num_frames - 1} frames) instead.")
                radius = num_frames - 1
                sigma = radius / 3
            if window // 2 > num_frames - 1:
                window = 2 * (num_frames - 1) - 1
                warn(f"MaximinSplineDetrend: {num_frames} frames is too short for the window; "
                     f"using a {window}-frame window instead.")
        else:
            if window % 2 == 0:
                window += 1
            # _maximin pads each trace once by 2 * (window // 2) + radius frames, and a reflection can only
            # reach num_frames - 1 frames back from either end. Shrink the smoothing, then the window, to fit.
            max_padding = num_frames - 1
            if radius > max_padding - 2:  # leave room for the smallest window (3 frames, padding 2)
                radius = max_padding - 2
                warn(f"MaximinSplineDetrend: {num_frames} frames is too short for sigma={sigma:g}; "
                     f"using sigma={radius / 3:g} (smoothing radius {radius} frames) instead.")
                sigma = radius / 3
            max_window = 2 * ((max_padding - radius) // 2) + 1
            if window > max_window:
                warn(f"MaximinSplineDetrend: {num_frames} frames is too short for a {window}-frame window; "
                     f"using a {max_window}-frame window instead.")
                window = max_window

        self.window = window
        self.window_padding = window // 2

        # Gaussian smoothing kernel
        self.gaussian_padding = radius
        t = torch.arange(2 * radius + 1, dtype=torch.float32) - radius
        gauss = torch.exp(-(t ** 2) / (2 * sigma ** 2))
        gauss = gauss / gauss.sum()
        gauss = gauss.to(device).view(1, 1, -1)  # (1, 1, conv_size) for conv1d

        # B-spline basis
        interior_knots = self._resolve_interior_knots(num_knots, knot_positions)
        basis = self._build_basis(num_frames, interior_knots, degree)
        basis_torch = torch.tensor(basis, dtype=torch.float32, device=device)

        # Precomputed solve matrix: (B^T B)^{-1} B^T  of shape (spline rank, num_frames)
        solve_matrix = torch.linalg.solve(basis_torch.T @ basis_torch, basis_torch.T)

        self.register_buffer("basis", basis_torch)              # (num_frames, spline_rank)
        self.register_buffer("solve_matrix", solve_matrix)      # (spline_rank, num_frames)
        self.register_buffer("gaussian_kernel", gauss)             # (1, 1, conv_size)

    @staticmethod
    def _resolve_interior_knots(
        num_knots: int,
        knot_positions: Sequence[float] | None,
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
        data: (num_frames, num_pixels) -> (num_frames, num_pixels).
        """
        if self.edge_mode == "line":
            return self._maximin_line(data)
        return self._maximin_mirror(data)

    def _maximin_mirror(self, data: torch.Tensor) -> torch.Tensor:
        """The original envelope: each step mirrors the trace at its ends."""
        input_traces = data.t().unsqueeze(1)  # (num_pixels, 1, num_frames)

        # Gaussian smooth along time
        traces_smoothed = torch.nn.functional.conv1d(
            torch.nn.functional.pad(input_traces, (self.gaussian_padding, self.gaussian_padding), mode="reflect"),
            self.gaussian_kernel,
        )

        # Rolling min: negate, max-pool, negate
        traces_min = -torch.nn.functional.max_pool1d(
            torch.nn.functional.pad(-traces_smoothed, (self.window_padding, self.window_padding), mode="reflect"),
            kernel_size=self.window,
            stride=1,
        )
        # Rolling max of the rolling min
        traces_max_min = torch.nn.functional.max_pool1d(
            torch.nn.functional.pad(traces_min, (self.window_padding, self.window_padding), mode="reflect"),
            kernel_size=self.window,
            stride=1,
        )

        return traces_max_min.squeeze(1).t()  # (num_frames, num_pixels)

    def _maximin_line(self, data: torch.Tensor) -> torch.Tensor:
        """The envelope with each trace point-reflected once about a line fit to its first / last half-window."""
        input_traces = data.t().unsqueeze(1)  # (num_pixels, 1, num_frames)

        # Pad once, wide enough for the smoothing and both rolling steps, so the rolling max works on the padded
        # envelope rather than a re-padded one. The line fit over half a window keeps per-pixel noise out of the pad.
        padded = _point_reflect_pad(input_traces, 2 * self.window_padding + self.gaussian_padding,
                                    anchor_frames=max(2, self.window_padding))

        # Gaussian smooth along time: length num_frames + 4 * window_padding
        traces_smoothed = torch.nn.functional.conv1d(padded, self.gaussian_kernel)

        # Rolling min (negate, max-pool, negate): length num_frames + 2 * window_padding
        traces_min = -torch.nn.functional.max_pool1d(-traces_smoothed, kernel_size=self.window, stride=1)

        # Rolling max of the rolling min: length num_frames
        traces_max_min = torch.nn.functional.max_pool1d(traces_min, kernel_size=self.window, stride=1)

        return traces_max_min.squeeze(1).t()  # (num_frames, num_pixels)

    def forward(self, data: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            data: (num_frames, num_pixels) tensor.

        Returns:
            detrended: (num_frames, num_pixels), data minus the fitted baseline.
            coeffs:    (spline_rank, num_pixels), spline coefficients defining
                the baseline. Reconstruct the baseline as `self.basis @ coeffs`.
        """
        baseline_est = self._maximin(data)          # (num_frames, num_pixels)
        coeffs = self.solve_matrix @ baseline_est   # (spline_rank, num_pixels)
        trend = self.basis @ coeffs                 # (num_frames, num_pixels)
        return data - trend, coeffs