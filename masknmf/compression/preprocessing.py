import torch
import torch.nn as nn
import numpy as np
from scipy.interpolate import BSpline


class SplineDetrend(nn.Module):
    """
    Preprocessor that removes slow trends from time series data by projecting
    out a B-spline temporal basis.

    Stores only the basis B (num_timepoints, spline_rank) and the Gram inverse (B^T B)^{-1} (spline_rank, spline_rank).

    Usage:
        detrend = SplineDetrend(num_frames=100000, num_knots=10, degree=3, device='cuda')
        data_detrended = detrend(data)  # (pixels, frames) -> (pixels, frames), (pixels, spline_rank)
            First output is the detrended dataset, second are the regression coefficients
    """

    def __init__(
            self,
            num_frames: int,
            num_knots: int = 10,
            degree: int = 3,
            device: str = "cpu",
    ):
        """
        Args:

        num_frames (int): Number of frames (length of the time axis).
        num_knots (int): Number of interior knot points. Total basis dimension will be
            num_knots + degree - 1.
        degree (int): B-spline degree (3 = cubic, the standard choice).
        device (str): Which device to run computations on
        """
        super().__init__()

        basis = self._build_basis(num_frames, num_knots, degree)  # (T, d)
        basis_torch = torch.tensor(basis, dtype=torch.float32, device=device)

        # Precompute (B^T B)^{-1}
        BtB = basis_torch.T @ basis_torch
        BtB_inv = torch.linalg.inv(BtB)

        self.register_buffer("basis", basis_torch)  # (frames, basis rank)
        self.register_buffer("BtB_inv", BtB_inv)  # (basis rank, basis rank)

    @staticmethod
    def _build_basis(num_frames: int, num_knots: int, degree: int) -> np.ndarray:
        """
        Construct B-spline basis matrix of shape (num_frames, num_basis).
        """
        t = np.linspace(0, 1, num_frames)
        interior_knots = np.linspace(0, 1, num_knots + 2)[1:-1]
        knots = np.concatenate([
            np.zeros(degree + 1),
            interior_knots,
            np.ones(degree + 1),
        ])
        num_basis = len(knots) - degree - 1
        basis = np.zeros((num_frames, num_basis))
        for i in range(num_basis):
            coeffs = np.zeros(num_basis)
            coeffs[i] = 1.0
            spline = BSpline(knots, coeffs, degree, extrapolate=False)
            basis[:, i] = spline(t)
        return basis

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """
        Detrend input data by subtracting the spline fit.

        Args:
            data (torch.Tensor): Shape (num_frames, num_pixels)

        Returns
        - torch.Tensor
            Detrended data, same shape as input. (num_frames, num_pixels)
        - torch.Tensor
            Shape (spline_rank, num_pixels). The spline basis
        """

        coeffs = self.BtB_inv @ (self.basis.T @ data)  # (spline_rank, pixels)
        trend = self.basis @ coeffs  # (frames, pixels)
        detrended = data - trend
        return detrended, coeffs