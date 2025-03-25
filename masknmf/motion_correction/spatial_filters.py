import torch
import numpy as np
from typing import *


def compute_highpass_filter_kernel(gaussian_sigma: list[int]) -> torch.tensor:
    """
    Computes a high-pass filter kernel using a Gaussian filter.

    Args:
        gaussian_sigma (list[int]): Standard deviations for the Gaussian kernel.

    Returns:
        torch.Tensor: High-pass filter kernel.
    """
    # Validate input
    if not isinstance(gaussian_sigma[0], int):
        raise TypeError(f"gaussian_sigma must be a list containing an integer, but got {type(gaussian_sigma[0])}")
    if gaussian_sigma[0] < 1:
        raise ValueError("gaussian_sigma must contain a positive integer")

    # Compute kernel size: ksize = (3 * sigma) rounded to nearest odd integer
    ksize = [(3 * i) // 2 * 2 + 1 for i in gaussian_sigma]

    # Create 1D Gaussian kernel
    x = torch.arange(ksize[0]) - ksize[0] // 2
    gauss_1d = torch.exp(-0.5 * (x / gaussian_sigma[0]) ** 2)
    gauss_1d /= gauss_1d.sum()  # Normalize

    # Create 2D Gaussian kernel
    ker2D = gauss_1d[:, None] @ gauss_1d[None, :]  # Outer product

    # Find nonzero indices where kernel is greater than its first column max
    nz = ker2D >= ker2D[:, 0].max()
    zz = ker2D < ker2D[:, 0].max()

    # Modify kernel values for high-pass filtering
    ker2D[nz] -= ker2D[nz].mean()
    ker2D[zz] = 0

    return ker2D

def gaussian_kernel(kernel_size: int=3,
                    sigma: float=1.0) -> torch.tensor:
    """Generates a 2D Gaussian kernel."""
    x = torch.arange(kernel_size) - kernel_size // 2
    y = torch.arange(kernel_size) - kernel_size // 2
    xx, yy = torch.meshgrid(x, y, indexing='ij')

    kernel = torch.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    kernel /= kernel.sum()  # Normalize to sum to 1
    return kernel


def image_filter(frames: torch.tensor,
                kernel: torch.tensor) -> torch.tensor:
    """
    Generic Image filter function; given a kernel an image stack, convolve every image with the kernel.

    Args:
        frames (torch.Tensor): Shape (num_frames, fov_dim1, fov_dim2)
        kernel (torch.Tensor): Shape (kH, kW)

    Returns:
        torch.Tensor: Convolved frames with shape (num_frames, fov_dim1, fov_dim2)
    """

    num_frames, fov_dim1, fov_dim2 = frames.shape
    kH, kW = kernel.shape

    # Reshape frames to fit conv2d input format: (batch=frames, channels=1, height, width)
    frames = frames.unsqueeze(1)  # Shape: (num_frames, 1, fov_dim1, fov_dim2)

    # Reshape kernel to fit conv2d weight format: (out_channels=1, in_channels=1, kH, kW)
    kernel = kernel.unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, kH, kW)

    # Apply convolution (padding='same' ensures output size matches input size)
    convolved_frames = torch.nn.functional.conv2d(frames, kernel, padding="same")  # Shape: (num_frames, 1, fov_dim1, fov_dim2)

    # Remove channel dimension
    return convolved_frames.squeeze(1)  # Shape: (num_frames, fov_dim1, fov_dim2)
