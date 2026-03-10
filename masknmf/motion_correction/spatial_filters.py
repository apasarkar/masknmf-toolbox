import torch
import numpy as np
from typing import *
import math

from typing import List

def compute_highpass_filter_kernel(gaussian_sigma: List[float]) -> torch.Tensor:
    """
    Computes a high-pass filter kernel using a Gaussian filter.

    Args:
        gaussian_sigma (list[int]): Standard deviations for the Gaussian kernel.

    Returns:
        torch.Tensor: High-pass filter kernel.
    """

    if len(gaussian_sigma) != 2:
        raise ValueError("gaussian_sigma must have length 2")

    if any(s <= 0 for s in gaussian_sigma):
        raise ValueError("gaussian_sigma must contain positive values")

    sigma_h, sigma_w = gaussian_sigma

    radius_h = int(3 * sigma_h)
    radius_w = int(3 * sigma_w)

    coords_h = torch.arange(-radius_h, radius_h + 1, dtype=torch.float32)
    coords_w = torch.arange(-radius_w, radius_w + 1, dtype=torch.float32)

    g_h = torch.exp(-0.5 * (coords_h ** 2) / (sigma_h ** 2))
    g_w = torch.exp(-0.5 * (coords_w ** 2) / (sigma_w ** 2))

    kernel = g_h[:, None] @ g_w[None, :]
    kernel /= kernel.sum()

    kernel = -kernel
    kernel[radius_h, radius_w] += 1.0

    return kernel

def gaussian_kernel(kernel_size: int = 3, sigma: float = 1.0) -> torch.tensor:
    """Generates a 2D Gaussian kernel."""
    x = torch.arange(kernel_size) - kernel_size // 2
    y = torch.arange(kernel_size) - kernel_size // 2
    xx, yy = torch.meshgrid(x, y, indexing="ij")

    kernel = torch.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    kernel /= kernel.sum()  # Normalize to sum to 1
    return kernel


def image_filter(frames: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    """
    Generic Image filter function; given a kernel and an image stack,
    convolve every image with the kernel using reflect padding.

    Args:
        frames (torch.Tensor): Shape (num_frames, fov_dim1, fov_dim2)
        kernel (torch.Tensor): Shape (kH, kW)

    Returns:
        torch.Tensor: Convolved frames with shape (num_frames, fov_dim1, fov_dim2)
    """

    num_frames, fov_dim1, fov_dim2 = frames.shape
    kH, kW = kernel.shape

    # Reshape frames to (batch, channels, height, width)
    frames = frames.unsqueeze(1)  # (num_frames, 1, H, W)

    # Reshape kernel to (out_channels, in_channels, kH, kW)
    kernel = kernel.unsqueeze(0).unsqueeze(0)  # (1, 1, kH, kW)

    # Compute padding for "same" output size
    pad_h = kH // 2
    pad_w = kW // 2

    # Reflect padding (OpenCV BORDER_REFLECT-style)
    frames = torch.nn.functional.pad(
        frames,
        pad=(pad_w, pad_w, pad_h, pad_h),
        mode="reflect"
    )

    # Convolution (no padding here)
    convolved_frames = torch.nn.functional.conv2d(
        frames,
        kernel,
        padding=0
    )  # (num_frames, 1, H, W)

    # Remove channel dimension
    return convolved_frames.squeeze(1)