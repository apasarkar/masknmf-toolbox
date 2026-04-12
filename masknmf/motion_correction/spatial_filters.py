import torch
import numpy as np
from typing import *
import math
import cv2
from typing import List

def compute_highpass_filter_kernel(sigma: List[float]):
    "Idea attributed to Giovanucci et al (Caiman)"
    ksize = tuple([(3 * i) // 2 * 2 + 1 for i in sigma])
    ker = cv2.getGaussianKernel(ksize[0], sigma[0])
    ker2D = ker.dot(ker.T)
    nz = np.nonzero(ker2D >= ker2D[:, 0].max())
    zz = np.nonzero(ker2D < ker2D[:, 0].max())
    ker2D[nz] -= ker2D[nz].mean()
    ker2D[zz] = 0
    return torch.tensor(ker2D, dtype=torch.float32)


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