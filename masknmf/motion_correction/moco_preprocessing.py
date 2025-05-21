
import torch
import numpy as np
from typing import *
import scipy.ndimage as ndi
import skimage
import math

def compute_saturation_mask(data: np.ndarray,
                            saturation_limit: Union[float, int]) -> np.ndarray:
    """
    Identifies pixels that become saturated at any point of the movie
    Args:
        data (np.ndarray): Shape (num_frames, fov_dim1, fov_dim2)
        saturation_limit (Union[float, int]): The minimum pixel value that qualifies as saturation
    Returns:
        mask (np.ndarray): Shape (fov_dim1, fov_dim2). Boolean pixel mask indicating whether a pixel becomes saturated (True) or not (False).
    """
    mask = np.any(data > saturation_limit, axis = 0)
    return mask

def dilate_saturation_mask(mask: np.ndarray,
                           expansion_value: int = 3) -> np.ndarray:
    """
    Basic routine to expansion a binary mask uniformly across the FOV by a certain expansion radius

    Args:
        mask (np.ndarray): Shape (fov_dim1, fov_dim2)
        expansion_value (int): The expansion radius for the mask dilation
    """
    mask_dilated = skimage.morphology.dilation(mask, skimage.morphology.disk(expansion_value))
    return mask_dilated

def _fast_inpaint_conv_routine(movie: torch.tensor,
                              num_iters: int) -> torch.tensor:
    """
    Batch inpaint using iterative diffusion.
    Args:
        movie (torch.tensor): Shape (frames, fov_dim1, fov_dim2)
        num_iters (int): The number of convolutions we do to fill in "zero" values
    Returns:
        torch.tensor: The inpainted data
    """
    device = movie.device

    # Convert to (B, C, H, W)
    movie = movie.unsqueeze(1).clone()  # (T, 1, H, W)

    # Define 4-neighbor convolution kernel
    kernel = torch.tensor([[1.0, 1.0, 1.0],
                           [1.0, 0.0, 1.0],
                           [1.0, 1.0, 1.0]], device=device).view(1, 1, 3, 3) / 8.0

    for _ in range(num_iters):
        movie = torch.nn.functional.conv2d(movie, kernel, padding=1)

    return movie.squeeze(1)  # Return (T, H, W)

def mask_inpainting_routine(data: np.ndarray,
                            mask: np.ndarray,
                            device: str = "cpu",
                            batch_size = 200) -> np.ndarray:
    """
    Routine to zero out (and inpaint) some pixels in a dataset
    Args:
        data (np.ndarray): Shape (num_frames, fov_dim1, fov_dim2). The dataset to process
        mask (np.ndarray): Shape (fov_dim1, fov_dim2). The boolean mask where True values are the pixels that
            need to be set to zero and inpainted
        device (str): The device string for data processing. 'cuda', 'cpu', etc.
        batch_size (int): The number of frames to process at a time. 
    """
    # Infer the number of convolutions you will need to do
    dist_vals = ndi.distance_transform_cdt(mask, metric = 'taxicab', return_distances = True) # float dtype
    max_dist = np.amax(dist_vals) # Number of iterations to run
    row, col = mask.nonzero()
    final_stack = data.copy().astype('float32')
    final_stack[:, row, col] = 0
    num_iters = math.ceil(data.shape[0] / batch_size)
    for k in range(num_iters):
        # Make a new tensor
        start = batch_size * k
        end = min(start + batch_size, data.shape[0])
        data_subset = torch.from_numpy(final_stack[start:end, :, :]).to(device).float()
        data_inpainted = _fast_inpaint_conv_routine(data_subset, max_dist)
        final_stack[start:end, row, col] = data_inpainted.cpu().numpy()[:, row, col]

    return final_stack