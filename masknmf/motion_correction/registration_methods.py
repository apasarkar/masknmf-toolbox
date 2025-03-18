import torch
import numpy as np
from typing import *

def apply_rigid_shifts(imgs: torch.tensor,
                       shifts: torch.tensor) -> torch.tensor:
    """
    Applies rigid shifts in dimension 1 (height) and dimension 2 (width) for each image.
    Args:
        imgs (torch.tensor): Shape (num_frames, fov_dim1, fov_dim2). Images to which we apply shifts
        shifts (torch.tensor): Shape (num_frames, 2). Index [i, :] gives the (i-1)-th shift in dim 1 (height) and dim 2 (width).
    Returns:
        shifted_imgs (torch.tensor): Shape (num_frames, fov_dim1, fov_dim2).
    """
    if imgs.ndim == 2:
        imgs = imgs[None, :, :]
    if shifts.ndim == 1:
        shifts = shifts[None, :]
    if imgs.shape[0] != shifts.shape[0]:
        raise ValueError(
            f"Provide same number of images and shifts. You provided {imgs.shape[0]} images and {shifts.shape[0]} shifts")
    if imgs.device != shifts.device:
        raise ValueError(f"images are on {imgs.device} and shifts are on {shifts.device}. Place on same device first")
    device = imgs.device

    H, W = imgs.shape[1], imgs.shape[2]

    ## Apply the shifts directly in batch
    freq_imgs = torch.fft.fft2(imgs)

    ## Identify frequency-specific multipliers to generate the appropriate "shifts" in the Fourier Domain
    shift_dim1_terms = shifts[:, [0]].to(torch.complex64)
    dim1_frequency = (-1 * 1j * (2 * torch.pi / H) * torch.arange(H, device=device))[None, :]
    term_dim1 = shift_dim1_terms @ dim1_frequency
    term_dim1 = torch.exp(term_dim1)
    freq_imgs *= term_dim1[:, :, None]

    shift_dim2_terms = shifts[:, [1]].to(torch.complex64)
    dim2_frequency = (-1 * 1j * (2 * torch.pi / W) * torch.arange(W, device=device))[None, :]
    term_dim2 = shift_dim2_terms @ dim2_frequency
    term_dim2 = torch.exp(term_dim2)
    freq_imgs *= term_dim2[:, None, :]

    # Run the inverse transform, with normalization (since the direct transform did not do any normalization)
    shifted_imgs = torch.fft.ifft2(freq_imgs, norm="backward")

    return torch.real(shifted_imgs)


def estimate_rigid_shifts(image_stack: torch.tensor,
                          template: torch.tensor,
                          max_shifts: tuple[int, int]) -> torch.tensor:
    """
    Estimate rigid shifts to apply to a given image stack to best align each frame to a template

    Args:
        image_stack (torch.tensor): Shape (num_frames, fov dim1, fov dim2).
        template (torch.tensor): Shape (fov dim1, fov dim2).
        max_shifts (tuple[int, int]): Maximum shifts we can apply in each direction
    Returns:
        corrected_stack (torch.tensor): Shape (num_frames, fov dim1, fov dim2)
    """

    num_frames, d1, d2 = image_stack.shape
    device = image_stack.device
    fft_image_stack = torch.fft.fft2(image_stack)
    fft_template = torch.conj(torch.fft.fft2(template))
    max_shifts = torch.tensor(max_shifts).to(device)

    fft_cross_correlation = fft_image_stack * fft_template[None, :, :]
    spatial_domain_cross_correlation = torch.real(torch.fft.ifft2(fft_cross_correlation, norm="backward"))

    dim1_valid_shifts = torch.arange(d1, device=device)
    dim1_valid_locations = torch.logical_or(dim1_valid_shifts >= d1 - 1 - torch.abs(max_shifts[0]),
                                            dim1_valid_shifts <= torch.abs(max_shifts[0])).float()

    dim2_valid_shifts = torch.arange(d2, device=device)
    dim2_valid_locations = torch.logical_or(dim2_valid_shifts >= d2 - 1 - torch.abs(max_shifts[1]),
                                            dim2_valid_shifts <= torch.abs(max_shifts[1])).float()

    valid_locations = dim1_valid_locations[:, None] @ dim2_valid_locations[None, :]
    invalid_locations = (~(valid_locations.bool())).float()
    invalid_subtraction = invalid_locations * torch.abs(torch.amax(spatial_domain_cross_correlation))
    cross_correlation_values = spatial_domain_cross_correlation * valid_locations[None, :, :].float()

    # Guarantees that the max cross correlation happens at |shift value| < |max shifts| in both dimensions
    cross_correlation_values -= invalid_subtraction

    max_indices = torch.argmax(cross_correlation_values.reshape((num_frames, -1)), dim=1)
    shifts_dim1, shifts_dim2 = torch.unravel_index(max_indices, (d1, d2))

    values_to_subtract_dim1 = (torch.abs(d1 - shifts_dim1) < torch.abs(shifts_dim1)).long()
    shifts_dim1 -= values_to_subtract_dim1 * (d1 - 1)

    values_to_subtract_dim2 = (torch.abs(d2 - shifts_dim2) < torch.abs(shifts_dim2)).long()
    shifts_dim2 -= values_to_subtract_dim2 * (d2 - 1)

    shifts = torch.stack([shifts_dim1, shifts_dim2], dim=1)

    # These shifts keep the image fixed and find the optimal template shift; we want the opposite (shift image --> match to template)
    shifts *= -1

    return shifts

def _interpolate_to_border(shifted_imgs: torch.tensor,
                           shifts: torch.tensor):
    """
    After applying rigid shifts via FFT methods, the resulting image will have some artifacts at the edges (wrap-around artifacts).
    This approach overwrites those pixels with the (approximately) nearest "valid" pixel.

    Args:
        shifted_imgs (torch.tensor): Shape (num_frames, fov dim1, fov dim2). The images after shifts have been applied
        shifts (torch.tensor): The shifts that were applied to each image
    """

    # Establish device
    device = shifted_imgs.device

    num_frames, H, W = shifted_imgs.shape

    ## If the shift in some dimension is 2, then the index we want is 3. Similarly if it is -2, then the index is -3
    shifted_indices = shifts + torch.nan_to_num(shifts / torch.abs(shifts), nan=0)
    frame_indices = torch.arange(shifted_imgs.shape[0], device=device)
    shifted_indices = torch.fix(shifted_indices).long()
    index_row_values = shifted_imgs[frame_indices, shifted_indices[:, 0], :]
    index_col_values = shifted_imgs[frame_indices, :, shifted_indices[:, 1]]

    # Decide which pixels actually need to be rewritten
    height_indicator = torch.arange(H, device=device)
    dim1_indicator = torch.broadcast_to(height_indicator, (shifted_imgs.shape[0], H))

    # If shifts are positive, we're interested in indices where shifts > index
    condition1 = torch.logical_and(shifts[:, [0]] >= dim1_indicator, shifts[:, [0]] > 0)
    # If shifts are negative, we're interested in indices where shifts + H < index
    condition2 = torch.logical_and(shifts[:, [0]] + torch.tensor([H], device=device) <= dim1_indicator,
                                   shifts[:, [0]] < 0)
    combined_dim1_condition = torch.logical_or(condition1, condition2)
    inverted_dim1_condition = ~combined_dim1_condition

    width_indicator = torch.arange(W, device=device)
    dim2_indicator = torch.broadcast_to(width_indicator, (shifted_imgs.shape[0], W))
    # If shifts are positive, we're interested in indices where shifts > index
    condition1 = torch.logical_and(shifts[:, [1]] >= dim2_indicator, shifts[:, [1]] > 0)
    # If shifts are negative, we're interested in indices where shifts + H < index
    condition2 = torch.logical_and(shifts[:, [1]] + torch.tensor([W], device=device) <= dim2_indicator,
                                   shifts[:, [1]] < 0)
    combined_dim2_condition = torch.logical_or(condition1, condition2)
    inverted_dim2_condition = ~combined_dim2_condition

    shifted_imgs *= inverted_dim2_condition[:, None, :].float()
    shifted_imgs += (combined_dim2_condition[:, None, :].expand(num_frames, H, W)).float() * index_col_values[:, :,
                                                                                             None]
    shifted_imgs *= inverted_dim1_condition[:, :, None].float()
    shifted_imgs += (combined_dim1_condition[:, :, None].expand(num_frames, H, W)).float() * index_row_values[:, None,
                                                                                             :]

    return shifted_imgs



