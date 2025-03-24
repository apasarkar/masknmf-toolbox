import torch
import numpy as np
from typing import *

from typing import Tuple

def register_frames_rigid(reference_frames: torch.tensor,
                          template: torch.tensor,
                          max_shifts: Tuple[int, int],
                          target_frames: Optional[torch.tensor]):
    """
    Runs full rigid motion correction pipeline: estimating shifts, applying shifts to the iamge stack, and using a copying scheme
    to deal with edge artifacts.

    Args:
        reference_frames (torch.tensor): Shape (num_frames, fov_dim1, fov_dim2)
        template (torch.tensor): Shape either (fov_dim1, fov_dim2) or (num_frames, fov_dim1, fov_dim2). The template(s) to which we align the images
        max_shifts (Tuple[int, int]): The max shift in dimension 1 (height) and dimension 2 (width) respectively.
        target_frames (Optional[torch.tensor]): If specified, we learn the shifts to optimally align reference frames to the template(s) and
            apply those shifts to this set of target frames. Useful for dual-color imaging settings.
    Returns:
        registered_images (torch.tensor): Shape (num_frames, fov_dim1, fov_dim2).
        estimated_shifts (torch.tensor): Shape (num_frames, fov_dim1, fov_dim2).
    """
    if target_frames is None:
        target_frames = reference_frames

    #Compute shifts to align reference frame to template(s)
    rigid_shifts = estimate_rigid_shifts(reference_frames, template, max_shifts)

    #Apply these shifts to target frame
    updated_stack = apply_rigid_shifts(target_frames, rigid_shifts)
    updated_stack = interpolate_to_border(updated_stack, rigid_shifts)
    return updated_stack, rigid_shifts

def apply_rigid_shifts(imgs: torch.tensor,
                       shifts: torch.tensor) -> torch.tensor:
    """
    Applies rigid shifts in dimension 1 (height) and dimension 2 (width) for each image.
    Critical: implementation must use torch.complex128 for numerical precision.

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

    # Compute FFT of images
    freq_imgs = torch.fft.fft2(imgs, norm="ortho")

    # Compute frequency grids using fftfreq
    dim1_frequency = (-1j * 2 * torch.pi * torch.fft.fftfreq(H, d=1, device=device))[None, :].to(torch.complex128)
    dim2_frequency = (-1j * 2 * torch.pi * torch.fft.fftfreq(W, d=1, device=device))[None, :].to(torch.complex128)

    # Compute phase shift multipliers
    shift_dim1_terms = shifts[:, [0]].to(torch.complex128)
    term_dim1 = torch.exp(shift_dim1_terms @ dim1_frequency)
    freq_imgs *= term_dim1[:, :, None]

    shift_dim2_terms = shifts[:, [1]].to(torch.complex128)
    term_dim2 = torch.exp(shift_dim2_terms @ dim2_frequency)
    freq_imgs *= term_dim2[:, None, :]

    # Inverse FFT
    shifted_imgs = torch.fft.ifft2(freq_imgs, norm="ortho")

    return torch.real(shifted_imgs)

def estimate_rigid_shifts(image_stack: torch.tensor,
                          template: torch.tensor,
                          max_shifts: Tuple[int, int]) -> torch.tensor:
    """
    Estimate rigid shifts to apply to a given image stack to best align each frame to template(s)

    Args:
        image_stack (torch.tensor): Shape (num_frames, fov dim1, fov dim2).
        template (torch.tensor): Shape (fov dim1, fov dim2) or (num_frames, fov_dim1, fov_dim2).
        max_shifts (tuple[int, int]): Maximum shifts we can apply in each direction
    Returns:
        rigid_shifts (torch.tensor): Shape (num_frames, 2). rigid_shifts[i, :] gives the (fov dim1, fov dim2) shifts,
            in that order, for frame "i"
    """

    if len(template.shape) == 2: #One template, all frames
        template = template[None, :, :]
    elif len(template.shape) == 3:
        if template.shape[0] == 1:
            pass
        elif template.shape[0] != image_stack.shape[0]:
            raise ValueError(f"The number of templates {template.shape[0]} does not match number of frames {image_stack.shape[0]}")

    num_frames, d1, d2 = image_stack.shape
    device = image_stack.device
    fft_image_stack = torch.fft.fft2(image_stack)
    fft_template = torch.conj(torch.fft.fft2(template))
    max_shifts = torch.abs(torch.tensor(max_shifts).to(device))

    fft_cross_correlation = fft_image_stack * fft_template
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

    shifts = torch.stack([shifts_dim1, shifts_dim2], dim=1)
    shifts = subpixel_shift_method(shifts, fft_cross_correlation)
    shifts_dim1, shifts_dim2 = shifts[:, 0], shifts[:, 1]

    values_to_subtract_dim1 = (torch.abs(d1 - shifts_dim1) <= torch.abs(shifts_dim1)).long()
    shifts_dim1 -= values_to_subtract_dim1 * d1

    values_to_subtract_dim2 = (torch.abs(d2 - shifts_dim2) <= torch.abs(shifts_dim2)).long()
    shifts_dim2 -= values_to_subtract_dim2 * d2

    #Make sure the final shifts are strictly within the max_shifts interval (we allow the superpixel estimator
    torch.clip_(shifts_dim1, -1*max_shifts[0], max_shifts[0])
    torch.clip_(shifts_dim2, -1*max_shifts[1], max_shifts[1])

    shifts = torch.stack([shifts_dim1, shifts_dim2], dim=1)

    # These shifts keep the image fixed and find the optimal template shift; we want the opposite (shift image --> match to template)
    shifts *= -1
    return shifts


def subpixel_shift_method(opt_integer_shifts: torch.tensor,
                          fft_cross_correlation: torch.tensor) -> torch.tensor:
    """
    Use fourier interpolation (up to the "upsample_factor") to find the optimal "subpixel" shift, within 0.1 of a pixel

    Args:
        opt_integer_shifts (torch.tensor): Shape (num_frames, 2). Tensor describing for each frame the optimal integer
            dim1 and dim2 shifts. This function searches for subpixel shifts in a local neighborbood of the optimal integer shifts.
        fft_cross_correlation (torch.tensor): Shape (num_frames, fov_dim1, fov_dim2).
            The FFT of the spatial cross correlation between each frame and the template

    Returns:
        subpixel_estimates (torch.tensor): Shape (num_frames, 2). The optimal subpixel shifts
    """
    num_frames, d1, d2 = fft_cross_correlation.shape
    upsample_factor = 10
    offset_value = 0.7
    device = fft_cross_correlation.device
    division_rate = 1 / upsample_factor

    dim_spread = torch.arange(-1 * offset_value, offset_value, step=division_rate, device=device)
    integer_pixel_indices = torch.argmin(torch.abs(dim_spread))
    dim1_subpixel_indices = opt_integer_shifts[:, [0]].float() + dim_spread[None, :]  # Shape (num_frames, spread_dim1)
    dim1_subpixel_indices *= upsample_factor
    dim1_multiplier_vector = 2 * 1j * torch.pi * torch.fft.fftfreq(d1, d=upsample_factor, device=device).to(
        torch.complex128)
    # Shape (num_frames, spread_dim1, d1)
    dim1_multiplier_matrix = dim1_subpixel_indices.to(torch.complex128).unsqueeze(2) @ dim1_multiplier_vector[None, :]
    torch.exp_(dim1_multiplier_matrix)

    dim2_subpixel_indices = opt_integer_shifts[:, [1]].float() + dim_spread[None, :]  # Shape (num_frames, spread_dim2)
    dim2_subpixel_indices *= upsample_factor
    dim2_multiplier_vector = 2 * 1j * torch.pi * torch.fft.fftfreq(d2, d=upsample_factor, device=device).to(
        torch.complex128)
    dim2_multiplier_matrix = dim2_subpixel_indices.to(torch.complex128).unsqueeze(2) @ dim2_multiplier_vector[None, :]
    dim2_multiplier_matrix = dim2_multiplier_matrix.permute(0, 2, 1)  # Shape (num_frames, d2, spread_dim2)
    torch.exp_(dim2_multiplier_matrix)

    local_cross_corr = torch.bmm(dim1_multiplier_matrix, fft_cross_correlation)
    local_cross_corr = torch.bmm(local_cross_corr, dim2_multiplier_matrix)
    local_cross_corr = torch.real(local_cross_corr)
    local_cross_corr /= d1 * d2 * upsample_factor ** 2

    max_corr_values, max_indices = torch.max(local_cross_corr.reshape(num_frames, -1), dim=1)
    max_indices_dim1, max_indices_dim2 = torch.unravel_index(max_indices, (local_cross_corr.shape[1],
                                                                           local_cross_corr.shape[2]))

    frame_indexer = torch.arange(local_cross_corr.shape[0], device = device)
    #Decide whether the subpixel shift in dim1 (keeping dim2 fixed at its original integer shift value) improves things
    dim1_subpixel_improvement_indicator = (
                local_cross_corr[frame_indexer, integer_pixel_indices, max_indices_dim2] >= max_corr_values)
    max_indices_dim1[dim1_subpixel_improvement_indicator] = integer_pixel_indices
    #Decide whether the subpixel shift in dim2 (keeping dim1 fixed at its original integer shift value) improves things
    dim2_subpixel_improvement_indicator = (
                local_cross_corr[frame_indexer, max_indices_dim1, integer_pixel_indices] >= max_corr_values)
    max_indices_dim2[dim2_subpixel_improvement_indicator] = integer_pixel_indices

    #Only incorporate subpixel shifts in each dimension if it actually improves the results
    shifts_dim1 = opt_integer_shifts[:, 0] + ((max_indices_dim1 / upsample_factor) - offset_value)
    shifts_dim2 = opt_integer_shifts[:, 1] + ((max_indices_dim2 / upsample_factor) - offset_value)

    return torch.stack([shifts_dim1, shifts_dim2], dim=1)

def interpolate_to_border(shifted_imgs: torch.tensor,
                           shifts: torch.tensor):
    """
    After applying rigid shifts via FFT methods, the resulting image will have some artifacts at the edges (wrap-around artifacts).
    This approach overwrites those pixels with the (approximately) nearest "valid" pixel.
    Note: this is an in-place operation.

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


def extract_patches(img: torch.tensor,
                    patches: Tuple[int, int],
                    overlaps: Tuple[int, int]) -> torch.tensor:
    """
    Batched routine that extracted a proper "sliding window" of patches for piecewise rigid registration.

    Args:
        img (torch.Tensor): Shape (num_frames, height, width).
        patches (tuple[int, int]): The height and width patch dimensions
        overlaps (tuple[int, int]): The overlap between adjacent patches, in both height and width dimensions.
        overlap_h (int): Overlap in height.
        overlap_w (int): Overlap in width.

    Returns:
        patches (torch.tensor): Extracted patches with shape (num_frames, patch_grid_dim1, patch_grid_dim2, patch_height, patch_width).
            patch_grid_dim1, patch_grid_dim2 gives the dimensions of the grid of overlapping patches (in the way they tile the actual FOV).
    """
    num_frames, h, w = img.shape

    patch_h, patch_w = patches
    overlap_h, overlap_w = overlaps
    first_dim, second_dim = _get_indices((h, w), patch_h, patch_w, overlap_h, overlap_w)

    # Create all start positions using meshgrid
    grid_x, grid_y = torch.meshgrid(first_dim, second_dim, indexing="ij")
    patch_grid_dimensions = grid_x.shape

    start_positions = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=1)
    num_patches = start_positions.shape[0]

    # Generate patch indices
    patch_dim1 = torch.arange(patch_h).view(-1, 1) + start_positions[:, 0].view(-1, 1, 1)  # (num_patches, patch_h, 1)
    patch_dim2 = torch.arange(patch_w).view(1, -1) + start_positions[:, 1].view(-1, 1, 1)  # (num_patches, 1, patch_w)


    patches = img[:, patch_dim1, patch_dim2]  # (num_frames, num_patches, patch_h, patch_w)
    return patches.reshape((num_frames, patch_grid_dimensions[0], patch_grid_dimensions[1], patch_h, patch_w))

def _get_indices(img_shape: Tuple[int, int],
                 patch_h: int,
                 patch_w: int,
                 overlap_h: int,
                 overlap_w: int) -> Tuple[torch.tensor, torch.tensor]:
    """
    Compute the start indices for extracting patches with given patch sizes and overlaps.

    Args:
        img_shape (tuple): Shape of the image (height, width).
        patch_h (int): Patch height.
        patch_w (int): Patch width.
        overlap_h (int): Overlap along height.
        overlap_w (int): Overlap along width.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: Start indices for each patch along height and width.
    """
    h, w = img_shape

    # Compute strides from overlaps
    stride_h = patch_h - overlap_h
    stride_w = patch_w - overlap_w

    first_dim = torch.arange(0, h - patch_h, stride_h)
    first_dim = torch.cat([first_dim, torch.tensor([max(h - patch_h, 0)])])

    second_dim = torch.arange(0, w - patch_w, stride_w)
    second_dim = torch.cat([second_dim, torch.tensor([max(w - patch_w, 0)])])

    return first_dim, second_dim

def apply_displacement_vector_field(imgs: torch.tensor,
                                    shift_vector_field: torch.tensor) -> torch.tensor:
    """
    Apply displacements from a given displacement vector field to an image stack.

    Args:
        imgs (torch.tensor): Shape (num_frames, fov_dim1, fov_dim2). The images stack to which shifts are applied
        shift_vector_field (torch.tensor): Shape (num_frames, fov_dim1, fov_dim2, 2). For each frame a vector field
            describing the motion correct shifts at each pixel. shift_vector_field[0, i, j, :] gives (dim1, dim2)
            coordinates in python indexing.

    Returns:
        corrected_imgs (torch.tensor): Shape (num_frames, fov_dim1, fov_dim2). The motion corrected images.

    """
    num_frames, fov_dim1, fov_dim2 = imgs.shape
    device = imgs.device

    # Generate coordinate grid
    fovdim1_grid, fovdim2_grid = torch.meshgrid(
        torch.arange(0.0, fov_dim1, dtype=torch.float32, device=device),
        torch.arange(0.0, fov_dim2, dtype=torch.float32, device=device),
        indexing="ij"  # "ij" produces (row, col) ordering
    )
    base_grid = torch.stack((fovdim1_grid, fovdim2_grid), dim=-1)  # Shape: (fov_dim1, fov_dim2, 2)

    ## Need to negate the shifts here because the grid_sample routine is really moving the "grid" (not the image)
    remapped_coords = base_grid.unsqueeze(0) + -1*shift_vector_field

    # Normalize remapped coordinates to [-1, 1]
    remapped_coords[..., 0] *= (2 / (fov_dim1 - 1))
    remapped_coords[..., 0] -= 1

    remapped_coords[..., 1] *= (2 / (fov_dim2 - 1))
    remapped_coords[..., 1] -= 1

    # Swap the coordinate order to match (x, y) expectation of grid_sample
    remapped_coords = torch.flip(remapped_coords, dims=[-1])

    imgs = imgs.unsqueeze(1)  # Shape (num_frames, 1, H, W)

    corrected_imgs = torch.nn.functional.grid_sample(
        imgs.double(), remapped_coords.double(), mode="bilinear", padding_mode="border", align_corners=False
    )

    return corrected_imgs.squeeze(1).float()


def generate_motion_field_from_pwrigid_shifts(shifts: torch.Tensor, fov_dim1: int, fov_dim2: int) -> torch.Tensor:
    """
    Pwrigid motion correction partitions the FOV into, say, (k1, k2) patches (each with dimension (patch_dimension)),
    and estimates rigid shifts at each patch.

    The below method uses these rigid shifts to generate a smooth vector field describing the shifts to be applied at each pixel.

    Function is batched over dimension 0 of shifts

    Args:
        shifts (torch.Tensor): Motion field estimates, shape (batch_size, k1, k2, 2),
                               where the last dimension contains (dy, dx) shifts.
        fov_dim1 (int): Height of full field.
        fov_dim2 (int): Width of full field.

    Returns:
        torch.Tensor: Remapped coordinate grids, shape (batch_size, fov_dim1, fov_dim2, 2).
    """
    device = shifts.device
    batch_size, k1, k2, _ = shifts.shape

    # Resize motion fields using bicubic interpolation
    pixelwise_motion_vector = torch.nn.functional.interpolate(
        shifts.permute(0, 3, 1, 2),  # Move channels to (batch_size, 2, k1, k2)
        size=(fov_dim1, fov_dim2),
        mode="bicubic",
        align_corners=True
    ).permute(0, 2, 3, 1)  # Restore to (batch_size, fov_dim1, fov_dim2, 2)

    return pixelwise_motion_vector


def _valid_pixel_identifier(shift_lower_bounds: torch.Tensor,
                            shift_upper_bounds: torch.Tensor,
                            fov_dim1: int,
                            fov_dim2: int):
    """
    Given the amounts of "valid" shifts for each frame, this function returns indicators
    describing which rows/columns in space are valid. This is useful when searching for
    shifts that maximize the cross-correlation.

    Args:
        shift_lower_bounds (torch.Tensor): Shape (num_frames, 2). The lower bound shifts
                                           in spatial dimension 1 and 2 respectively.
        shift_upper_bounds (torch.Tensor): Shape (num_frames, 2). The upper bound shifts
                                           in spatial dimension 1 and 2 respectively.
        fov_dim1 (int): The height of the field of view (FOV).
        fov_dim2 (int): The width of the field of view (FOV).

    Returns:
        - valid_rows (torch.Tensor): Shape (num_frames, fov_dim1).
                                     Indicates valid row indices for each frame.
        - valid_cols (torch.Tensor): Shape (num_frames, fov_dim2).
                                     Indicates valid column indices for each frame.
    """
    device = shift_lower_bounds.device
    num_frames = shift_lower_bounds.shape[0]

    # Clone to avoid modifying original tensors
    shift_lower_bounds_adj = shift_lower_bounds.clone()
    shift_upper_bounds_adj = shift_upper_bounds.clone()

    # Convert negative indices to valid positive indices using modular wrapping
    # If the interval is (a, b) with a < 0, then the new interval should be
    shift_lower_bounds_adj[:, 0] += fov_dim1
    shift_upper_bounds_adj[:, 0] += fov_dim1  # The interval is now [0, 2*fov_dim1)

    shift_lower_bounds_adj[:, 1] += fov_dim2
    shift_upper_bounds_adj[:, 1] += fov_dim2

    # Generate row and column indices
    row_indices = torch.arange(fov_dim1 * 2, device=device).expand(num_frames, -1)
    col_indices = torch.arange(fov_dim2 * 2, device=device).expand(num_frames, -1)

    # Compute valid row/column masks
    valid_rows = (row_indices >= shift_lower_bounds_adj[:, 0, None]) & (
                row_indices <= shift_upper_bounds_adj[:, 0, None])
    valid_cols = (col_indices >= shift_lower_bounds_adj[:, 1, None]) & (
                col_indices <= shift_upper_bounds_adj[:, 1, None])

    valid_rows[:, :fov_dim1] += valid_rows[:, fov_dim1:]
    valid_cols[:, :fov_dim2] += valid_cols[:, fov_dim2:]

    return valid_rows[:, :fov_dim1], valid_cols[:, :fov_dim2]


def _estimate_patchwise_rigid_shifts(image_stack_patchwise: torch.tensor,
                                     template_patchwise: torch.tensor,
                                     shift_lower_bounds: torch.tensor,
                                     shift_upper_bounds: torch.tensor) -> torch.tensor:
    """
    Estimate rigid shifts to apply to a given image stack to best align each frame to template(s)

    Args:
        image_stack_patchwise (torch.tensor): Shape (num_frames, num_patches, patch_dim1, patch_dim2).
        template_patchwise (torch.tensor): Shape either (num_frames, num_patches, patch_dim1, patch_dim2) or (num_patches, patch_dim1, patch_dim2).
            The template to which we align each patch.
        shift_lower_bounds (torch.tensor): Shape (num_frames, 2). The minimum fov_dim1 and fov_dim2 shifts respectively
            that we use for aligning patches to their templates.
        shift_upper_bounds (torch.tensor): Shape (num_frames, 2). The maximum fov_dim1, fov_dim2 shifts respectively
            that we use for aligning patches to their templates.
    Returns:
        patchwise_rigid_shifts (torch.tensor): Shape (num_frames, num_patches, 2). Describes the rigid shift in dim1 and dim2 that needs to be applied
            at each patch at each frame to optimally align it with the appropriate template.
    """

    if len(template_patchwise.shape) == 3:  # One set of patchwise templates for all frames
        template_patchwise = template_patchwise.unsqueeze(0)
    elif len(template_patchwise.shape) == 4:
        if template_patchwise.shape[0] == 1:
            pass
        elif template_patchwise.shape[0] != image_stack_patchwise.shape[0]:
            raise ValueError(
                f"The number of templates {template_patchwise.shape[0]} does not match number of frames {image_stack_patchwise.shape[0]}")

    num_frames, num_patches, patch_dim1, patch_dim2 = image_stack_patchwise.shape
    device = image_stack_patchwise.device
    fft_image_stack = torch.fft.fft2(image_stack_patchwise)
    fft_template = torch.conj(torch.fft.fft2(template_patchwise))
    # max_shifts = torch.abs(torch.tensor(max_shifts).to(device))

    fft_cross_correlation = fft_image_stack * fft_template
    spatial_domain_cross_correlation = torch.real(torch.fft.ifft2(fft_cross_correlation, norm="backward"))

    ## For each frame, there is a valid "interval" in dim1 and dim2 of indices we care about. This varies across frames now.
    shift_lower_bounds = shift_lower_bounds.clone()
    shift_upper_bounds = shift_upper_bounds.clone()

    valid_rows, valid_cols = _valid_pixel_identifier(shift_lower_bounds, shift_upper_bounds, patch_dim1, patch_dim2)
    valid_locations = torch.bmm(valid_rows.unsqueeze(2).float(),
                                valid_cols.unsqueeze(1).float())  # Shape (num_frames, patch_dim1, patch_dim2)
    invalid_locations = (~(valid_locations.bool())).float()
    cross_correlation_values = spatial_domain_cross_correlation * valid_locations[:, None, :, :]
    invalid_subtraction = invalid_locations * torch.abs(torch.amax(spatial_domain_cross_correlation))

    print(
        f"shape invalid subtraction is {invalid_subtraction.shape} while cc values is {cross_correlation_values.shape}")
    cross_correlation_values -= invalid_subtraction.unsqueeze(
        1)  # Guarantees that the maximum correlation value is not at an invalid pixel

    ## We can move from num_frames x num_patches x patchdim1 x patchdim2 to (num_frames x num_patches) x patchdim1 x patchdim2
    cross_correlation_values = cross_correlation_values.reshape((num_frames * num_patches, patch_dim1, patch_dim2))
    max_indices = torch.argmax(cross_correlation_values.reshape((cross_correlation_values.shape[0], -1)), dim=1)
    shifts_dim1, shifts_dim2 = torch.unravel_index(max_indices, (patch_dim1, patch_dim2))
    shifts = torch.stack([shifts_dim1, shifts_dim2], dim=1)
    print(f"shift upper and lower bds are {shift_lower_bounds} and {shift_upper_bounds} but shifts are {shifts}")
    shifts = subpixel_shift_method(shifts, fft_cross_correlation.reshape((num_frames * num_patches,
                                                                          patch_dim1,
                                                                          patch_dim2)))
    print(f"after subpixel, shifts are {shifts}")
    shifts_dim1, shifts_dim2 = shifts[:, 0], shifts[:, 1]

    values_to_subtract_dim1 = (torch.abs(patch_dim1 - shifts_dim1) <= torch.abs(shifts_dim1)).long()
    shifts_dim1 -= values_to_subtract_dim1 * patch_dim1

    values_to_subtract_dim2 = (torch.abs(patch_dim2 - shifts_dim2) <= torch.abs(shifts_dim2)).long()
    shifts_dim2 -= values_to_subtract_dim2 * patch_dim2

    print(f"after subtraction, shifts are {shifts_dim1} and {shifts_dim2}")
    # No need to be strict about max shift here (within fractional pixels)
    shifts = torch.stack([shifts_dim1, shifts_dim2], dim=1)
    shifts = shifts.reshape(num_frames, num_patches, 2)
    shifts *= -1

    return shifts


def register_frames_pwrigid(reference_frames: torch.tensor,
                            template: torch.tensor,
                            strides: Tuple[int, int],
                            overlaps: Tuple[int, int],
                            max_rigid_shifts: Tuple[int, int],
                            max_deviation_rigid: int,
                            target_frames: Optional[torch.tensor] = None):
    """
    Performs piecewise rigid normcorre registration. Method estimates a motion vector field that quantifies motion of
    references frames relative to template, and applies relevant transform to correct the motion.

    Args:
        reference_frames (torch.tensor): Shape (num_frames, fov_dim1, fov_dim2). We estimate shifts that optimally align reference_frames to
            the template
        template (torch.tensor): Shape (fov_dim1, fov_dim2)  or (num_frames, fov_dim1, fov_dim2). The template(s) used for alignment.
        strides (tuple[int, int]): Two integers, used to specify patch dimensions for pwrigid registration
        overlaps (tuple[int, int]): Two integers, used to specify the degree of overlap between patches.
            Together, (strides[0] + overlaps[0], strides[1] + overlaps[1]) defines the patch size for pw rigid registration.
        max_rigid_shifts (tuple[int, int]): The maximum (full-fov) rigid shifts, used to perform rigid motion correction prior to piecewise
            rigid registration.
        max_deviation_rigid (tuple[int, int]): The maximum number of pixels (in the height, width directions respectively) that a patch
            can shift relative to the estimate global rigid shifts of the frame.
        target_frames (Optional): The relevant shift estimation is computed between the references frames and the template(s). But the shifts can be
            applied to any other stack. To do this, specify a stack in target_frames.

    Returns:
        registered_frames (torch.tensor): Shape (num_frames, fov_dim1, fov_dim2). The motion corrected frames.
        shift_vector_field (torch.tensor): Shape (num_frames, num_patches_dim1, num_patches_dim2, 2). During piecewise motion correction,
            we break the field of view into overlapping patches and estimate a 2D rigid shift per patch.
            The function "generate_motion_field_from_piecewise_rigid_shifts" transforms these patchwise rigid shifts into a (num_frames, fov_dim1, fov_dim2, 2)
            shaped shift vector field. It is more memory efficient to return the (num_patches_dim1, num_patches_dim2, 2) "lowrank"
            version of the shift vector field.

    """
    device = reference_frames.device
    num_frames, fov_dim1, fov_dim2 = reference_frames.shape

    if target_frames is None:
        target_frames = reference_frames  # We are not applying shifts to another stack here

    rigid_shifts = estimate_rigid_shifts(reference_frames, template, max_rigid_shifts)
    print(f"rigid shifts are {rigid_shifts}")

    """
    Do patchwise rigid registration. What does this method need? 
    For each frame, there is a rigid shift. Given this rigid shift, we can establish an interval (in each dimension) 
    of patchwise rigid shifts that are tolerable. 

    These patchwise shifts are placed into a final vector, of shape (num_frames, k1, k2, 2)
    """
    max_deviation_rigid = torch.tensor([max_deviation_rigid[0], max_deviation_rigid[1]]).to(device)
    lb_shifts = rigid_shifts - max_deviation_rigid.unsqueeze(0)
    ub_shifts = rigid_shifts + max_deviation_rigid.unsqueeze(0)
    print(f"lb shifts is {lb_shifts}")
    print(f"ub_shifts is {ub_shifts}")

    patches = (strides[0] + overlaps[0], strides[1] + overlaps[1])
    patched_data = extract_patches(reference_frames, patches, overlaps)
    patched_templates = extract_patches(template, patches, overlaps)

    patch_grid_dim1 = patched_data.shape[1]
    patch_grid_dim2 = patched_data.shape[2]

    lowrank_patchwise_rigid_shifts = _estimate_patchwise_rigid_shifts(
        patched_data.reshape(num_frames, -1, patches[0], patches[1]),
        patched_templates.reshape(patched_templates.shape[0], -1, patches[0], patches[1]),
        lb_shifts,
        ub_shifts)
    # Reshape to (num_frames, patch_grid_dim1, patch_grid_dim2, 2)
    lowrank_patchwise_rigid_shifts = lowrank_patchwise_rigid_shifts.reshape(
        (num_frames, patch_grid_dim1, patch_grid_dim2, 2))
    """
    Output: A num_frames x patch_grid_dim1 x patch_grid_dim2 x 2 tensor describing the patchwise estimated rigid shifts in the height and width dimension.
    From here it is straightforward: (1) go from patch_grid_dim1 x patch_grid_dim2 x 2 --> fov dim 1, fov dim 2 x 2 motion field
    (2) apply relevant shifts to target img
    """
    shift_field_batch = generate_motion_field_from_pwrigid_shifts(lowrank_patchwise_rigid_shifts, fov_dim1, fov_dim2)

    registered_imgs = apply_displacement_vector_field(target_frames, shift_field_batch)
    return registered_imgs, lowrank_patchwise_rigid_shifts

