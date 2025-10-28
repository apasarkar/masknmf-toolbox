import torch
import numpy as np
import math
from typing import *

from typing import Tuple


def register_frames_rigid(
    reference_frames: torch.tensor,
    template: torch.tensor,
    max_shifts: Tuple[int, int],
    target_frames: Optional[torch.tensor] = None,
    pixel_weighting: Optional[torch.tensor] = None,
) -> Tuple[torch.tensor, torch.tensor]:
    """
    Runs full rigid motion correction pipeline: estimating shifts, applying shifts to the iamge stack, and using a copying scheme
    to deal with edge artifacts.

    Args:
        reference_frames (torch.tensor): Shape (num_frames, fov_dim1, fov_dim2)
        template (torch.tensor): Shape either (fov_dim1, fov_dim2) or (num_frames, fov_dim1, fov_dim2). The template(s) to which we align the images
        max_shifts (Tuple[int, int]): The max shift in dimension 1 (height) and dimension 2 (width) respectively.
        target_frames (Optional[torch.tensor]): If specified, we learn the shifts to optimally align reference frames to the template(s) and
            apply those shifts to this set of target frames. Useful for dual-color imaging settings.
        pixel_weighting (Optional[torch.tensor]): Shape (fov_dim1, fov_dim2). If specified, the weight (importance) of
            each pixel in the rigid shift estimation.
    Returns:
        registered_images (torch.tensor): Shape (num_frames, fov_dim1, fov_dim2).
        estimated_shifts (torch.tensor): Shape (num_frames, fov_dim1, fov_dim2).
    """
    if target_frames is None:
        target_frames = reference_frames

    # Compute shifts to align reference frame to template(s)
    rigid_shifts = estimate_rigid_shifts(
        reference_frames, template, max_shifts, pixel_weighting=pixel_weighting
    )

    # Apply these shifts to target frame
    updated_stack = apply_rigid_shifts(target_frames, rigid_shifts)
    updated_stack = interpolate_to_border(updated_stack, rigid_shifts)
    return updated_stack, rigid_shifts


def apply_rigid_shifts(imgs: torch.tensor, shifts: torch.tensor) -> torch.tensor:
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
            f"Provide same number of images and shifts. You provided {imgs.shape[0]} images and {shifts.shape[0]} shifts"
        )
    if imgs.device != shifts.device:
        raise ValueError(
            f"images are on {imgs.device} and shifts are on {shifts.device}. Place on same device first"
        )

    device = imgs.device
    H, W = imgs.shape[1], imgs.shape[2]

    # Compute FFT of images
    freq_imgs = torch.fft.fft2(imgs, norm="ortho")

    # Compute frequency grids using fftfreq
    dim1_frequency = (-1j * 2 * torch.pi * torch.fft.fftfreq(H, d=1, device=device))[
        None, :
    ].to(torch.complex128)
    dim2_frequency = (-1j * 2 * torch.pi * torch.fft.fftfreq(W, d=1, device=device))[
        None, :
    ].to(torch.complex128)

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


def estimate_rigid_shifts(
    image_stack: torch.tensor,
    template: torch.tensor,
    max_shifts: Tuple[int, int],
    pixel_weighting: Optional[torch.tensor] = None,
) -> torch.tensor:
    """
    Estimate rigid shifts to apply to a given image stack to best align each frame to template(s)

    Args:
        image_stack (torch.tensor): Shape (num_frames, fov dim1, fov dim2).
        template (torch.tensor): Shape (fov dim1, fov dim2) or (num_frames, fov_dim1, fov_dim2).
        max_shifts (tuple[int, int]): Maximum shifts we can apply in each direction
        pixel_weighting (torch.tensor): A weighting of each pixel of the FOV. If provided, this means we are
            solving a weighted L2 problem, where we prioritize alignment of certain pixels over others.
    Returns:
        rigid_shifts (torch.tensor): Shape (num_frames, 2). rigid_shifts[i, :] gives the (fov dim1, fov dim2) shifts,
            in that order, for frame "i"
    """

    if len(template.shape) == 2:  # One template, all frames
        template = template[None, :, :]
    elif len(template.shape) == 3:
        if template.shape[0] == 1:
            pass
        elif template.shape[0] != image_stack.shape[0]:
            raise ValueError(
                f"The number of templates {template.shape[0]} does not match number of frames {image_stack.shape[0]}"
            )

    num_frames, d1, d2 = image_stack.shape
    device = image_stack.device

    if pixel_weighting is None:
        fft_image_stack = torch.fft.fft2(image_stack)
        fft_template = torch.conj(torch.fft.fft2(template))

        fft_l2_objective = fft_image_stack * fft_template
        spatial_domain_cross_correlation = torch.real(
            torch.fft.ifft2(fft_l2_objective, norm="backward")
        )
    else:
        if len(pixel_weighting.shape) == 2:
            pixel_weighting = pixel_weighting[None, :, :]
        else:
            raise ValueError(f"Must pass in a 2D pixel weighting tensor")
        fft_image_stack = torch.fft.fft2(image_stack)
        fft_image_stack_sq = torch.fft.fft2(torch.square(image_stack))
        fft_weighted_template = torch.conj(
            torch.fft.fft2(torch.square(pixel_weighting) * template)
        )
        fft_pixel_weight_sq = torch.conj(torch.fft.fft2(torch.square(pixel_weighting)))
        fft_l2_objective = (
            2 * fft_weighted_template * fft_image_stack
            - fft_pixel_weight_sq * fft_image_stack_sq
        )
        spatial_domain_cross_correlation = torch.real(
            torch.fft.ifft2(fft_l2_objective, norm="backward")
        )

    max_shifts = torch.abs(torch.tensor(max_shifts).to(device))
    dim1_valid_shifts = torch.arange(d1, device=device)
    dim1_valid_locations = torch.logical_or(
        dim1_valid_shifts >= d1 - 1 - torch.abs(max_shifts[0]),
        dim1_valid_shifts <= torch.abs(max_shifts[0]),
    ).float()

    dim2_valid_shifts = torch.arange(d2, device=device)
    dim2_valid_locations = torch.logical_or(
        dim2_valid_shifts >= d2 - 1 - torch.abs(max_shifts[1]),
        dim2_valid_shifts <= torch.abs(max_shifts[1]),
    ).float()

    valid_locations = dim1_valid_locations[:, None] @ dim2_valid_locations[None, :]
    invalid_locations = (~(valid_locations.bool())).float()
    invalid_subtraction = invalid_locations * torch.abs(
        torch.amax(spatial_domain_cross_correlation)
    )
    cross_correlation_values = (
        spatial_domain_cross_correlation * valid_locations[None, :, :].float()
    )

    # Guarantees that the max cross correlation happens at |shift value| < |max shifts| in both dimensions
    cross_correlation_values -= invalid_subtraction

    max_indices = torch.argmax(
        cross_correlation_values.reshape((num_frames, -1)), dim=1
    )
    shifts_dim1, shifts_dim2 = torch.unravel_index(max_indices, (d1, d2))
    shifts = torch.stack([shifts_dim1, shifts_dim2], dim=1)

    for precision in [0.1, 0.01, 0.001]:
        shifts = subpixel_shift_method(shifts, fft_l2_objective, precision)

    shifts_dim1, shifts_dim2 = shifts[:, 0], shifts[:, 1]

    values_to_subtract_dim1 = (
        torch.abs(d1 - shifts_dim1) <= torch.abs(shifts_dim1)
    ).long()
    shifts_dim1 -= values_to_subtract_dim1 * d1

    values_to_subtract_dim2 = (
        torch.abs(d2 - shifts_dim2) <= torch.abs(shifts_dim2)
    ).long()
    shifts_dim2 -= values_to_subtract_dim2 * d2

    # Make sure the final shifts are strictly within the max_shifts interval (we allow the superpixel estimator
    torch.clip_(shifts_dim1, -1 * max_shifts[0], max_shifts[0])
    torch.clip_(shifts_dim2, -1 * max_shifts[1], max_shifts[1])

    shifts = torch.stack([shifts_dim1, shifts_dim2], dim=1)

    # These shifts keep the image fixed and find the optimal template shift; we want the opposite (shift image --> match to template)
    shifts *= -1
    return shifts


def subpixel_shift_method(
    opt_shifts: torch.tensor, fft_l2_objective: torch.tensor, precision: float
) -> torch.tensor:
    """
    Use fourier interpolation (up to the "upsample_factor") to find the optimal "subpixel" shift, within 0.1 of a pixel

    Args:
        opt_shifts (torch.tensor): Shape (num_frames, 2). Tensor describing for each frame the optimal integer
            dim1 and dim2 shifts. This function searches for subpixel shifts in a local neighborbood of the optimal integer shifts.
        fft_l2_objective (torch.tensor): Shape (num_frames, fov_dim1, fov_dim2).
            The FFT of the objective function (over "shifts") which we seek to optimize
        precision (float): Only accepts these values: [0.1, 0.01, 0.001]. The accuracy to which we estimate the subpixel shift, relative to the
            opt_integer shifts.

    Returns:
        subpixel_estimates (torch.tensor): Shape (num_frames, 2). The optimal subpixel shifts
    """
    if precision not in [0.1, 0.01, 0.001]:
        raise ValueError(
            f"Precision can only be 0.1, 0.01, 0.001. Input was {precision}"
        )

    num_frames, d1, d2 = fft_l2_objective.shape
    division_rate = precision
    offset_value = (
        6 * precision
    )  # If precision is 0.1, we want to look at a (-0.6, 0.6) interval, etc.
    device = fft_l2_objective.device
    upsample_factor = 1 / division_rate

    dim_spread = torch.arange(
        -1 * offset_value, offset_value, step=division_rate, device=device
    )
    integer_pixel_indices = torch.argmin(torch.abs(dim_spread))
    dim1_subpixel_indices = (
        opt_shifts[:, [0]].float() + dim_spread[None, :]
    )  # Shape (num_frames, spread_dim1)
    # dim1_subpixel_indices = upsample_factor
    dim1_multiplier_vector = (
        2
        * 1j
        * torch.pi
        * torch.fft.fftfreq(d1, d=1.0, device=device).to(torch.complex128)
    )
    # Shape (num_frames, spread_dim1, d1)
    dim1_multiplier_matrix = (
        dim1_subpixel_indices.to(torch.complex128).unsqueeze(2)
        @ dim1_multiplier_vector[None, :]
    )
    torch.exp_(dim1_multiplier_matrix)

    dim2_subpixel_indices = (
        opt_shifts[:, [1]].float() + dim_spread[None, :]
    )  # Shape (num_frames, spread_dim2)
    dim2_multiplier_vector = (
        2
        * 1j
        * torch.pi
        * torch.fft.fftfreq(d2, d=1.0, device=device).to(torch.complex128)
    )
    dim2_multiplier_matrix = (
        dim2_subpixel_indices.to(torch.complex128).unsqueeze(2)
        @ dim2_multiplier_vector[None, :]
    )
    dim2_multiplier_matrix = dim2_multiplier_matrix.permute(
        0, 2, 1
    )  # Shape (num_frames, d2, spread_dim2)
    torch.exp_(dim2_multiplier_matrix)

    local_cross_corr = torch.bmm(
        dim1_multiplier_matrix, fft_l2_objective.to(torch.complex128)
    )
    local_cross_corr = torch.bmm(local_cross_corr, dim2_multiplier_matrix)
    local_cross_corr = torch.real(local_cross_corr)
    local_cross_corr /= d1 * d2 * upsample_factor**2

    max_corr_values, max_indices = torch.max(
        local_cross_corr.reshape(num_frames, -1), dim=1
    )
    max_indices_dim1, max_indices_dim2 = torch.unravel_index(
        max_indices, (local_cross_corr.shape[1], local_cross_corr.shape[2])
    )

    frame_indexer = torch.arange(local_cross_corr.shape[0], device=device)
    # Decide whether the subpixel shift in dim1 (keeping dim2 fixed at its original integer shift value) improves things
    dim1_subpixel_improvement_indicator = (
        local_cross_corr[frame_indexer, integer_pixel_indices, max_indices_dim2]
        >= max_corr_values
    )
    max_indices_dim1[dim1_subpixel_improvement_indicator] = integer_pixel_indices
    # Decide whether the subpixel shift in dim2 (keeping dim1 fixed at its original integer shift value) improves things
    dim2_subpixel_improvement_indicator = (
        local_cross_corr[frame_indexer, max_indices_dim1, integer_pixel_indices]
        >= max_corr_values
    )
    max_indices_dim2[dim2_subpixel_improvement_indicator] = integer_pixel_indices

    # Only incorporate subpixel shifts in each dimension if it actually improves the results
    shifts_dim1 = opt_shifts[:, 0] + dim_spread[max_indices_dim1]
    shifts_dim2 = opt_shifts[:, 1] + dim_spread[max_indices_dim2]
    # shifts_dim1 = opt_shifts[:, 0] + ((max_indices_dim1 / upsample_factor) - offset_value)
    # shifts_dim2 = opt_shifts[:, 1] + ((max_indices_dim2 / upsample_factor) - offset_value)

    return torch.stack([shifts_dim1, shifts_dim2], dim=1)


def interpolate_to_border(shifted_imgs: torch.tensor, shifts: torch.tensor):
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
    condition2 = torch.logical_and(
        shifts[:, [0]] + torch.tensor([H], device=device) <= dim1_indicator,
        shifts[:, [0]] < 0,
    )
    combined_dim1_condition = torch.logical_or(condition1, condition2)
    inverted_dim1_condition = ~combined_dim1_condition

    width_indicator = torch.arange(W, device=device)
    dim2_indicator = torch.broadcast_to(width_indicator, (shifted_imgs.shape[0], W))
    # If shifts are positive, we're interested in indices where shifts > index
    condition1 = torch.logical_and(shifts[:, [1]] >= dim2_indicator, shifts[:, [1]] > 0)
    # If shifts are negative, we're interested in indices where shifts + H < index
    condition2 = torch.logical_and(
        shifts[:, [1]] + torch.tensor([W], device=device) <= dim2_indicator,
        shifts[:, [1]] < 0,
    )
    combined_dim2_condition = torch.logical_or(condition1, condition2)
    inverted_dim2_condition = ~combined_dim2_condition

    shifted_imgs *= inverted_dim2_condition[:, None, :].float()
    shifted_imgs += (
        combined_dim2_condition[:, None, :].expand(num_frames, H, W)
    ).float() * index_col_values[:, :, None]
    shifted_imgs *= inverted_dim1_condition[:, :, None].float()
    shifted_imgs += (
        combined_dim1_condition[:, :, None].expand(num_frames, H, W)
    ).float() * index_row_values[:, None, :]

    return shifted_imgs


def compute_stride_routine(shape: Tuple[int, int, int],
                           num_blocks: Tuple[int, int],
                           overlaps: Tuple[int, int]):
    """
    Args
        num_blocks (Tuple[int, int]): The number of blocks in each dimension that we use to partition the FOV
        overlaps (Tuple[int, int]): The amount of overlap in each dimension between adjacent blocks
    Returns:
        Tuple[Tuple[int, int], torch.tensor, torch.tensor]: A tuple describing the (a) strides in both dimensions and the start points for
            each block.
    """
    fov_dim1, fov_dim2 = shape[1], shape[2]
    if fov_dim1 < overlaps[0] or fov_dim2 < overlaps[1]:
        raise ValueError(f"overlap values are bigger than the corresponding FOV dimensions")
    if math.ceil((fov_dim1 - overlaps[0]) / num_blocks[0]) < overlaps[0]:
        raise ValueError(f"This configuration guarantees that the stride in dimenion 0 is less than the overlaps, which is not allowed")
    if math.ceil((fov_dim1 - overlaps[1]) / num_blocks[1]) < overlaps[1]:
        raise ValueError(f"This configuration guarantees that the stride in dimenion 1 is less than the overlaps, which is not allowed")

    ## Add some error catching logic later
    dim1_start_pts = torch.floor(torch.linspace(0, fov_dim1 - overlaps[0], num_blocks[0] + 1))[:-1]
    dim1_stride = fov_dim1 - overlaps[0] - dim1_start_pts[-1]

    dim2_start_pts = torch.floor(torch.linspace(0, fov_dim2 - overlaps[1], num_blocks[1] + 1))[:-1]
    dim2_stride = fov_dim2 - overlaps[1] - dim2_start_pts[-1]

    return (dim1_stride, dim2_stride), dim1_start_pts, dim2_start_pts


def extract_patches(
    img: torch.tensor,
    start_pts_dim1: torch.tensor,
    start_pts_dim2: torch.tensor,
    patch_dims: Tuple[int, int]
) -> torch.tensor:
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
    device = img.device
    patch_h, patch_w = patch_dims
    first_dim, second_dim = start_pts_dim1, start_pts_dim2

    # Create all start positions using meshgrid
    grid_x, grid_y = torch.meshgrid(first_dim, second_dim, indexing="ij")
    patch_grid_dimensions = grid_x.shape

    start_positions = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=1)
    num_patches = start_positions.shape[0]

    # Generate patch indices
    patch_dim1 = torch.arange(patch_h, device=device).view(-1, 1) + start_positions[:, 0].view(
        -1, 1, 1
    ) # (num_patches, patch_h, 1)
    patch_dim2 = torch.arange(patch_w, device=device).view(1, -1) + start_positions[:, 1].view(
        -1, 1, 1
    )  # (num_patches, 1, patch_w)

    patches = img[
        :, patch_dim1.long(), patch_dim2.long()
    ]  # (num_frames, num_patches, patch_h, patch_w)
    return patches.reshape(
        (
            num_frames,
            patch_grid_dimensions[0],
            patch_grid_dimensions[1],
            patch_h,
            patch_w,
        )
    )

def apply_displacement_vector_field(
    imgs: torch.tensor, shift_vector_field: torch.tensor
) -> torch.tensor:
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
        indexing="ij",  # "ij" produces (row, col) ordering
    )
    base_grid = torch.stack(
        (fovdim1_grid, fovdim2_grid), dim=-1
    )  # Shape: (fov_dim1, fov_dim2, 2)

    ## Need to negate the shifts here because the grid_sample routine is really moving the "grid" (not the image)
    remapped_coords = base_grid.unsqueeze(0) + -1 * shift_vector_field

    # Normalize remapped coordinates to [-1, 1]
    remapped_coords[..., 0] *= 2 / (fov_dim1 - 1)
    remapped_coords[..., 0] -= 1

    remapped_coords[..., 1] *= 2 / (fov_dim2 - 1)
    remapped_coords[..., 1] -= 1

    # Swap the coordinate order to match (x, y) expectation of grid_sample
    remapped_coords = torch.flip(remapped_coords, dims=[-1])

    imgs = imgs.unsqueeze(1)  # Shape (num_frames, 1, H, W)

    corrected_imgs = torch.nn.functional.grid_sample(
        imgs.double(),
        remapped_coords.double(),
        mode="bicubic",
        padding_mode="border",
        align_corners=False,
    )

    return corrected_imgs.squeeze(1).float()


def generate_motion_field_from_pwrigid_shifts(
    shifts: torch.Tensor, fov_dim1: int, fov_dim2: int
) -> torch.Tensor:
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
        align_corners=True,
    ).permute(
        0, 2, 3, 1
    )  # Restore to (batch_size, fov_dim1, fov_dim2, 2)

    return pixelwise_motion_vector


def _valid_pixel_identifier(
    shift_lower_bounds: torch.Tensor,
    shift_upper_bounds: torch.Tensor,
    fov_dim1: int,
    fov_dim2: int,
):
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
        row_indices <= shift_upper_bounds_adj[:, 0, None]
    )
    valid_cols = (col_indices >= shift_lower_bounds_adj[:, 1, None]) & (
        col_indices <= shift_upper_bounds_adj[:, 1, None]
    )

    valid_rows[:, :fov_dim1] += valid_rows[:, fov_dim1:]
    valid_cols[:, :fov_dim2] += valid_cols[:, fov_dim2:]

    return valid_rows[:, :fov_dim1], valid_cols[:, :fov_dim2]


def _estimate_patchwise_rigid_shifts(
    image_stack_patchwise: torch.tensor,
    template_patchwise: torch.tensor,
    max_deviation_rigid: Tuple[int, int],
    rigid_shifts: torch.tensor,
    pixel_weighting: Optional[torch.tensor] = None,
) -> torch.tensor:
    """
    Estimate rigid shifts to apply to a given image stack to best align each frame to template(s)

    Args:
        image_stack_patchwise (torch.tensor): Shape (num_frames, num_patches, patch_dim1, patch_dim2).
        template_patchwise (torch.tensor): Shape either (num_frames, num_patches, patch_dim1, patch_dim2) or (num_patches, patch_dim1, patch_dim2).
            The template to which we align each patch.
        max_deviation_rigid (Tuple[int, int]): The maximum deviation of each patch from its optimal integer rigid shift
        rigid_shifts (torch.tensor): Shape (num_frames, 2)
        pixel_weighting (Optional[torch.tensor]): Shape (num_frames, num_patches, patch_dim1, patch_dim2).
    Returns:
        patchwise_rigid_shifts (torch.tensor): Shape (num_frames, num_patches, 2). Describes the rigid shift in dim1 and dim2 that needs to be applied
            at each patch at each frame to optimally align it with the appropriate template.
    """

    if (
        len(template_patchwise.shape) == 3
    ):  # One set of patchwise templates for all frames
        template_patchwise = template_patchwise.unsqueeze(0)
    elif len(template_patchwise.shape) == 4:
        if template_patchwise.shape[0] == 1:
            pass
        elif template_patchwise.shape[0] != image_stack_patchwise.shape[0]:
            raise ValueError(
                f"The number of templates {template_patchwise.shape[0]} does not match number of frames {image_stack_patchwise.shape[0]}"
            )

    num_frames, num_patches, patch_dim1, patch_dim2 = image_stack_patchwise.shape
    device = image_stack_patchwise.device

    if pixel_weighting is None:
        fft_image_stack = torch.fft.fft2(image_stack_patchwise)
        fft_template = torch.conj(torch.fft.fft2(template_patchwise))

        fft_l2_objective = fft_image_stack * fft_template
        spatial_domain_cross_correlation = torch.real(
            torch.fft.ifft2(fft_l2_objective, norm="backward")
        )
    else:
        fft_image_stack = torch.fft.fft2(image_stack_patchwise)
        fft_image_stack_sq = torch.fft.fft2(torch.square(image_stack_patchwise))
        fft_weighted_template = torch.conj(
            torch.fft.fft2(torch.square(pixel_weighting) * template_patchwise)
        )
        fft_pixel_weight_sq = torch.conj(torch.fft.fft2(torch.square(pixel_weighting)))
        fft_l2_objective = (
            2 * fft_weighted_template * fft_image_stack
            - fft_pixel_weight_sq * fft_image_stack_sq
        )
        spatial_domain_cross_correlation = torch.real(
            torch.fft.ifft2(fft_l2_objective, norm="backward")
        )

    """
    Critical: we negate the rigid shifts, because the all routines to estimate shifts first 
    find the optimal TEMPLATE --> Frame shift. So if we want to provide bounds, they need to be shifts applied
    to the template, not the frames.
    """
    max_deviation_rigid = torch.tensor(
        [max_deviation_rigid[0], max_deviation_rigid[1]]
    ).to(device)
    shift_lower_bounds = -1 * rigid_shifts - max_deviation_rigid.unsqueeze(0)
    shift_upper_bounds = -1 * rigid_shifts + max_deviation_rigid.unsqueeze(0)

    valid_rows, valid_cols = _valid_pixel_identifier(
        shift_lower_bounds, shift_upper_bounds, patch_dim1, patch_dim2
    )
    valid_locations = torch.bmm(
        valid_rows.unsqueeze(2).float(), valid_cols.unsqueeze(1).float()
    )  # Shape (num_frames, patch_dim1, patch_dim2)
    invalid_locations = (~(valid_locations.bool())).float()
    cross_correlation_values = (
        spatial_domain_cross_correlation * valid_locations[:, None, :, :]
    )
    invalid_subtraction = invalid_locations * torch.abs(
        torch.amax(spatial_domain_cross_correlation)
    )

    cross_correlation_values -= invalid_subtraction.unsqueeze(
        1
    )  # Guarantees that the maximum correlation value is not at an invalid pixel

    ## We can move from num_frames x num_patches x patchdim1 x patchdim2 to (num_frames x num_patches) x patchdim1 x patchdim2
    cross_correlation_values = cross_correlation_values.reshape(
        (num_frames * num_patches, patch_dim1, patch_dim2)
    )
    max_indices = torch.argmax(
        cross_correlation_values.reshape((cross_correlation_values.shape[0], -1)), dim=1
    )
    shifts_dim1, shifts_dim2 = torch.unravel_index(
        max_indices, (patch_dim1, patch_dim2)
    )
    shifts = torch.stack([shifts_dim1, shifts_dim2], dim=1)

    fft_corr_reshape = fft_l2_objective.reshape(
        (num_frames * num_patches, patch_dim1, patch_dim2)
    )

    for precision in [0.1, 0.01, 0.001]:
        shifts = subpixel_shift_method(shifts, fft_corr_reshape, precision)

    shifts_dim1, shifts_dim2 = shifts[:, 0], shifts[:, 1]

    values_to_subtract_dim1 = (
        torch.abs(patch_dim1 - shifts_dim1) <= torch.abs(shifts_dim1)
    ).long()
    shifts_dim1 -= values_to_subtract_dim1 * patch_dim1

    values_to_subtract_dim2 = (
        torch.abs(patch_dim2 - shifts_dim2) <= torch.abs(shifts_dim2)
    ).long()
    shifts_dim2 -= values_to_subtract_dim2 * patch_dim2

    # No need to be strict about max shift here (within fractional pixels)
    shifts = torch.stack([shifts_dim1, shifts_dim2], dim=1)
    shifts = shifts.reshape(num_frames, num_patches, 2)
    shifts *= -1

    return shifts


def construct_weighting_scheme(dim1: int, dim2: int, device="cpu") -> torch.Tensor:
    # Half sizes (center region)
    hbh = dim1 // 2
    hbw = dim2 // 2

    # Create the ramp matrices
    ramp_y = torch.arange(hbh, device=device).unsqueeze(1).expand(hbh, hbw)
    ramp_x = torch.arange(hbw, device=device).unsqueeze(0).expand(hbh, hbw)
    min_ramp = torch.minimum(ramp_x, ramp_y).float()

    # Initialize the full weighting matrix
    block_weights = torch.ones((dim1, dim2), device=device, dtype=torch.float32)

    # Fill quadrants
    block_weights[:hbh, :hbw] += min_ramp
    block_weights[:hbh, hbw:] = torch.fliplr(block_weights[:hbh, :dim2 - hbw])
    block_weights[hbh:, :] = torch.flipud(block_weights[:dim1 - hbh, :])

    return block_weights



def scatter_patches_to_fov(
    data_to_reformat: torch.Tensor,
    start_points_dim0: torch.Tensor,
    start_points_dim1: torch.Tensor,
    fov_dims: tuple,
):
    """
    Efficiently scatter patches into a full FOV tensor.

    Args:
        X: Tensor of shape (num_frames, num_patches_dim0, num_patches_dim1, patch_length_dim0, patch_length_dim1)
        start_points_dim0: LongTensor of shape (num_patches_dim0,) - start indices for patches along dim 0
        start_points_dim1: LongTensor of shape (num_patches_dim1,) - start indices for patches along dim 1
        patch_dims: Tuple (patch_length_dim0, patch_length_dim1)
        fov_dims: Tuple (fov_dim0, fov_dim1) - output FOV size

    Returns:
        full: Tensor of shape (num_frames, fov_dim0, fov_dim1)
    """
    device = data_to_reformat.device
    F, P0, P1, ph, pw = data_to_reformat.shape
    fov_dim0, fov_dim1 = fov_dims

    # Create patch-local grid
    dy = torch.arange(ph, device=device)
    dx = torch.arange(pw, device=device)
    grid_y, grid_x = torch.meshgrid(dy, dx, indexing='ij')  # shape (ph, pw)

    # Global positions for each patch
    start_y = start_points_dim0.to(device) # (P0,)
    start_x = start_points_dim1.to(device) # (P1,)

    # Compute global indices per patch
    gy = start_y[:, None, None] + grid_y[None, :, :]       # (P0, ph, pw)
    gx = start_x[:, None, None] + grid_x[None, :]       # (P1, ph, pw)

    # Expand to full shape
    gy = gy[None, :, None, :, :].expand(F, P0, P1, ph, pw)  # (F, P0, P1, ph, pw)
    gx = gx[None, None, :, :, :].expand(F, P0, P1, ph, pw)  # (F, P0, P1, ph, pw)
    gf = torch.arange(F, device=device)[:, None, None, None, None].expand(F, P0, P1, ph, pw)

    # Flatten everything for scatter
    data_to_reformat_flat = data_to_reformat.reshape(-1)
    gy_flat = gy.reshape(-1).long()
    gx_flat = gx.reshape(-1).long()
    gf_flat = gf.reshape(-1).long()

    # Output tensor
    full = torch.zeros((F, fov_dim0, fov_dim1), device=device)
    full.index_put_((gf_flat, gy_flat, gx_flat), data_to_reformat_flat, accumulate=True)

    return full

def register_frames_pwrigid(
    reference_frames: torch.tensor,
    template: torch.tensor,
    num_blocks: Tuple[int, int],
    overlaps: Tuple[int, int],
    max_rigid_shifts: Tuple[int, int],
    max_deviation_rigid: Tuple[int, int],
    target_frames: Optional[torch.tensor] = None,
    pixel_weighting: Optional[torch.tensor] = None,
):
    """
    Performs piecewise rigid normcorre registration. Method estimates a motion vector field that quantifies motion of
    references frames relative to template, and applies relevant transform to correct the motion.

    Args:
        reference_frames (torch.tensor): Shape (num_frames, fov_dim1, fov_dim2). We estimate shifts that optimally align reference_frames to
            the template
        template (torch.tensor): Shape (fov_dim1, fov_dim2)  or (num_frames, fov_dim1, fov_dim2). The template(s) used for alignment.
        num_blocks (tuple[int, int]): The number of patches in both the height and width dimensions that we partition the FOV into
        overlaps (tuple[int, int]): Two integers, used to specify the degree of overlap between patches.
            Together, (strides[0] + overlaps[0], strides[1] + overlaps[1]) defines the patch size for pw rigid registration.
        max_rigid_shifts (tuple[int, int]): The maximum (full-fov) rigid shifts, used to perform rigid motion correction prior to piecewise
            rigid registration.
        max_deviation_rigid (tuple[int, int]): The maximum number of pixels (in the height, width directions respectively) that a patch
            can shift relative to the estimate global rigid shifts of the frame.
        target_frames (Optional): The relevant shift estimation is computed between the references frames and the template(s). But the shifts can be
            applied to any other stack. To do this, specify a stack in target_frames.
        pixel_weighting (Optional): Shape (fov_dim1, fov_dim2). The weight of each pixel in the L2 loss. Used to encourage the algorithm to prioritize alignemnt
            of certain spatial regions of the data.

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

    if len(template.shape) == 2:  # One template, all frames
        template = template[None, :, :]
    elif len(template.shape) == 3:
        if template.shape[0] == 1:
            pass
        elif template.shape[0] != reference_frames.shape[0]:
            raise ValueError(
                f"The number of templates {template.shape[0]} does not match number of frames {reference_frames.shape[0]}"
            )

    if target_frames is None:
        target_frames = (
            reference_frames  # We are not applying shifts to another stack here
        )

    rigid_shifts = estimate_rigid_shifts(
        reference_frames, template, max_rigid_shifts, pixel_weighting=pixel_weighting
    )

    strides, dim1_start_pts, dim2_start_pts = compute_stride_routine(reference_frames.shape, num_blocks, overlaps)
    dim1_start_pts = dim1_start_pts.to(device)
    dim2_start_pts = dim2_start_pts.to(device)

    patches = (int(strides[0].item()) + overlaps[0], int(strides[1].item()) + overlaps[1])
    interpolation_weighting = construct_weighting_scheme(patches[0], patches[1], device=device)
    patched_data = extract_patches(reference_frames.float(),
                                   dim1_start_pts,
                                   dim2_start_pts,
                                   patches)
    patched_target_data = extract_patches(target_frames.float(),
                                          dim1_start_pts,
                                          dim2_start_pts,
                                          patches)
    if pixel_weighting is not None:
        patched_weights = extract_patches(pixel_weighting.unsqueeze(0).float(),
                                         dim1_start_pts,
                                         dim2_start_pts,
                                         patches)
    else:
        patched_weights = None
    patched_templates = extract_patches(template.float(),
                                        dim1_start_pts,
                                        dim2_start_pts,
                                        patches)

    patch_grid_dim1 = patched_data.shape[1]
    patch_grid_dim2 = patched_data.shape[2]

    lowrank_patchwise_rigid_shifts = _estimate_patchwise_rigid_shifts(
        patched_data.reshape(num_frames, -1, patches[0], patches[1]),
        patched_templates.reshape(
            patched_templates.shape[0], -1, patches[0], patches[1]
        ),
        max_deviation_rigid,
        rigid_shifts,
        pixel_weighting=patched_weights.reshape(
            patched_weights.shape[0], -1, patches[0], patches[1]
        ) if patched_weights is not None else None,
    )

    patched_target_data = apply_rigid_shifts(patched_target_data.reshape(-1, patches[0], patches[1]),
                                       lowrank_patchwise_rigid_shifts.reshape(-1, 2))
    patched_target_data = interpolate_to_border(patched_target_data,
                                          lowrank_patchwise_rigid_shifts.reshape(-1, 2))

    ## Multiply each patch by the interpolation matrix
    patched_target_data *= interpolation_weighting[None,...]

    ## Reshape everything to (num_frames, num_patches_dim0, num_patches_dim1, patch_height, patch_width)
    patched_target_data = patched_target_data.reshape(num_frames, patch_grid_dim1, patch_grid_dim2, patches[0], patches[1])

    interpolation_patches = torch.zeros(1, patch_grid_dim1, patch_grid_dim2, patches[0], patches[1], device=device)
    interpolation_patches += interpolation_weighting[None, None, None, :, :]

    ## Now efficiently scatter this data back to (num_frames, fov_dim1, fov_dim2) data
    pwrigid_results = scatter_patches_to_fov(patched_target_data, dim1_start_pts, dim2_start_pts, (fov_dim1, fov_dim2))

    pwrigid_net_weightings = scatter_patches_to_fov(interpolation_patches, dim1_start_pts, dim2_start_pts, (fov_dim1, fov_dim2))

    return torch.nan_to_num(pwrigid_results / pwrigid_net_weightings, nan=0.0), lowrank_patchwise_rigid_shifts.reshape(
        num_frames, patch_grid_dim1, patch_grid_dim2, 2
    )

def compute_pwrigid_patch_midpoints(num_blocks, overlaps, fov_height, fov_width):
    """
    Computes the midpoints of all pwrigid patches.
    Args:
        num_blocks (tuple[int, int]): The number of blocks which we partition the height/width into, respectively
        overlaps (tuple[int, int]): The number of pixels of overlap between adjacent blocks (in each spatial dimension)
        fov_height (int): The fov height
        fov_width (int): The fov width
    Returns:
        dim1_midpoints (torch.tensor): Shape (num_blocks[0], num_blocks[1]): The height-dimension midpoint coordinate for each block
        dim2_midpoints (torch.tensor): Shape (num_blocks[0], num_blocks[1]): The width dimension midpoint coordinate for each block
    """
    strides, dim1_start_pts, dim2_start_pts = compute_stride_routine((1, fov_height, fov_width), num_blocks, overlaps)
    dim1_end_pts = torch.clip(dim1_start_pts + strides[0], min=0, max=fov_height)
    dim2_end_pts = torch.clip(dim2_start_pts + strides[1], min=0, max=fov_width)
    dim1_midpoints = (dim1_start_pts + dim1_end_pts) / 2
    dim2_midpoints = (dim2_start_pts + dim2_end_pts) / 2

    dim1_coords, dim2_coords = torch.meshgrid(dim1_midpoints, dim2_midpoints, indexing='ij')
    return np.dstack([dim1_coords, dim2_coords])

def weighted_alignment_loss(
    template: torch.tensor,
    registered_images: torch.tensor,
    image_weighting: torch.tensor,
):
    """
    Args:
        template (torch.tensor): Shape (1, fov_dim1, fov_dim2) or (num_frames, fov_dim1, fov_dim2). The template(s) to which
            we align the registered_images.
        registered_images (torch.tensor): Shape (num_frames, fov_dim1, fov_dim2).
        image_weighting (torch.tensor): Shape (fov_dim1, fov_dim2).

    Returns:
        loss (torch.float)
    """
    num_pixels = template.shape[1] * template.shape[2]
    return (
        torch.sum(
            torch.square(template - registered_images) * image_weighting[None, :, :]
        )
        / num_pixels
    )
