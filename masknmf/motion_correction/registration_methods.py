import torch
import numpy as np
import math
from masknmf.arrays.array_interfaces import ArrayLike

def register_frames_rigid(
    reference_frames: torch.Tensor,
    template: torch.Tensor,
    max_shifts: tuple[int, int],
    target_frames: torch.Tensor | None  = None,
    pixel_weighting: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Runs full rigid motion correction pipeline: estimating shifts, applying shifts to the iamge stack, and using a copying scheme
    to deal with edge artifacts.

    Args:
        reference_frames (torch.Tensor): Shape (num_frames, fov_height, fov_width)
        template (torch.Tensor): Shape either (fov_height, fov_width) or (num_frames, fov_height, fov_width). The template(s) to which we align the images
        max_shifts (tuple[int, int]): The max shift in the spatial height and width dimensions respectively.
        target_frames (torch.Tensor | None): If specified, we learn the shifts to optimally align reference frames to the template(s) and
            apply those shifts to this set of target frames. Useful for dual-color imaging settings.
        pixel_weighting (torch.Tensor | None): Shape (fov_height, fov_width). If specified, the weight (importance) of
            each pixel in the rigid shift estimation.
    Returns:
        registered_images (torch.Tensor): Shape (num_frames, fov_height, fov_width).
        estimated_shifts (torch.Tensor): Shape (num_frames, 2).
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


def apply_rigid_shifts(images: torch.Tensor, shifts: torch.Tensor) -> torch.Tensor:
    """
    Applies rigid shifts in the height and width dimensions for each image.
    Critical: implementation must use torch.complex128 for numerical precision.

    Args:
        images (torch.Tensor): Shape (num_frames, fov_height, fov_width). Images to which we apply shifts
        shifts (torch.Tensor): Shape (num_frames, 2). Index [i, :] gives the (i-1)-th shift in the height/width dimensions respectively.

    Returns:
        shifted_images (torch.Tensor): Shape (num_frames, fov_height, fov_width).
    """
    if images.ndim == 2:
        images = images[None, :, :]
    if shifts.ndim == 1:
        shifts = shifts[None, :]
    if images.shape[0] != shifts.shape[0]:
        raise ValueError(
            f"Provide same number of images and shifts. You provided {images.shape[0]} images and {shifts.shape[0]} shifts"
        )
    if images.device != shifts.device:
        raise ValueError(
            f"images are on {images.device} and shifts are on {shifts.device}. Place on same device first"
        )

    device = images.device
    fov_height, fov_width = images.shape[1], images.shape[2]

    # Compute FFT of images
    frequency_images = torch.fft.fft2(images, norm="ortho")

    # Compute frequency grids using fftfreq
    dim1_frequency = (-1j * 2 * torch.pi * torch.fft.fftfreq(fov_height, d=1, device=device))[
        None, :
    ].to(torch.complex128)
    dim2_frequency = (-1j * 2 * torch.pi * torch.fft.fftfreq(fov_width, d=1, device=device))[
        None, :
    ].to(torch.complex128)

    # Compute phase shift multipliers
    shift_dim1_terms = shifts[:, [0]].to(torch.complex128)
    term_dim1 = torch.exp(shift_dim1_terms @ dim1_frequency)
    frequency_images *= term_dim1[:, :, None]

    shift_dim2_terms = shifts[:, [1]].to(torch.complex128)
    term_dim2 = torch.exp(shift_dim2_terms @ dim2_frequency)
    frequency_images *= term_dim2[:, None, :]

    # Inverse FFT
    shifted_images = torch.fft.ifft2(frequency_images, norm="ortho")

    return torch.real(shifted_images)


def estimate_rigid_shifts(
    image_stack: torch.Tensor,
    template: torch.Tensor,
    max_shifts: tuple[int, int],
    pixel_weighting: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Estimate rigid shifts to apply to a given image stack to best align each frame to template(s)

    Args:
        image_stack (torch.Tensor): Shape (num_frames, fov_height, fov_width).
        template (torch.Tensor): Shape (fov_height, fov_width) or (num_frames, fov_height, fov_width).
        max_shifts (tuple[int, int]): Maximum shifts we can apply in each direction
        pixel_weighting (torch.Tensor): A weighting of each pixel of the FOV. If provided, this means we are
            solving a weighted L2 problem, where we prioritize alignment of certain pixels over others.
    Returns:
        rigid_shifts (torch.Tensor): Shape (num_frames, 2). rigid_shifts[i, :] gives the (fov height dimension, fov width dimension) shifts,
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

    num_frames, fov_height, fov_width = image_stack.shape
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
    dim1_valid_shifts = torch.arange(fov_height, device=device)
    dim1_valid_locations = torch.logical_or(
        dim1_valid_shifts >= fov_height - 1 - torch.abs(max_shifts[0]),
        dim1_valid_shifts <= torch.abs(max_shifts[0]),
    ).float()

    dim2_valid_shifts = torch.arange(fov_width, device=device)
    dim2_valid_locations = torch.logical_or(
        dim2_valid_shifts >= fov_width - 1 - torch.abs(max_shifts[1]),
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
    shifts_dim1, shifts_dim2 = torch.unravel_index(max_indices, (fov_height, fov_width))
    shifts = torch.stack([shifts_dim1, shifts_dim2], dim=1)

    for precision in [0.1, 0.01, 0.001]:
        shifts = subpixel_shift_method(shifts, fft_l2_objective, precision)

    shifts_dim1, shifts_dim2 = shifts[:, 0], shifts[:, 1]

    values_to_subtract_dim1 = (
        torch.abs(fov_height - shifts_dim1) <= torch.abs(shifts_dim1)
    ).long()
    shifts_dim1 -= values_to_subtract_dim1 * fov_height

    values_to_subtract_dim2 = (
        torch.abs(fov_width - shifts_dim2) <= torch.abs(shifts_dim2)
    ).long()
    shifts_dim2 -= values_to_subtract_dim2 * fov_width

    # Make sure the final shifts are strictly within the max_shifts interval (we allow the superpixel estimator
    torch.clip_(shifts_dim1, -1 * max_shifts[0], max_shifts[0])
    torch.clip_(shifts_dim2, -1 * max_shifts[1], max_shifts[1])

    shifts = torch.stack([shifts_dim1, shifts_dim2], dim=1)

    # These shifts keep the image fixed and find the optimal template shift; we want the opposite (shift image --> match to template)
    shifts *= -1
    return shifts


def subpixel_shift_method(
    opt_shifts: torch.Tensor, fft_l2_objective: torch.Tensor, precision: float
) -> torch.Tensor:
    """
    Use fourier interpolation (up to the "upsample_factor") to find the optimal "subpixel" shift, within 0.1 of a pixel

    Args:
        opt_shifts (torch.Tensor): Shape (num_frames, 2). Tensor describing for each frame the optimal integer
            height and width shifts. This function searches for subpixel shifts in a local neighborbood of the optimal integer shifts.
        fft_l2_objective (torch.Tensor): Shape (num_frames, fov_height, fov_width).
            The FFT of the objective function (over "shifts") which we seek to optimize
        precision (float): Only accepts these values: [0.1, 0.01, 0.001]. The accuracy to which we estimate the subpixel shift, relative to the
            opt_integer shifts.

    Returns:
        subpixel_estimates (torch.Tensor): Shape (num_frames, 2). The optimal subpixel shifts
    """
    if precision not in [0.1, 0.01, 0.001]:
        raise ValueError(
            f"Precision can only be 0.1, 0.01, 0.001. Input was {precision}"
        )

    num_frames, fov_height, fov_width = fft_l2_objective.shape
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
    height_dim_subpixel_indices = (
        opt_shifts[:, [0]].float() + dim_spread[None, :]
    )  # Shape (num_frames, spread_height_dim)

    height_dim_multiplier_vector = (
        2
        * 1j
        * torch.pi
        * torch.fft.fftfreq(fov_height, d=1.0, device=device).to(torch.complex128)
    )
    # Shape (num_frames, spread_dim1, fov_height)
    height_dim_multiplier_matrix = (
        height_dim_subpixel_indices.to(torch.complex128).unsqueeze(2)
        @ height_dim_multiplier_vector[None, :]
    )
    torch.exp_(height_dim_multiplier_matrix)

    width_dim_subpixel_indices = (
        opt_shifts[:, [1]].float() + dim_spread[None, :]
    )  # Shape (num_frames, spread_dim2)
    width_dim_multiplier_vector = (
        2
        * 1j
        * torch.pi
        * torch.fft.fftfreq(fov_width, d=1.0, device=device).to(torch.complex128)
    )
    width_dim_multiplier_matrix = (
        width_dim_subpixel_indices.to(torch.complex128).unsqueeze(2)
        @ width_dim_multiplier_vector[None, :]
    )
    width_dim_multiplier_matrix = width_dim_multiplier_matrix.permute(
        0, 2, 1
    )  # Shape (num_frames, fov_width, spread_dim2)
    torch.exp_(width_dim_multiplier_matrix)

    local_cross_corr = torch.bmm(
        height_dim_multiplier_matrix, fft_l2_objective.to(torch.complex128)
    )
    local_cross_corr = torch.bmm(local_cross_corr, width_dim_multiplier_matrix)
    local_cross_corr = torch.real(local_cross_corr)
    local_cross_corr /= fov_height * fov_width * upsample_factor**2

    max_corr_values, max_indices = torch.max(
        local_cross_corr.reshape(num_frames, -1), dim=1
    )
    max_indices_height_dim, max_indices_width_dim = torch.unravel_index(
        max_indices, (local_cross_corr.shape[1], local_cross_corr.shape[2])
    )

    frame_indexer = torch.arange(local_cross_corr.shape[0], device=device)
    # Decide whether the subpixel shift in height dim (keeping width dim fixed at its original integer shift value) improves things
    height_dim_subpixel_improvement_indicator = (
        local_cross_corr[frame_indexer, integer_pixel_indices, max_indices_width_dim]
        >= max_corr_values
    )
    max_indices_height_dim[height_dim_subpixel_improvement_indicator] = integer_pixel_indices
    # Decide whether the subpixel shift in width dim (keeping height dim fixed at its original integer shift value) improves things
    width_dim_subpixel_improvement_indicator = (
        local_cross_corr[frame_indexer, max_indices_height_dim, integer_pixel_indices]
        >= max_corr_values
    )
    max_indices_width_dim[width_dim_subpixel_improvement_indicator] = integer_pixel_indices

    # Only incorporate subpixel shifts in each dimension if it actually improves the results
    shifts_height_dim = opt_shifts[:, 0] + dim_spread[max_indices_height_dim]
    shifts_width_dim = opt_shifts[:, 1] + dim_spread[max_indices_width_dim]

    return torch.stack([shifts_height_dim, shifts_width_dim], dim=1)


def interpolate_to_border(shifted_images: torch.Tensor, shifts: torch.Tensor):
    """
    After applying rigid shifts via FFT methods, the resulting image will have some artifacts at the edges (wrap-around artifacts).
    This approach overwrites those pixels with the (approximately) nearest "valid" pixel.
    Note: this is an in-place operation.

    Args:
        shifted_images (torch.Tensor): Shape (num_frames, fov_height, fov_width). The images after shifts have been applied
        shifts (torch.Tensor): The shifts that were applied to each image
    """

    # Establish device
    device = shifted_images.device

    num_frames, fov_height, fov_width = shifted_images.shape

    ## If the shift in some dimension is 2, then the index we want is 3. Similarly if it is -2, then the index is -3
    shifted_indices = shifts + torch.nan_to_num(shifts / torch.abs(shifts), nan=0)
    frame_indices = torch.arange(shifted_images.shape[0], device=device)
    shifted_indices = torch.fix(shifted_indices).long()
    index_row_values = shifted_images[frame_indices, shifted_indices[:, 0], :]
    index_col_values = shifted_images[frame_indices, :, shifted_indices[:, 1]]

    # Decide which pixels actually need to be rewritten
    height_indicator = torch.arange(fov_height, device=device)
    dim1_indicator = torch.broadcast_to(height_indicator, (shifted_images.shape[0], fov_height))

    # If shifts are positive, we're interested in indices where shifts > index
    condition1 = torch.logical_and(shifts[:, [0]] >= dim1_indicator, shifts[:, [0]] > 0)
    # If shifts are negative, we're interested in indices where shifts + H < index
    condition2 = torch.logical_and(
        shifts[:, [0]] + torch.tensor([fov_height], device=device) <= dim1_indicator,
        shifts[:, [0]] < 0,
    )
    combined_dim1_condition = torch.logical_or(condition1, condition2)
    inverted_dim1_condition = ~combined_dim1_condition

    width_indicator = torch.arange(fov_width, device=device)
    dim2_indicator = torch.broadcast_to(width_indicator, (shifted_images.shape[0], fov_width))
    # If shifts are positive, we're interested in indices where shifts > index
    condition1 = torch.logical_and(shifts[:, [1]] >= dim2_indicator, shifts[:, [1]] > 0)
    # If shifts are negative, we're interested in indices where shifts + H < index
    condition2 = torch.logical_and(
        shifts[:, [1]] + torch.tensor([fov_width], device=device) <= dim2_indicator,
        shifts[:, [1]] < 0,
    )
    combined_dim2_condition = torch.logical_or(condition1, condition2)
    inverted_dim2_condition = ~combined_dim2_condition

    shifted_images *= inverted_dim2_condition[:, None, :].float()
    shifted_images += (
        combined_dim2_condition[:, None, :].expand(num_frames, fov_height, fov_width)
    ).float() * index_col_values[:, :, None]
    shifted_images *= inverted_dim1_condition[:, :, None].float()
    shifted_images += (
        combined_dim1_condition[:, :, None].expand(num_frames, fov_height, fov_width)
    ).float() * index_row_values[:, None, :]

    return shifted_images


def compute_stride_routine(shape: tuple[int, int, int],
                           minimum_patch_sizes: tuple[int, int],
                           overlaps: tuple[int, int]) -> tuple[tuple[int, int], torch.Tensor, torch.Tensor]:
    """
    Args:
        shape (tuple[int, int, int]): Describes shape of imaging data (num_frames, fov_height, fov_width)
        minimum_patch_sizes (tuple[int, int]): The number of blocks in each dimension that we use to partition the FOV
        overlaps (tuple[int, int]): The amount of overlap in each dimension between adjacent blocks
    Returns:
        tuple[tuple[int, int], torch.Tensor, torch.Tensor]: A tuple describing the (a) strides in both dimensions and the start points for
            each block.
    """
    fov_height, fov_width = shape[1], shape[2]
    if fov_height <= overlaps[0] or fov_width <= overlaps[1]:
        raise ValueError(f"overlap values are bigger than the corresponding FOV dimensions")
    if fov_height <= minimum_patch_sizes[0]  or fov_width <= minimum_patch_sizes[1]:
        raise ValueError(f"patch size dimensions must be smaller than the actual FOV dimensions")
    if minimum_patch_sizes[0] <= 2 * overlaps[0] or minimum_patch_sizes[1] <= 2 * overlaps[1]:
        raise ValueError(f"the minimum patch size must be at least twice the size of the overlaps")

    min_strides = (minimum_patch_sizes[0] - overlaps[0], minimum_patch_sizes[1] - overlaps[1])


    ##Since minimum_patch_sizes is less than (fov_height, fov_width), both below terms are guaranteed to be at least 1
    num_blocks_height = math.floor((fov_height - overlaps[0]) / min_strides[0])
    num_blocks_width = math.floor((fov_width - overlaps[1]) / min_strides[1])

    ## Add some error catching logic later
    dim1_start_pts = torch.floor(torch.linspace(0, fov_height - overlaps[0], num_blocks_height + 1))[:-1]
    dim1_stride = fov_height - overlaps[0] - dim1_start_pts[-1]

    dim2_start_pts = torch.floor(torch.linspace(0, fov_width - overlaps[1], num_blocks_width + 1))[:-1]
    dim2_stride = fov_width - overlaps[1] - dim2_start_pts[-1]

    return (dim1_stride, dim2_stride), dim1_start_pts, dim2_start_pts


def extract_patches(
    images: torch.Tensor,
    start_points_height_dim: torch.Tensor,
    start_points_width_dim: torch.Tensor,
    patch_dims: tuple[int, int]
) -> torch.Tensor:
    """
    Batched routine that extracted a proper "sliding window" of patches for piecewise rigid registration.

    Args:
        images (torch.Tensor): Shape (num_frames, height, width).
        start_points_height_dim (torch.Tensor): A 1D torch Tensor specifying at which height indices the piecewise rigid patches start
        start_points_width_dim (torch.Tensor): A 1D torch Tensor specifying at which width indices the piecewise rigid patches start
        patch_dims (tuple[int, int]): The height, width dimensions of a single patch

    Returns:
        patches (torch.Tensor): Extracted patches with shape (num_frames, num_patches_height, num_patches_width, patch_height, patch_width).
            num_patches_height, num_patches_width gives the dimensions of the grid of overlapping patches (in the way they tile the actual FOV).
    """
    num_frames = images.shape[0]
    device = images.device
    patch_height, patch_width = patch_dims

    # Create all start positions using meshgrid
    grid_height, grid_width = torch.meshgrid(start_points_height_dim.to(device),
                                             start_points_width_dim.to(device),
                                             indexing="ij")
    patch_grid_dimensions = grid_height.shape

    start_positions = torch.stack([grid_height.flatten(), grid_width.flatten()], dim=1)

    # Generate patch indices
    patch_row_indices = torch.arange(patch_height, device=device).view(-1, 1) + start_positions[:, 0].view(
        -1, 1, 1
    ) # (num_patches, patch_height, 1)
    patch_column_indices = torch.arange(patch_width, device=device).view(1, -1) + start_positions[:, 1].view(
        -1, 1, 1
    )  # (num_patches, 1, patch_width)

    patches = images[
        :, patch_row_indices.long(), patch_column_indices.long()
              ]  # (num_frames, num_patches, patch_height, patch_width)
    return patches.reshape(
        (
            num_frames,
            patch_grid_dimensions[0],
            patch_grid_dimensions[1],
            patch_height,
            patch_width,
        )
    )

def _valid_pixel_identifier(
    shift_lower_bounds: torch.Tensor,
    shift_upper_bounds: torch.Tensor,
    fov_height: int,
    fov_width: int,
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
        fov_height (int): The height of the field of view (FOV).
        fov_width (int): The width of the field of view (FOV).

    Returns:
        - valid_rows (torch.Tensor): Shape (num_frames, fov_height).
                                     Indicates valid row indices for each frame.
        - valid_cols (torch.Tensor): Shape (num_frames, fov_width).
                                     Indicates valid column indices for each frame.
    """
    device = shift_lower_bounds.device
    num_frames = shift_lower_bounds.shape[0]

    # Clone to avoid modifying original tensors
    shift_lower_bounds_adj = shift_lower_bounds.clone()
    shift_upper_bounds_adj = shift_upper_bounds.clone()

    # Convert negative indices to valid positive indices using modular wrapping
    # If the interval is (a, b) with a < 0, then the new interval should be
    shift_lower_bounds_adj[:, 0] += fov_height
    shift_upper_bounds_adj[:, 0] += fov_height  # The interval is now [0, 2*fov_height)

    shift_lower_bounds_adj[:, 1] += fov_width
    shift_upper_bounds_adj[:, 1] += fov_width

    # Generate row and column indices
    row_indices = torch.arange(fov_height * 2, device=device).expand(num_frames, -1)
    col_indices = torch.arange(fov_width * 2, device=device).expand(num_frames, -1)

    # Compute valid row/column masks
    valid_rows = (row_indices >= shift_lower_bounds_adj[:, 0, None]) & (
        row_indices <= shift_upper_bounds_adj[:, 0, None]
    )
    valid_cols = (col_indices >= shift_lower_bounds_adj[:, 1, None]) & (
        col_indices <= shift_upper_bounds_adj[:, 1, None]
    )

    valid_rows[:, :fov_height] += valid_rows[:, fov_height:]
    valid_cols[:, :fov_width] += valid_cols[:, fov_width:]

    return valid_rows[:, :fov_height], valid_cols[:, :fov_width]


def _estimate_patchwise_rigid_shifts(
    image_stack_patchwise: torch.Tensor,
    template_patchwise: torch.Tensor,
    max_deviation_rigid: tuple[int, int],
    rigid_shifts: torch.Tensor,
    pixel_weighting: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Estimate rigid shifts to apply to a given image stack to best align each frame to template(s)

    Args:
        image_stack_patchwise (torch.Tensor): Shape (num_frames, num_patches, patch_dim1, patch_dim2).
        template_patchwise (torch.Tensor): Shape either (num_frames, num_patches, patch_dim1, patch_dim2) or (num_patches, patch_dim1, patch_dim2).
            The template to which we align each patch.
        max_deviation_rigid (tuple[int, int]): The maximum deviation of each patch from its optimal integer rigid shift
        rigid_shifts (torch.Tensor): Shape (num_frames, 2)
        pixel_weighting (torch.Tensor | None = None): Shape (num_frames, num_patches, patch_dim1, patch_dim2).
    Returns:
        patchwise_rigid_shifts (torch.Tensor): Shape (num_frames, num_patches, 2). Describes the rigid shift in dim1 and dim2 that needs to be applied
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
    start_points_height_dim: torch.Tensor,
    start_points_width_dim: torch.Tensor,
    fov_dims: tuple[int, int],
):
    """
    Efficiently scatter patches into a full FOV tensor.

    Args:
        data_to_reformat (torch.Tensor): (num_frames, num_patches_height, num_patches_width, 2)
        start_points_height_dim (torch.Tensor): LongTensor of shape (num_patches_height,) - start indices for patches along dim 0
        start_points_width_dim (torch.Tensor): LongTensor of shape (num_patches_width,) - start indices for patches along dim 1
        fov_dims: Tuple (fov_height, fov_width) - output FOV size

    Returns:
        full: Tensor of shape (num_frames, fov_height, fov_width)
    """
    device = data_to_reformat.device
    F, P0, P1, ph, pw = data_to_reformat.shape
    fov_dim0, fov_dim1 = fov_dims

    # Create patch-local grid
    dy = torch.arange(ph, device=device)
    dx = torch.arange(pw, device=device)
    grid_y, grid_x = torch.meshgrid(dy, dx, indexing='ij')  # shape (ph, pw)

    # Global positions for each patch
    start_y = start_points_height_dim.to(device) # (P0,)
    start_x = start_points_width_dim.to(device) # (P1,)

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


def pwrigid_shift_estimation_routine(
        reference_frames: torch.Tensor,
        template: torch.Tensor,
        minimum_patch_sizes: tuple[int, int],
        overlaps: tuple[int, int],
        max_rigid_shifts: tuple[int, int],
        max_deviation_rigid: tuple[int, int],
        pixel_weighting: torch.Tensor | None = None):
    """
    This routine is run to infer the piecewise rigid shift (per patch) to optimally align a reference movie to a template
    Args are outlined in register_frames_pwrigid
    """
    device = reference_frames.device
    num_frames, fov_height, fov_width = reference_frames.shape

    if len(template.shape) == 2:  # One template, all frames
        template = template[None, :, :]
    elif len(template.shape) == 3:
        if template.shape[0] == 1:
            pass
        elif template.shape[0] != reference_frames.shape[0]:
            raise ValueError(
                f"The number of templates {template.shape[0]} does not match number of frames {reference_frames.shape[0]}"
            )

    rigid_shifts = estimate_rigid_shifts(
        reference_frames, template, max_rigid_shifts, pixel_weighting=pixel_weighting
    )

    strides, dim1_start_pts, dim2_start_pts = compute_stride_routine(reference_frames.shape, minimum_patch_sizes, overlaps)
    dim1_start_pts = dim1_start_pts.to(device)
    dim2_start_pts = dim2_start_pts.to(device)

    patches = (int(strides[0].item()) + overlaps[0], int(strides[1].item()) + overlaps[1])
    patched_data = extract_patches(reference_frames.float(),
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

    lowrank_patchwise_rigid_shifts = lowrank_patchwise_rigid_shifts.reshape(num_frames, patch_grid_dim1,
                                                                            patch_grid_dim2, 2)

    return lowrank_patchwise_rigid_shifts

def subsample_shifts(shifts: torch.Tensor,
                     height: int,
                     width: int,
                     row_slice: slice,
                     col_slice: slice,
                     device='cpu'):
    """
    Routine to grid sample a low-dimensional vector field. This routine
    """
    height_ids = torch.linspace(-1 + (1 / height), 1 - (1 /  height), height, device=device)
    width_ids = torch.linspace(-1 + (1 / width), 1 - (1 / width), width, device=device)
    yy, xx = torch.meshgrid(height_ids, width_ids, indexing='ij')
    grid = torch.stack([xx, yy], dim=2)[row_slice, col_slice, :] ## grid sample needs x first, then y
    grid = grid.expand(shifts.shape[0], grid.shape[0], grid.shape[1], grid.shape[2])  # Now shape (1, H_grid, W_grid, 2)
    outputs = torch.nn.functional.grid_sample(shifts.permute(0, 3, 1, 2),
                                              grid,
                                              mode='bilinear',
                                              align_corners=False,
                                              padding_mode='border')
    return outputs.permute(0, 2, 3, 1)


def compute_pixel_coords_to_sample(row_slice: slice,
                                   col_slice: slice,
                                   shift_coordinates: torch.Tensor,
                                   height: int,
                                   width: int):
    r0, r1, rstep = row_slice.indices(height)
    c0, c1, cstep = col_slice.indices(width)

    row_tensor = torch.arange(r0,
                              r1,
                              rstep,
                              device=shift_coordinates.device)

    col_tensor = torch.arange(c0,
                              c1,
                              cstep,
                              device=shift_coordinates.device)
    yy, xx = torch.meshgrid(row_tensor, col_tensor, indexing='ij')
    updated_coords = torch.stack([yy, xx], dim=2)[None, ...]
    updated_coords = updated_coords.expand(shift_coordinates.shape)
    coords_to_sample = updated_coords - shift_coordinates

    return coords_to_sample


def compute_pixel_sample_lower_bounds(coords_to_sample: torch.Tensor,
                                      height: int,
                                      width: int):
    """
    We need to decide what frames of data to load
    Args:
        coords_to_sample (torch.Tensor): Shape (N, height_dim, width_dim, 2)
    """
    ## Critical: Add a 2 pixel (or however many pixels are possible for bicubic interpolation
    min_coordinate = torch.floor(torch.amin(coords_to_sample, dim=[0, 1, 2])) - 2  # Shape (N, 2)
    max_coordinate = torch.ceil(torch.amax(coords_to_sample, dim=[0, 1, 2])) + 3  # Shape (N, 2)

    height_low = max(0, int(min_coordinate[0]))
    height_high = min(height, int(max_coordinate[0]))

    width_low = max(0, int(min_coordinate[1]))
    width_high = min(width, int(max_coordinate[1]))
    return (height_low, height_high), (width_low, width_high)


def compute_pixel_to_pixel_resample(data: torch.Tensor,
                                    height_range: tuple[int, int],
                                    width_range: tuple[int, int],
                                    coords_sample: torch.Tensor):
    """
    Given ``data`` a spatiotemporal dataset whose pixels lie in height_range x width_range,
     this routine computes estimates at the coordinate locations given by coords_sample

    Args:
        data (torch.Tensor): Shape (num_frames, height_range, width_range)
        height_range (tuple[int, int]): A lower and (exclusive) upper bound for the row indices that are being sampled
        width_range (tuple[int, int]): A lower and (exclusive) upper bound for the column indices that are being sampled
        coords_sample (torch.Tensor): Shape (num_frames, H, W, 2), coordinate values that we want to sample
     """
    height = height_range[1] - height_range[0]
    width = width_range[1] - width_range[0]

    ## Convert the coordinates to the 1D grid where pixels are NOT aligned to the corners
    scaled_height_coords = -1 + (2*(coords_sample[:,:,:,0] - height_range[0]) + 1)/height
    scaled_width_coords = -1 + (2*(coords_sample[:,:,:,1] - width_range[0])  + 1)/width
    final_grid_coords = torch.stack([scaled_width_coords, scaled_height_coords], dim=3)
    final_output = torch.nn.functional.grid_sample(data[:, None, :, :],
                                                   final_grid_coords,
                                                   mode = 'bicubic',
                                                   align_corners = False,
                                                   padding_mode = 'border').squeeze(1)

    return final_output


## Routine for applying nonrigid shifts to any spatial subset of the data
def apply_pwrigid_shifts(data: ArrayLike,
                         shifts: torch.Tensor,
                         row_slice: slice,
                         col_slice: slice,
                         temporal_indices: torch.Tensor | slice | None = None,
                         device='cpu') -> torch.Tensor:
    """
    Routine for applying the estimated nonrigid patchwise shifts to any spatial subset of the data
    Args:
        data (ArrayLike): Shape (num_frames, height, width)
        shifts (torch.Tensor): Shape (num_frames, num_height_patches, num_width_patches, 2)
        row_slice (slice): The continuous set of rows to apply these shifts to
        col_slice (slice): The continuous set of cols to apply these shifts to
        temporal_indices (slice | torch.Tensor | None): Specifies whether we only want to get the results for a subset of timepoints.
            If None, then data and shifts must have the same number of timepoints

    Returns:
        - The motion corrected data that corresponds to data[:, row_slice, col_slice]

    """
    if temporal_indices is not None:
        shifts = shifts[temporal_indices]

    height, width = data.shape[1], data.shape[2]
    subsampled_shifts = subsample_shifts(shifts.to(device), height, width, row_slice, col_slice, device=device)
    coords_sample = compute_pixel_coords_to_sample(row_slice,
                                                   col_slice,
                                                   subsampled_shifts,
                                                   height,
                                                   width)

    height_range, width_range = compute_pixel_sample_lower_bounds(coords_sample, height, width)

    if temporal_indices is not None:
        data_subset = torch.as_tensor(data[temporal_indices, height_range[0]:height_range[1], width_range[0]:width_range[1]],
                                      device=device, dtype=torch.float32)
    else:
        data_subset = torch.as_tensor(data[:, height_range[0]:height_range[1], width_range[0]:width_range[1]],
                                      device=device, dtype=torch.float32)

    corrected_data = compute_pixel_to_pixel_resample(data_subset,
                                                     height_range,
                                                     width_range,
                                                     coords_sample)
    return corrected_data


def register_frames_pwrigid(
        reference_frames: torch.Tensor,
        template: torch.Tensor,
        minimum_patch_sizes: tuple[int, int],
        overlaps: tuple[int, int],
        max_rigid_shifts: tuple[int, int],
        max_deviation_rigid: tuple[int, int],
        target_frames: torch.Tensor | None = None,
        pixel_weighting: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Performs piecewise rigid normcorre registration. Method partitions the imaging field of view into overlapping
    rectangular patches, estimates rigid motion shifts within these patches, and then accordingly applies a nonrigid
    transformation to the full field of view to correct motion.

    Args:
        reference_frames (torch.Tensor): Shape (num_frames, fov_height, fov_width). We estimate shifts that optimally align reference_frames to
            the template
        template (torch.Tensor): Shape (fov_height, fov_width)  or (num_frames, fov_height, fov_width). The template(s) used for alignment.
        minimum_patch_sizes (tuple[int, int]): A lower bound on the (height, width) dimensions of the patch size. The actual patch sizes
            used to perform motion correction will be approximately equal to these  and are guaranteed to be at least as large.
        overlaps (tuple[int, int]): Two integers, used to specify the degree of overlap between patches.
            Together, (strides[0] + overlaps[0], strides[1] + overlaps[1]) defines the patch size for pw rigid registration.
        max_rigid_shifts (tuple[int, int]): The maximum (full-fov) rigid shifts, used to perform rigid motion correction prior to piecewise
            rigid registration.
        max_deviation_rigid (tuple[int, int]): The maximum number of pixels (in the height, width directions respectively) that a patch
            can shift relative to the estimate global rigid shifts of the frame.
        target_frames (torch.Tensor | None = None): The relevant shift estimation is computed between the references frames and the template(s). But the shifts can be
            applied to any other stack. To do this, specify a stack in target_frames.
        pixel_weighting (torch.Tensor | None = None): Shape (fov_height, fov_width). The weight of each pixel in the L2 loss. Used to encourage the algorithm to prioritize alignemnt
            of certain spatial regions of the data.

    Returns:
        registered_frames (torch.Tensor): Shape (num_frames, fov_height, fov_width). The motion corrected frames.
        shift_vector_field (torch.Tensor): Shape (num_frames, num_patches_dim1, num_patches_dim2, 2). During piecewise motion correction,
            we break the field of view into overlapping patches and estimate a 2D rigid shift per patch.
            The function "generate_motion_field_from_piecewise_rigid_shifts" transforms these patchwise rigid shifts into a (num_frames, fov_height, fov_width, 2)
            shaped shift vector field. It is more memory efficient to return the (num_patches_dim1, num_patches_dim2, 2) "lowrank"
            version of the shift vector field.

    """
    lowrank_patchwise_rigid_shifts = pwrigid_shift_estimation_routine(reference_frames,
                                                                      template,
                                                                      minimum_patch_sizes,
                                                                      overlaps,
                                                                      max_rigid_shifts,
                                                                      max_deviation_rigid,
                                                                      pixel_weighting)

    if target_frames is None:
        target_frames = reference_frames
    if target_frames.shape[1] != reference_frames.shape[1] or target_frames.shape[2] != reference_frames.shape[2]:
        raise ValueError("Target and Reference frames must have the same spatial dimensions")

    corrected_data = apply_pwrigid_shifts(target_frames,
                                          lowrank_patchwise_rigid_shifts,
                                          slice(0, target_frames.shape[1]),
                                          slice(0, target_frames.shape[2]),
                                          device=lowrank_patchwise_rigid_shifts.device)
    return corrected_data, lowrank_patchwise_rigid_shifts


def compute_pwrigid_patch_midpoints(minimum_patch_sizes: tuple[int, int],
                                    overlaps: tuple[int, int],
                                    fov_height: int,
                                    fov_width: int) -> torch.Tensor:
    """
    Computes the midpoints of all pwrigid patches.
    Args:
        minimum_patch_sizes (tuple[int, int]): The lower bound (height,width) patch size dimensions used to estimate
            piecewise rigid shifts over the entire field of view.
        overlaps (tuple[int, int]): The number of pixels of overlap between adjacent blocks (in each spatial dimension)
        fov_height (int): The fov height
        fov_width (int): The fov width
    Returns:
        midpoints (torch.Tensor): Shape (num_blocks_height, num_blocks_width, 2). Gives the height/width dimensions for the height and width
            midpoints respectively
    """
    strides, dim1_start_pts, dim2_start_pts = compute_stride_routine(
        (1, fov_height, fov_width), minimum_patch_sizes, overlaps
    )
    patch_h = strides[0] + overlaps[0]
    patch_w = strides[1] + overlaps[1]

    dim1_midpoints = dim1_start_pts + (patch_h - 1) / 2
    dim2_midpoints = dim2_start_pts + (patch_w - 1) / 2

    dim1_coords, dim2_coords = torch.meshgrid(dim1_midpoints, dim2_midpoints, indexing="ij")
    return torch.stack([dim1_coords, dim2_coords], dim=-1)

def weighted_alignment_loss(
    template: torch.Tensor,
    registered_images: torch.Tensor,
    image_weighting: torch.Tensor,
):
    """
    Args:
        template (torch.Tensor): Shape (1, fov_height, fov_width) or (num_frames, fov_height, fov_width). The template(s) to which
            we align the registered_images.
        registered_images (torch.Tensor): Shape (num_frames, fov_height, fov_width).
        image_weighting (torch.Tensor): Shape (fov_height, fov_width).

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
