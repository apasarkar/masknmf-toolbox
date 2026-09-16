from typing import Optional
import numpy as np
import torch


def component_at_pixel(a, centers, fov_shape, pick_index, mask=None, radius=None) -> Optional[int]:
    """
    Index of the component whose footprint contains the picked pixel; nearest center wins.

    Args:
        a: coalesced sparse tensor of shape (pixels, components)
        centers: (components, 2) tensor of (row, col) centers
        fov_shape: (height, width)
        pick_index: (col, row) as given by fastplotlib pick_info["index"]
        mask: optional bool tensor over components restricting the candidates
        radius: if no footprint contains the pixel, fall back to the nearest center within this distance
    """
    col, row = pick_index
    height, width = fov_shape
    if not (0 <= row < height and 0 <= col < width):
        return None
    pixel_ind = int(row) * width + int(col)

    a_row, a_col = a.indices()
    a_val = a.values()
    conditions = (a_row == pixel_ind) & (a_val > 0)
    if mask is not None:
        conditions &= mask.to(a_col.device)[a_col]
    candidates = torch.unique(a_col[conditions])
    max_distance = None
    if candidates.numel() == 0:
        if radius is None:
            return None
        if mask is not None:
            candidates = torch.nonzero(mask.to(centers.device), as_tuple=True)[0]
        else:
            candidates = torch.arange(centers.shape[0], device=centers.device)
        if candidates.numel() == 0:
            return None
        max_distance = float(radius)
    point = torch.tensor([row, col], dtype=centers.dtype, device=centers.device)
    distances = torch.linalg.norm(centers[candidates] - point[None, :], dim=1)
    best = int(torch.argmin(distances))
    if max_distance is not None and float(distances[best]) > max_distance:
        return None
    return int(candidates[best])


def contours_to_bbox(fov_shape, contour, extra_space=10):
    min_y, min_x = np.amin(contour, axis=0)
    max_y, max_x = np.amax(contour, axis=0)

    bound_y = (max(0, int(min_y) - extra_space), min(int(fov_shape[0]), int(max_y) + extra_space))
    bound_x = (max(0, int(min_x) - extra_space), min(int(fov_shape[1]), int(max_x) + extra_space))

    return (bound_x[0], bound_y[0], 1), (bound_x[1], bound_y[1], 1)


def zoom_to_bbox(subplot, graphic, lower_bound, upper_bound):
    world_coord_lower = graphic.map_model_to_world(lower_bound)
    world_coord_upper = graphic.map_model_to_world(upper_bound)
    subplot.x_range = (float(world_coord_lower[0]), float(world_coord_upper[0]))
    subplot.y_range = (float(world_coord_lower[1]), float(world_coord_upper[1]))
