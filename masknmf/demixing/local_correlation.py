"""
The local correlation image: each pixel's mean correlation with its eight neighbours, computed from MAD-thresholded
traces in pixel tiles. The demixer runs it on its residual (UV - AC - background); BaseResults runs it on any
residual movie an adapter exposes.
"""
from typing import Literal, Iterator

import math

import numpy as np
import torch


def threshold_data_inplace(movie_chunk,
                           mad_threshold_value: int = 2,
                           dim: int = 2,
                           sign: Literal["positive", "negative", "unconstrained"] = "unconstrained"):
    """
    Threshold data: in each pixel, compute the median and median absolute deviation (MAD),
    then zero all bins (x,t) such that Yd(x,t) < med(x) + th * MAD(x).
    Min subtract and return the result.
    Args:
        movie_chunk (torch.tensor). Shape (fov dim 1, fov dim 2, frames)
        mad_threshold_value (int): "th", as described above. The number of median absolute deviations in the threshold.
        dim (int): The axis over which operations are applied.
    Returns:
        Yd: A MAD thresholded version of the movie, where values that do not make it past the threhold are set to nan
    """
    movie_median = torch.median(movie_chunk, dim=dim, keepdim=True)[0]
    diff = torch.abs(movie_chunk - movie_median)
    mad_values = torch.median(diff, dim=dim, keepdim=True)[0]

    if sign == "unconstrained":
        condition = diff >= mad_values * mad_threshold_value
    elif sign == "positive":
        condition = (diff >= mad_values * mad_threshold_value) & (movie_chunk > movie_median)
    elif sign == "negative":
        condition = (diff >= mad_values * mad_threshold_value) & (movie_chunk < movie_median)
    else:
        raise ValueError("Invalid sign parameter")
    weight_matrix = torch.where(condition, 1.0, torch.nan)

    return movie_chunk * weight_matrix



def local_mad_correlation_mat(
        dim1_coordinates: torch.tensor,
        dim2_coordinates: torch.tensor,
        correlations: torch.tensor,
        dims: tuple[int, int, ...],
        order: str = "C",
) -> np.ndarray:
    """
    We MAD-threshold each pixel and compute correlations between neighboring pixels in the superpixel step
    Start with an index-index-value representation of the correlations (i.e. (3, 4, 0.2) means the corr between pixels 3, 4 is 0.2).
    The below function removes duplicates. It then computes the average correlation for each pixel "i" with all of its neighbors.

    Args:
        dim1_coordinates (torch.tensor): Shape (N). Pixel coordinates in the field of view
        dim2_coordinates (torch.tensor): Shape (N). Pixel coordinates in the field of view
        correlations (torch.tensor): Shape (N). Correlation values for dim1_coordinates[i], dim2_coordinates[i]
        dims (tuple): Integer specifying shape of data (field of view dim1, field of view dim2, and maybe frames).
        order (str): either "F" or "C" indicating how to reshape flattened data

    Returns:
        correlation_image (np.ndarray): Shape (fov_height, fov_width).
    """
    coordinate_pairs = torch.stack((dim1_coordinates, dim2_coordinates), dim=1)
    sorted_coordinates, _ = torch.sort(coordinate_pairs, dim=1)

    num_pixels_fov = np.prod(dims[:2])

    multiplicity_tracker = 100
    correlations = (
            correlations + multiplicity_tracker
    )  # Now every correlation is between 99 and 101

    correlations_mat = torch.sparse_coo_tensor(
        sorted_coordinates.T, correlations, (num_pixels_fov, num_pixels_fov)
    ).coalesce()

    rows, cols = correlations_mat.indices()
    correlation_sums = correlations_mat.values()
    multiplicity = torch.round(correlation_sums / multiplicity_tracker)
    correlation_sums = (
                               correlation_sums - multiplicity_tracker * multiplicity
                       ) / multiplicity

    # Now repeat this algorithm again
    correlation_sums = correlation_sums + multiplicity_tracker

    final_rows = torch.concatenate([rows, cols], dim=0)
    final_cols = torch.zeros_like(final_rows, device=final_rows.device)
    final_values = torch.concatenate([correlation_sums, correlation_sums])

    correlations_mat = torch.sparse_coo_tensor(
        torch.stack([final_rows, final_cols]), final_values, (num_pixels_fov, 1)
    ).coalesce()

    rows, _ = correlations_mat.indices()
    final_values = correlations_mat.values()
    multiplicity = torch.round(final_values / multiplicity_tracker)
    final_values = (final_values - multiplicity_tracker * multiplicity) / multiplicity

    dense_correlation_mat = torch.zeros(num_pixels_fov, device=rows.device)
    dense_correlation_mat[rows] = final_values

    return dense_correlation_mat.cpu().numpy().reshape((dims[0], dims[1]), order=order)


def fov_tiles(fov_height: int, fov_width: int, pixel_batch_size: int) -> Iterator[tuple[int, int, int, int]]:
    """
    (row_start, row_end, col_start, col_end) of square tiles of at most pixel_batch_size pixels covering the FOV.

    Consecutive tiles overlap by one row or column, so every pair of 8-connected neighbours lies inside some tile.
    """
    tilesize = max(2, math.floor(math.sqrt(pixel_batch_size)))  # a tile of 1 pixel would make the step zero
    for row_start in range(0, fov_height, tilesize - 1):
        for col_start in range(0, fov_width, tilesize - 1):
            yield row_start, min(row_start + tilesize, fov_height), col_start, min(col_start + tilesize, fov_width)


def neighbor_correlations(movie_tile: torch.Tensor,
                          pixel_ids: torch.Tensor,
                          noise_std: torch.Tensor,
                          mad_threshold: int,
                          sign: Literal["positive", "negative", "unconstrained"] = "unconstrained",
                          tol: float = 0.000001) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Robust correlations between every pair of 8-connected neighbours in one tile.

    Each trace is MAD-thresholded (mad_threshold 0 skips it), mean-subtracted and divided by
    sqrt(sum of squares + num_frames * noise_std ** 2), so a pixel whose variance is near the noise floor correlates
    weakly with everything.

    Parameters
    ----------
    movie_tile : torch.Tensor
        Shape (height, width, frames).
    pixel_ids : torch.Tensor
        Shape (height, width). Index of each pixel in the flattened FOV.
    noise_std : torch.Tensor
        Shape (height, width). The robust term per pixel.

    Returns
    -------
    point1, point2 : torch.Tensor
        Flat FOV indices of the two pixels of each pair.
    correlations : torch.Tensor
        One correlation per pair, NaN as 0.
    """
    num_frames = movie_tile.shape[2]
    if mad_threshold != 0:
        movie_tile = threshold_data_inplace(movie_tile, mad_threshold, sign=sign)

    movie_tile = movie_tile.permute(2, 0, 1)
    movie_tile -= torch.nanmean(movie_tile, dim=0, keepdim=True)
    divisor = torch.nansum(movie_tile * movie_tile, dim=0, keepdim=True) + num_frames * (noise_std ** 2)[None, :]
    divisor = torch.sqrt(divisor)
    divisor = torch.nan_to_num(divisor, nan=1.0)
    divisor[divisor < 0] = 1.0
    final_divisor = divisor.clone()

    # a divisor of 0 means a 0-mean pixel with 0 variance: the pixel is 0 everywhere, so its normalized trace is 0
    final_divisor[divisor < tol] = 1.0
    torch.reciprocal(final_divisor, out=final_divisor)
    final_divisor[divisor < tol] = 0.0
    torch.mul(movie_tile, final_divisor, out=movie_tile)

    pairs = (((slice(None, -1), slice(None)), (slice(1, None), slice(None))),  # vertical
             ((slice(None), slice(None, -1)), (slice(None), slice(1, None))),  # horizontal
             ((slice(None, -1), slice(None, -1)), (slice(1, None), slice(1, None))),  # top left to bottom right
             ((slice(1, None), slice(None, -1)), (slice(None, -1), slice(1, None))))  # bottom left to top right
    point1, point2, correlations = [], [], []
    for first, second in pairs:
        rho = torch.nansum(movie_tile[(slice(None),) + first] * movie_tile[(slice(None),) + second], dim=0)
        point1.append(pixel_ids[first].flatten())
        point2.append(pixel_ids[second].flatten())
        correlations.append(torch.nan_to_num(rho.flatten(), nan=0.0))
    return torch.cat(point1), torch.cat(point2), torch.cat(correlations)


def local_correlation_image(movie,
                            noise_std: torch.Tensor | None = None,
                            mad_threshold: int = 1,
                            pixel_batch_size: int = 10000,
                            sign: Literal["positive", "negative", "unconstrained"] = "unconstrained") -> torch.Tensor:
    """
    Each pixel's mean correlation with its eight neighbours after MAD thresholding, read from movie in pixel tiles.

    Parameters
    ----------
    movie : ArrayLike or np.ndarray
        Shape (frames, height, width); movie[:, rows, cols] gives a tile over every frame.
    noise_std : torch.Tensor, optional
        Shape (height, width). ``None`` uses the demixer's robust term: the 10th percentile of the pixel standard
        deviations, on every pixel.
    pixel_batch_size : int
        The most pixels expanded over every frame at once.

    Returns
    -------
    torch.Tensor
        Shape (height, width), on the cpu.
    """
    num_frames, fov_height, fov_width = movie.shape
    tiles = list(fov_tiles(fov_height, fov_width, pixel_batch_size))
    if noise_std is None:
        std = torch.zeros(fov_height, fov_width)
        for row_start, row_end, col_start, col_end in tiles:
            tile = torch.as_tensor(movie[:, row_start:row_end, col_start:col_end], dtype=torch.float32)
            std[row_start:row_end, col_start:col_end] = torch.std(tile, dim=0, correction=0).cpu()
        noise_std = torch.full((fov_height, fov_width), float(torch.quantile(std.flatten(), 0.1)))

    pixel_ids = torch.arange(fov_height * fov_width).reshape(fov_height, fov_width)
    point1, point2, correlations = [], [], []
    for row_start, row_end, col_start, col_end in tiles:
        tile = torch.as_tensor(movie[:, row_start:row_end, col_start:col_end], dtype=torch.float32).permute(1, 2, 0)
        out = neighbor_correlations(tile.contiguous(),
                                    pixel_ids[row_start:row_end, col_start:col_end].to(tile.device),
                                    noise_std[row_start:row_end, col_start:col_end].to(tile.device),
                                    mad_threshold,
                                    sign=sign)
        point1.append(out[0].cpu())
        point2.append(out[1].cpu())
        correlations.append(out[2].cpu())
    image = local_mad_correlation_mat(torch.cat(point1), torch.cat(point2), torch.cat(correlations),
                                      (fov_height, fov_width))
    return torch.from_numpy(image)
