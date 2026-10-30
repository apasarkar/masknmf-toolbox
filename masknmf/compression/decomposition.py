import torch
import psutil
import masknmf
from masknmf.compression.compression_array import CompressionArray
import logging
import math
import numpy as np
from collections.abc import Callable
from typing import Literal, NamedTuple
from masknmf.utils.display import progress

from masknmf import display
from masknmf.utils import torch_select_device, SparseCOOTensor, SparseCSRTensor
from masknmf.compression.preprocessing import SplineDetrend, SplineDetrenderBase

logger = logging.getLogger(__name__)


def decompress_csr_rows(crow: torch.Tensor, num_nonzero: int):
    indices = torch.arange(crow.numel() - 1, device=crow.device, dtype=crow.dtype)
    return torch.repeat_interleave(indices, crow.diff(), output_size=num_nonzero)

def truncated_random_svd(
        input_matrix: torch.Tensor,
        rank: int,
        num_oversamples: int = 5,
        device: str = "cpu"
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Assumptions:
    (1) input_matrix has been adequately mean-subtracted (so every column has mean 0, at least over the full dataset)
    (2) rank + num_oversamples is less than all dimensions of input_matrix.
    """
    num_frames = input_matrix.shape[1]
    input_matrix = input_matrix.to(device)
    random_data = torch.randn(num_frames, rank + num_oversamples, device=device)
    projected = input_matrix @ random_data
    q, r = torch.linalg.qr(projected, mode="reduced")
    b = q.T @ input_matrix
    u, s, v = torch.linalg.svd(b, full_matrices=False)
    u_final = q @ u
    v_final = v
    return u_final[:, :rank], s[:rank], v_final[:rank, :]


"""
The below functions are for spatial and temporal roughness penalties
"""


def temporal_roughness_statistic(temporal_traces: torch.Tensor) -> torch.Tensor:
    """
    Computes the temporal roughness statistics, batched over all the traces of interest
    Args:
        temporal_traces (torch.Tensor): shape (num_traces, num_frames).
    Returns:
        stats (torch.Tensor): shape (num_traces)
    """
    left_term = temporal_traces[:, :-2]
    right_term = temporal_traces[:, 2:]
    center_term = temporal_traces[:, 1:-1]

    numerator = torch.mean(torch.abs(left_term + right_term - 2 * center_term), dim=1)
    denominator = torch.mean(torch.abs(temporal_traces), dim=1)
    denominator[denominator == 0] = 1.0
    return numerator / denominator


def spatial_roughness_statistic(spatial_comps: torch.Tensor) -> torch.Tensor:
    """
    Computes spatial roughness statistic, batched over all spatial comps of interest
    Args:
        spatial_comps (torch.Tensor): shape (fov_height, fov_width, num_components)
    Returns:
        stats (torch.tensor): shape (num_components)
    """
    # Compute abs(vertical differences) :
    top_vertical = spatial_comps[:-1, :, :]
    bottom_vertical = spatial_comps[1:, :, :]
    vertical_diffs = torch.abs(top_vertical - bottom_vertical)

    # Compute abs(horizontal differences)
    left_horizontal = spatial_comps[:, :-1, :]
    right_horizontal = spatial_comps[:, 1:, :]
    horizontal_diffs = torch.abs(left_horizontal - right_horizontal)

    # Compute abs(top left --> bottom right differences)
    top_left = spatial_comps[:-1, :-1, :]
    bottom_right = spatial_comps[1:, 1:, :]
    top_bottom_diag_diffs = torch.abs(top_left - bottom_right)

    # Compute abs(bottomleft --> topright differences)
    top_right = spatial_comps[1:, 1:, :]
    bottom_left = spatial_comps[:-1, :-1, :]
    bottom_top_diag_diffs = torch.abs(top_right - bottom_left)

    total_terms = (
            torch.prod(torch.tensor(vertical_diffs.shape[:2]))
            + torch.prod(torch.tensor(horizontal_diffs.shape[:2]))
            + torch.prod(torch.tensor(top_bottom_diag_diffs.shape[:2]))
            + torch.prod(torch.tensor(bottom_top_diag_diffs.shape[:2]))
    )

    avg_diff = (
            torch.sum(vertical_diffs, dim=(0, 1))
            + torch.sum(horizontal_diffs, dim=(0, 1))
            + torch.sum(top_bottom_diag_diffs, dim=(0, 1))
            + torch.sum(top_bottom_diag_diffs, dim=(0, 1))
    )
    avg_diff /= total_terms

    return avg_diff / torch.mean(torch.abs(spatial_comps), dim=(0, 1))


def evaluate_fitness(
        spatial_comps: torch.Tensor,
        temporal_traces: torch.Tensor,
        spatial_statistic_threshold: float,
        temporal_statistic_threshold: float,
) -> torch.Tensor:
    """
    Args:
        spatial_comps (torch.Tensor): shape (fov_height, fov_width, num_components)
        temporal_traces (torch.Tensor): shape (num_components, num_frames)
        spatial_statistic_threshold (float): All accepted comps have a spatial roughness LESS than this threshold
        temporal_statistic_threshold (float): All accepted comps have a temporal roughness LESS than this threshold
    Returns:

    """
    evaluated_spatial_stats = spatial_roughness_statistic(spatial_comps)
    evaluated_temporal_stats = temporal_roughness_statistic(temporal_traces)

    spatial_decisions = evaluated_spatial_stats < spatial_statistic_threshold
    temporal_decisions = evaluated_temporal_stats < temporal_statistic_threshold

    return torch.logical_and(spatial_decisions, temporal_decisions)

def filter_by_failures(
        decisions: torch.Tensor,
        max_consecutive_failures: int
) -> torch.Tensor:
    """
    Filters decisions based on maximum consecutive failures.

    Args:
        decisions (torch.Tensor): 1-dimensional array of boolean values representing decisions.
        max_consecutive_failures (int): Maximum number of consecutive failures (ie decisions[i] == 0) allowed.

    Returns:
        np.ndarray: Filtered decisions with the same shape and type as input decisions.
    """

    false_tensor = (~decisions).to(dtype=torch.float32)
    kernel = torch.ones(
        max_consecutive_failures, device=false_tensor.device, dtype=torch.float32
    )[
             None, None, :
             ]  # Shape (1,1,n)
    seq = false_tensor.to(torch.float32)[None, None, :]

    # Convolve to find runs of n consecutive False values
    conv_result = torch.nn.functional.conv1d(
        seq, kernel, stride=1, padding=max_consecutive_failures - 1
    ).squeeze(0, 1)[: false_tensor.shape[0]]

    over_threshold = (conv_result >= max_consecutive_failures).to(torch.float32)
    keep_comps = (
            torch.cumsum(torch.cumsum(over_threshold, dim=0), dim=0) <= 1
    )  ##Two cumulative sums guarantee we pick the last element properly

    return keep_comps


def identify_window_chunks(
        frame_range: int,
        total_frames: int,
        window_chunks: int
) -> list:
    """
    Args:
        frame_range (int): Number of frames to fit
        total_frames (int): Total number of frames in the movie
        window_chunks (int): We sample continuous chunks of data throughout the movie.
            Each chunk is of size "window_chunks"

    Returns:
        (list): Contains the starting point of the intervals
            (each of length "window_chunk") on which we do the decomposition.

    Key requirements:
        (1) frame_range should be less than total number of frames
        (2) window_chunks should be less than or equal to frame_range
    """
    if frame_range > total_frames:
        raise ValueError("Requested more frames than available")
    if window_chunks > frame_range:
        raise ValueError("The size of each temporal chunk is bigger than frame range")

    num_intervals = math.ceil(frame_range / window_chunks)

    available_intervals = np.arange(0, total_frames, window_chunks)
    if available_intervals[-1] > total_frames - window_chunks:
        available_intervals[-1] = total_frames - window_chunks
    starting_points = np.random.choice(
        available_intervals, size=num_intervals, replace=False
    )
    starting_points = np.sort(starting_points)
    logger.debug("sampled from the following regions: {}".format(starting_points))

    net_frames = []
    for k in starting_points:
        curr_start = k
        curr_end = min(k + window_chunks, total_frames)

        curr_frame_list = [i for i in range(curr_start, curr_end)]
        net_frames.extend(curr_frame_list)
    return net_frames


def check_fov_size(fov_dims: tuple[int, int], min_allowed_value: int = 10) -> None:
    """
    Checks if the field of view (FOV) dimensions are too small.

    Args:
        fov_dims (tuple): Two integers specifying the FOV dimensions.
        min_allowed_value (int): The minimum allowed value for FOV dimensions. Defaults to 10.

    Returns:
        None

    Raises:
        ValueError: If either field of view dimension is less than the minimum allowed value.
    """
    for k in fov_dims:
        if k < min_allowed_value:
            raise ValueError(
                "At least one FOV dimension is lower than {}, "
                "too small to process".format(min_allowed_value)
            )


def update_block_sizes(
        blocks: tuple,
        fov_shape: tuple,
        min_block_value: int = 4,
) -> list:
    """
    If user specifies block sizes that are too large, this approach truncates the blocksizes appropriately

    Args:
        blocks (tuple): Two integers, specifying the height and width blocksizes used in compression
        fov_shape (tuple): The height and width of the FOV
        min_block_value (int): The minimum value of a block in either spatial dimension.

    Returns:
        list: A list containing the updated block sizes

    Raises:
        ValueError if either block dimension is less than min allowed value.
    """
    if blocks[0] < min_block_value or blocks[1] < min_block_value:
        raise ValueError(
            "One of the block dimensions was less than min allowed value of {}, "
            "set to a larger value".format(min_block_value)
        )
    final_blocks = []
    if blocks[0] > fov_shape[0]:
        display(
            "Height blocksize was set to {} but corresponding dimension has size {}. Truncating to {}".format(
                blocks[0], fov_shape[0], fov_shape[0]
            )
        )
        final_blocks.append(fov_shape[0])
    else:
        final_blocks.append(blocks[0])
    if blocks[1] > fov_shape[1]:
        display(
            "Height blocksize was set to {} but corresponding dimension has size {}. Truncating to {}".format(
                blocks[1], fov_shape[1], fov_shape[1]
            )
        )
        final_blocks.append(fov_shape[1])
    else:
        final_blocks.append(blocks[1])
    return final_blocks


def compute_mean_and_normalizer_dataset(
        dataset: masknmf.ArrayLike | masknmf.LazyFrameLoader,
        compute_normalizer: bool,
        pixel_batch_size: int,
        device: str,
        dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Computes a pixelwise mean and a noise variance estimate. For now, the noise var estimate is turned off
    Args:
        dataset (masknmf.ArrayLike | masknmf.LazyFrameLoader): The dataloader object we use to access the dataset. Anything that supports
            numpy-like __getitem__ indexing can be used here.
        compute_normalizer (bool): Whether or not we compute the noise normalizer; for now this variable has no effect.
        pixel_batch_size (int): The number of pixels in each dimension to load at a time
        dtype (torch.dtype): The dtype of the data once it has been moved to accelerator.
        device (str): The
    Returns:
        mean_image (torch.Tensor): The (fov_height, fov_width) shaped mean image.
        noise_variance_image (torch.Tensor): The (fov_height, fov_width) noise variance image.
    """
    num_frames, fov_height, fov_width = dataset.shape

    height_iters = math.ceil(fov_height / pixel_batch_size)
    width_iters = math.ceil(fov_width / pixel_batch_size)

    noise_normalizer = torch.zeros((fov_height, fov_width), dtype=dtype, device=device)
    mean_value = torch.zeros((fov_height, fov_width), dtype=dtype, device=device)

    for i in range(height_iters):
        for j in range(width_iters):
            start_height = pixel_batch_size * i
            end_height = min(fov_height, start_height + pixel_batch_size)
            start_width = pixel_batch_size * j
            end_width = min(fov_width, start_width + pixel_batch_size)

            data_subset = torch.from_numpy(dataset[:, start_height:end_height, start_width:end_width]).to(device).to(dtype)
            mean_value[start_height:end_height, start_width:end_width] = torch.mean(data_subset, dim=0)
            if compute_normalizer:
                data_subset -= mean_value[start_height:end_height, start_width:end_width][None, :, :]
                noise_term = ((torch.sum(data_subset**2, dim=0)/num_frames) -
                              (torch.sum(data_subset[1:, :, :]*data_subset[:-1, :, :], dim=0)) / (num_frames-1))
                noise_normalizer[start_height:end_height, start_width:end_width] = torch.sqrt(noise_term)

    noise_normalizer[noise_normalizer <= 0] = 1.0
    return mean_value.cpu(), noise_normalizer.cpu()


def compute_full_fov_spatial_basis(
        dataset: masknmf.ArrayLike | masknmf.LazyFrameLoader,
        mean_img: torch.Tensor,
        noise_variance_img: torch.Tensor,
        background_rank: int,
        dtype: torch.dtype = torch.float32,
        device: str = "cpu",
        num_samples: int = 1000,
):
    """
    Routine for approximating a full FOV low-rank spatial basis; useful for estimating full FOV trends

    Args:
        dataset (masknmf.ArrayLike | masknmf.LazyFrameLoader): Input dataset
        mean_img (torch.Tensor): Shape (fov_height, fov_width). Mean image of the data
        noise_variance_img (torch.Tensor): Shape (fov_height, fov_width). The noise variance estimate at each pixel
        background_rank (int): The rank of the background term we are trying to estimate
        frame_batch_size (int): The number of frames we can load into memory at a time
        dtype (torch.dtype): The dtype of data used in the actual computations
        num_oversamples (int): The number of oversamples for the subspace estimation method
        device (str): Which device the computations are performed on
        num_samples (int): The number of frames of the dataset that we randomly sample to estimate the full FOV
            subspace.

    Returns:
        spatial_basis (torch.Tensor). Shape (fov_height, fov_width, net_rank). The orthonormal spatial basis vectors
    """
    num_frames, fov_height, fov_width = dataset.shape
    if background_rank <= 0:
        return torch.zeros((fov_height, fov_width, 1)).to(dtype)
    sample_list = [i for i in range(0, num_frames)]
    random_data = np.random.choice(
        sample_list, replace=False, size=min(num_samples, num_frames)
    )

    mean_img = mean_img.to(device).to(dtype)
    noise_variance_img = noise_variance_img.to(device).to(dtype)
    my_data = torch.tensor(dataset[random_data]).to(device).to(dtype)
    my_data = (my_data - mean_img[None, :, :]) / noise_variance_img[None, :, :]
    my_data = my_data.permute(1, 2, 0).reshape((fov_height * fov_width, -1))

    my_data = my_data.to(device).to(dtype)
    spatial_basis, _, _ = truncated_random_svd(my_data, background_rank, device=device)

    return spatial_basis.cpu().reshape((fov_height, fov_width, -1))


def compute_full_fov_temporal_basis(
        dataset: torch.Tensor,
        mean_img: torch.Tensor,
        noise_variance_img: torch.Tensor,
        full_fov_spatial_basis: torch.Tensor,
        dtype: torch.dtype,
        frame_batch_size: int,
        device: str = "cpu",
        temporal_denoiser: torch.nn.Module | None = None
) -> torch.Tensor:
    """
    Regress some portion of the data onto the spatial basis.
    Args:
        dataset (torch.Tensor). A dataset of shape (frames, fov_height, fov_width)
        mean_img (torch.Tensor). The mean image of the data. Shape (fov_height, fov width)
        noise_variance_img (torch.Tensor). The noise variance image of the data. Shape (fov_height, fov_width)
        full_fov_spatial_basis (torch.Tensor). A full FOV spatial basis for the data (Shape (fov_height, fov_width, rank).
        dtype (torch.dtype): The dtype on which we do computations. Should be torch.float32.
        frame_batch_size (int): The max number of frames we want to load onto GPU at a time.
        device (str): Either "cpu" or "cuda". Specifies whether we can do computations on GPU or not.
        temporal_denoiser (torch.nn.Module | None): A function which denoises batches of time series. Input is a tensor of shape
            (batch_size, num_timesteps), output is same.
    Returns:
        spatial_basis (torch.Tensor). Shape (num_pixels, rank). The orthogonal spatial basis
        temporal_basis (torch.Tensor). Shape (rank, num_frames). Projection of standardized data onto spatial basis.
    """
    num_frames, fov_height, fov_width = dataset.shape
    mean_img = mean_img.to(device).to(dtype)
    noise_variance_img = noise_variance_img.to(device).to(dtype)
    num_iters = math.ceil(dataset.shape[0] / frame_batch_size)
    final_results = []

    mean_img_r = mean_img.reshape((fov_height * fov_width, 1)).T
    noise_variance_img_r = noise_variance_img.reshape((fov_height * fov_width, 1))
    spatial_basis_r = (
        full_fov_spatial_basis.to(device).to(dtype).reshape((fov_height * fov_width, -1))
    )
    spatial_basis_r_weighted_by_variance = spatial_basis_r * torch.reciprocal(
        noise_variance_img_r
    )
    for k in range(num_iters):
        if frame_batch_size >= dataset.shape[0]:
            curr_dataset = dataset.to(device).to(dtype)
        else:
            start_pt = k * frame_batch_size
            end_pt = min(dataset.shape[0], start_pt + frame_batch_size)
            curr_dataset = dataset[start_pt:end_pt].to(device).to(dtype)
        curr_dataset_r = curr_dataset.reshape((-1, fov_height * fov_width))
        projection = (
                curr_dataset_r @ spatial_basis_r_weighted_by_variance
                - mean_img_r @ spatial_basis_r_weighted_by_variance
        )
        final_results.append(projection)

    temporal_basis = torch.concatenate(final_results, dim=0).T  #Shape (rank, num_timesteps)

    if temporal_denoiser is not None:
        temporal_basis = temporal_denoiser(temporal_basis)
        temporal_basis = _temporal_basis_pca(temporal_basis, explained_var_cutoff=.99)
        rank = temporal_basis.shape[0]

        full_fov_spatial_basis = torch.zeros((rank, fov_height * fov_width), device=device)
        temporal_sum = temporal_basis @ torch.ones((temporal_basis.shape[1], 1), device=device,
                                                   dtype=temporal_basis.dtype)
        full_fov_spatial_basis -= temporal_sum @ mean_img_r
        for k in range(num_iters):
            if frame_batch_size >= dataset.shape[0]:
                curr_dataset = dataset.to(device).to(dtype)
                start_pt = 0
                end_pt = dataset.shape[0]
            else:
                start_pt = k * frame_batch_size
                end_pt = min(dataset.shape[0], start_pt + frame_batch_size)
                curr_dataset = dataset[start_pt:end_pt].to(device).to(dtype)
            curr_dataset_r = curr_dataset.reshape((-1, fov_height * fov_width))
            full_fov_spatial_basis += temporal_basis[:, start_pt:end_pt] @ curr_dataset_r
        full_fov_spatial_basis *= torch.reciprocal(noise_variance_img_r.T)
        left_sing, sing, right_sing = torch.linalg.svd(full_fov_spatial_basis, full_matrices=False)
        temporal_basis = temporal_basis.T @ (left_sing * sing[None, :])
        return right_sing.T.reshape(fov_height, fov_width, -1), temporal_basis.T
    else:
        return full_fov_spatial_basis, temporal_basis


def compute_factorized_svd_with_leftbasis(
        p: SparseCOOTensor, v: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Use case: you have a factorized movie, UPV where spatial_compressed is sparse (and you don't want to change that), and
    UP has orthonormal columns. This function reformats the factorization into UPV = (UR)sV_{new} where (UR) are left
    singular vecotrs, s describes singular values, V_new describes right singular vectors.

    Args:
        spatial_compressed (SparseCOOTensor): shape (pixels, rank)
        p (torch.Tensor): shape (rank, rank)
        v: (torch.Tensor): shape (rank, num_frames)
    """
    q, m = [i.float().T for i in torch.linalg.qr(v.T, mode="reduced")]  # Now temporal_compressed = mq

    # Note that (upm)^T(upm) = (m^T)m
    mtm = m.T @ m
    eig_vecs, eig_vals, _ = [
        i.float() for i in torch.linalg.svd(mtm, full_matrices=True)
    ]

    logger.debug(f"mtm symmetric: {torch.allclose(mtm, mtm.T)}")
    logger.debug(
        f"When we ran the  leftbasis eigh routine, the smallest value we saw was {np.amin(eig_vals.cpu().numpy())}"
    )

    logger.debug(
        f"When we ran the eigh routine, the smallest value we saw was {np.amin(eig_vals.cpu().numpy())}"
    )
    good_components = eig_vals > 0
    eig_vecs = eig_vecs[:, good_components]
    eig_vals = eig_vals[good_components]

    s = torch.sqrt(eig_vals)

    r = p @ (eig_vecs / s[None, :])
    v = eig_vecs.T @ q
    return r, s, v


def compute_lowrank_factorized_svd(
        spatial_compressed: SparseCOOTensor,
        temporal_compressed: torch.Tensor,
):
    """
    Compute the factorized Singular Value Decomposition (SVD) of a low-rank matrix factorization.

    This function computes the SVD of a matrix `spatial_compressed @ temporal_compressed`, where `spatial_compressed` is sparse and `temporal_compressed` is dense,
    both representing a low-rank factorization. It efficiently computes a reduced or partial SVD
    based on this factorization. The function allows returning just the left singular vectors
    (spatial mixing matrix) if specified.

    Args:
        spatial_compressed (torch.sparse_coo_tensor):
            Sparse left matrix of the factorization with shape `(pixels, low_rank)`.
        temporal_compressed (torch.Tensor):
            Dense right matrix of the factorization with shape `(low_rank, frames)`.

    Returns:
        torch.Tensor:
            `spatial_mixing_matrix`: An orthonormal column basis for the factorization `spatial_compressed @ temporal_compressed`.
            This matrix represents the spatial components of the original data.
        torch.Tensor:
            `singular_values`: 1D vector of singular values, representing the scaling factors
            for the corresponding orthonormal directions.
        torch.Tensor:
            `right_singular_vectors`: Orthonormal column vectors representing the temporal
            components of the matrix `temporal_compressed`.

    Notes:
        - This is not a full SVD; the result is truncated to preserve efficiency, especially
        for large matrices. The orthogonality of the left singular vectors holds within the
        reduced space of the factorization.
        - This routine uses eigh on the spatial basis to exploit the low rank of the decomposition. This will lead to bad results if the matrix is ill-conditioned.
        compression gives us a reasonable guarantee that this is not true (due to the blockwise decompositions).
    """
    q, p = [
        i.float().T for i in torch.linalg.qr(temporal_compressed.T, mode="reduced")
    ]  # Here, temporal_compressed = pq, with q having orth rows
    ut_u = torch.sparse.mm(spatial_compressed.T, spatial_compressed).to_dense()

    ptut_up = (p.T @ ut_u) @ p

    eig_vals, eig_vecs = [i.float() for i in torch.linalg.eigh(ptut_up.double())]

    good_components = eig_vals > 0
    eig_vals = eig_vals[good_components]
    eig_vecs = eig_vecs[:, good_components]

    eig_vals = torch.flip(eig_vals, dims=[0])
    eig_vecs = torch.flip(eig_vecs, dims=[1])

    s = torch.sqrt(eig_vals)
    r = p @ (eig_vecs / s[None, :])
    new_v = eig_vecs.T @ q

    return r, s, new_v


def regress_onto_spatial_basis(
        dataset: masknmf.ArrayLike | masknmf.LazyFrameLoader,
        spatial_compressed: SparseCOOTensor,
        frame_batch_size: int,
        dataset_mean: torch.Tensor,
        dataset_noise_variance: torch.Tensor,
        dtype: torch.dtype,
        device: str = "cpu",
) -> torch.Tensor:
    """
    We have a spatial basis from blockwise decompositions. This function will do two things, in a single pass through the
    data:
        (1) It will project the data onto each block's orthogonal basis (this is NOT equivalent to a linear subspace projection onto the
        spatial basis!)
        (2) It will perform a linear subspace projection of the centered+standardized data onto the full FOV data

    The computation to perform here is:
    temporal_compressed = spatial_compressed^T (I_{norms} * (Data - Mean) - Spatial_Full_FOV_Bkgd * Temporal_Full_FOV_Bkgd)
    Here, I_{norms} is a diagonal matrix containing the reciprocal of the dataset_noise_variance. The term
    I_{norms}(Data - Mean) does pixelwise centering + standardization of the data.
    In the below routine, we exploit the low rank of spatial_compressed and conduct operations in an order that minimizes data size/number of computations.

    Args:
        dataset (masknmf.ArrayLike | masknmf.LazyFrameLoader): Any array-like object that supports __getitem__ for fast frame retrieval.
        spatial_compressed (torch.sparse_coo_tensor): The spatial basis, where components from the same block are orthonormal.
        frame_batch_size (int): The number of frames we load at any point in time
        dataset_mean (torch.Tensor): Shape (fov_height, fov_width). The mean across all pixels
        dataset_noise_variance (torch.Tensor): Shape (fov_height, fov_width). The noise variance across all pixels.
        dtype (torch.dtype): The dtype to which we convert the data for processing; should be torch.float32, or float64.
        device (str): The platform on which processing occurs ("cuda" or "cpu")
    """
    num_frames, fov_height, fov_width = dataset.shape
    num_iters = math.ceil(dataset.shape[0] / frame_batch_size)
    dataset_mean = dataset_mean.to(device).to(dtype).reshape((fov_height * fov_width, 1))
    dataset_noise_variance = (
        dataset_noise_variance.to(device).to(dtype).reshape((fov_height * fov_width, 1))
    )

    u_t = spatial_compressed.T.coalesce()

    row_indices, col_indices = u_t.indices()
    ut_values = u_t.values()

    new_values = ut_values / dataset_noise_variance[col_indices].squeeze()

    u_t_normalized = torch.sparse_coo_tensor(
        u_t.indices(), new_values, u_t.shape
    ).coalesce()

    mean_projected = torch.sparse.mm(u_t_normalized, dataset_mean)

    temporal_results = []
    for k in progress(range(num_iters), desc="projecting frames"):
        start_pt = k * frame_batch_size
        end_pt = min(start_pt + frame_batch_size, num_frames)
        curr_data = dataset[start_pt:end_pt]
        if isinstance(curr_data, np.ndarray):
            curr_data = torch.from_numpy(curr_data).to(device).to(dtype).permute(1, 2, 0).reshape((fov_height*fov_width,-1))
        elif isinstance(curr_data, torch.Tensor):
            curr_data = curr_data.to(device).to(dtype).permute(1, 2, 0).reshape((fov_height*fov_width,-1))
        else:
            raise ValueError(f"Dataset returns data of type {type(curr_data)}. Only valid return types are np.ndarray and torch.tensor")

        projection = torch.sparse.mm(u_t_normalized, curr_data)
        projection -= mean_projected


        ## Add the full fov and blockwise temporal components that we estimate above to a list to concatenate later
        temporal_results.append(projection)
    return torch.concatenate(temporal_results, dim=1)


def temporal_downsample(tensor: torch.Tensor,
                        temporal_avg_factor: int) -> torch.Tensor:
    """
    Temporally downsamples a (fov_height, fov_width, num_frames) tensor using avg_pool1d.

    Args:
        tensor (torch.Tensor): Input tensor of shape (num_frames, fov_height, fov_width).
        temporal_avg_factor (int): Downsampling factor (number of frames per block).

    Returns:
        Downsampled tensor of shape (height, width, ceil(num_frames / n)).
    """
    height, width, num_frames = tensor.shape
    tensor = tensor.reshape(height * width, num_frames).unsqueeze(1)
    downsampled = torch.nn.functional.avg_pool1d(
        tensor,
        kernel_size=temporal_avg_factor,
        stride=temporal_avg_factor,
        ceil_mode=True,
    )

    # Reshape back to (num_frames // n, height, width)
    return downsampled.squeeze().reshape(height, width, -1)


def downsample_sparse(sparse_tensor: SparseCSRTensor,
                      fov_dims: tuple[int, int],
                      downsample_factor: int):
    """
    Given a 2D sparse tensor, describing a (fov_height, fov_width, num_frames) array, this routine performs spatial downsampling
    (averaging). Assumes the (fov_height, fov_width) is vectorized in row-major order
    Args:
        sparse_tensor (torch.sparse_coo_tensor): Shape (height*width, columns)
    Returns:
        torch.sparse_coo_tensor: Shape ((height//factor) * (width//factor), columns)
    """
    # if not sparse_tensor.is_coalesced():
    #     sparse_tensor = sparse_tensor.coalesce()

    divisor = downsample_factor ** 2
    height, width = fov_dims[0], fov_dims[1]
    rows, cols = decompress_csr_rows(sparse_tensor.crow_indices(), sparse_tensor.values().numel()), sparse_tensor.col_indices()
    sparse_values = sparse_tensor.values()
    h_vals = rows // width # torch.floor(rows / width)
    w_vals = rows % width # - h_vals * width

    new_height, new_width = height // downsample_factor, width // downsample_factor
    h_vals = h_vals // downsample_factor
    w_vals = w_vals // downsample_factor
    good_indices = (h_vals < new_height).long()
    good_indices *= (w_vals < new_width).long()
    new_rows = h_vals * new_width + w_vals

    new_rows *= good_indices

    final_indices = torch.stack([new_rows, cols], dim=0)
    downsampled_sparse_tensor = torch.sparse_coo_tensor(
        final_indices,
        (sparse_values * good_indices) / divisor,
        (new_height * new_width, sparse_tensor.shape[1]),
    ).coalesce().to_sparse_csr()

    return downsampled_sparse_tensor

def spatial_downsample(
        image_stack: torch.Tensor,
        spatial_avg_factor: int
) -> torch.Tensor:
    """
    Downsamples a (fov_height, fov_width, num_frames) image stack via n x n binning.

    Args:
        image_stack: Tensor of shape (fov_height, fov_width, num_frames).

    Returns:
        Downsampled tensor of shape (H//factor, W//factor, T).
    """

    image_stack = image_stack.permute(2, 0, 1).unsqueeze(
        1
    )  # (num_frames, 1, height, width)
    downsampled = torch.nn.functional.avg_pool2d(
        image_stack, kernel_size=spatial_avg_factor, stride=spatial_avg_factor
    )  # (T, 1, H//2, W//2)
    return downsampled.squeeze(1).permute(1, 2, 0)


def _temporal_basis_pca(temporal_basis: torch.Tensor,
                        explained_var_cutoff: float = 0.99) -> torch.Tensor:
    """
    Keeps the top "k" PCA components of temporal_basis that explain >= explained_var_cutoff of the variance
    Args:
        temporal_basis (torch.Tensor): Shape (number_of_timeseries, num_timesteps).
        explained_var_cutoff (float): Float between 0 and 1, describes how much of the net variance these comps
            should explain. Default value of 0.99 makes sense when you are processing data that has been smoothed
            already via some denoising algorithm - this should shift the signal subspace to the top of the spectrum.
    Returns:
        truncated_temporal_basis (torch.Tensor): Shape (truncated_num_of_timeseries, num_timesteps). Truncated basis
    """
    temporal_basis_mean = torch.mean(temporal_basis, dim=1, keepdim=True)
    temporal_basis_meansub = temporal_basis - temporal_basis_mean
    _, sing, right_vec = torch.linalg.svd(temporal_basis_meansub, full_matrices=False)
    explained_ratios = torch.cumsum(sing ** 2, dim=0) / torch.sum(sing ** 2, dim=0)

    ind = torch.nonzero(explained_ratios >= explained_var_cutoff, as_tuple=True)
    if ind[0].numel() == 0:
        ind = temporal_basis.shape[0]
    else:
        ind = ind[0][0].item() + 1
    new_basis = right_vec[:ind, :]
    return new_basis


def blockwise_decomposition(
        video_subset: torch.Tensor,
        subset_pixel_weighting: torch.Tensor,
        subset_frame_weighting: torch.Tensor,
        max_components: int,
        spatial_avg_factor: int,
        temporal_avg_factor: int,
        dtype: torch.dtype,
        spatial_denoiser: Callable | None = None,
        temporal_denoiser: Callable | None = None,
        device: str = "cpu",
        subset_mean: torch.Tensor | None = None,
        subset_noise_std: torch.Tensor | None = None
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

    first_spatial, first_temporal, subset_mean, subset_noise_std = blockwise_decomposition_singlepass(video_subset,
                                                                       subset_pixel_weighting,
                                                                       subset_frame_weighting,
                                                                       max_components,
                                                                       spatial_avg_factor,
                                                                       temporal_avg_factor,
                                                                       dtype,
                                                                       spatial_denoiser=spatial_denoiser,
                                                                       temporal_denoiser=temporal_denoiser,
                                                                       device=device,
                                                                       subset_mean=subset_mean,
                                                                       subset_noise_std=subset_noise_std)

    return first_spatial, first_temporal, subset_mean, subset_noise_std
    # Below is an experimental rank-1 deflation procedure
    #
    # else:
    #     #Rank 1 deflation routine
    #     final_spatial = []
    #     final_temporal = []
    #     num_frames, h, w = video_subset.shape
    #     empty_noise = torch.ones_like(subset_noise_variance)
    #     empty_mean = torch.zeros_like(subset_noise_variance)
    #     for k in range(max_components):
    #
    #         curr_spatial, curr_temporal = blockwise_decomposition_singlepass(video_subset,
    #                                                                 subset_mean,
    #                                                                 subset_noise_variance,
    #                                                                 subset_pixel_weighting,
    #                                                                 1,
    #                                                                 spatial_avg_factor,
    #                                                                 temporal_avg_factor,
    #                                                                 dtype,
    #                                                                 spatial_denoiser=spatial_denoiser,
    #                                                                 temporal_denoiser=temporal_denoiser,
    #                                                                 device=device)
    #         video_subset = video_subset.to(device)
    #         video_subset -= subset_mean.to(device).to(dtype)[None, :, :]
    #         video_subset /= subset_noise_variance.to(device).to(dtype)[None, :, :]
    #         video_subset -= (curr_spatial @ curr_temporal).permute(2, 0, 1)
    #         subset_noise_variance = empty_noise
    #         subset_mean = empty_mean
    #         # return first_spatial, first_temporal
    #         final_spatial.append(curr_spatial)
    #         final_temporal.append(curr_temporal)
    #
    #     final_spatial = torch.concatenate(final_spatial, dim=2)
    #     final_temporal = torch.concatenate(final_temporal, dim = 0)
    #     final_spatial = final_spatial.reshape(-1, final_spatial.shape[2])
    #     q, r = torch.linalg.qr(final_spatial, mode='reduced')
    #
    #     final_spatial = q.reshape(h, w, -1)
    #     final_temporal = r @ final_temporal
    #     return final_spatial, final_temporal

def blockwise_decomposition_singlepass(
        video_subset: torch.Tensor,
        subset_pixel_weighting: torch.Tensor,
        subset_frame_weighting: torch.Tensor,
        max_components: int,
        spatial_avg_factor: int,
        temporal_avg_factor: int,
        dtype: torch.dtype,
        spatial_denoiser: Callable | None= None,
        temporal_denoiser: Callable | None = None,
        device: str = "cpu",
        subset_mean: torch.Tensor | None= None,
        subset_noise_std: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

    num_frames, fov_height, fov_width = video_subset.shape
    empty_values = torch.zeros((fov_height, fov_width, 1), device=device, dtype=dtype), torch.zeros((1, num_frames),
                                                                                                 device=device,
                                                                                                 dtype=dtype)

    subset = video_subset.to(device).to(dtype)
    if subset_mean is None:
        subset_mean = subset.mean(dim = 0)
    subset = subset - subset_mean.to(device).to(dtype)[None, :, :]

    if subset_noise_std is None:
        subset_noise_std = ((torch.sum(subset ** 2, dim=0) / num_frames) -
                      (torch.sum(subset[1:, :, :] * subset[:-1, :, :], dim=0)) / (num_frames - 1))
        subset_noise_std[subset_noise_std < 1e-8] = 0.0 #Avoid division errors
        subset_noise_std = torch.sqrt(subset_noise_std)
        subset_noise_std[subset_noise_std < 1e-8] = 1.0

    subset /= subset_noise_std.to(device).to(dtype)[None, :, :]

    subset = subset.permute(1, 2, 0)
    subset_weighted = subset * subset_pixel_weighting.to(device).to(dtype)[:, :, None]
    subset_weighted *= subset_frame_weighting[None, None, :]

    if spatial_avg_factor != 1:
        spatial_pooled_subset = spatial_downsample(subset_weighted, spatial_avg_factor)
    else:
        spatial_pooled_subset = subset_weighted

    spatial_pooled_subset_r = spatial_pooled_subset.reshape(
        (-1, spatial_pooled_subset.shape[2])
    )

    if temporal_avg_factor != 1:
        spatiotemporal_pooled_subset = temporal_downsample(
            spatial_pooled_subset, temporal_avg_factor
        )
        spatiotemporal_pooled_subset -= torch.mean(spatiotemporal_pooled_subset, dim=2, keepdim=True)
    else:
        spatiotemporal_pooled_subset = spatial_pooled_subset

    spatiotemporal_pooled_subset_r = spatiotemporal_pooled_subset.reshape(
        (-1, spatiotemporal_pooled_subset.shape[2])
    )

    lowres_spatial_basis_r, interm_sing_values, _ = truncated_random_svd(
        spatiotemporal_pooled_subset_r, max_components, device=device
    )
    if torch.count_nonzero(interm_sing_values) == 0:
        return empty_values[0], empty_values[1], subset_mean, subset_noise_std

    temporal_projection_from_downsample = (
            lowres_spatial_basis_r.T @ spatial_pooled_subset_r
    )
    if temporal_denoiser is not None:
        temporal_projection_from_downsample = temporal_denoiser(
            temporal_projection_from_downsample
        )
    std_values = torch.std(temporal_projection_from_downsample, dim=1)
    if torch.count_nonzero(std_values) == 0:
        return empty_values[0], empty_values[1], subset_mean, subset_noise_std
    else:
        temporal_projection_from_downsample = temporal_projection_from_downsample[std_values != 0]

    if temporal_denoiser is None or temporal_projection_from_downsample.shape[0] <= 1:
        temporal_basis_from_downsample = _temporal_basis_pca(temporal_projection_from_downsample,
                                                             explained_var_cutoff=1.0)
    else:
        temporal_basis_from_downsample = _temporal_basis_pca(temporal_projection_from_downsample,
                                                             explained_var_cutoff=0.99)

    subset_weighted_r = subset_weighted.reshape((-1, subset_weighted.shape[2]))
    spatial_basis_fullres = subset_weighted_r @ temporal_basis_from_downsample.T

    if spatial_denoiser is not None:
        spatial_basis_fullres = spatial_basis_fullres.reshape((fov_height, fov_width, -1))
        spatial_basis_fullres = spatial_denoiser(spatial_basis_fullres)
        spatial_basis_fullres = spatial_basis_fullres.reshape((fov_height * fov_width, -1))

    spatial_basis_orthogonal, interm_sing_vals, _ = torch.linalg.svd(
        spatial_basis_fullres, full_matrices=False
    )
    if torch.count_nonzero(interm_sing_vals) == 0:
        return empty_values[0], empty_values[1], subset_mean, subset_noise_std
    else:
        spatial_basis_orthogonal = spatial_basis_orthogonal[:, interm_sing_vals != 0]

    # Regress the original (unweighted) data onto this basis
    subset_r = subset.reshape((-1, subset.shape[2]))
    if temporal_denoiser is None:
        #In this case, we want to take advantage of the existing
        final_temporal_projection = (spatial_basis_orthogonal.T @ (subset_r @ temporal_basis_from_downsample.T)) @ temporal_basis_from_downsample
    else:
        final_temporal_projection = spatial_basis_orthogonal.T @ subset_r #In this case we want to project onto this basis and re-apply the denoiser
    left, sing, right = torch.linalg.svd(final_temporal_projection, full_matrices=False)
    if torch.count_nonzero(sing) == 0:
        return empty_values[0], empty_values[1], subset_mean, subset_noise_std
    else:
        indices_to_keep = sing != 0
        left = left[:, indices_to_keep]
        right = right[indices_to_keep, :]
        sing = sing[indices_to_keep]
    local_spatial_basis = (spatial_basis_orthogonal @ left).reshape(
        (fov_height, fov_width, -1)
    )
    local_temporal_basis = sing[:, None] * right

    if temporal_denoiser is not None:
        local_temporal_basis = temporal_denoiser(local_temporal_basis)
        if torch.count_nonzero(local_temporal_basis) == 0:
            return empty_values[0], empty_values[1], subset_mean, subset_noise_std
        else:
            local_temporal_basis -= torch.mean(local_temporal_basis, dim=1, keepdims=True)

    return local_spatial_basis, local_temporal_basis, subset_mean, subset_noise_std

def residual_std_calculation(spatial_decomposition: torch.Tensor,
                             temporal_decomposition: torch.Tensor,
                             data_block: torch.Tensor):
    output = spatial_decomposition @ temporal_decomposition # shape (block dim1, block dim2, frames)
    resid = data_block - output.permute(2, 0, 1)
    return torch.std(resid, dim = 0)

def blockwise_decomposition_with_rank_selection(
        video_subset: torch.Tensor,
        subset_pixel_weighting: torch.Tensor,
        subset_frame_weighting: torch.Tensor,
        max_components: int,
        max_consecutive_failures: int,
        spatial_roughness_threshold: float,
        temporal_roughness_threshold: float,
        spatial_avg_factor: int,
        temporal_avg_factor: int,
        dtype: torch.dtype,
        spatial_denoiser: torch.nn.Module | None = None,
        temporal_denoiser: torch.nn.Module | None = None,
        device: str = "cpu",
        subset_mean: torch.Tensor | None = None,
        subset_noise_std: torch.Tensor = None
):
    local_spatial_basis, local_temporal_basis, subset_mean, subset_noise_std = blockwise_decomposition(
        video_subset,
        subset_pixel_weighting,
        subset_frame_weighting,
        max_components,
        spatial_avg_factor,
        temporal_avg_factor,
        dtype,
        spatial_denoiser=spatial_denoiser,
        temporal_denoiser=temporal_denoiser,
        device=device,
        subset_mean=subset_mean,
        subset_noise_std=subset_noise_std
    )

    decisions = evaluate_fitness(
        local_spatial_basis,
        local_temporal_basis,
        spatial_roughness_threshold,
        temporal_roughness_threshold,
    )

    decisions = filter_by_failures(decisions, max_consecutive_failures)


    ## Also eliminate bad components
    std_dev = local_temporal_basis.std(dim=1)
    nonzero_temporal = std_dev > 1e-6

    std_dev = local_spatial_basis.std(dim=(0,1))
    nonzero_spatial = std_dev > 1e-6

    nonzero_comps = torch.logical_and(nonzero_temporal, nonzero_spatial)
    decisions = torch.logical_and(decisions, nonzero_comps)
    return local_spatial_basis[:, :, decisions], local_temporal_basis[decisions, :], subset_mean, subset_noise_std


def threshold_heuristic(
        dimensions: tuple[int, int, int],
        spatial_avg_factor: int,
        temporal_avg_factor: int,
        spatial_denoiser: torch.nn.Module | None,
        temporal_denoiser: torch.nn.Module | None,
        dtype: torch.dtype,
        num_comps: int = 1,
        iters: int = 250,
        percentile_threshold: float = 5,
        device: str = "cpu",
) -> tuple[float, float]:
    """
    Generates a histogram of spatial and temporal roughness statistics from running the decomposition on random noise.
    This is used to decide how "smooth" the temporal and spatial components need to be in order to contain signal.

    Args:
        dimensions (tuple): Tuple describing the dimensions of the blocks which we will
            decompose. Contains (fov_height, fov_width, num_frames), the two spatial field of view dimensions and the number of frames
        spatial_avg_factor (int): The factor (in both spatial dimensions) by which we downsample the data to get higher SNR estimates
        temporal_avg_factor (int): The factor (in time dimension) by which we downsample the data to get higher SNR estimates
        spatial_denoiser (torch.nn.Module | None): A spatial denoiser module for denoising spatial basis vectors
        temporal_denoiser (torch.nn.Module | None): A temporal denoiser module for denoising (batch_size, timeseries_length)
            shaped time series data
        dtype (torch.dtype): the dtype that all tensors must have
        num_comps (int): The number of components which we identify in the decomposition
        iters (int): The number of times we run this simulation procedure to collect a histogram of spatial and temporal
            roughness statistics
        percentile_threshold (float): The threshold we use to decide whether the spatial and temporal roughness stats of
            decomposition are "smooth" enough to contain signal.

    Returns:
        tuple[float, float]: The spatial and temporal "cutoffs" for deciding whether a spatial-temporal decomposition
            contains signals.

    """
    spatial_list = []
    temporal_list = []

    fov_height, fov_width, num_frames = dimensions
    sim_mean = torch.zeros((fov_height, fov_width), device=device, dtype=dtype)
    sim_noise_normalizer = torch.ones((fov_height, fov_width), device=device, dtype=dtype)
    pixel_weighting = torch.ones((fov_height, fov_width), device=device, dtype=dtype)
    frame_weighting = torch.ones(num_frames, device=device, dtype=dtype)
    max_components = num_comps

    for k in progress(range(iters), desc="noise thresholds"):
        sim_data = torch.randn(num_frames, fov_height * fov_width, device=device, dtype=dtype).reshape(
            (num_frames, fov_height, fov_width)
        )

        spatial, temporal, _, _ = blockwise_decomposition(
            sim_data,
            pixel_weighting,
            frame_weighting,
            max_components,
            spatial_avg_factor,
            temporal_avg_factor,
            dtype,
            spatial_denoiser=spatial_denoiser,
            temporal_denoiser=temporal_denoiser,
            device=device,
            subset_mean=sim_mean,
            subset_noise_std=sim_noise_normalizer
        )

        spatial_stat = spatial_roughness_statistic(spatial)
        temporal_stat = temporal_roughness_statistic(temporal)
        spatial_list.append(spatial_stat)
        temporal_list.append(temporal_stat)

    spatial_list = torch.concatenate(spatial_list, dim=0).cpu().numpy()
    temporal_list = torch.concatenate(temporal_list, dim=0).cpu().numpy()

    spatial_threshold = np.percentile(spatial_list.flatten(), percentile_threshold)
    temporal_threshold = np.percentile(temporal_list.flatten(), percentile_threshold)
    return spatial_threshold, temporal_threshold


def construct_weighting_scheme(dim1, dim2) -> torch.Tensor:
    # Define the block weighting matrix
    block_weights = np.ones((dim1, dim2), dtype=np.float32)
    hbh = dim1 // 2
    hbw = dim2 // 2
    # Increase weights to value block centers more than edges
    block_weights[:hbh, :hbw] += np.minimum(
        np.tile(np.arange(0, hbw), (hbh, 1)), np.tile(np.arange(0, hbh), (hbw, 1)).T
    )
    block_weights[:hbh, hbw:] = np.fliplr(block_weights[:hbh, :hbw])
    block_weights[hbh:, :] = np.flipud(block_weights[:hbh, :])
    block_weights = torch.from_numpy(block_weights)
    return block_weights


# ---------------------------------------------------------------------------------------------------------------------
# Memory planning for compression_routine
#
# compression_routine decomposes overlapping blocks. Rather than reading every block from `dataset` on its own, it
# reads a "tile" covering a group of neighboring blocks once and slices the blocks out of it. Larger tiles mean fewer
# redundant reads (and less redundant on-the-fly motion correction) but more memory. The planner picks how many blocks
# each tile groups, and whether the tile is held in host RAM or on the compute device, from the memory budgets.
# ---------------------------------------------------------------------------------------------------------------------

_GIGABYTE = 2 ** 30  # "GB" in the memory budgets, matching how GPU memory is usually reported

_BYTES_PER_VALUE = 4  # tiles and blocks are processed as float32

# Host copies of a tile at its peak: the tile, plus the array `dataset[...]` returned before conversion. A
# RegistrationArray also concatenates its per-batch outputs, so two full copies briefly coexist there too.
_HOST_TILE_COPIES = 2.0

# Memory one block decomposition needs, in multiples of the block's float32 size: the block, its mean-subtracted
# copy, the weighted copy, the lag-1 product of the noise estimate, and slack for the detrender and the SVDs.
# This is an estimate; refine it with torch.cuda.max_memory_allocated() around one block.
_BLOCK_WORKSPACE_COPIES = 6.0

# Fixed device headroom: cuBLAS / cuSOLVER workspaces, denoiser activations, allocator fragmentation.
_DEVICE_HEADROOM_BYTES = 512 * 2 ** 20

# With no budget given, the fraction of currently available memory compression allows itself.
_AUTO_HOST_FRACTION = 0.5
_AUTO_DEVICE_FRACTION = 0.8

# A tile on the compute device skips the per-block host-to-device copies (which move ~4x the data, since blocks
# overlap), but device memory usually makes it smaller than a host tile, so it needs more reads. Prefer the device
# tile unless it reads more than this factor of the data the best host tile reads.
_DEVICE_TILE_MAX_READ_RATIO = 1.25


class TilePlan(NamedTuple):
    """How compression_routine groups blocks into tiles. Produced by plan_tiles."""

    data_batching: tuple[int, int]
    """Most block positions a tile covers along (height, width)"""
    tile_device: str
    """Where tiles are held: "cpu", or the compute device (e.g. "cuda")"""
    max_tile_shape: tuple[int, int]
    """Largest tile, in pixels"""
    max_tile_bytes: int
    """Size of the largest tile across all frames, as float32"""
    num_tiles: int
    read_amplification: float
    """Pixels read from `dataset` divided by the pixels in the field of view; 1.0 means every pixel is read once"""
    host_budget_bytes: int
    device_budget_bytes: int | None

    def describe(self) -> str:
        budgets = f"host budget {self.host_budget_bytes / _GIGABYTE:.2f} GB"
        if self.device_budget_bytes is not None:
            budgets += f", device budget {self.device_budget_bytes / _GIGABYTE:.2f} GB"
        return (f"compression tiles: up to {self.data_batching[0]}x{self.data_batching[1]} blocks "
                f"({self.max_tile_shape[0]}x{self.max_tile_shape[1]} px, {self.max_tile_bytes / _GIGABYTE:.2f} GB) "
                f"held on {self.tile_device}; {self.num_tiles} tiles, each pixel read "
                f"{self.read_amplification:.2f}x on average; {budgets}")


def _tile_groups(positions: list[int], blocks_per_tile: int) -> list[list[int]]:
    """
    Split sorted block start positions along one axis into consecutive groups of at most blocks_per_tile, sized as
    evenly as possible. compression_routine and plan_tiles both use this, so the plan describes the tiles exactly.
    """
    num_positions = len(positions)
    num_groups = math.ceil(num_positions / blocks_per_tile)
    bounds = [round(i * num_positions / num_groups) for i in range(num_groups + 1)]
    return [positions[bounds[i]:bounds[i + 1]] for i in range(num_groups)]


def _axis_groupings(positions: list[int], block_size: int) -> dict[int, tuple[int, int, int]]:
    """
    Every distinct way to group one axis, keyed by the most positions a group holds, as
    (sum of the tiles' extents, largest extent, number of tiles), extents in pixels.
    """
    groupings = {}
    for blocks_per_tile in range(1, len(positions) + 1):
        groups = _tile_groups(positions, blocks_per_tile)
        extents = [group[-1] - group[0] + block_size for group in groups]
        largest_group = max(len(group) for group in groups)
        groupings[largest_group] = (sum(extents), max(extents), len(groups))
    return groupings


def plan_tiles(
        dim1_positions: list[int],
        dim2_positions: list[int],
        block_sizes: tuple[int, int],
        num_frames: int,
        compute_device: str,
        host_budget_bytes: int,
        device_budget_bytes: int | None,
        data_batching: tuple[int, int] | None = None,
        tile_device: Literal["cpu", "device"] | None = None,
) -> TilePlan:
    """
    Choose how compression_routine groups blocks into tiles: the grouping that reads the least data from `dataset`
    within the memory budgets, and whether tiles are held in host RAM or on the compute device.

    Tiles group blocks along each axis independently, so the total read is (sum of tile heights) x (sum of tile
    widths) x frames, and the largest tile is (largest height) x (largest width) x frames. The planner searches every
    pair of axis groupings for the smallest total read whose largest tile fits.

    Memory model, per tile placement:
        - Host tile, device compute: _HOST_TILE_COPIES x tile <= host budget, and one block's workspace <= device budget
        - Host tile, CPU compute: _HOST_TILE_COPIES x tile + one block's workspace <= host budget
        - Device tile: tile + one block's workspace <= device budget, and _HOST_TILE_COPIES x tile <= host budget (the
          array `dataset[...]` returns sits in host RAM before it is copied over)

    Args:
        dim1_positions (list[int]): Sorted start rows of the blocks
        dim2_positions (list[int]): Sorted start columns of the blocks
        block_sizes (tuple[int, int]): Block height and width in pixels
        num_frames (int): Frames loaded per pixel (the frames used for the fit)
        compute_device (str): The device the decompositions run on, e.g. "cpu" or "cuda"
        host_budget_bytes (int): Host RAM compression may use for tiles and, when computing on CPU, block workspace
        device_budget_bytes (int | None): Device memory compression may use. None when computing on CPU.
        data_batching (tuple[int, int] | None): Fix the most block positions per tile instead of planning them
        tile_device ("cpu" | "device" | None): Fix where tiles are held instead of choosing. "device" means the
            compute device.

    Returns:
        TilePlan: The chosen grouping and placement, with its tile sizes and read cost
    """
    if tile_device not in (None, "cpu", "device"):
        raise ValueError(f'tile_device must be "cpu", "device" or None, got {tile_device}')
    on_device = not str(compute_device).startswith("cpu")
    if tile_device == "device" and not on_device:
        raise ValueError('tile_device="device" needs a compute device other than the CPU')
    # Tiles can only be held on the device when its memory budget is known
    device_tiles_allowed = on_device and device_budget_bytes is not None
    if tile_device == "device" and not device_tiles_allowed:
        raise ValueError('tile_device="device" needs a device memory budget (device_memory_gb) for this device')

    column_bytes = num_frames * _BYTES_PER_VALUE  # one pixel across all frames
    block_workspace = _BLOCK_WORKSPACE_COPIES * block_sizes[0] * block_sizes[1] * column_bytes
    if on_device:
        # Block decompositions run on the device, so host RAM holds only tiles
        host_tile_limit = host_budget_bytes / _HOST_TILE_COPIES
    else:
        host_tile_limit = (host_budget_bytes - block_workspace) / _HOST_TILE_COPIES
    device_tile_limit = None
    if device_tiles_allowed:
        block_workspace_device = block_workspace + _DEVICE_HEADROOM_BYTES
        if device_budget_bytes < block_workspace_device:
            logger.warning(f"The device memory budget ({device_budget_bytes / _GIGABYTE:.2f} GB) is below what one "
                           f"block decomposition needs (~{block_workspace_device / _GIGABYTE:.2f} GB for "
                           f"{num_frames} frames); compression may run out of device memory.")
        device_tile_limit = min(device_budget_bytes - block_workspace_device, host_tile_limit)

    axis1 = _axis_groupings(dim1_positions, block_sizes[0])
    axis2 = _axis_groupings(dim2_positions, block_sizes[1])
    fov_pixels = (max(dim1_positions) + block_sizes[0]) * (max(dim2_positions) + block_sizes[1])

    def make_plan(per_tile: tuple[int, int], held_on: str) -> TilePlan:
        sum1, max1, count1 = axis1[per_tile[0]]
        sum2, max2, count2 = axis2[per_tile[1]]
        return TilePlan(data_batching=per_tile,
                        tile_device=compute_device if held_on == "device" else "cpu",
                        max_tile_shape=(max1, max2),
                        max_tile_bytes=int(max1 * max2 * column_bytes),
                        num_tiles=count1 * count2,
                        read_amplification=sum1 * sum2 / fov_pixels,
                        host_budget_bytes=int(host_budget_bytes),
                        device_budget_bytes=None if device_budget_bytes is None else int(device_budget_bytes))

    def best_grouping(tile_limit: float) -> tuple[int, int] | None:
        """The grouping with the smallest total read whose largest tile fits tile_limit; ties go to fewer tiles."""
        best, best_key = None, None
        for per_tile1, (sum1, max1, count1) in axis1.items():
            for per_tile2, (sum2, max2, count2) in axis2.items():
                if max1 * max2 * column_bytes > tile_limit:
                    continue
                key = (sum1 * sum2, count1 * count2)
                if best_key is None or key < best_key:
                    best, best_key = (per_tile1, per_tile2), key
        return best

    if data_batching is not None:
        if min(data_batching) < 1:
            raise ValueError(f"data_batching entries must be positive, got {data_batching}")
        # Snap to the grouping the loop will actually use (groups never exceed the requested size)
        per_tile = tuple(max(len(group) for group in _tile_groups(positions, int(count)))
                         for positions, count in zip((dim1_positions, dim2_positions), data_batching))
        tile_bytes = axis1[per_tile[0]][1] * axis2[per_tile[1]][1] * column_bytes
        if tile_device is None:
            fits_device = device_tiles_allowed and tile_bytes <= device_tile_limit
            tile_device = "device" if fits_device else "cpu"
        limit = device_tile_limit if tile_device == "device" else host_tile_limit
        if tile_bytes > limit:
            logger.warning(f"data_batching={data_batching} makes tiles of {tile_bytes / _GIGABYTE:.2f} GB, over the "
                           f"{max(limit, 0) / _GIGABYTE:.2f} GB the memory budgets allow on {tile_device}")
        return make_plan(per_tile, tile_device)

    host_grouping = best_grouping(host_tile_limit) if tile_device in (None, "cpu") else None
    device_grouping = (best_grouping(device_tile_limit)
                       if device_tiles_allowed and tile_device in (None, "device") else None)

    if host_grouping is None and device_grouping is None:
        held_on = tile_device if tile_device is not None else "cpu"
        logger.warning("The memory budgets cannot hold even one block per tile; reading each block separately, "
                       "which may exceed the budgets.")
        return make_plan((1, 1), held_on)
    if device_grouping is None:
        return make_plan(host_grouping, "cpu")
    if host_grouping is None:
        return make_plan(device_grouping, "device")

    host_plan = make_plan(host_grouping, "cpu")
    device_plan = make_plan(device_grouping, "device")
    if device_plan.read_amplification <= _DEVICE_TILE_MAX_READ_RATIO * host_plan.read_amplification:
        return device_plan
    return host_plan


def _available_host_bytes() -> int | None:
    """Host RAM available to this process now, or None when it cannot be measured."""
    return int(psutil.virtual_memory().available)


def _resolve_memory_budgets(device: str,
                            host_memory_gb: float | None,
                            device_memory_gb: float | None) -> tuple[int, int | None]:
    """
    The host and device budgets in bytes. A budget left as None is a fraction of the memory available now. The
    device budget is None when computing on a device whose memory cannot be measured (the CPU included).
    """
    if host_memory_gb is not None:
        host_budget = int(host_memory_gb * _GIGABYTE)
    else:
        available = _available_host_bytes()
        if available is None:
            available = 16 * _GIGABYTE
            logger.warning("Could not measure available host RAM; assuming 16 GB. Set host_memory_gb to override.")
        host_budget = int(_AUTO_HOST_FRACTION * available)

    torch_device = torch.device(device)
    if torch_device.type != "cuda":
        if torch_device.type != "cpu" and device_memory_gb is not None:
            return host_budget, int(device_memory_gb * _GIGABYTE)
        return host_budget, None

    if device_memory_gb is not None:
        return host_budget, int(device_memory_gb * _GIGABYTE)
    free, _ = torch.cuda.mem_get_info(torch_device)
    # Memory PyTorch's caching allocator holds but is not using is free to us, though the driver counts it as used
    cached_unused = torch.cuda.memory_reserved(torch_device) - torch.cuda.memory_allocated(torch_device)
    return host_budget, int(_AUTO_DEVICE_FRACTION * (free + cached_unused))


def compression_routine(
        dataset: np.ndarray | masknmf.ArrayLike,
        block_sizes: tuple[int, int],
        frame_range: int | None = None,
        max_components: int = 20,
        sim_conf: int = 5,
        max_consecutive_failures=1,
        spatial_avg_factor: int = 1,
        temporal_avg_factor: int = 1,
        window_chunks: int | None = None,
        compute_normalizer: bool = True,
        pixel_weighting: np.ndarray | None = None,
        frame_weighting: np.ndarray | torch.Tensor | None = None,
        spatial_denoiser: torch.nn.Module | None = None,
        temporal_denoiser: torch.nn.Module | None = None,
        detrender: SplineDetrenderBase | None = None,
        data_batching: tuple[int, int] | None = None,
        device: Literal["auto", "cuda", "cpu"] = "auto",
        device_memory_gb: float | None = None,
        host_memory_gb: float | None = None,
        tile_device: Literal["cpu", "device"] | None = None,
) -> CompressionArray:
    """
    General masknmf compression method
    Args:
        dataset (np.ndarray | masknmf.ArrayLike): An array-like object that supports fast slicing in all dimensions
            with shape (frames, fov_height, fov_width) that loads frames of raw data
        block_sizes (tuple[int, int]): The block sizes of the compression. Block size should be big enough to fit around the largest
            somatic signal in your data (roughly).
        frame_range (int): Number of frames of raw data used to fit the spatial basis.
        max_components (int): Max number of components we use to decompose any individual spatial block of the raw data.
        sim_conf (int): The percentile value used to define spatial and temporal roughness thresholds for rejecting/keeping SVD components
        max_consecutive_failures (int): In each blockwise decomposition, we stop accepting SVD components after we see this many "bad" components.
        spatial_avg_factor (int): In the blockwise decompositions, we can spatially downsample the data to estimate a cleaner temporal basis. We can use this to iteratively estimate a better
            full-resolution basis for the data. If signal sits on only a few pixels, keep this parameter at 1 (spatially downsampling is undesirable in this case).
        temporal_avg_factor (int): In the blockwise decompositions, we can temporally downsample the data to estimate a cleaner spatial basis. We can use this to iteratively estimate a better
            full-resolution basis for the data. If your signal "events" are very sparse (i.e. every event appears for only 1 frame) keep this parameter at 1 (temporal downsampling is undesirable in this case).
         window_chunks (int): To be removed
         compute_normalizer (bool): Whether or not we estimate a pixelwise noise variance. If False, the normalizer is set to 1 (no normalization).
         pixel_weighting (np.ndarray | None): Shape (fov_height, fov_width). We weight the data by this value to estimate a cleaner spatial basis. The pixel_weighting
            should intuitively boost the relative variance of pixels containing signal to those that do not contain signal.
        frame_Weighting (np.ndarray | torch.Tensor | None): Shape (num_frames,). Weight certain frames of data as being "more important" in learning the low-rank data subspace
        spatial_denoiser (torch.nn.Module | None): A function that operates on (height, width, num_components)-shaped images, denoising each of the images.
        temporal_denoiser (torch.nn.Module | None): A function that operates on (num_components, num_frames)-shaped traces, denoising each of the traces.
        detrender (SplineDetrenderBase): Module for detrending data
        data_batching (tuple[int, int] | None): Blocks are read from `dataset` in tiles of up to data_batching[0] x
            data_batching[1] neighboring blocks, so overlapping blocks share one read (and one on-the-fly motion
            correction). None (recommended) lets plan_tiles choose it from the memory budgets; set it to fix the
            tiling, e.g. for benchmarking.
        device ("auto" | "cuda" | "cpu"): Which device the computations should be performed on.
        device_memory_gb (float | None): How much memory, in gigabytes, compression may use on the compute
            device (the GPU when device is "cuda"). None uses a fraction of the device memory free when it starts.
        host_memory_gb (float | None): How much host RAM, in gigabytes, compression may use for the data it
            holds while processing. None uses a fraction of the host RAM available when it starts. The compression
            results themselves are kept in host RAM in addition to this.
        tile_device ("cpu" | "device" | None): Where tiles are held. "device" keeps them on the compute device,
            avoiding host-to-device copies per block; "cpu" keeps them in host RAM, allowing larger tiles. None
            (recommended) lets plan_tiles choose.

    Returns:
        compression_array (masknmf.CompressionArray): A CompressionArray object capturing the compression results.
    """

    device = torch_select_device(device)

    logger.debug("Starting compression")
    num_frames, fov_height, fov_width = dataset.shape
    dtype = torch.float32  # This is the target dtype we use for doing computations
    check_fov_size((dataset.shape[1], dataset.shape[2]))

    # Move denoisers to device
    if temporal_denoiser is not None:
        temporal_denoiser.to(device)
    if spatial_denoiser is not None:
        spatial_denoiser.to(device)

    # Decide which chunks of the data you will use for the spatial compression blockwise fits
    if frame_range is None:
        frame_range = dataset.shape[0]

    if window_chunks is None:
        window_chunks = frame_range

    if dataset.shape[0] < frame_range:
        display("WARNING: Specified using more frames than there are in the dataset.")
        frame_range = dataset.shape[0]
        start = 0
        end = dataset.shape[0]
        frames = [i for i in range(start, end)]
        if frame_range <= window_chunks:
            window_chunks = frame_range
    else:
        if frame_range <= window_chunks:
            window_chunks = frame_range
        frames = identify_window_chunks(frame_range, dataset.shape[0], window_chunks)
    logger.debug("We are initializing on a total of {} frames".format(len(frames)))

    block_sizes = update_block_sizes(block_sizes, (dataset.shape[1], dataset.shape[2]))

    overlap = [math.ceil(block_sizes[0] / 2), math.ceil(block_sizes[1] / 2)]

    dataset_mean = torch.zeros(dataset.shape[1], dataset.shape[2]).to(device).to(dtype)

    if not compute_normalizer:
        dataset_noise_std = torch.ones(dataset.shape[1], dataset.shape[2]).to(device).to(dtype)
    else:
        dataset_noise_std = torch.zeros(dataset.shape[1], dataset.shape[2]).to(device).to(dtype)

    logger.debug("Loading data to estimate complete spatial basis")

    # First make sure the number of frames loaded is divisible by the temporal average factor
    if temporal_avg_factor >= len(frames):
        raise ValueError("Need at least {} frames".format(temporal_avg_factor))

    logger.debug("skipping the pruning step for frame cutoff")
    #frame_cutoff = (len(frames) // temporal_avg_factor) * temporal_avg_factor
    # frames = frames[:frame_cutoff]
    if len(frames) // temporal_avg_factor <= max_components:
        string_to_disp = (
            f"WARNING: temporal avg factor is too big, max rank per block adjusted to {len(frames) // temporal_avg_factor}.\n"
            "To avoid this, initialize with more frames or reduce temporal avg factor"
        )
        display(string_to_disp)
        max_components = int(len(frames) // temporal_avg_factor)

    if pixel_weighting is None:
        pixel_weighting = torch.ones((fov_height, fov_width), device=device, dtype=dtype)
    else:
        pixel_weighting = torch.as_tensor(pixel_weighting, device=device, dtype=dtype)

    if frame_weighting is None:
        frame_weighting = torch.ones(num_frames, device=device, dtype=dtype)
    else:
        frame_weighting = torch.as_tensor(frame_weighting, device=device, dtype=dtype)

    ## Define
    dim_1_iters = list(
        range(
            0,
            fov_height - block_sizes[0] + 1,
            block_sizes[0] - overlap[0],
        )
    )
    if (
            dim_1_iters[-1] != fov_height - block_sizes[0]
            and fov_height - block_sizes[0] != 0
    ):
        dim_1_iters.append(fov_height - block_sizes[0])

    dim_2_iters = list(
        range(
            0,
            fov_width - block_sizes[1] + 1,
            block_sizes[1] - overlap[1],
        )
    )
    if (
            dim_2_iters[-1] != fov_width - block_sizes[1]
            and fov_width - block_sizes[1] != 0
    ):
        dim_2_iters.append(fov_width - block_sizes[1])

    # Define the block weighting matrix
    block_weights = (
        construct_weighting_scheme(block_sizes[0], block_sizes[1]).to(device).to(dtype)
    )

    sparse_indices = torch.arange(
        fov_height * fov_width, dtype=torch.long, device=device
    ).reshape((fov_height, fov_width))

    column_number = 0
    final_row_indices = []
    final_column_indices = []
    spatial_overall_values = []
    spatial_overall_unweighted_values = []
    cumulative_weights = torch.zeros((fov_height, fov_width), dtype=dtype, device=device)
    total_temporal_fit = []

    logger.debug("Finding spatiotemporal roughness thresholds")
    spatial_roughness_threshold, temporal_roughness_threshold = threshold_heuristic(
        [block_sizes[0], block_sizes[1], window_chunks],
        spatial_avg_factor,
        temporal_avg_factor,
        spatial_denoiser,
        temporal_denoiser,
        dtype,
        num_comps=1,
        iters=250,
        percentile_threshold=sim_conf,
        device=device,
    )

    if detrender is not None:
        detrender = detrender.to(device)
        spatial_preprocess_basis = torch.zeros(
            fov_height, fov_width, detrender.spline_rank, device=device
        )
        temporal_preprocess_basis = detrender.basis.T  # (spline_rank, frames)
    else:
        spatial_preprocess_basis = None
        temporal_preprocess_basis = None

    if len(frames) == dataset.shape[0]:
        effective_frame_indexer = slice(0, int(dataset.shape[0]))
    else:
        effective_frame_indexer = frames

    # Plan the tiles last, so the device memory measured excludes what the denoisers and thresholds hold
    host_budget_bytes, device_budget_bytes = _resolve_memory_budgets(device, host_memory_gb, device_memory_gb)
    plan = plan_tiles(dim_1_iters,
                      dim_2_iters,
                      block_sizes,
                      num_frames=len(frames),
                      compute_device=device,
                      host_budget_bytes=host_budget_bytes,
                      device_budget_bytes=device_budget_bytes,
                      data_batching=data_batching,
                      tile_device=tile_device)
    logger.info(plan.describe())
    tiles = [(dim1_group, dim2_group)
             for dim1_group in _tile_groups(dim_1_iters, plan.data_batching[0])
             for dim2_group in _tile_groups(dim_2_iters, plan.data_batching[1])]

    logger.debug("Running Blockwise Decompositions")
    for dim1_batch_start, dim2_batch_start in progress(tiles, desc="decomposing blocks"):
        ## Now compute the slice into the dataset array required to load this batch
        dataset_slice_lower_bound = (int(dim1_batch_start[0]), int(dim2_batch_start[0]))
        dataset_slice_upper_bound = (int(dim1_batch_start[-1]) + block_sizes[0], int(dim2_batch_start[-1]) + block_sizes[1])
        dim1_batch_slice = slice(dataset_slice_lower_bound[0], dataset_slice_upper_bound[0])
        dim2_batch_slice = slice(dataset_slice_lower_bound[1], dataset_slice_upper_bound[1])

        # Release the previous tile first: the right-hand side below runs before the assignment, so otherwise
        # two tiles would coexist while the next one loads
        curr_dataset = None
        curr_dataset = torch.as_tensor(dataset[effective_frame_indexer, dim1_batch_slice, dim2_batch_slice],
                                       device=plan.tile_device, dtype=dtype)
        for k in dim1_batch_start:
            for j in dim2_batch_start:
                slice_dim1 = slice(k, k + block_sizes[0])
                slice_dim2 = slice(j, j + block_sizes[1])

                offset_slice_dim1 = slice(k - dataset_slice_lower_bound[0], k - dataset_slice_lower_bound[0] + block_sizes[0])
                offset_slice_dim2 = slice(j - dataset_slice_lower_bound[1], j - dataset_slice_lower_bound[1] + block_sizes[1])

                current_data_for_fit = curr_dataset[:, offset_slice_dim1, offset_slice_dim2].to(device)

                if detrender is not None:
                    curr_shape = current_data_for_fit.shape
                    current_data_for_fit, curr_spline_basis = detrender(
                        current_data_for_fit.reshape(curr_shape[0], -1)
                    )
                    spatial_preprocess_basis[slice_dim1, slice_dim2, :] = (
                        curr_spline_basis.T.reshape(curr_shape[1], curr_shape[2], -1)
                    )
                    current_data_for_fit = current_data_for_fit.reshape(*curr_shape)

                (
                    unweighted_local_spatial_basis,
                    local_temporal_basis,
                    subset_mean,
                    subset_noise_std
                ) = blockwise_decomposition_with_rank_selection(
                    current_data_for_fit,
                    pixel_weighting[slice_dim1, slice_dim2],
                    frame_weighting,
                    max_components,
                    max_consecutive_failures,
                    spatial_roughness_threshold,
                    temporal_roughness_threshold,
                    spatial_avg_factor,
                    temporal_avg_factor,
                    dtype,
                    spatial_denoiser=spatial_denoiser,
                    temporal_denoiser=temporal_denoiser,
                    device=device,
                    subset_noise_std=dataset_noise_std[slice_dim1, slice_dim2] if not compute_normalizer else None
                )

                dataset_mean[slice_dim1, slice_dim2] = subset_mean
                dataset_noise_std[slice_dim1, slice_dim2] = subset_noise_std

                if local_temporal_basis.shape[0] == 0:
                    continue

                # Results move to host RAM as they are produced; kept on the device they would grow through the
                # whole run outside any memory budget (they are moved to the CPU at the end regardless)
                total_temporal_fit.append(local_temporal_basis.cpu())

                # Weight the spatial components here

                local_spatial_basis = (
                        unweighted_local_spatial_basis * block_weights[:, :, None]
                )
                current_cumulative_weight = block_weights
                cumulative_weights[
                k: k + block_sizes[0], j: j + block_sizes[1]
                ] += current_cumulative_weight

                curr_spatial_row_indices = sparse_indices[
                                           k: k + block_sizes[0], j: j + block_sizes[1]
                                           ][:, :, None]
                curr_spatial_row_indices = curr_spatial_row_indices + torch.zeros(
                    (1, 1, local_spatial_basis.shape[2]), device=device, dtype=torch.long
                )

                curr_spatial_col_indices = torch.zeros_like(
                    curr_spatial_row_indices, device=device
                )
                addend = torch.arange(
                    column_number,
                    column_number + local_spatial_basis.shape[2],
                    device=device,
                    dtype=torch.long,
                )[None, None, :]
                curr_spatial_col_indices = curr_spatial_col_indices + addend

                sparse_row_indices_f = curr_spatial_row_indices.flatten()
                sparse_col_indices_f = curr_spatial_col_indices.flatten()
                spatial_values_f = local_spatial_basis.flatten()

                final_row_indices.append(sparse_row_indices_f.cpu())
                final_column_indices.append(sparse_col_indices_f.cpu())
                spatial_overall_values.append(spatial_values_f.cpu())
                spatial_overall_unweighted_values.append(
                    unweighted_local_spatial_basis.flatten().cpu()
                )
                column_number += local_spatial_basis.shape[2]

    curr_dataset = None  # release the last tile before assembling the results

    # Construct the spatial_compressed matrix up to this point (the per-block results are in host RAM)
    final_row_indices = torch.concatenate(final_row_indices, dim=0)
    final_column_indices = torch.concatenate(final_column_indices, dim=0)
    spatial_overall_values = torch.concatenate(spatial_overall_values, dim=0)
    final_indices = torch.stack([final_row_indices, final_column_indices], dim=0)

    spatial_overall_unweighted_values = torch.concatenate(
        spatial_overall_unweighted_values, dim=0
    )
    u_spatial_fit = torch.sparse_coo_tensor(
        final_indices,
        spatial_overall_unweighted_values,
        (fov_height * fov_width, column_number),
    ).coalesce()

    if total_temporal_fit[0].shape[1] != num_frames:
        raise ValueError("Fitting to part of the data is no longer supported")

    else:
        v_aggregated = torch.concatenate(total_temporal_fit, dim=0)

    interpolation_weightings = torch.reciprocal(
        cumulative_weights.flatten().cpu()[final_row_indices]
    )
    spatial_overall_values *= interpolation_weightings

    num_cols = column_number
    num_rows = fov_height * fov_width
    u_local_projector = torch.sparse_coo_tensor(
        final_indices, spatial_overall_unweighted_values, (num_rows, num_cols)
    ).coalesce()

    final_indices = torch.stack([final_row_indices, final_column_indices], dim=0)
    u_aggregated = torch.sparse_coo_tensor(
        final_indices, spatial_overall_values, (num_rows, num_cols)
    ).coalesce()
    logger.debug(f"Constructed spatial_compressed matrix. Rank of spatial_compressed is {u_aggregated.shape[1]}")


    ## If the preprocessing basis was used, re-assign the mean of that decomposition to the compression array
    ## Also re-shape the spatial preprocess basis
    if spatial_preprocess_basis is not None and temporal_preprocess_basis is not None:
        temporal_basis_mean = torch.mean(temporal_preprocess_basis, dim = 1, keepdims = True)
        curr_mean = spatial_preprocess_basis @ temporal_basis_mean
        curr_mean = curr_mean.squeeze(-1)
        dataset_mean += curr_mean
        temporal_preprocess_basis = temporal_preprocess_basis - temporal_basis_mean
        spatial_preprocess_basis = spatial_preprocess_basis.reshape(-1, spatial_preprocess_basis.shape[2])

    final_compression_arr = CompressionArray.from_tensors(
        (num_frames, fov_height, fov_width),
        u_aggregated.cpu(),
        v_aggregated.cpu(),
        dataset_mean,
        dataset_noise_std,
        spatial_compressed_local_projector=u_local_projector.cpu()
        if u_local_projector is not None
        else None,
        spatial_trend_basis=spatial_preprocess_basis.cpu() if spatial_preprocess_basis is not None else None,
        temporal_trend_basis = temporal_preprocess_basis.cpu() if temporal_preprocess_basis is not None else None,
        device="cpu",
    )
    logger.debug("Compression Object constructed")
    return final_compression_arr