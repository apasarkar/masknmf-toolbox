import torch
from typing import List

import masknmf
from masknmf.compression.pmd_array import PMDArray
from masknmf.arrays.array_interfaces import LazyFrameLoader
import math
import numpy as np

from tqdm import tqdm

from masknmf import display
from typing import *


def truncated_random_svd(
        input_matrix: torch.tensor,
        rank: int,
        num_oversamples: int = 5,
        device: str = "cpu"
) -> Tuple[torch.tensor, torch.tensor, torch.tensor]:
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
    v_final = s[:, None] * v
    return u_final[:, :rank], s[:rank], v_final[:rank, :]


"""
The below functions are for spatial and temporal roughness penalties
"""


def temporal_roughness_statistic(temporal_traces: torch.tensor) -> torch.tensor:
    """
    Computes the temporal roughness statistics, batched over all the traces of interest
    Args:
        temporal_traces (torch.tensor): shape (num_traces, num_frames).
    Returns:
        stats (torch.tensor): shape (num_traces)
    """
    left_term = temporal_traces[:, :-2]
    right_term = temporal_traces[:, 2:]
    center_term = temporal_traces[:, 1:-1]

    numerator = torch.mean(torch.abs(left_term + right_term - 2 * center_term), dim=1)
    denominator = torch.mean(torch.abs(temporal_traces), dim=1)
    denominator[denominator == 0] = 1.0
    return numerator / denominator


def spatial_roughness_statistic(spatial_comps: torch.tensor) -> torch.tensor:
    """
    Computes spatial roughness statistic, batched over all spatial comps of interest
    Args:
        spatial_comps (torch.tensor): shape (fov dim1, fov dim2, num_components)
    Returns:
        stats (torch.tensor): shape (num_components)
    """
    d1, d2 = spatial_comps.shape[0], spatial_comps.shape[1]
    #Compute abs(vertical differences) :
    top_vertical = spatial_comps[:-1, :, :]
    bottom_vertical = spatial_comps[1:, :, :]
    vertical_diffs = torch.abs(top_vertical - bottom_vertical)

    #Compute abs(horizontal differences)
    left_horizontal = spatial_comps[:, :-1, :]
    right_horizontal = spatial_comps[:, 1:, :]
    horizontal_diffs = torch.abs(left_horizontal - right_horizontal)

    #Compute abs(top left --> bottom right differences)
    top_left = spatial_comps[:-1, :-1, :]
    bottom_right = spatial_comps[1:, 1:, :]
    top_bottom_diag_diffs = torch.abs(top_left - bottom_right)

    #Compute abs(bottomleft --> topright differences)
    top_right = spatial_comps[1:, 1:, :]
    bottom_left = spatial_comps[:-1, :-1, :]
    bottom_top_diag_diffs = torch.abs(top_right - bottom_left)

    total_terms = (torch.prod(torch.tensor(vertical_diffs.shape[:2])) + torch.prod(
        torch.tensor(horizontal_diffs.shape[:2])) +
                   torch.prod(torch.tensor(top_bottom_diag_diffs.shape[:2])) + torch.prod(
                torch.tensor(bottom_top_diag_diffs.shape[:2])))

    avg_diff = (torch.sum(vertical_diffs, dim=(0, 1)) + torch.sum(horizontal_diffs, dim=(0, 1))
                + torch.sum(top_bottom_diag_diffs, dim=(0, 1)) + torch.sum(top_bottom_diag_diffs, dim=(0, 1)))
    avg_diff /= total_terms

    return avg_diff / torch.mean(torch.abs(spatial_comps), dim=(0, 1))


def evaluate_fitness(spatial_comps: torch.tensor,
                     temporal_traces: torch.tensor,
                     spatial_statistic_threshold: float,
                     temporal_statistic_threshold: float) -> torch.tensor:
    """
    Args:
        spatial_comps (torch.tensor): shape (fov dim1, fov dim2, num_components)
        temporal_traces (torch.tensor): shape (num_components, num_frames)
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
        decisions: torch.tensor, max_consecutive_failures: int
) -> torch.tensor:
    """
    Filters decisions based on maximum consecutive failures.

    Args:
        decisions (np.ndarray): 1-dimensional array of boolean values representing decisions.
        max_consecutive_failures (int): Maximum number of consecutive failures (ie decisions[i] == 0) allowed.

    Returns:
        np.ndarray: Filtered decisions with the same shape and type as input decisions.
    """

    false_tensor = (~decisions).to(dtype=torch.float32)
    kernel = torch.ones(max_consecutive_failures, device=false_tensor.device, dtype=torch.float32)[None, None,
             :]  # Shape (1,1,n)
    seq = false_tensor.to(torch.float32)[None, None, :]

    # Convolve to find runs of n consecutive False values
    conv_result = torch.nn.functional.conv1d(seq, kernel, stride=1, padding=max_consecutive_failures - 1).squeeze()[
                  :false_tensor.shape[0]]

    over_threshold = (conv_result >= max_consecutive_failures).to(torch.float32)
    keep_comps = torch.cumsum(torch.cumsum(over_threshold, dim=0),
                              dim=0) <= 1  ##Two cumulative sums guarantee we pick the last element properly

    return keep_comps


def identify_window_chunks(
        frame_range: int, total_frames: int, window_chunks: int
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
    display("sampled from the following regions: {}".format(starting_points))

    net_frames = []
    for k in starting_points:
        curr_start = k
        curr_end = min(k + window_chunks, total_frames)

        curr_frame_list = [i for i in range(curr_start, curr_end)]
        net_frames.extend(curr_frame_list)
    return net_frames


def check_fov_size(fov_dims: Tuple[int, int],
                   min_allowed_value: int = 10) -> None:
    """
    Checks if the field of view (FOV) dimensions are too small.

    Args:
        fov_dims (tuple): Two integers specifying the FOV dimensions.
        min_allowed_value (int, optional): The minimum allowed value for FOV dimensions. Defaults to 10.

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
        blocks: tuple, fov_shape: tuple, min_block_value: int = 4,
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


def compute_mean_and_normalizer_dataset(dataset: LazyFrameLoader,
                                        compute_normalizer: bool,
                                        frame_batch_size: int,
                                        device: str,
                                        dtype: torch.dtype
                                        ) -> Tuple[torch.tensor, torch.tensor]:
    """
    Computes a pixelwise mean and a noise variance estimate. For now, the noise var estimate is turned off
    Args:
        dataset: masknmf.lazy_data_loader. The dataloader object we use to access the dataset. Anything that supports
            numpy-like __getitem__ indexing can be used here.
        compute_normalizer (bool): Whether or not we compute the noise normalizer; for now this variable has no effect.
        frame_batch_size (int): The max number of full frames we load onto the device performing computations at any point.
        pixel_batch_size (int): The max number of full pixels (length = frames of movie) we load onto the device
            performing computations ay any point.
        dtype (torch.dtype): The dtype of the data once it has been moved to accelerator.
        device (str): The
    Returns:
        mean_img (torch.tensor): The (fov dim1, fov dim2) shaped mean image.
        var_img (torch.tensor): The (fvo dim1, fov dim2) noise variance image.
    """
    num_frames, fov_dim1, fov_dim2 = dataset.shape
    noise_normalizer = torch.ones((fov_dim1, fov_dim2), dtype=dtype)

    num_batches = math.ceil(num_frames / frame_batch_size)
    curr_sum = torch.zeros((fov_dim1, fov_dim2), dtype=dtype, device=device)
    for k in range(num_batches):
        start_pt = frame_batch_size * k
        end_pt = min(start_pt + frame_batch_size, num_frames)
        curr_data = dataset[start_pt:end_pt]
        curr_tensor = torch.from_numpy(curr_data).to(device).to(dtype)
        curr_sum += torch.sum(curr_tensor / num_frames, dim=0)

    return curr_sum.cpu(), noise_normalizer.cpu()


def compute_full_fov_spatial_basis(dataset: LazyFrameLoader,
                                   mean_img: torch.tensor,
                                   noise_variance_img: torch.tensor,
                                   background_rank: int,
                                   dtype: torch.dtype = torch.float32,
                                   device: str = "cpu",
                                   num_samples: int = 1000):
    """
    Routine for approximating a full FOV low-rank spatial basis; useful for estimating full FOV trends

    Args:
        dataset (masknmf.LazyFrameLoader)
        mean_img (torch.tensor): Shape (fov_dim1, fov_dim2). Mean image of the data
        noise_variance_img (torch.tensor): Shape (fov_dim1, fov_dim2). The noise variance estimate at each pixel
        background_rank (int): The rank of the background term we are trying to estimate
        frame_batch_size (int): The number of frames we can load into memory at a time
        dtype (torch.dtype): The dtype of data used in the actual computations
        num_oversamples (int): The number of oversamples for the subspace estimation method
        device (str): Which device the computations are performed on
        num_samples (int): The number of frames of the dataset that we randomly sample to estimate the full FOV
            subspace.

    Returns:
        spatial_basis (torch.tensor). Shape (fov_dim1, fov_dim2, net_rank). The orthonormal spatial basis vectors
    """
    num_frames, fov_dim1, fov_dim2 = dataset.shape
    if background_rank <= 0:
        return torch.zeros((fov_dim1, fov_dim2, 1)).to(dtype)
    sample_list = [i for i in range(0, num_frames)]
    random_data = np.random.choice(
        sample_list, replace=False, size=min(num_samples, num_frames)
    )

    mean_img = mean_img.to(device).to(dtype)
    noise_variance_img = noise_variance_img.to(device).to(dtype)
    my_data = torch.tensor(dataset[random_data]).to(device).to(dtype)
    my_data = (my_data - mean_img[None, :, :]) / noise_variance_img[None, :, :]
    my_data = my_data.permute(1, 2, 0).reshape((fov_dim1 * fov_dim2, -1))

    my_data = my_data.to(device).to(dtype)
    spatial_basis, _, _ = truncated_random_svd(my_data,
                                               background_rank,
                                               device=device)

    return spatial_basis.cpu().reshape((fov_dim1, fov_dim2, -1))


def compute_full_fov_temporal_basis(dataset: torch.tensor,
                                    mean_img: torch.tensor,
                                    noise_variance_img: torch.tensor,
                                    full_fov_spatial_basis: torch.tensor,
                                    dtype: torch.dtype,
                                    frame_batch_size: int,
                                    device: str = "cpu") -> torch.tensor:
    """
    Regress some portion of the data onto the spatial basis.
    Args:
        dataset (torch.tensor). A dataset of shape (frames, fov_dim1, fov_dim2)
        mean_img (torch.tensor). The mean image of the data. Shape (fov dim1, fov dim2)
        noise_variance_img (torch.tensor). The noise variance image of the data. Shape (fov dim1, fov dim2)
        full_fov_spatial_basis (torch.tensor). A full FOV spatial basis for the data (Shape (fov dim1, fov dim2, rank).
        dtype (torch.dtype): The dtype on which we do computations. Should be torch.float32.
        frame_batch_size (int): The max number of frames we want to load onto GPU at a time.
        device (str): Either "cpu" or "cuda". Specifies whether we can do computations on GPU or not.
    Returns:
        temporal_basis (torch.tensor). Shape (rank, num_frames). Projection of standardized data onto spatial basis.
    """
    num_frames, fov_dim1, fov_dim2 = dataset.shape
    mean_img = mean_img.to(device).to(dtype)
    noise_variance_img = noise_variance_img.to(device).to(dtype)
    num_iters = math.ceil(dataset.shape[0] / frame_batch_size)
    final_results = []

    mean_img_r = mean_img.reshape((fov_dim1 * fov_dim2, 1)).T
    noise_variance_img_r = noise_variance_img.reshape((fov_dim1 * fov_dim2, 1))
    spatial_basis_r = full_fov_spatial_basis.to(device).to(dtype).reshape((fov_dim1 * fov_dim2, -1))
    spatial_basis_r_weighted_by_variance = (spatial_basis_r * torch.reciprocal(noise_variance_img_r))
    for k in range(num_iters):
        if frame_batch_size >= dataset.shape[0]:
            curr_dataset = dataset.to(device).to(dtype)
        else:
            start_pt = k * frame_batch_size
            end_pt = min(dataset.shape[0], start_pt + frame_batch_size)
            curr_dataset = dataset[start_pt:end_pt].to(device).to(dtype)
        curr_dataset_r = curr_dataset.reshape((-1, fov_dim1 * fov_dim2))
        projection = curr_dataset_r @ spatial_basis_r_weighted_by_variance - mean_img_r @ spatial_basis_r_weighted_by_variance
        final_results.append(projection)

    final_tensor = torch.concatenate(final_results, dim=0)
    return final_tensor.T


def compute_factorized_svd_with_leftbasis(p: torch.sparse_coo_tensor,
                                          v: torch.tensor) -> Tuple[torch.tensor, torch.tensor, torch.tensor]:
    """
    Use case: you have a factorized movie, UPV where U is sparse (and you don't want to change that), and
    UP has orthonormal columns. This function reformats the factorization into UPV = (UR)sV_{new} where (UR) are left
    singular vecotrs, s describes singular values, V_new describes right singular vectors.

    Args:
        u (torch.sparse_coo_tensor): shape (pixels, rank)
        p (torch.tensor): shape (rank, rank)
        v: (torch.tensor): shape (rank, num_frames)
    """
    q, m = [i.float().T for i in torch.linalg.qr(v.T, mode="reduced")]  #Now v = mq

    #Note that (upm)^T(upm) = (m^T)m
    mtm = m.T @ m
    # mtm = (mtm + mtm.T) / 2
    # eig_vals, eig_vecs = [i.float() for i in torch.linalg.eigh(mtm.double())]
    eig_vecs, eig_vals, _ = [i.float() for i in torch.linalg.svd(mtm, full_matrices=True)]

    print(f"{torch.allclose(mtm, mtm.T)}")
    print(f"When we ran the  leftbasis eigh routine, the smallest value we saw was {np.amin(eig_vals.cpu().numpy())}")

    # eig_vecs = torch.flip(eig_vecs, dims = [1])
    # eig_vals = torch.flip(eig_vals, dims = [0])

    print(f"When we ran the eigh routine, the smallest value we saw was {np.amin(eig_vals.cpu().numpy())}")
    good_components = eig_vals > 0
    eig_vecs = eig_vecs[:, good_components]
    eig_vals = eig_vals[good_components]

    s = torch.sqrt(eig_vals)

    r = p @ (eig_vecs / s[None, :])
    v = eig_vecs.T @ q
    return r, s, v


def compute_lowrank_factorized_svd(
        u: torch.sparse_coo_tensor,
        v: torch.tensor,
):
    """
    Compute the factorized Singular Value Decomposition (SVD) of a low-rank matrix factorization.

    This function computes the SVD of a matrix `u @ v`, where `u` is sparse and `v` is dense,
    both representing a low-rank factorization. It efficiently computes a reduced or partial SVD
    based on this factorization. The function allows returning just the left singular vectors
    (spatial mixing matrix) if specified.

    Args:
        u (torch.sparse_coo_tensor):
            Sparse left matrix of the factorization with shape `(pixels, low_rank)`.
        v (torch.tensor):
            Dense right matrix of the factorization with shape `(low_rank, frames)`.
        only_left (bool, optional):
            If `True`, only the left singular vectors (spatial mixing matrix) are returned. If we return a tensor, P,
            with the property that U@P has orthonormal columns. Defaults to `False`.

    Returns:
        np.ndarray:
            `spatial_mixing_matrix`: An orthonormal column basis for the factorization `u @ v`.
            This matrix represents the spatial components of the original data.

        If `only_left` is False, it also returns:
        np.ndarray:
            `singular_values`: 1D vector of singular values, representing the scaling factors
            for the corresponding orthonormal directions.
        np.ndarray:
            `right_singular_vectors`: Orthonormal column vectors representing the temporal
            components of the matrix `v`.

    Notes:
        - This is not a full SVD; the result is truncated to preserve efficiency, especially
        for large matrices. The orthogonality of the left singular vectors holds within the
        reduced space of the factorization.
        - This routine uses eigh on the spatial basis to exploit the low rank of the decomposition. This will lead to bad results if the matrix is ill-conditioned.
        PMD gives us a reasonable guarantee that this is not true (due to the blockwise decompositions).
    """
    q, p = [i.float().T for i in torch.linalg.qr(v.T, mode="reduced")]  # Here, v = pq, with q having orth rows
    ut_u = torch.sparse.mm(u.T, u).to_dense()

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


def regress_onto_spatial_basis(dataset: LazyFrameLoader,
                               u_aggregated: torch.sparse_coo_tensor,
                               frame_batch_size: int,
                               dataset_mean: torch.tensor,
                               dataset_noise_variance: torch.tensor,
                               full_fov_spatial_basis: torch.tensor,
                               dtype: torch.dtype,
                               device: str = "cpu") -> torch.tensor:
    """
    We have a spatial basis from blockwise decompositions. This function will do two things, in a single pass through the
    data:
        (1) It will project the data onto each block's orthogonal basis (this is NOT equivalent to a linear subspace projection onto the
        spatial basis!)
        (2) It will perform a linear subspace projection of the centered+standardized data onto the full FOV data

    The computation to perform here is:
    v_aggregate = u^T (I_{norms} * (Data - Mean) - Spatial_Full_FOV_Bkgd * Temporal_Full_FOV_Bkgd)
    Here, I_{norms} is a diagonal matrix containing the reciprocal of the dataset_noise_variance. The term
    I_{norms}(Data - Mean) does pixelwise centering + standardization of the data.
    In the below routine, we exploit the low rank of u and conduct operations in an order that minimizes data size/number of computations.

    Args:
        dataset (masknmf.LazyFrameLoader): Any array-like object that supports __getitem__ for fast frame retrieval.
        u_aggregated (torch.sparse_coo_tensor): The spatial basis, where components from the same block are orthonormal.
        frame_batch_size (int): The number of frames we load at any point in time
        dataset_mean (torch.tensor): Shape (fov_dim1, fov_dim2). The mean across all pixels
        dataset_noise_variance (torch.tensor): Shape (fov_dim1, fov_dim2). The noise variance across all pixels.
        full_fov_spatial_basis (torch.tensor): Shape (fov_dim1, fov_dim2, full_fov_rank): The rank of the full fov spatial basis term.
            This basis is orthonormal.
        dtype (torch.dtype): The dtype to which we convert the data for processing; should be torch.float32, or float64.
        device (str): The platform on which processing occurs ("cuda" or "cpu")
    """
    num_frames, fov_dim1, fov_dim2 = dataset.shape
    num_iters = math.ceil(dataset.shape[0] / frame_batch_size)
    dataset_mean = dataset_mean.to(device).to(dtype).reshape((fov_dim1 * fov_dim2, 1))
    dataset_noise_variance = dataset_noise_variance.to(device).to(dtype).reshape((fov_dim1 * fov_dim2, 1))
    full_fov_spatial_basis = full_fov_spatial_basis.to(device).to(dtype).reshape((fov_dim1 * fov_dim2, -1))

    u_t = u_aggregated.T.coalesce()

    row_indices, col_indices = u_t.indices()
    ut_values = u_t.values()

    new_values = ut_values / dataset_noise_variance[col_indices].squeeze()

    u_t_normalized = torch.sparse_coo_tensor(u_t.indices(), new_values, u_t.shape).coalesce()

    full_fov_spatial_projected = torch.sparse.mm(u_t, full_fov_spatial_basis)
    mean_projected = torch.sparse.mm(u_t_normalized, dataset_mean)

    temporal_results = []
    temporal_background_results = []
    for k in tqdm(range(num_iters)):
        start_pt = k * frame_batch_size
        end_pt = min(start_pt + frame_batch_size, num_frames)
        curr_data = torch.from_numpy(dataset[start_pt:end_pt]).to(device).to(dtype).permute(1, 2, 0).reshape(
            (fov_dim1 * fov_dim2, -1))
        projection = torch.sparse.mm(u_t_normalized, curr_data)
        projection -= mean_projected
        temporal_full_fov_comp = full_fov_spatial_basis.T @ curr_data
        full_fov_projected_term = full_fov_spatial_projected @ temporal_full_fov_comp
        projection -= full_fov_projected_term

        ## Add the full fov and blockwise temporal components that we estimate above to a list to concatenate later
        temporal_background_results.append(temporal_full_fov_comp)
        temporal_results.append(projection)
    return torch.concatenate(temporal_results, dim=1), torch.concatenate(temporal_background_results, dim=1)


def temporal_downsample(tensor: torch.Tensor, temporal_avg_factor: int) -> torch.Tensor:
    """
    Temporally downsamples a (height, width, num_frames) tensor using avg_pool1d.

    Args:
        tensor: Input tensor of shape (num_frames, height, width).
        n: Downsampling factor (number of frames per block).

    Returns:
        Downsampled tensor of shape (height, width, ceil(num_frames / n)).
    """
    height, width, num_frames = tensor.shape
    tensor = tensor.reshape(height * width, num_frames).unsqueeze(1)
    downsampled = torch.nn.functional.avg_pool1d(tensor,
                                                 kernel_size=temporal_avg_factor,
                                                 stride=temporal_avg_factor,
                                                 ceil_mode=True)

    # Reshape back to (num_frames // n, height, width)
    return downsampled.squeeze().reshape(height, width, -1)


def spatial_downsample(image_stack: torch.Tensor, spatial_avg_factor: int) -> torch.Tensor:
    """
    Downsamples a (height, width, num_frames) image stack via n x n binning.

    Args:
        image_stack: Tensor of shape (height, width, num_frames).

    Returns:
        Downsampled tensor of shape (H//factor, W//factor, T).
    """

    image_stack = image_stack.permute(2, 0, 1).unsqueeze(1)  # (num_frames, 1, height, width)
    downsampled = torch.nn.functional.avg_pool2d(image_stack, kernel_size=spatial_avg_factor,
                                                 stride=spatial_avg_factor)  # (T, 1, H//2, W//2)
    return downsampled.squeeze(1).permute(1, 2, 0)


def blockwise_decomposition(video_subset: torch.tensor,
                            full_fov_spatial_basis: torch.tensor,
                            full_fov_temporal_basis: torch.tensor,
                            subset_mean: torch.tensor,
                            subset_noise_variance: torch.tensor,
                            subset_pixel_weighting: torch.tensor,
                            max_components: int,
                            spatial_avg_factor: int,
                            temporal_avg_factor: int,
                            dtype: torch.dtype,
                            spatial_denoiser: Optional[Callable] = None,
                            temporal_denoiser: Optional[Callable] = None,
                            device: str = "cpu") -> Tuple[torch.tensor, torch.tensor]:
    num_frames, fov_dim1, fov_dim2 = video_subset.shape
    subset = video_subset.to(device).to(dtype)
    subset = subset - subset_mean.to(device).to(dtype)[None, :, :]
    subset /= subset_noise_variance.to(device).to(dtype)[None, :, :]
    spatial_basis_product = full_fov_spatial_basis.to(device).to(dtype) @ full_fov_temporal_basis.to(device).to(dtype)
    subset = subset.permute(1, 2, 0) - spatial_basis_product
    subset_weighted = subset * subset_pixel_weighting.to(device).to(dtype)[:, :, None]

    if spatial_avg_factor != 1:
        spatial_pooled_subset = spatial_downsample(subset_weighted, spatial_avg_factor)
    else:
        spatial_pooled_subset = subset_weighted

    spatial_pooled_subset_r = spatial_pooled_subset.reshape((-1, spatial_pooled_subset.shape[2]))

    if temporal_avg_factor != 1:
        spatiotemporal_pooled_subset = temporal_downsample(spatial_pooled_subset, temporal_avg_factor)
    else:
        spatiotemporal_pooled_subset = spatial_pooled_subset

    spatiotemporal_pooled_subset_r = spatiotemporal_pooled_subset.reshape((-1, spatiotemporal_pooled_subset.shape[2]))
    lowres_spatial_basis_r, _, _ = truncated_random_svd(spatiotemporal_pooled_subset_r,
                                                        max_components,
                                                        device=device)

    temporal_projection_from_downsample = lowres_spatial_basis_r.T @ spatial_pooled_subset_r
    if temporal_denoiser is not None:
        temporal_projection_from_downsample = temporal_denoiser(temporal_projection_from_downsample)

    temporal_basis_from_downsample = torch.linalg.svd(temporal_projection_from_downsample, full_matrices=False)[2]
    subset_weighted_r = subset_weighted.reshape((-1, subset_weighted.shape[2]))
    spatial_basis_fullres = subset_weighted_r @ temporal_basis_from_downsample.T

    if spatial_denoiser is not None:
        spatial_basis_fullres = spatial_basis_fullres.reshape((fov_dim1, fov_dim2, -1))
        spatial_basis_fullres = spatial_denoiser(spatial_basis_fullres)
        spatial_basis_fullres = spatial_basis_fullres.reshape((fov_dim1 * fov_dim2, -1))

    spatial_basis_orthogonal, _, _ = torch.linalg.svd(spatial_basis_fullres, full_matrices=False)

    # Regress the original (unweighted) data onto this basis
    subset_r = subset.reshape((-1, subset.shape[2]))
    final_temporal_projection = spatial_basis_orthogonal.T @ subset_r
    left, sing, right = torch.linalg.svd(final_temporal_projection, full_matrices=False)
    local_spatial_basis = (spatial_basis_orthogonal @ left).reshape((fov_dim1, fov_dim2, -1))
    local_temporal_basis = sing[:, None] * right

    return local_spatial_basis, local_temporal_basis


def blockwise_decomposition_with_rank_selection(video_subset: torch.tensor,
                                                full_fov_spatial_basis: torch.tensor,
                                                full_fov_temporal_basis: torch.tensor,
                                                subset_mean: torch.tensor,
                                                subset_noise_variance: torch.tensor,
                                                subset_pixel_weighting: torch.tensor,
                                                max_components: int,
                                                max_consecutive_failures: int,
                                                spatial_roughness_threshold: float,
                                                temporal_roughness_threshold: float,
                                                spatial_avg_factor: int,
                                                temporal_avg_factor: int,
                                                dtype: torch.dtype,
                                                spatial_denoiser: Optional[Callable] = None,
                                                temporal_denoiser: Optional[Callable] = None,
                                                device: str = "cpu"):
    local_spatial_basis, local_temporal_basis = blockwise_decomposition(video_subset,
                                                                        full_fov_spatial_basis,
                                                                        full_fov_temporal_basis,
                                                                        subset_mean,
                                                                        subset_noise_variance,
                                                                        subset_pixel_weighting,
                                                                        max_components,
                                                                        spatial_avg_factor,
                                                                        temporal_avg_factor,
                                                                        dtype,
                                                                        spatial_denoiser=spatial_denoiser,
                                                                        temporal_denoiser=temporal_denoiser,
                                                                        device=device)

    decisions = evaluate_fitness(local_spatial_basis,
                                 local_temporal_basis,
                                 spatial_roughness_threshold,
                                 temporal_roughness_threshold)

    decisions = filter_by_failures(decisions, max_consecutive_failures)
    return local_spatial_basis[:, :, decisions], local_temporal_basis[decisions, :]


def threshold_heuristic(
        dimensions: tuple[int, int, int],
        spatial_avg_factor: int,
        temporal_avg_factor: int,
        spatial_denoiser: Callable,
        temporal_denoiser: Callable,
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
            decompose. Contains (d1, d2, T), the two spatial field of view dimensions and the number of frames
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

    d1, d2, t = dimensions
    sim_mean = torch.zeros((d1, d2), device=device, dtype=dtype)
    sim_noise_normalizer = torch.ones((d1, d2), device=device, dtype=dtype)
    full_fov_spatial_basis = torch.zeros((d1, d2, 1), device=device, dtype=dtype)
    full_fov_temporal_basis = torch.zeros((1, t), device=device, dtype=dtype)
    pixel_weighting = torch.ones((d1, d2), device=device, dtype=dtype)
    max_components = num_comps

    for k in tqdm(range(iters)):
        sim_data = torch.randn(t, d1 * d2, device=device, dtype=dtype).reshape((t, d1, d2))

        spatial, temporal = blockwise_decomposition(sim_data,
                                                    full_fov_spatial_basis,
                                                    full_fov_temporal_basis,
                                                    sim_mean,
                                                    sim_noise_normalizer,
                                                    pixel_weighting,
                                                    max_components,
                                                    spatial_avg_factor,
                                                    temporal_avg_factor,
                                                    dtype,
                                                    spatial_denoiser=spatial_denoiser,
                                                    temporal_denoiser=temporal_denoiser,
                                                    device=device)

        spatial_stat = spatial_roughness_statistic(spatial)
        temporal_stat = temporal_roughness_statistic(temporal)
        spatial_list.append(spatial_stat)
        temporal_list.append(temporal_stat)

    spatial_list = torch.concatenate(spatial_list, dim=0).cpu().numpy()
    temporal_list = torch.concatenate(temporal_list, dim=0).cpu().numpy()

    spatial_threshold = np.percentile(
        spatial_list.flatten(), percentile_threshold)
    temporal_threshold = np.percentile(
        temporal_list.flatten(), percentile_threshold
    )
    return spatial_threshold, temporal_threshold


def pmd_decomposition(
        dataset: LazyFrameLoader,
        block_sizes: tuple[int, int],
        frame_range: int,
        max_components: int = 50,
        background_rank: int = 15,
        sim_conf: int = 5,
        frame_batch_size: int = 10000,
        max_consecutive_failures=1,
        spatial_avg_factor: int = 1,
        temporal_avg_factor: int = 1,
        window_chunks: Optional[int] = None,
        compute_normalizer: bool = True,
        pixel_weighting: Optional[np.ndarray] = None,
        spatial_denoiser: Optional[Callable] = None,
        temporal_denoiser: Optional[Callable] = None,
        device: str = "cpu"
) -> PMDArray:
    """
    General PMD Compression method
    Args:
        dataset (masknmf.LazyFrameLoader): An array-like object with shape (frames, fov_dim1, fov_dim2) that loads frames of raw data
        block_sizes (tuple[int, int]): The block sizes of the compression. Cannot be smaller than 10 in each dimension.
        frame_range (int): Number of frames or raw data used to fit the spatial basis.
            KEY: We assume that your system can store this many frames of raw data in RAM.
        max_components (int): Max number of components we use to decompose any individual spatial block of the raw data.
        background_rank (int): Before doing spatial blockwise decompositions, we estimate a full FOV truncated SVD; this often helps estimate global background trends
            before blockwise decompositions; leading to better compression.
        sim_conf (int): The percentile value used to define spatial and temporal roughness thresholds for rejecting/keeping SVD components
        frame_batch_size (int): The maximum number of frames we load onto the computational device (CPU or GPU) at any point in time.
        max_consecutive_failures (int): In each blockwise decomposition, we stop accepting SVD components after we see this many "bad" components.
        spatial_avg_factor (int): In the blockwise decompositions, we can spatially downsample the data to estimate a cleaner temporal basis. We can use this to iteratively estimate a better
            full-resolution basis for the data. If signal sits on only a few pixels, keep this parameter at 1 (spatially downsampling is undesirable in this case).
        temporaL_avg_factor (int): In the blockwise decompositions, we can temporally downsample the data to estimate a cleaner spatial basis. We can use this to iteratively estimate a better
            full-resolution basis for the data. If your signal "events" are very sparse (i.e. every event appears for only 1 frame) keep this parameter at 1 (temporal downsampling is undesirable in this case).
         window_chunks (int): To be removed
         compute_normalizer (bool): Whether or not we estimate a pixelwise noise variance. If False, the normalizer is set to 1 (no normalization).
         pixel_weighting (Optional[np.ndarray]): Shape (fov_dim1, fov_dim2). We weight the data by this value to estimate a cleaner spatial basis. The pixel_weighting
            should intuitively boost the relative variance of pixels containing signal to those that do not contain signal.
        spatial_denoiser (Optional[Callable]): A function that operates on (height, width, num_components)-shaped images, denoising each of the images.
        temporal_denoiser (Optional[Callable]): A function that operates on (num_components, num_frames)-shaped traces, denoising each of the traces.
        device (str): Which device the computations should be performed on. Options: "cuda" or "cpu".

    Returns:
        pmd_arr (masknmf.PMDArray): A PMD Array object capturing the compression results.
    """
    display("Starting compression")
    num_frames, fov_dim1, fov_dim2 = dataset.shape
    if frame_batch_size < 1024:
        raise ValueError(f"frame_batch_size is too small ({frame_batch_size}). "
                         f"Please set it to at 1024. Your device (cpu or cuda) must have space for at least this much data")
    background_rank_limit = 30
    if background_rank > background_rank_limit:
        raise ValueError(f"Background rank is too large. Should not be larger than {background_rank_limit}")
    dtype = torch.float32  #This is the target dtype we use for doing computations
    check_fov_size((dataset.shape[1], dataset.shape[2]))

    if window_chunks is None:
        window_chunks = frame_range
    # Decide which chunks of the data you will use for the spatial PMD blockwise fits
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
    display("We are initializing on a total of {} frames".format(len(frames)))

    block_sizes = update_block_sizes(
        block_sizes, (dataset.shape[1], dataset.shape[2])
    )

    overlap = [math.ceil(block_sizes[0] / 2), math.ceil(block_sizes[1] / 2)]

    dataset_mean, dataset_noise_variance = compute_mean_and_normalizer_dataset(dataset,
                                                                               compute_normalizer,
                                                                               frame_batch_size,
                                                                               device,
                                                                               dtype)

    dataset_mean = dataset_mean.to(device).to(dtype)
    dataset_noise_variance = dataset_noise_variance.to(device).to(dtype)

    display("Approximating full FOV basis terms")
    full_fov_spatial_basis = compute_full_fov_spatial_basis(dataset,
                                                            dataset_mean,
                                                            dataset_noise_variance,
                                                            background_rank,
                                                            dtype=dtype,
                                                            device=device,
                                                            num_samples=1000)

    full_fov_spatial_basis = full_fov_spatial_basis.to(device).to(dtype)

    display("Loading data to estimate complete spatial basis")

    #First make sure the number of frames loaded is divisible by the temporal average factor
    if temporal_avg_factor >= len(frames):
        raise ValueError("Need at least {} frames".format(temporal_avg_factor))

    frame_cutoff = (len(frames) // temporal_avg_factor) * temporal_avg_factor
    frames = frames[:frame_cutoff]
    if len(frames) // temporal_avg_factor <= max_components:
        string_to_disp = (
            f"WARNING: temporal avg factor is too big, max rank per block adjusted to {len(frames) // temporal_avg_factor}.\n"
            "To avoid this, initialize with more frames or reduce temporal avg factor"
        )
        display(string_to_disp)
        max_components = int(len(frames) // temporal_avg_factor)

    data_for_spatial_fit = torch.from_numpy(dataset[frames])
    if frame_batch_size >= data_for_spatial_fit.shape[0]:
        data_for_spatial_fit = data_for_spatial_fit.to(device).to(dtype)

    full_fov_temporal_basis = compute_full_fov_temporal_basis(data_for_spatial_fit,
                                                              dataset_mean,
                                                              dataset_noise_variance,
                                                              full_fov_spatial_basis,
                                                              dtype,
                                                              frame_batch_size,
                                                              device=device)

    if pixel_weighting is None:
        pixel_weighting = torch.ones((fov_dim1, fov_dim2), device=device, dtype=dtype)
    else:
        pixel_weighting = torch.from_numpy(pixel_weighting).to(device).to(dtype)

    ## Define
    dim_1_iters = list(
        range(0, data_for_spatial_fit.shape[1] - block_sizes[0] + 1, block_sizes[0] - overlap[0])
    )
    if (
            dim_1_iters[-1] != data_for_spatial_fit.shape[1] - block_sizes[0]
            and data_for_spatial_fit.shape[1] - block_sizes[0] != 0
    ):
        dim_1_iters.append(data_for_spatial_fit.shape[1] - block_sizes[0])

    dim_2_iters = list(
        range(0, data_for_spatial_fit.shape[2] - block_sizes[1] + 1, block_sizes[1] - overlap[1])
    )
    if (
            dim_2_iters[-1] != data_for_spatial_fit.shape[2] - block_sizes[1]
            and data_for_spatial_fit.shape[2] - block_sizes[1] != 0
    ):
        dim_2_iters.append(data_for_spatial_fit.shape[2] - block_sizes[1])

    # Define the block weighting matrix
    block_weights = np.ones((block_sizes[0], block_sizes[1]), dtype=np.float32)
    hbh = block_sizes[0] // 2
    hbw = block_sizes[1] // 2
    # Increase weights to value block centers more than edges
    block_weights[:hbh, :hbw] += np.minimum(
        np.tile(np.arange(0, hbw), (hbh, 1)), np.tile(np.arange(0, hbh), (hbw, 1)).T
    )
    block_weights[:hbh, hbw:] = np.fliplr(block_weights[:hbh, :hbw])
    block_weights[hbh:, :] = np.flipud(block_weights[:hbh, :])
    block_weights = torch.from_numpy(block_weights).to(device).to(dtype)

    sparse_indices = torch.arange(fov_dim1 * fov_dim2, dtype=torch.long, device=device).reshape(
        (fov_dim1, fov_dim2))

    column_number = 0
    final_row_indices = []
    final_column_indices = []
    spatial_overall_values = []
    spatial_overall_unweighted_values = []
    cumulative_weights = torch.zeros((fov_dim1, fov_dim2), dtype=dtype, device=device)
    total_temporal_fit = []

    display("Finding spatiotemporal roughness thresholds")
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
        device=device)

    display("Running Blockwise Decompositions")
    for k in dim_1_iters:
        for j in dim_2_iters:
            slice_dim1 = slice(k, k + block_sizes[0])
            slice_dim2 = slice(j, j + block_sizes[1])
            unweighted_local_spatial_basis, local_temporal_basis = blockwise_decomposition_with_rank_selection(
                data_for_spatial_fit[:, slice_dim1, slice_dim2],
                full_fov_spatial_basis[slice_dim1, slice_dim2, :],
                full_fov_temporal_basis,
                dataset_mean[slice_dim1, slice_dim2],
                dataset_noise_variance[slice_dim1, slice_dim2],
                pixel_weighting[slice_dim1, slice_dim2],
                max_components,
                max_consecutive_failures,
                spatial_roughness_threshold,
                temporal_roughness_threshold,
                spatial_avg_factor,
                temporal_avg_factor,
                dtype,
                spatial_denoiser=spatial_denoiser,
                temporal_denoiser=temporal_denoiser,
                device=device)

            total_temporal_fit.append(local_temporal_basis)

            # Weight the spatial components here

            local_spatial_basis = unweighted_local_spatial_basis * block_weights[:, :, None]
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

            curr_spatial_col_indices = torch.zeros_like(curr_spatial_row_indices, device=device)
            addend = torch.arange(column_number, column_number + local_spatial_basis.shape[2], device=device,
                                  dtype=torch.long)[
                     None, None, :
                     ]
            curr_spatial_col_indices = curr_spatial_col_indices + addend

            sparse_row_indices_f = curr_spatial_row_indices.flatten()
            sparse_col_indices_f = curr_spatial_col_indices.flatten()
            spatial_values_f = local_spatial_basis.flatten()

            final_row_indices.append(sparse_row_indices_f)
            final_column_indices.append(sparse_col_indices_f)
            spatial_overall_values.append(spatial_values_f)
            spatial_overall_unweighted_values.append(unweighted_local_spatial_basis.flatten())
            column_number += local_spatial_basis.shape[2]

    #Construct the U matrix up to this point
    final_row_indices = torch.concatenate(final_row_indices, dim=0)
    final_column_indices = torch.concatenate(final_column_indices, dim=0)
    spatial_overall_values = torch.concatenate(spatial_overall_values, dim=0)
    final_indices = torch.stack([final_row_indices, final_column_indices], dim=0)

    spatial_overall_unweighted_values = torch.concatenate(spatial_overall_unweighted_values, dim=0)
    u_spatial_fit = torch.sparse_coo_tensor(final_indices, spatial_overall_unweighted_values,
                                            (fov_dim1 * fov_dim2, column_number)).coalesce()

    if data_for_spatial_fit.shape[0] < dataset.shape[0]:
        display("Regressing the full dataset onto the learned spatial basis")
        v_regression, full_dataset_temporal_basis = regress_onto_spatial_basis(dataset, u_spatial_fit, frame_batch_size,
                                                                               dataset_mean, dataset_noise_variance,
                                                                               full_fov_spatial_basis, dtype, device)
    else:
        v_regression = torch.concatenate(total_temporal_fit, dim=0)
        full_dataset_temporal_basis = full_fov_temporal_basis

    interpolation_weightings = torch.reciprocal(cumulative_weights.flatten()[final_row_indices])
    spatial_overall_values *= interpolation_weightings

    ## Now add the full FOV spatial background
    v_aggregated = [v_regression]
    if background_rank <= 0:
        num_cols = column_number
        num_rows = fov_dim1 * fov_dim2
        u_global_projector = None
        u_local_projector = torch.sparse_coo_tensor(final_indices,
                                                    spatial_overall_unweighted_values,
                                                    (num_rows, num_cols))
    else:
        background_sparse_row_indices = sparse_indices[:, :, None] + torch.zeros_like(full_fov_spatial_basis,
                                                                                      device=device, dtype=torch.long)
        column_range = torch.arange(column_number, column_number + background_rank, device=device, dtype=torch.long)[
                       None, None, :]
        num_cols = column_number + background_rank
        num_rows = fov_dim1 * fov_dim2
        background_sparse_column_indices = torch.zeros((sparse_indices.shape[0], sparse_indices.shape[1], 1),
                                                       device=device, dtype=torch.long) + column_range

        background_sparse_column_indices_standalone = background_sparse_column_indices - column_number

        u_global_projector = torch.sparse_coo_tensor(torch.stack([background_sparse_row_indices.flatten(),
                                                                  background_sparse_column_indices_standalone.flatten()],
                                                                 dim=0),
                                                     full_fov_spatial_basis.flatten(),
                                                     (num_rows, background_rank))

        u_local_projector = torch.sparse_coo_tensor(torch.stack([final_row_indices,
                                                                 final_column_indices], dim=0),
                                                    spatial_overall_unweighted_values,
                                                    (num_rows, column_number))

        final_row_indices = torch.concatenate([final_row_indices, background_sparse_row_indices.flatten()], dim=0)
        final_column_indices = torch.concatenate([final_column_indices, background_sparse_column_indices.flatten()],
                                                 dim=0)
        spatial_overall_values = torch.concatenate([spatial_overall_values, full_fov_spatial_basis.flatten()], dim=0)

        v_aggregated.append(full_dataset_temporal_basis)

    final_indices = torch.stack([final_row_indices, final_column_indices], dim=0)
    u_aggregated = torch.sparse_coo_tensor(final_indices, spatial_overall_values, (num_rows, num_cols))
    v_aggregated = torch.concatenate(v_aggregated, dim=0)
    display(f"Constructed U matrix. Rank of U is {u_aggregated.shape[1]}")

    final_pmd_arr = PMDArray((num_frames, fov_dim1, fov_dim2),
                             u_aggregated.cpu(),
                             v_aggregated.cpu(),
                             dataset_mean,
                             dataset_noise_variance,
                             u_local_projector=u_local_projector.cpu() if u_local_projector is not None else None,
                             u_global_projector=u_global_projector.cpu() if u_global_projector is not None else None,
                             device="cpu")
    display("PMD Objected constructed")
    return final_pmd_arr


def pmd_batch(dataset: LazyFrameLoader,
              batch_dimensions: tuple[int, int],
              batch_overlaps: tuple[int, int],
              block_sizes: tuple[int, int],
              frame_range: int,
              max_components: int = 50,
              background_rank: int = 15,
              sim_conf: int = 5,
              frame_batch_size: int = 10000,
              max_consecutive_failures=1,
              spatial_avg_factor: int = 1,
              temporal_avg_factor: int = 1,
              window_chunks: Optional[int] = None,
              compute_normalizer: bool = True,
              pixel_weighting: Optional[np.ndarray] = None,
              spatial_denoiser: Optional[Callable] = None,
              temporal_denoiser: Optional[Callable] = None,
              device: str = "cpu"
              ) -> PMDArray:
    """
    Method for running PMD across huge fields of view. This method breaks the large FOV into smaller regions (say, 300 x 300 pixels),
    runs PMD on each smaller region, and pieces the results together to get a fullFOV decomposition.

    Args:
        dataset (masknmf.LazyFrameLoader): An array-like object with shape (frames, fov_dim1, fov_dim2) that loads frames of raw data
        block_sizes (tuple[int, int]): The block sizes of the compression. Cannot be smaller than 10 in each dimension.
        frame_range (int): Number of frames or raw data used to fit the spatial basis.
            KEY: We assume that your system can store this many frames of raw data in RAM.
        max_components (int): Max number of components we use to decompose any individual spatial block of the raw data.
        background_rank (int): Before doing spatial blockwise decompositions, we estimate a full FOV truncated SVD; this often helps estimate global background trends
            before blockwise decompositions; leading to better compression.
        sim_conf (int): The percentile value used to define spatial and temporal roughness thresholds for rejecting/keeping SVD components
        frame_batch_size (int): The maximum number of frames we load onto the computational device (CPU or GPU) at any point in time.
        max_consecutive_failures (int): In each blockwise decomposition, we stop accepting SVD components after we see this many "bad" components.
        spatial_avg_factor (int): In the blockwise decompositions, we can spatially downsample the data to estimate a cleaner temporal basis. We can use this to iteratively estimate a better
            full-resolution basis for the data. If signal sits on only a few pixels, keep this parameter at 1 (spatially downsampling is undesirable in this case).
        temporaL_avg_factor (int): In the blockwise decompositions, we can temporally downsample the data to estimate a cleaner spatial basis. We can use this to iteratively estimate a better
            full-resolution basis for the data. If your signal "events" are very sparse (i.e. every event appears for only 1 frame) keep this parameter at 1 (temporal downsampling is undesirable in this case).
         window_chunks (int): To be removed
         compute_normalizer (bool): Whether or not we estimate a pixelwise noise variance. If False, the normalizer is set to 1 (no normalization).
         pixel_weighting (Optional[np.ndarray]): Shape (fov_dim1, fov_dim2). We weight the data by this value to estimate a cleaner spatial basis. The pixel_weighting
            should intuitively boost the relative variance of pixels containing signal to those that do not contain signal.
        spatial_denoiser (Optional[Callable]): A function that operates on (height, width, num_components)-shaped images, denoising each of the images.
        temporal_denoiser (Optional[Callable]): A function that operates on (num_components, num_frames)-shaped traces, denoising each of the traces.
        device (str): Which device the computations should be performed on. Options: "cuda" or "cpu".

    Returns:
        pmd_arr (PMDArray): A PMDArray capturing the compression results across the full FOV.
    """

    """
    How do we interpolate and stitch together the PMD patches? 
    (1) The u_local_projector and u_global_projector terms need to have their row indices modified. All other
    concatenation follows easily
    (2) We should "globally" weight each 300 x 300 patch with an interpolation function. This is important for
    stitching together actual U matrices. 
    (3) Stacking V is easy. 
    
    Workflow:
    (1) Just run PMD across all the sub-patches of the data. Return a list of PMD objects. 
    (2) Once that works, add the weighting functions, combine everything into one file. 
    (3) Once that works, think about the optimal way to do dataloading. 
    
    
    Design decision: if you have a huge FOV, it does not make sense to keep all of U on the GPU. So moving off GPU is good.
    
    Things to circle back on: 
        Make sure the dataloader is being told to read the right stuff
        The blockwise iteration scheme creates a lot of redundancy at the boundaries right now. Should improve that. 
    """
    num_frames, fov_dim1, fov_dim2 = dataset.shape

    spatial_dim1_start_pts = list(
        range(0, dataset.shape[1] - batch_dimensions[0] + 1, batch_dimensions[0] - batch_overlaps[0])
    )
    if (
            spatial_dim1_start_pts[-1] != dataset.shape[1] - batch_dimensions[0]
            and dataset.shape[1] - batch_dimensions[0] > 0
    ):
        spatial_dim1_start_pts.append(dataset.shape[1] - batch_dimensions[0])

    spatial_dim2_start_pts = list(
        range(0, dataset.shape[2] - batch_dimensions[1] + 1, batch_dimensions[1] - batch_overlaps[1])
    )
    if (
            spatial_dim2_start_pts[-1] != dataset.shape[2] - batch_dimensions[1]
            and dataset.shape[2] - batch_dimensions[1] > 0
    ):
        spatial_dim2_start_pts.append(dataset.shape[2] - batch_dimensions[1])

    #Collect all the spatial information into the following lists
    #First pass: don't worry at all about
    u_row_indices = []
    u_col_indices = []
    u_values = []
    u_local_projector_row_indices = []
    u_local_projector_col_indices = []
    u_local_projector_values = []
    u_global_projector_row_indices = [] if background_rank > 0 else None
    u_global_projector_col_indices = [] if background_rank > 0 else None
    u_global_projector_values = [] if background_rank > 0 else None

    v_aggregate = []

    dataset_mean_img = torch.zeros(fov_dim1, fov_dim2, device="cpu", dtype=torch.float)
    dataset_variance_img = torch.zeros(fov_dim1, fov_dim2, device="cpu", dtype=torch.float)
    pixel_mat = torch.arange(dataset.shape[1] * dataset.shape[2],
                             device="cpu", dtype=torch.long).reshape(dataset.shape[1], dataset.shape[2])

    rank_tracker = 0
    for i in range(len(spatial_dim1_start_pts)):
        for j in range(len(spatial_dim2_start_pts)):
            curr_dim1_start_pt = spatial_dim1_start_pts[i]
            curr_dim1_end_pt = spatial_dim1_start_pts[i] + batch_dimensions[0]

            curr_dim2_start_pt = spatial_dim2_start_pts[j]
            curr_dim2_end_pt = spatial_dim2_start_pts[j] + batch_dimensions[1]

            display(f"Processing {curr_dim1_start_pt}:{curr_dim1_end_pt} "
                    f"to {curr_dim2_start_pt}:{curr_dim2_end_pt}")
            dataset_crop = dataset[:,
                           curr_dim1_start_pt:curr_dim1_end_pt,
                           curr_dim2_start_pt:curr_dim2_end_pt]

            current_row_map = pixel_mat[curr_dim1_start_pt:curr_dim1_end_pt,
                              curr_dim2_start_pt:curr_dim2_end_pt].flatten()

            curr_pmd_array = pmd_decomposition(dataset_crop,
                                               block_sizes,
                                               frame_range,
                                               max_components,
                                               background_rank,
                                               sim_conf,
                                               frame_batch_size,
                                               max_consecutive_failures,
                                               spatial_avg_factor,
                                               temporal_avg_factor,
                                               window_chunks=window_chunks,
                                               compute_normalizer=compute_normalizer,
                                               pixel_weighting=pixel_weighting,
                                               spatial_denoiser=spatial_denoiser,
                                               temporal_denoiser=temporal_denoiser,
                                               device=device)

            #Update the dataset mean/variance in this region:
            dataset_mean_img[curr_dim1_start_pt:curr_dim1_end_pt, curr_dim2_start_pt:curr_dim2_end_pt] = curr_pmd_array.mean_img
            dataset_variance_img[curr_dim1_start_pt:curr_dim1_end_pt, curr_dim2_start_pt:curr_dim2_end_pt] = curr_pmd_array.var_img

            computed_rank = curr_pmd_array.u.shape[1]
            v_aggregate.append(curr_pmd_array.v)

            #Process the indices/values of u
            curr_u = curr_pmd_array.u
            curr_u_indices = curr_u.indices()
            curr_u_rows, curr_u_cols = curr_u_indices[0], curr_u_indices[1]
            curr_u_rows = current_row_map[curr_u_rows]
            curr_u_cols += rank_tracker
            u_row_indices.append(curr_u_rows)
            u_col_indices.append(curr_u_cols)
            u_values.append(curr_u.values())

            #Process the indices/values of u_local_proj
            curr_u_local_proj = curr_pmd_array.u_local_projector
            curr_u_local_proj_indices = curr_u_local_proj.indices()
            curr_u_local_proj_rows, curr_u_local_proj_cols = curr_u_local_proj_indices[0], curr_u_local_proj_indices[1]
            curr_u_local_proj_rows = current_row_map[curr_u_local_proj_rows]
            curr_u_local_proj_cols += rank_tracker
            u_local_projector_row_indices.append(curr_u_local_proj_rows)
            u_local_projector_col_indices.append(curr_u_local_proj_cols)
            u_local_projector_values.append(curr_u_local_proj.values())

            if background_rank > 0:
                curr_u_global_proj = curr_pmd_array.u_global_projector
                curr_u_global_proj_indices = curr_u_global_proj.indices()
                curr_u_global_proj_rows, curr_u_global_proj_cols = curr_u_global_proj_indices[0], \
                curr_u_global_proj_indices[1]
                curr_u_global_proj_rows = current_row_map[curr_u_global_proj_rows]
                curr_u_global_proj_cols += rank_tracker
                u_global_projector_row_indices.append(curr_u_global_proj_rows)
                u_global_projector_col_indices.append(curr_u_global_proj_cols)
                u_global_projector_values.append(curr_u_global_proj.values())

            rank_tracker += computed_rank

    v_aggregate = torch.concatenate(v_aggregate, dim = 0)

    u_final_rows = torch.concatenate(u_row_indices, dim = 0)
    u_final_cols = torch.concatenate(u_col_indices, dim = 0)
    u_final_values = torch.concatenate(u_values, dim = 0)

    u_final = torch.sparse_coo_tensor(torch.stack([u_final_rows, u_final_cols], dim = 0),
                                      u_final_values,
                                      (dataset.shape[1] * dataset.shape[2], rank_tracker))

    u_final_local_projector_rows = torch.concatenate(u_local_projector_row_indices, dim = 0)
    u_final_local_projector_cols = torch.concatenate(u_local_projector_col_indices, dim = 0)
    u_final_local_projector_values = torch.concatenate(u_local_projector_values, dim = 0)
    u_local_projector = torch.sparse_coo_tensor(torch.stack([u_final_local_projector_rows, u_final_local_projector_cols], dim=0),
                                                u_final_local_projector_values,
                                                (dataset.shape[1] * dataset.shape[2], rank_tracker))

    if background_rank > 0:
        u_final_global_projector_rows = torch.concatenate(u_global_projector_row_indices, dim=0)
        u_final_global_projector_cols = torch.concatenate(u_global_projector_col_indices, dim=0)
        u_final_global_projector_values = torch.concatenate(u_global_projector_values, dim=0)
        u_global_projector = torch.sparse_coo_tensor(
            torch.stack([u_final_global_projector_rows, u_final_global_projector_cols], dim=0),
            u_final_global_projector_values,
            (dataset.shape[1] * dataset.shape[2], rank_tracker))

    else:
        u_global_projector = None


    return PMDArray((num_frames, fov_dim1, fov_dim2),
                             u_final.cpu(),
                             v_aggregate.cpu(),
                             dataset_mean_img,
                             dataset_variance_img,
                             u_local_projector=u_local_projector.cpu(),
                             u_global_projector=u_global_projector.cpu() if u_global_projector is not None else None,
                             device="cpu")


