import numpy as np
from scipy.signal import butter, lfilter, filtfilt
import masknmf
import math
import torch
from tqdm import tqdm

def construct_gaussian_highpass_filter_kernel(gaussian_sigma: list[float]) -> torch.Tensor:
    """
    Computes a high-pass filter kernel using a Gaussian filter. The Kernel is I - Gauss(sigma)

    Args:
        gaussian_sigma (list[int]): Standard deviations for the Gaussian kernel.

    Returns:
        torch.Tensor: High-pass filter kernel.
    """

    if len(gaussian_sigma) != 2:
        raise ValueError("gaussian_sigma must have length 2")

    if any(s <= 0 for s in gaussian_sigma):
        raise ValueError("gaussian_sigma must contain positive values")

    sigma_h, sigma_w = gaussian_sigma

    radius_h = int(3 * sigma_h)
    radius_w = int(3 * sigma_w)

    coords_h = torch.arange(-radius_h, radius_h + 1, dtype=torch.float32)
    coords_w = torch.arange(-radius_w, radius_w + 1, dtype=torch.float32)

    g_h = torch.exp(-0.5 * (coords_h ** 2) / (sigma_h ** 2))
    g_w = torch.exp(-0.5 * (coords_w ** 2) / (sigma_w ** 2))

    kernel = g_h[:, None] @ g_w[None, :]
    kernel /= kernel.sum()

    kernel = -kernel
    kernel[radius_h, radius_w] += 1.0

    return kernel

def spatial_filter_compressed_array(compression_array: masknmf.CompressionArray,
                                    batch_size: int = 200,
                                    filter_sigma: int = 3,
                                    target_device: torch.device | str = 'cpu') -> masknmf.CompressionArray:

    #We can change the state of the below compression array without any issue
    compression_array = masknmf.CompressionArray.from_flyweight(compression_array.shape,
                                                                compression_array.flyweight,
                                                                rescale = True,
                                                                include_trend = False)
    device = compression_array.device
    num_frames, fov_height, fov_width = compression_array.shape
    hp_filter_kernel = construct_gaussian_highpass_filter_kernel(
        [filter_sigma, filter_sigma]).to(device)
    num_batches = math.ceil(compression_array.shape[0] / batch_size)
    relu_obj = torch.nn.ReLU()
    results = []
    for k in tqdm(range(num_batches)):
        start = k * batch_size
        end = min(start + batch_size, compression_array.shape[0])
        curr_frames = compression_array.getitem_tensor(slice(start, end))
        if curr_frames.ndim == 2:
            curr_frames = curr_frames[None, ...]

        filtered_frames = masknmf.motion_correction.spatial_filters.image_filter(curr_frames, hp_filter_kernel)
        filtered_frames = relu_obj(filtered_frames)
        filtered_frames = filtered_frames.permute(1, 2, 0)
        projection = compression_array.project_frames(filtered_frames, standardize=False)
        results.append(projection.to(target_device))
    compression_array.to(target_device)
    final_temporal_compressed = torch.cat(results, dim=1).to(target_device)

    new_mean = torch.sparse.mm(compression_array.spatial_compressed, torch.mean(final_temporal_compressed.to(target_device), dim=1, keepdim = True))
    new_mean = new_mean.reshape(fov_height, fov_width)
    final_temporal_compressed -= torch.mean(final_temporal_compressed, dim=1, keepdim=True)

    final_arr = masknmf.CompressionArray.from_tensors(compression_array.shape,
                                                      compression_array.spatial_compressed,
                                                      final_temporal_compressed,
                                                      new_mean,
                                                      torch.ones_like(new_mean),
                                                      spatial_compressed_local_projector=compression_array.spatial_compressed_local_projector,
                                                      device=target_device)

    return final_arr


def truncated_random_svd_compressed_array(
    spatial_compressed: torch.Tensor,
    temporal_compressed: torch.Tensor,
    rank: int,
    num_oversamples: int = 5,
    device: str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Randomized SVD of F = spatial_compressed @ temporal_compressed without materializing the full dataset.

    The random projection must act on F, not just temporal_compressed, so we thread spatial_compressed
    through both the forward and adjoint passes.

    Returns global_spatial_basis (num_pixels, rank), singular_values (rank,), right_singular_vectors (rank, num_frames)
    where F ≈ (spatial_compressed @ global_spatial_basis) @ diag(singular_values) @ right_singular_vectors
    """
    compression_rank, num_frames = temporal_compressed.shape

    omega = torch.randn(num_frames, rank + num_oversamples, device=device)
    y = torch.sparse.mm(spatial_compressed, temporal_compressed @ omega)

    q, _ = torch.linalg.qr(y, mode="reduced")  # (num_pixels, compression_rank + oversamples)

    qt_spatial_compressed = torch.sparse.mm(spatial_compressed.T, q).T            # (rank+os, compression_rank)
    b = qt_spatial_compressed @ temporal_compressed                                 # (rank+os, num_frames)

    left_singular_vectors, singular_values, right_singular_vectors = torch.linalg.svd(b, full_matrices=False)

    global_spatial_basis = q @ left_singular_vectors                        # (num_pixels, rank+os)

    return global_spatial_basis[:, :rank], singular_values[:rank], right_singular_vectors[:rank, :]


def filter_global_signal_compression_array(
    compression_array: masknmf.CompressionArray,
    rank: int = 3,
    num_oversamples: int = 5,
) -> tuple[masknmf.CompressionArray, torch.Tensor, torch.Tensor, torch.Tensor]:

    device = compression_array.device
    spatial_compressed = compression_array.spatial_compressed         # (num_pixels, compression_rank)
    temporal_compressed = compression_array.temporal_compressed          # (compression_rank, num_frames), dense

    global_spatial_basis, singular_values, right_singular_vectors = truncated_random_svd_compressed_array(spatial_compressed, temporal_compressed, rank, num_oversamples, device)

    # --- Global signal in pixel space, multiplication order exploits low-rank structure ---
    temporal_compressed_global = torch.sparse.mm(
        compression_array.spatial_compressed_local_projector.T, global_spatial_basis * singular_values[None, :]
    ) @ right_singular_vectors

    temporal_compressed_global_subtracted = temporal_compressed - temporal_compressed_global

    temporal_compressed_global_subtracted -= torch.mean(temporal_compressed_global_subtracted, dim = 1, keepdims=True)

    # --- Build residual CompressionArray (spatial_compressed unchanged, temporal_compressed replaced) ---
    num_frames, fov_height, fov_width = compression_array.shape
    new_mean = torch.zeros(fov_height, fov_width, device=device)
    residual_compression_array = masknmf.CompressionArray.from_tensors(
        compression_array.shape,
        spatial_compressed,
        temporal_compressed_global_subtracted,
        new_mean,
        torch.ones_like(new_mean),
        spatial_compressed_local_projector=compression_array.spatial_compressed_local_projector,
        device="cpu",
    )
    return residual_compression_array

##Define the filtering operation
def high_pass_filter(data: np.ndarray,
                     cutoff: float,
                     sampling_rate: float,
                     order=5) -> np.ndarray:
    """
    data (np.ndarray): 1D time series
    cutoff (float): The frequency cutoff in hertz
    sampling_rate (float): The sampling rate of the data
    order (int): Order of the butterworth filter for the sampling rate

    Returns:
        filtered_data (np.ndarray): Shape (T,). 1D high-pass filtered time series
    """
    nyquist = 0.5 * sampling_rate
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    filtered_data = filtfilt(b, a, data)
    return filtered_data


def high_pass_filter_batch(temporal_matrix: np.ndarray,
                           cutoff: float,
                           sampling_rate: float) -> np.ndarray:
    """
    Runs a high pass filter on all rows of a matrix

    Args:
        temporal_matrix (np.ndarray): Shape (Compression Rank, Number of Frames). Compression temporal basis
        cutoff (float): The frequency cutoff in hertz
        sampling_rate (float): The sampling rate of the data

    Returns:
        temporal_hp (np.ndarray): Shape (Compression Rank, Number of Frames). High-pass filtered matrix
    """
    temporal_hp = np.zeros_like(temporal_matrix)

    for k in range(temporal_matrix.shape[0]):
        temporal_hp[k, :] = high_pass_filter(temporal_matrix[k, :], cutoff, sampling_rate)
    return temporal_hp

def bandstop_filter(data: np.ndarray,
                    low_cutoff: float,
                    high_cutoff: float,
                    sampling_rate: float,
                    order: int = 5):
    """
    Args:
        data (np.ndarray): 1D time series
        low_cutoff (float): Lower bound of the stop band in hertz
        high_cutoff (float): Upper bound of the stop band in hertz
        sampling_rate (float): The sampling rate of the data
        order (int): Order of the Butterworth filter

    Returns:
        filtered_data (np.ndarray): Shape (T,). 1D bandstop-filtered time series
    """
    nyquist = 0.5 * sampling_rate
    low = low_cutoff / nyquist
    high = high_cutoff / nyquist
    b, a = butter(order, [low, high], btype='bandstop', analog=False)
    filtered_data = filtfilt(b, a, data)
    return filtered_data

def bandstop_filter_batch(temporal_matrix: np.ndarray,
                          low_cutoff: float,
                          high_cutoff: float,
                          sampling_rate: float,
                          order: int = 5) -> np.ndarray:
    temporal_filtered = np.zeros_like(temporal_matrix)
    for k in range(temporal_matrix.shape[0]):
        temporal_filtered[k, :] = bandstop_filter(temporal_matrix[k, :], low_cutoff, high_cutoff, sampling_rate, order)
    return temporal_filtered

def bandstop_filter_compression_array(compression_array: masknmf.CompressionArray,
                                      low_cutoff: float,
                                      high_cutoff: float,
                                      sampling_rate: float,
                                      order: int = 5) -> masknmf.CompressionArray:
    """
    Apply a bandstop filter to the temporal components of a CompressionArray object.

    Args:
        compression_array (masknmf.CompressionArray): Input CompressionArray object
        low_cutoff (float): Lower bound of the stop band in hertz
        high_cutoff (float): Upper bound of the stop band in hertz
        sampling_rate (float): The sampling rate of the data in hertz
        order (int): Order of the Butterworth filter

    Returns:
        masknmf.CompressionArray: Updated CompressionArray object with bandstop-filtered temporal components
    """
    temporal_compressed = compression_array.temporal_compressed  # (compression_rank, num_frames)

    # Filter on CPU as numpy
    temporal_compressed_numpy = temporal_compressed.cpu().numpy()
    temporal_compressed_filtered = bandstop_filter_batch(temporal_compressed_numpy, low_cutoff, high_cutoff, sampling_rate, order)
    final_temporal_compressed = torch.as_tensor(temporal_compressed_filtered, device=temporal_compressed.device, dtype=temporal_compressed.dtype)

    # Recompute mean image from filtered temporal_compressed, then zero-mean temporal_compressed
    mean = torch.sparse.mm(compression_array.spatial_compressed, torch.mean(final_temporal_compressed, dim=1, keepdim=True))
    new_mean = mean.reshape(compression_array.shape[1], compression_array.shape[2])
    final_temporal_compressed -= torch.mean(final_temporal_compressed, dim=1, keepdim=True)

    device = compression_array.device
    return masknmf.CompressionArray.from_tensors(compression_array.shape,
                                                 compression_array.spatial_compressed.to(device),
                                                 final_temporal_compressed.to(device),
                                                 new_mean.to(device),
                                                 torch.ones_like(new_mean),
                                                 spatial_compressed_local_projector=compression_array.spatial_compressed_local_projector,
                                                 device=device)

