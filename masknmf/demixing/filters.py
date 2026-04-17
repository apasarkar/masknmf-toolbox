import numpy as np
from scipy.signal import butter, lfilter, filtfilt
import masknmf
import math
import torch
from tqdm import tqdm
def spatial_filter_pmd(pmd_obj: masknmf.PMDArray,
                       batch_size: int = 200,
                       filter_sigma: int = 3,
                       device: str = 'cpu') -> masknmf.PMDArray:
    if pmd_obj.rescale is False:
        switch = True
        pmd_obj.rescale = True
    else:
        switch = False
    t, d1, d2 = pmd_obj.shape
    hp_filter_kernel = masknmf.motion_correction.spatial_filters.compute_highpass_filter_kernel(
        [filter_sigma, filter_sigma]).to(device)
    num_batches = math.ceil(pmd_obj.shape[0] / batch_size)
    pmd_obj.to(device)
    relu_obj = torch.nn.ReLU()
    results = []
    for k in tqdm(range(num_batches)):
        start = k * batch_size
        end = min(start + batch_size, pmd_obj.shape[0])
        curr_frames = pmd_obj.getitem_tensor(slice(start, end))
        if curr_frames.ndim == 2:
            curr_frames = curr_frames[None, ...]

        filtered_frames = masknmf.motion_correction.spatial_filters.image_filter(curr_frames, hp_filter_kernel)
        filtered_frames = relu_obj(filtered_frames)
        filtered_frames = filtered_frames.permute(1, 2, 0)
        projection = pmd_obj.project_frames(filtered_frames, standardize=False)
        results.append(projection)
    final_v = torch.cat(results, dim=1)

    new_mean = torch.sparse.mm(pmd_obj.u.to(device), torch.mean(final_v.to(device), dim=1, keepdim = True))

    new_mean = new_mean.reshape(d1, d2)
    final_v -= torch.mean(final_v, dim=1, keepdim=True)
    final_arr = masknmf.PMDArray.from_tensors(pmd_obj.shape,
                                 pmd_obj.u.to(device),
                                 final_v.to(device),
                                 new_mean.to(device),
                                 torch.ones_like(new_mean),
                                 u_local_projector=pmd_obj.u_local_projector,
                                 device='cpu')

    if switch:
        pmd_obj.rescale = False

    return final_arr


def detrend_pmd(pmd_obj: masknmf.PMDArray,
                n_knots: int = 10) -> torch.Tensor:
    """
    Remove spline baseline from V using natural cubic spline basis.

    Args:
        V:       (rank, T) temporal components
        n_knots: number of knots (evenly spaced)

    Returns:
        V_detrended: (rank, T)
    """
    V = pmd_obj.v
    T = V.shape[1]
    t = torch.linspace(0, 1, T, device=V.device, dtype=V.dtype)
    knots = torch.linspace(0, 1, n_knots, device=V.device, dtype=V.dtype)

    # --- Build truncated power spline basis (T, n_knots + 4) ---
    # Cubic polynomial part
    poly_basis = torch.stack([t ** d for d in range(4)], dim=1)  # (T, 4)

    # Truncated cubic terms for each interior knot: max(t - knot, 0)^3
    spline_basis = torch.stack(
        [torch.clamp(t - k, min=0.0) ** 3 for k in knots], dim=1
    )  # (T, n_knots)

    A = torch.cat([poly_basis, spline_basis], dim=1)  # (T, 4 + n_knots)

    # --- Fit all rank traces simultaneously ---
    coeffs = torch.linalg.lstsq(A, V.T).solution  # (4 + n_knots, rank)

    baseline = A @ coeffs  # (T, rank)
    final_v = V - baseline.T  # (rank, T)

    mean = torch.sparse.mm(pmd_obj.u, torch.mean(final_v, dim = 1, keepdims=True))
    new_mean = mean.reshape(pmd_obj.shape[1], pmd_obj.shape[2])
    final_v -= torch.mean(final_v, dim = 1, keepdims=True)

    device = pmd_obj.device
    final_arr = masknmf.PMDArray.from_tensors(pmd_obj.shape,
                                              pmd_obj.u.to(device),
                                              final_v.to(device),
                                              new_mean.to(device),
                                              torch.ones_like(new_mean),
                                              u_local_projector=pmd_obj.u_local_projector,
                                              device=device)

    return final_arr




##Define the filtering operation
def high_pass_filter(data: np.ndarray,
                     cutoff: float,
                     sampling_rate: float, order=5):
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
                       sampling_rate: float):
    """
    Runs a high pass filter on all rows of a matrix

    Args:
        temporal_matrix (np.ndarray): Shape (PMD Rank, Number of Frames). PMD temporal basis
        cutoff (float): The frequency cutoff in hertz
        sampling_rate (float): The sampling rate of the data

    Returns:
        temporal_hp (np.ndarray): Shape (PMD Rank, Number of Frames). High-pass filtered matrix
    """
    temporal_hp = np.zeros_like(temporal_matrix)

    for k in range(temporal_matrix.shape[0]):
        temporal_hp[k, :] = high_pass_filter(temporal_matrix[k, :], cutoff, sampling_rate)
    return temporal_hp