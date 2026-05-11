import numpy as np
from scipy.signal import butter, lfilter, filtfilt
import masknmf
import math
import torch
from tqdm import tqdm
from typing import *


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


def truncated_random_svd_pmd(
    U: torch.Tensor,      # (n_pixels, r), sparse — U_pmd
    V: torch.Tensor,      # (r, T),        dense  — V_pmd
    rank: int,
    num_oversamples: int = 5,
    device: str = "cpu",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Randomized SVD of F = U @ V without materializing (n_pixels, T).

    The random projection must act on F, not just V, so we thread U
    through both the forward and adjoint passes.

    Returns U_svd (r, rank), S (rank,), Vt (rank, T)
    where F ≈ (U @ U_svd) @ diag(S) @ Vt
    """
    r, T = V.shape

    Omega = torch.randn(T, rank + num_oversamples, device=device)
    Y = torch.sparse.mm(U, V @ Omega)          # (n_pixels, rank+os) — only dense op

    Q, _ = torch.linalg.qr(Y, mode="reduced")  # (n_pixels, rank+os)

    QtU = torch.sparse.mm(U.T, Q).T            # (rank+os, r)
    B = QtU @ V                                 # (rank+os, T)

    U_b, S, Vt = torch.linalg.svd(B, full_matrices=False)

    U_svd = Q @ U_b                        # (r, rank+os) — in PMD basis

    return U_svd[:, :rank], S[:rank], Vt[:rank, :]


def filter_global_signal_pmd(
    pmd_obj: masknmf.PMDArray,
    rank: int = 3,
    num_oversamples: int = 5,
    device: str = "cpu",
) -> tuple[masknmf.PMDArray, torch.Tensor, torch.Tensor, torch.Tensor]:

    pmd_obj.to(device)
    U_pmd = pmd_obj.u.to(device)          # (n_pixels, r), sparse
    V_pmd = pmd_obj.v.to(device)          # (r, T), dense

    U_svd, S, Vt = truncated_random_svd_pmd(U_pmd, V_pmd, rank, num_oversamples, device)

    # --- Global signal in pixel space (never densified to n_pixels x T) ---
    # pixel_global = U_pmd @ U_svd @ diag(S)   shape: (n_pixels, rank)
    U_global_pixels = U_svd * S[None, :]  # (n_pixels, rank)

    # --- Project back into PMD basis using u_local_projector ---
    # This is exactly what project_frames does, without standardization
    V_global = torch.sparse.mm(
        pmd_obj.u_local_projector.T, U_global_pixels
    ) @ Vt                                 # (r, rank) @ (rank, T) -> (r, T)

    V_residual = V_pmd - V_global          # (r, T)

    V_residual -= torch.mean(V_residual, dim = 1, keepdims=True)

    # --- Build residual PMDArray (U unchanged, V replaced) ---
    T, H, W = pmd_obj.shape
    new_mean = torch.zeros(H, W, device=device)
    residual_pmd = masknmf.PMDArray.from_tensors(
        pmd_obj.shape,
        U_pmd,
        V_residual,
        new_mean,
        torch.ones_like(new_mean),
        u_local_projector=pmd_obj.u_local_projector,
        device="cpu",
    )
    return residual_pmd

def detrend_pmd(pmd_obj: masknmf.PMDArray,
                n_knots: int = 10) -> torch.Tensor:
    """

    TODO: Remove this now that we detrend before PMD?
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

def bandstop_filter_pmd(pmd_obj: masknmf.PMDArray,
                        low_cutoff: float,
                        high_cutoff: float,
                        sampling_rate: float,
                        order: int = 5) -> masknmf.PMDArray:
    """
    Apply a bandstop filter to the temporal components of a PMD object.

    Args:
        pmd_obj (masknmf.PMDArray): Input PMD object
        low_cutoff (float): Lower bound of the stop band in hertz
        high_cutoff (float): Upper bound of the stop band in hertz
        sampling_rate (float): The sampling rate of the data in hertz
        order (int): Order of the Butterworth filter

    Returns:
        masknmf.PMDArray: Updated PMD object with bandstop-filtered temporal components
    """
    V = pmd_obj.v  # (rank, T)

    # Filter on CPU as numpy
    V_np = V.cpu().numpy()
    V_filtered = bandstop_filter_batch(V_np, low_cutoff, high_cutoff, sampling_rate, order)
    final_v = torch.tensor(V_filtered, device=V.device, dtype=V.dtype)

    # Recompute mean image from filtered V, then zero-mean V
    mean = torch.sparse.mm(pmd_obj.u, torch.mean(final_v, dim=1, keepdim=True))
    new_mean = mean.reshape(pmd_obj.shape[1], pmd_obj.shape[2])
    final_v -= torch.mean(final_v, dim=1, keepdim=True)

    device = pmd_obj.device
    return masknmf.PMDArray.from_tensors(pmd_obj.shape,
                                         pmd_obj.u.to(device),
                                         final_v.to(device),
                                         new_mean.to(device),
                                         torch.ones_like(new_mean),
                                         u_local_projector=pmd_obj.u_local_projector,
                                         device=device)

