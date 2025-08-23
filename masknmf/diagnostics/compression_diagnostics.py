import torch
import numpy as np
import masknmf
from masknmf.compression import PMDArray
from typing import *
import math
from tqdm import tqdm

def compute_general_spatial_correlation_map(
        stack: Union[np.ndarray, masknmf.LazyFrameLoader, masknmf.FactorizedVideo],
        device: str = 'cpu',
        batch_size: int = 200) -> torch.tensor:
    """
    General routine to compute a spatial correlation map for a single stack.
    Args:
        stack (Union[np.ndarray, masknmf.LazyFrameLoader, masknmf.FactorizedVideo]): A (num_frames, fov_dim1, fov_dim2)
            shaped imaging stack
        device (str): Which device we use for computations ('cpu', 'cuda', etc.)
        batch_size (int): The number of frames to process at a time
    Returns:
        (torch.tensor): The spatial correlation map for this stack
    """
    num_frames, fov_dim1, fov_dim2 = stack.shape

    stack_mean = torch.zeros((fov_dim1, fov_dim2), device=device).float()
    stack_std = torch.zeros((fov_dim1, fov_dim2), device=device).float()

    num_iters = math.ceil(num_frames / batch_size)

    for k in tqdm(range(num_iters)):
        start = k * batch_size
        end = min(start + batch_size, num_frames)
        curr_frames = torch.from_numpy(stack[start:end]).to(device).float()
        if curr_frames.ndim == 2:
            curr_frames = curr_frames[None, :, :]
        stack_mean += (torch.sum(curr_frames, dim=0) / num_frames)

    for k in tqdm(range(num_iters)):
        start = k * batch_size
        end = min(start + batch_size, num_frames)
        curr_frames = torch.from_numpy(stack[start:end]).to(device).float()
        curr_frames -= stack_mean[None, :, :]
        if curr_frames.ndim == 2:
            curr_frames = curr_frames[None, :, :]
        stack_std += torch.sum(curr_frames * curr_frames, dim=0) / num_frames

    stack_std = stack_std.clamp(
        min=0.0)  # Easy way to avoid numerical issues with division by values close to 0 in torch
    stack_std = torch.sqrt(stack_std)
    stack_std = stack_std.clamp(
        min=0.0)  # Easy way to avoid numerical issues with division by values close to 0 in torch

    top_left_bottom_right = torch.zeros((1, fov_dim1 - 1, fov_dim2 - 1), device=device).float()
    horizontal = torch.zeros((1, fov_dim1, fov_dim2 - 1), device=device).float()
    vertical = torch.zeros((1, fov_dim1 - 1, fov_dim2), device=device).float()
    top_right_bottom_left = torch.zeros((1, fov_dim1 - 1, fov_dim2 - 1), device=device).float()

    num_iters = math.ceil(num_frames / batch_size)
    for k in tqdm(range(num_iters)):
        start = batch_size * k
        end = min(start + batch_size, num_frames)
        stack_subset = stack[start:end, :, :]

        if stack_subset.ndim == 2:
            stack_subset = stack_subset[None, :, :]
        stack_subset = (torch.from_numpy(stack_subset).to(device).float() - stack_mean[None, :, :])
        stack_subset /= stack_std[None, :, :]
        stack_subset = torch.nan_to_num(stack_subset, nan=0.0)
        top_left_bottom_right[0, :, :] += torch.sum(stack_subset[:, :-1, :-1] *
                                                    stack_subset[:, 1:, 1:], dim=0) / num_frames

        horizontal[0, :, :] += torch.sum(stack_subset[:, :, :-1] *
                                         stack_subset[:, :, 1:], dim=0) / num_frames


        vertical[0, :, :] += torch.sum(stack_subset[:, :-1, :] *
                                       stack_subset[:, 1:, :], dim=0) / num_frames

        top_right_bottom_left[0, :, :] += torch.sum(stack_subset[:, :-1, 1:] *
                                                    stack_subset[:, 1:, :-1], dim=0) / num_frames

    counter_matrix = torch.zeros((fov_dim1, fov_dim2), device=device).float()
    stack_final_img = torch.zeros((fov_dim1, fov_dim2), device=device).float()

    stack_final_img[:-1, :-1] += top_left_bottom_right[0, ...]
    stack_final_img[1:, 1:] += top_left_bottom_right[0, ...]

    counter_matrix[:-1, :-1] += 1
    counter_matrix[1:, 1:] += 1

    stack_final_img[:, :-1] += horizontal[0, ...]
    stack_final_img[:, 1:] += horizontal[0, ...]
    counter_matrix[:, :-1] += 1
    counter_matrix[:, 1:] += 1

    stack_final_img[:-1, :] += vertical[0, ...]
    stack_final_img[1:, :] += vertical[0, ...]
    counter_matrix[:-1, :] += 1
    counter_matrix[1:, :] += 1

    stack_final_img[:-1, 1:] += top_right_bottom_left[0, ...]
    stack_final_img[1:, :-1] += top_right_bottom_left[0, ...]
    counter_matrix[:-1, 1:] += 1
    counter_matrix[1:, :-1] += 1

    stack_final_img /= counter_matrix
    return stack_final_img


def compute_pmd_spatial_correlation_maps(raw_stack: Union[np.ndarray, masknmf.LazyFrameLoader, masknmf.FactorizedVideo],
                                         pmd_stack: masknmf.PMDArray,
                                         device='cpu',
                                         batch_size: int = 200) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes spatial correlation heatmaps for the raw data, PMD reconstruction, and residuals.
    For each pair of adjacent pixels (in horizontal, vertical, and diagonal directions),
    the function calculates the normalized covariance (correlation) between pixel values over time.
    The normalization is based on the raw variance so that comparisons between raw, PMD, and residual
    signals reflect how well PMD decorrelates spatial structure.

    Args:
        raw_stack (Union[np.ndarray, masknmf.LazyFrameLoader, masknmf.FactorizedVideo]):
            The raw video stack with shape (frames, height, width).
        pmd_stack (masknmf.PMDArray):
            The PMD reconstruction object, which includes factorized temporal and spatial components.
        device (str):
            The device on which computations will be performed ('cpu' or 'cuda').
        batch_size (int):
            Number of frames to process per batch.
        mode (str):
            Currently unused; placeholder for future support of different aggregation modes.
    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]:
            Three spatial correlation maps (height, width) for:
              - raw video
              - PMD reconstruction
              - residual (raw - PMD)
    """
    pmd_stack.to(device)
    pmd_stack.rescale = True
    num_frames, fov_dim1, fov_dim2 = raw_stack.shape

    raw_var_img = torch.from_numpy(np.std(raw_stack, axis=0, keepdims=True)).to(device)
    raw_mean = torch.from_numpy(np.mean(raw_stack, axis=0, keepdims=True)).to(device)
    pmd_mean = pmd_stack.mean_img.to(device)[None, :, :] + torch.sparse.mm(pmd_stack.u, torch.mean(pmd_stack.v, dim=1,
                                                                                                   keepdim=True)).reshape(
        (1, fov_dim1, fov_dim2)).to(device)
    resid_mean = raw_mean - pmd_mean

    top_left_bottom_right = torch.zeros((3, fov_dim1 - 1, fov_dim2 - 1), device=device).float()
    horizontal = torch.zeros((3, fov_dim1, fov_dim2 - 1), device=device).float()
    vertical = torch.zeros((3, fov_dim1 - 1, fov_dim2), device=device).float()
    top_right_bottom_left = torch.zeros((3, fov_dim1 - 1, fov_dim2 - 1), device=device).float()

    num_iters = math.ceil(num_frames / batch_size)
    for k in tqdm(range(num_iters)):
        start = batch_size * k
        end = min(start + batch_size, num_frames)
        raw_subset = (torch.from_numpy(raw_stack[start:end, :, :]).to(device) - raw_mean) / raw_var_img
        pmd_subset = (torch.from_numpy(pmd_stack[start:end, :, :]).to(device) - pmd_mean) / raw_var_img

        pmd_subset = torch.nan_to_num(pmd_subset, nan=0)
        raw_subset = torch.nan_to_num(raw_subset, nan=0)
        residual_subset = raw_subset - pmd_subset

        top_left_bottom_right[0, :, :] += torch.sum(raw_subset[:, :-1, :-1] *
                                                    raw_subset[:, 1:, 1:], dim=0) / num_frames
        top_left_bottom_right[1, :, :] += torch.sum(pmd_subset[:, :-1, :-1] *
                                                    pmd_subset[:, 1:, 1:], dim=0) / num_frames
        top_left_bottom_right[2, :, :] += torch.sum(residual_subset[:, :-1, :-1] *
                                                    residual_subset[:, 1:, 1:], dim=0) / num_frames

        horizontal[0, :, :] += torch.sum(raw_subset[:, :, :-1] *
                                         raw_subset[:, :, 1:], dim=0) / num_frames
        horizontal[1, :, :] += torch.sum(pmd_subset[:, :, :-1] *
                                         pmd_subset[:, :, 1:], dim=0) / num_frames
        horizontal[2, :, :] += torch.sum(residual_subset[:, :, :-1] *
                                         residual_subset[:, :, 1:], dim=0) / num_frames

        vertical[0, :, :] += torch.sum(raw_subset[:, :-1, :] *
                                       raw_subset[:, 1:, :], dim=0) / num_frames
        vertical[1, :, :] += torch.sum(pmd_subset[:, :-1, :] *
                                       pmd_subset[:, 1:, :], dim=0) / num_frames
        vertical[2, :, :] += torch.sum(residual_subset[:, :-1, :] *
                                       residual_subset[:, 1:, :], dim=0) / num_frames

        top_right_bottom_left[0, :, :] += torch.sum(raw_subset[:, :-1, 1:] *
                                                    raw_subset[:, 1:, :-1], dim=0) / num_frames
        top_right_bottom_left[1, :, :] += torch.sum(pmd_subset[:, :-1, 1:] *
                                                    pmd_subset[:, 1:, :-1], dim=0) / num_frames
        top_right_bottom_left[2, :, :] += torch.sum(residual_subset[:, :-1, 1:] *
                                                    residual_subset[:, 1:, :-1], dim=0) / num_frames

    counter_matrix = torch.zeros((fov_dim1, fov_dim2), device=device).float()
    raw_final_img = torch.zeros((fov_dim1, fov_dim2), device=device).float()
    pmd_final_img = torch.zeros((fov_dim1, fov_dim2), device=device).float()
    resid_final_img = torch.zeros((fov_dim1, fov_dim2), device=device).float()

    raw_final_img[:-1, :-1] += top_left_bottom_right[0, ...]
    raw_final_img[1:, 1:] += top_left_bottom_right[0, ...]
    pmd_final_img[:-1, :-1] += top_left_bottom_right[1, ...]
    pmd_final_img[1:, 1:] += top_left_bottom_right[1, ...]
    resid_final_img[:-1, :-1] += top_left_bottom_right[2, ...]
    resid_final_img[1:, 1:] += top_left_bottom_right[2, ...]

    counter_matrix[:-1, :-1] += 1
    counter_matrix[1:, 1:] += 1

    raw_final_img[:, :-1] += horizontal[0, ...]
    raw_final_img[:, 1:] += horizontal[0, ...]
    pmd_final_img[:, :-1] += horizontal[1, ...]
    pmd_final_img[:, 1:] += horizontal[1, ...]
    resid_final_img[:, :-1] += horizontal[2, ...]
    resid_final_img[:, 1:] += horizontal[2, ...]
    counter_matrix[:, :-1] += 1
    counter_matrix[:, 1:] += 1

    raw_final_img[:-1, :] += vertical[0, ...]
    raw_final_img[1:, :] += vertical[0, ...]
    pmd_final_img[:-1, :] += vertical[1, ...]
    pmd_final_img[1:, :] += vertical[1, ...]
    resid_final_img[:-1, :] += vertical[2, ...]
    resid_final_img[1:, :] += vertical[2, ...]
    counter_matrix[:-1, :] += 1
    counter_matrix[1:, :] += 1

    raw_final_img[:-1, 1:] += top_right_bottom_left[0, ...]
    raw_final_img[1:, :-1] += top_right_bottom_left[0, ...]
    pmd_final_img[:-1, 1:] += top_right_bottom_left[1, ...]
    pmd_final_img[1:, :-1] += top_right_bottom_left[1, ...]
    resid_final_img[:-1, 1:] += top_right_bottom_left[2, ...]
    resid_final_img[1:, :-1] += top_right_bottom_left[2, ...]
    counter_matrix[:-1, 1:] += 1
    counter_matrix[1:, :-1] += 1

    raw_final_img /= counter_matrix
    pmd_final_img /= counter_matrix
    resid_final_img /= counter_matrix

    return raw_final_img.cpu().numpy(), pmd_final_img.cpu().numpy(), resid_final_img.cpu().numpy()


def pmd_autocovariance_diagnostics(raw_movie: Union[np.ndarray, masknmf.LazyFrameLoader],
                                   pmd_movie: PMDArray,
                                   batch_size: int = 200,
                                   device: str = 'cpu') -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes a normalized version of the lag-1 autocovariance for the raw, pmd, and residual stacks.
    Args:
        raw_movie (masknmf.LazyFrameLoader)
        pmd_movie (masknmf.PMDArray)
        batch_size (int): Number of frames we process at a time
        device (str): 'cpu' or 'cuda' depending on where computations occur

    Returns:
        - np.ndarray: The lag-1 autocorrelation image of the raw/motion corrected movie
        - np.ndarray: The lag-1 autocovariance of the pmd movie, normalized by l2 norms used in the raw lag-1 statistics
        - np.ndarray: The lag-1 autocovariance of the resid movie, normalized by l2 norms used in the raw lag-1 statistics

    Key assumptions in calculation:
        - raw_movie mean is pmd_movie.mean_img
        - resid movie is therefore mean 0
    """
    num_frames, fov_dim1, fov_dim2 = raw_movie.shape
    if num_frames == 1:
        raise ValueError("Only 1 frame passed in, can't compute autocorrelation")
    num_iters = math.ceil(num_frames / batch_size)

    pmd_movie.to(device)
    if pmd_movie.rescale is False:
        switch = True
    else:
        switch = False
    pmd_movie.rescale = True
    raw_autocov = torch.zeros(fov_dim1, fov_dim2, device=device).float()
    left_raw_mean = pmd_movie.mean_img * (num_frames / (num_frames - 1)) - (
            torch.from_numpy(raw_movie[-1]).float().to(device) / (
            num_frames - 1))
    right_raw_mean = pmd_movie.mean_img * (num_frames / (num_frames - 1)) - (
            torch.from_numpy(raw_movie[0]).float().to(device) / (
            num_frames - 1))

    pmd_autocov = torch.zeros(fov_dim1, fov_dim2, device=device).float()
    left_pmd_mean = pmd_movie.mean_img * (num_frames / (num_frames - 1)) - (
            pmd_movie.getitem_tensor([num_frames - 1]).float().to(device) / (
            num_frames - 1))
    right_pmd_mean = pmd_movie.mean_img * (num_frames / (num_frames - 1)) - (
            pmd_movie.getitem_tensor([0]).float().to(device) / (
            num_frames - 1))

    resid_autocov = torch.zeros(fov_dim1, fov_dim2, device=device).float()
    left_resid_mean = left_raw_mean - left_pmd_mean
    right_resid_mean = right_raw_mean - right_pmd_mean

    start_pts = np.arange(0, num_frames, batch_size)
    if start_pts.shape[0] > 1 and start_pts[-1] == num_frames - 1:
        start_pts = start_pts[:-1]

    left_raw_sq_sum = torch.zeros_like(raw_autocov)
    right_raw_sq_sum = torch.zeros_like(raw_autocov)
    for start in start_pts:
        end = min(start + batch_size, num_frames)
        raw_subset = torch.from_numpy(raw_movie[start:end]).to(device)
        raw_left = (raw_subset[:-1] - left_raw_mean)
        raw_right = (raw_subset[1:] - right_raw_mean)
        left_raw_sq_sum += torch.sum(raw_left * raw_left, dim=0)
        right_raw_sq_sum += torch.sum(raw_right * raw_right, dim=0)

        pmd_subset = pmd_movie.getitem_tensor(slice(start, end)).float().to(device)
        pmd_left = (pmd_subset[:-1] - left_pmd_mean)
        pmd_right = (pmd_subset[1:] - right_pmd_mean)

        resid = raw_subset - pmd_subset
        resid_left = (resid[:-1] - left_resid_mean)
        resid_right = (resid[1:] - right_resid_mean)

        raw_autocov += torch.sum(raw_left * raw_right, dim=0)
        pmd_autocov += torch.sum(pmd_left * pmd_right, dim=0)
        resid_autocov += torch.sum(resid_left * resid_right, dim=0)

    left_raw_norm = torch.sqrt(left_raw_sq_sum)
    right_raw_norm = torch.sqrt(right_raw_sq_sum)

    raw_autocov = raw_autocov.cpu().numpy()
    pmd_autocov = pmd_autocov.cpu().numpy()
    resid_autocov = resid_autocov.cpu().numpy()
    left_raw_norm = left_raw_norm.cpu().numpy()
    right_raw_norm = right_raw_norm.cpu().numpy()
    left_raw_norm[left_raw_norm < 1e-6] = 1
    right_raw_norm[right_raw_norm < 1e-6] = 1

    raw_autocov /= (left_raw_norm * right_raw_norm)
    pmd_autocov /= (left_raw_norm * right_raw_norm)
    resid_autocov /= (left_raw_norm * right_raw_norm)

    raw_autocov = np.nan_to_num(raw_autocov, nan=0.0)
    pmd_autocov = np.nan_to_num(pmd_autocov, nan=0.0)
    resid_autocov = np.nan_to_num(resid_autocov, nan=0.0)

    if switch:
        pmd_movie.rescale = False
    return raw_autocov, pmd_autocov, resid_autocov