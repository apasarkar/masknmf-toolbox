import torch
import numpy as np
import masknmf
from typing import *
import math

def pmd_autocovariance_diagnostics(raw_movie: masknmf.LazyFrameLoader,
                                   pmd_movie: masknmf.PMDArray,
                                   batch_size: int = 200,
                                   device: str = 'cpu') -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes a normalized version of the lag-1 autocovariance for the raw, pmd, and residual stacks.
    Args:
        raw_movie (masknmf.LazFrameLoader)
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
    pmd_movie.rescale = True
    raw_autocov = torch.zeros(fov_dim1, fov_dim2, device=device).float()
    left_raw_mean = (torch.zeros_like(raw_autocov) - torch.from_numpy(raw_movie[-1]).float().to(device)) / (
            num_frames - 1)
    right_raw_mean = (torch.zeros_like(raw_autocov) - torch.from_numpy(raw_movie[0]).float().to(device)) / (
            num_frames - 1)

    pmd_autocov = torch.zeros(fov_dim1, fov_dim2, device=device).float()
    left_pmd_mean = (torch.zeros_like(raw_autocov) - pmd_movie.getitem_tensor([num_frames - 1]).float().to(device)) / (
            num_frames - 1)
    right_pmd_mean = (torch.zeros_like(raw_autocov) - pmd_movie.getitem_tensor([0]).float().to(device)) / (
            num_frames - 1)

    resid_autocov = torch.zeros(fov_dim1, fov_dim2, device=device).float()
    left_resid_mean = (torch.zeros_like(raw_autocov) - (
            torch.from_numpy(raw_movie[-1]).float().to(device) - pmd_movie.getitem_tensor(
        [num_frames - 1]).float().to(device))) / (num_frames - 1)
    right_resid_mean = (torch.zeros_like(raw_autocov) - (
            torch.from_numpy(raw_movie[0]).float().to(device) - pmd_movie.getitem_tensor([0]).float().to(
        device))) / (num_frames - 1)

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

    raw_autocov /= (left_raw_norm * right_raw_norm)
    pmd_autocov /= (left_raw_norm * right_raw_norm)
    resid_autocov /= (left_raw_norm * right_raw_norm)

    return raw_autocov.cpu().numpy(), pmd_autocov.cpu().numpy(), resid_autocov.cpu().numpy()
