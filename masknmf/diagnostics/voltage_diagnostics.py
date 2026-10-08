import torch
import numpy as np
import masknmf
from typing import *
from scipy.signal import find_peaks
from scipy.ndimage import median_filter
import math


def detect_spikes(trace: np.ndarray,
                  fps: int=800,
                  median_ms: int=10,
                  threshold_factor=6,
                  min_interval_frames=3,
                  negative_going=False):
    """
    Coarse temporal spike detection applied to a 1D trace.
    Median filter the trace, look at time points with large mean absolute deviation, find peaks among those time points.

    Median filter removes slower trends (subthreshold signal, optostim, etc.)

    Args:
        trace (np.ndarray): Shape (num_frames,). The voltage imaging time series
        fps (int): The video frame rate (in Hz)
        median_ms (int): This defines the window (in milliseconds) for the median filter
        threshold_factor (float): How many mean absolute deviations above the mean the threshold lies for findng spike peaks
        min_interval_frames (int): The min number of frames between two spikes
        negative_going (bool): Whether the input trace has spikes that are negative oing or not. maskNMF will invert
            negatively tuned data, so this should always be False

    Returns:
        - peaks (np.ndarray): An array containing frame indices for the spike detections
        - detr (np.ndarray): The detrended time series used to perform the detection
        - threshold (float): The threshold value applied to the detrended time series. Peaks above this threshold are considered spikes
    """
    trace = np.asarray(trace, float).ravel()
    W = int(round(median_ms / 1000 * fps))  # 8 frames @800Hz = 10 ms
    W_temporal = W * 4  # detrend the trace at 4x (=40 ms)

    base = median_filter(trace, size=W_temporal, mode='nearest')
    detr = (base - trace) if negative_going else (trace - base)  # spikes -> positive

    edge = int(round(W / 2) * 4)  # ignore frames at the edges
    detr[:edge] = 0
    detr[-edge:] = 0

    noise = np.mean(np.abs(detr - detr.mean()))  # mean absolute deviation plots
    threshold = detr.mean() + threshold_factor * noise
    detr[detr < threshold] = 0

    peaks, _ = find_peaks(detr)

    if peaks.size:  # refractory: drop spikes < interval apart
        gaps = np.concatenate(([min_interval_frames], np.diff(peaks)))
        peaks = peaks[gaps >= min_interval_frames]
    return peaks, detr, threshold


def spike_reassignment(trace: np.ndarray,
                       noisy_trace: np.ndarray,
                       peaks: np.ndarray,
                       fps=800,
                       ms_radius: float = 2):
    """
    Routine to assign signals from the raw data back to the denoised traces to avoid any spike height loss
    We detect spikes on the smoothed PMD trace and then look for missed signal in the noisy raw data trace

    Args:
        trace (np.ndarray): Shape (num_frames,). The denoised trace obtained from demixing the PMD compressed representation of the movie
        noisy_trace: Shape (num_frmaes,). The trace obtained by running temporal HALS on the motion corrected (raw) data, keeping
            the spatial footprints fixed
        peaks (np.ndarray): An array of frame indices where a spike is detected
        fps (int): The sampling speed of the movie in hz
        ms_radius (float): The time radius (in milliseconds) around each spike where signal is
            reassigned from the noisy trace to the denoised trace.

    Returns:
        - final_trace: The de-biased trace which re-incorporates any missing spike information

    """
    frame_interval = math.ceil(fps * 0.001 * ms_radius)

    offset = np.mean(trace - noisy_trace)
    noisy_trace_centered = noisy_trace + offset

    intervals = np.arange(-1 * frame_interval, frame_interval)
    peak_indices = (peaks[:, None] + intervals[None, :])

    peak_indices = peak_indices.flatten()
    peak_indices = np.unique(np.arange(trace.shape[0])[peak_indices])

    final_trace = trace.copy()
    final_trace[peak_indices] = noisy_trace_centered[peak_indices]
    return final_trace

