import torch
import numpy as np
import masknmf
from typing import *
from scipy.signal import find_peaks

def onephoton_voltage_noise_heuristic(my_trace: np.ndarray,
                sampling_speed: int,
               cutoff_for_snr_estimate: float = 300):
    """
    This code estimates shot noise variance by (a) high-pass filtering the temporal trace (b) computing the signal power in this interval
    (c) rescaling this to get a general shot noise variance estimate.

    Args:
        my_trace (np.ndarray): Shape (num_frames,)
        sampling_speed (int): Frame rate of the imaging
        cutoff_for_snr_estimate (float): The cutoff for the high pass filter
    """
    if sampling_speed / 2 <= cutoff_for_snr_estimate:
        raise ValueError(f"The cutoff has to be lower than nyquist frequency; in this case that frequency is {sampling_speed / 2}")
    filtered_trace = masknmf.demixing.filters.high_pass_filter_batch(my_trace[None, :],
                                                                    cutoff_for_snr_estimate,
                                                                    sampling_speed).squeeze()

    high_pass_var = np.var(filtered_trace)
    return high_pass_var * (sampling_speed / 2) / (sampling_speed / 2 - cutoff_for_snr_estimate)


def mad_filter_for_spikes(filtered_spike: np.ndarray,
                         mad_factor: float=1.5):
    median_abs_dev = np.median(np.abs(filtered_spike - np.median(filtered_spike)))
    new_trace = np.maximum(0, filtered_spike - np.median(filtered_spike) - mad_factor * median_abs_dev)
    return new_trace

def detect_peaks(trace, height=None, distance=None, prominence=None):
    """
    Detect peaks in a 1D time series.

    Parameters:
    - trace (array-like): The input 1D signal.
    - height (float or tuple, optional): Required height of peaks.
    - distance (int, optional): Minimum distance between peaks (in samples).
    - prominence (float or tuple, optional): Required prominence of peaks.

    Returns:
    - peaks (np.ndarray): Indices of detected peaks.
    - properties (dict): Properties of the peaks (heights, prominences, etc).
    """
    peaks, properties = find_peaks(trace, height=height, distance=distance, prominence=prominence)
    return peaks, properties
