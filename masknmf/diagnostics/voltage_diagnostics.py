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


def onephoton_voltage_snr_stats(full_moco_dense: np.ndarray,
                                pmd_arr_moco: masknmf.PMDArray,
                                spatial_footprint: np.ndarray,
                                temporal_trace: np.ndarray,
                                trend_filter_cutoff_freq: float = 1,
                                spike_detect_cutoff_freq: float = 3,
                                mad_cutoff=8,
                                sampling_rate: float = 800,
                                reverse_polarity: bool = True):
    """
    General pipeline for studying the spikes present in the same ROI average across two image stacks.
    Assumption: the pmd array (pmd_arr_moco) has been "flipped" if the indicator is negative (i.e. all spikes point upwards in that stack).
    Args:
        full_moco_dense: dense stack (frames, fov_dim1, fov_dim2)
        pmd_arr_moco: masknmf.PMDArray object
        spatial_footprint (np.ndarray). Shape (fov_dim1, fov_dim2)
        temporal_trace (np.ndarray): Shape (num_frames,)
        trend_filter_cutoff_freq (float): the frequency cutoff for highpass filtering to remove optostim related smooth trends
        spike_detect_cutoff_freq (float): the highpass filter frequency used before running MAD thresholding to identify spikes
        mad_cutoff (float): the mad threshold used to identify spiking locations (and eventually find the spike peak times).
        sampling_rate (float): The sampling rate of the data in Hz
        reverse_polarity (bool): True if the raw stack contains a negatively tuned indicator.

    Returns:
        - np.ndarray: raw roi average time series
        - np.ndarray: raw pmd roi average time series
        - np.ndarray: time series indicating estimated spike peak frame locations
        - np.ndarray: time series indicating pmd spike height estimates
        - np.ndarray: time series indicating raw data spike height estimates
        - np.ndarray: time series indicating the spike height 'attenuation' estimate
        - np.ndarray: time series indicating the spike height attenuation as fraction of raw spike height (we disregard negative values here)
        - np.ndarray: the spike height attenuation divided by the residual standard deviation
    """

    raw_roi_avg, pmd_roi_avg = masknmf.visualization.plots.roi_compare_pmd_raw(full_moco_dense,
                                                                               pmd_arr_moco,
                                                                               spatial_footprint)

    if reverse_polarity:
        raw_roi_avg *= -1
        raw_roi_avg -= np.mean(raw_roi_avg)

    pmd_roi_avg -= np.mean(pmd_roi_avg)

    pmd_roi_lowpass = pmd_roi_avg - masknmf.demixing.filters.high_pass_filter_batch(pmd_roi_avg[None, :],
                                                                                    trend_filter_cutoff_freq,
                                                                                    sampling_rate).squeeze()

    raw_roi_lowpass = raw_roi_avg - masknmf.demixing.filters.high_pass_filter_batch(raw_roi_avg[None, :],
                                                                                    trend_filter_cutoff_freq,
                                                                                    sampling_rate).squeeze()

    resid_roi_avg = raw_roi_avg - pmd_roi_avg

    raw_roi_avg_noisevar = masknmf.diagnostics.voltage_diagnostics.onephoton_voltage_noise_heuristic(raw_roi_avg, 800,
                                                                                                     300)
    resid_std = np.std(resid_roi_avg)

    ## Next step: let's take the C matrix and compute the spike locations (after running a 3Hz high pass filter)
    hp_filter_c = masknmf.demixing.filters.high_pass_filter_batch(temporal_trace[None, :],
                                                                  spike_detect_cutoff_freq,
                                                                  sampling_rate).squeeze()

    thres_c = masknmf.diagnostics.voltage_diagnostics.mad_filter_for_spikes(hp_filter_c, mad_cutoff)
    c_peaks, _ = masknmf.diagnostics.voltage_diagnostics.detect_peaks(thres_c, distance=10)

    # For each peak, estimate the splike amplitude. Strategy: subtract the 1Hz low-pass
    pmd_spike_heights = pmd_roi_avg[c_peaks] - pmd_roi_lowpass[c_peaks]
    raw_spike_heights = raw_roi_avg[c_peaks] - raw_roi_lowpass[c_peaks]

    attenuation_estimate = raw_spike_heights - pmd_spike_heights
    attenuation_estimate[attenuation_estimate < 0] = 0
    fractional_attenuation = np.nan_to_num(np.abs(attenuation_estimate) / raw_spike_heights, nan=0.0)
    zscore_attenuation = np.nan_to_num(np.abs(attenuation_estimate) / resid_std, nan=0.0)

    return raw_roi_avg, pmd_roi_avg, c_peaks, pmd_spike_heights, raw_spike_heights, attenuation_estimate, fractional_attenuation, zscore_attenuation


