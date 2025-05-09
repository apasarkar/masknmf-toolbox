import numpy as np
from scipy.signal import butter, lfilter, filtfilt


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