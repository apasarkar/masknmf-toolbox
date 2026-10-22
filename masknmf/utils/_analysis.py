import torch
import numpy as np

def infill_bad_timepoints(analyzed_timeseries: torch.Tensor | None,
                          good_timepoint_indicator: torch.Tensor,
                          device: torch.device | str = "cpu") -> torch.Tensor | None:
    """
    Expands time series data derived from "good timepoints" to series that include "bad timepoints".
    The value is 0 at those points.
    Args:
        analyzed_timeseries (torch.Tensor): Shape (num_series, num_good_timepoints) or (num_good_timepoints,)
        good_timepoint_indicator (torch.Tensor): Boolean tensor whose length is the total number of time points, good and bad
    """
    if analyzed_timeseries is None:
        return None
    analyzed_timeseries = torch.as_tensor(analyzed_timeseries).to(device) #no-op if tensor already
    good_timepoint_indicator = torch.as_tensor(good_timepoint_indicator).to(torch.bool).to(device) #no-op if already tensor

    if analyzed_timeseries.ndim == 1:
        squeeze_flag = True
        analyzed_timeseries = analyzed_timeseries[None, :]
    else:
        squeeze_flag = False

    if good_timepoint_indicator.ndim != 1:
        raise ValueError("good_timepoint_indicator must be one-dimensional")

    num_series = analyzed_timeseries.shape[0]
    num_data_frames = analyzed_timeseries.shape[1]
    indicator_num_good_frames = torch.count_nonzero(good_timepoint_indicator)
    if num_data_frames != indicator_num_good_frames:
        raise ValueError(f"Number of good frames specified by the frame labelling, {indicator_num_good_frames}, "
                         f"does not match the number of frames in the data {num_data_frames}")

    final_tensor = torch.zeros(num_series, good_timepoint_indicator.numel(), dtype=analyzed_timeseries.dtype, device=analyzed_timeseries.device)
    final_tensor[:, good_timepoint_indicator] = analyzed_timeseries

    return final_tensor.squeeze(0) if squeeze_flag else final_tensor