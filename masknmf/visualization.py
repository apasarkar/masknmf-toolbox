import localnmf
from typing import *
import numpy as np
import torch
import os
import sys
import matplotlib.pyplot as plt

import fastplotlib as fpl
from fastplotlib.widgets import ImageWidget


def make_demixing_video(results: localnmf.DemixingResults,
                        device: str,
                        v_range: Optional[tuple[float, float]]=None,
                        show_histogram: bool=False) -> ImageWidget:


    results.to(device)

    ac_arr = results.ac_array
    fluctuating_arr = results.fluctuating_background_array
    pmd_arr = results.pmd_array
    residual_arr = results.residual_array
    colorful_arr = results.colorful_ac_array
    static_bg = results.baseline.cpu().numpy()

    iw = ImageWidget(data=[pmd_arr,
                           ac_arr,
                           fluctuating_arr,
                           residual_arr,
                           colorful_arr,
                           static_bg],
                     names=['pmd',
                            'signals',
                            'fluctuating bkgd',
                            'residual',
                            'colorful signals',
                            'static Bkgd'],
                     rgb=[False, False, False, False, True, False],
                     histogram_widget=show_histogram)

    if v_range is not None:
        values_to_set = [0, 1, 2, 3, 5]
        for i, subplot in enumerate(iw.figure):
            if i in values_to_set:
                ig = subplot["image_widget_managed"]
                ig.vmin = v_range[0]
                ig.vmax = v_range[1]

    return iw


# For every signal, need to look at the temporal trace and the PMD average, superimposed
def get_roi_avg(array, p1, p2, normalize=True):
    """
    Given nonzero dim1 and dim2 indices p1 and p2, get the ROI average
    """
    selected_pixels = array[:, np.amin(p1):np.amax(p1) + 1, np.amin(p2):np.amax(p2) + 1]
    data_2d = selected_pixels[:, p1 - np.amin(p1), p2 - np.amin(p2)]
    avg_trace = np.mean(data_2d, axis=1)
    if normalize:
        return avg_trace / np.amax(avg_trace)
    else:
        return avg_trace


def plot_ith_roi(i: int,
                 results: localnmf.DemixingResults,
                 folder=".",
                 name="neuron.png"):
    """
    Generates a diagnostic plot of the i-th ROI
    Args:
        i (int): The neuron data
        results (localnmf.DemixingResults): the results from the demixing procedure
        folder (str): folder where the data is saved. This folder must exist already.
        name (str): The name of the output .png file
    """
    if os.path.exists(folder):
        raise ValueError(f"folder {folder} already exists. delete it or pick different folder name")

    os.mkdir(folder)

    order = results.order
    current_a = torch.index_select(results.a, 1,
                                   torch.arange(i, i + 1).to(results.device)).to_dense().cpu().numpy()
    a = current_a.reshape((results.shape[1], results.shape[2]), order=order)

    p1, p2 = a.nonzero()
    pmd_roi_avg = get_roi_avg(results.pmd_array, p1, p2, normalize=False)
    static_bg_roi_avg = np.ones_like(pmd_roi_avg) * np.mean(results.baseline[p1, p2].cpu().numpy().flatten())
    fluctuating_bg_roi_avg = get_roi_avg(results.fluctuating_background_array, p1, p2, normalize=False)
    signal_roi_avg = np.mean(a[a > 0]) * results.c[:, i].cpu().numpy()
    residual_roi_avg = get_roi_avg(results.residual_array, p1, p2, normalize=False)

    max_c = np.amax(signal_roi_avg)

    # We want to show the fluctuating background and the signal at the same scale
    fluctuating_bg_roi_avg = (fluctuating_bg_roi_avg + static_bg_roi_avg) / max_c

    fig, ax = plt.subplots(4, 1, figsize=(20, 20))

    # Need to show the right set of pixels
    ax[0].imshow(a[np.amin(p1):np.amax(p1), np.amin(p2):np.amax(p2)],
                 extent=[np.amin(p2), np.amax(p2), np.amin(p1), np.amax(p1)])
    ax[0].set_title("Spatial Footprint")

    ax[1].plot(signal_roi_avg)
    ax[1].set_title("Temporal Trace")

    ax[2].plot(residual_roi_avg)
    ax[2].set_title("Residual")

    ax[3].plot(4 + pmd_roi_avg / np.amax(pmd_roi_avg), label="PMD")
    ax[3].plot(2 + signal_roi_avg / np.amax(signal_roi_avg), label="Source")
    ax[3].plot(2 + fluctuating_bg_roi_avg, label="Net Bkgd")
    ax[3].set_yticks([])
    ax[3].set_title("ROI Average, Extracted Signal, Net Background")
    ax[3].legend()

    savename = os.path.join(folder, name)
    plt.savefig(savename)
    plt.show()


