"""
Utility functions for inspecting suite2p results.
For now, we only work with 
"""
import numpy as np
import os
import sys
from typing import *
import masknmf
from masknmf import utils
import torch

def make_masks_from_suite2p_statfile(stat: dict,
                                     Ly: int, 
                                     Lx: int) -> Tuple[np.ndarray, np.ndarray]:
    signal_arr = np.zeros((Ly*Lx, len(stat)))
    neuropil_signal_arr = np.zeros((Ly*Lx, len(stat)))

    for k in range(len(stat)):
        cell_mask = np.ravel_multi_index((stat[k]["ypix"], stat[k]["xpix"]), (Ly, Lx))
        lam_val = stat[k]['lam'] / np.sum(stat[k]['lam'])
        signal_arr[cell_mask, k] = lam_val

        # Let's do the same thing for the neuropil now
        neuropil_pixels = np.ones_like(lam_val)
        neuropil_pixels = neuropil_pixels / np.sum(neuropil_pixels)
        neuropil_signal_arr[cell_mask, k] = neuropil_pixels
    
    return signal_arr, neuropil_signal_arr

def demix_with_s2p_outputs(folder: str,
                           pmd_object: masknmf.PMDArray,
                           device='cuda') -> masknmf.DemixingResults:

    """
    Goal here is to compare the suite2p results directly w.r.t. the PMD array: how well does suite2p demix the PMD representation of the data? 
    Args:
        folder (str): An absolute folder path describing a folder containing all of the suite2p outputs (named in the usual way)
        pmd_object (masknmf.PMDArray) 
    """
    is_cell = np.load(os.path.join(folder, "iscell.npy"))
    c_traces = np.load(os.path.join(folder, "F.npy"), allow_pickle=True)
    c_neuropil = np.load(os.path.join(folder, "Fneu.npy"), allow_pickle=True)
    stat = np.load(os.path.join(folder, "stat.npy"), allow_pickle=True)
    ops = np.load(os.path.join(folder,'ops.npy'), allow_pickle=True).item()
    
    # Now let's do neuropil correction
    c_traces = c_traces - ops['neucoeff']*c_neuropil
    
    #Define the neuropil estimate (at each cell) as a scaled version of the neuropil mask ROI average:
    # c_neuropil = ops['neucoeff']*c_neuropil
    
    c_traces = c_traces.T
    c_neuropil = c_neuropil.T

    #Load the spatial data
    signal_spatial, neuropil_spatial = make_masks_from_suite2p_statfile(stat, ops['Ly'], ops['Lx'])
    signal_spatial = signal_spatial.reshape((ops['Ly'], ops['Lx'], -1))


    num_frames, fov_dim1, fov_dim2 = pmd_object.shape
    frame_batch_size = 300
    unfiltered_pmd_demixer = masknmf.demixing.signal_demixer.SignalDemixer(
                                                    pmd_object,
                                                    device=device,
                                                    frame_batch_size=frame_batch_size)

    unfiltered_pmd_demixer.initialize_signals(is_custom=True,
                                              spatial_footprints=signal_spatial)
    unfiltered_pmd_demixer.lock_results_and_continue()

    
    ## Demixing State
    num_iters = 25
    ## Now run demixing...
    localnmf_params = {
        'maxiter':num_iters,
        'support_threshold':np.linspace(0.95, 0.5, num_iters).tolist(),
        'deletion_threshold':0.2,
        'ring_model_start_pt':0,
        'ring_radius':15,
        'background_downsampling_factor': 20,
        'merge_threshold':0.8,
        'merge_overlap_threshold':0.8,
        'update_frequency':num_iters + 1,
        'c_nonneg':True,
        'denoise':False,
        'plot_en': False
    }

    with torch.no_grad():
        unfiltered_pmd_demixer.demix(**localnmf_params)

    return unfiltered_pmd_demixer.results



    
    

def build_s2p_demixingresults(folder: str,
                              pmd_object: masknmf.PMDArray,
                             device = 'cpu') -> masknmf.DemixingResults:
    """
    Goal here is to compare the suite2p results directly w.r.t. the PMD array: how well does suite2p demix the PMD representation of the data? 
    Args:
        folder (str): An absolute folder path describing a folder containing all of the suite2p outputs (named in the usual way)
        pmd_object (masknmf.PMDArray) 
    """
    is_cell = np.load(os.path.join(folder, "iscell.npy"))
    c_traces = np.load(os.path.join(folder, "F.npy"), allow_pickle=True)
    c_neuropil = np.load(os.path.join(folder, "Fneu.npy"), allow_pickle=True)
    stat = np.load(os.path.join(folder, "stat.npy"), allow_pickle=True)
    ops = np.load(os.path.join(folder,'ops.npy'), allow_pickle=True).item()
    
    # Now let's do neuropil correction
    c_traces = c_traces - ops['neucoeff']*c_neuropil
    
    #Define the neuropil estimate (at each cell) as a scaled version of the neuropil mask ROI average:
    c_neuropil = ops['neucoeff']*c_neuropil
    
    c_traces = c_traces.T
    c_neuropil = c_neuropil.T

    #Load the spatial data
    signal_spatial, neuropil_spatial = make_masks_from_suite2p_statfile(stat, ops['Ly'], ops['Lx'])
    signal_spatial = signal_spatial.reshape((ops['Ly'], ops['Lx'], -1))
    neuropil_spatial = neuropil_spatial.reshape((ops['Ly'], ops['Lx'], -1))

    a_suite2p = masknmf.ndarray_to_torch_sparse_coo(signal_spatial.reshape((-1, signal_spatial.shape[2]))).to(device)
    c_suite2p = torch.from_numpy(c_traces).float().to(device)
    
    c_suite2p_rescale, b_rescale = masknmf.demixing.regression_update.alternating_least_squares_affine_fit(pmd_object.u.to(device),
                                                                                                       pmd_object.v.to(device),
                                                                                                       a_suite2p.to(device),
                                                                                                       c_suite2p.to(device),
                                                                                                      scale_nonneg=True)

    results = masknmf.DemixingResults(pmd_object.shape,
                                  pmd_object.u,
                                  pmd_object.v,
                                  a_suite2p,
                                  c_suite2p_rescale,
                                  b=b_rescale,
                                  device=device)
    return results



######
## Below code is for matching neurons between suite2p and masknmf results
######

def find_unmatched_neurons(first_a: np.ndarray,
                           first_c: np.ndarray,
                           second_a: np.ndarray,
                           second_c: np.ndarray,
                           spatial_similarity_ceiling: float=0.6,
                           temporal_similarity_ceiling: float=0.6):
    """"
    Args:
        first_a (np.ndarray): Shape (height, width, num_neurons_from_pipeline1)
        first_c (np.ndarray): Shape (num_frames, num_neurons_from_pipeline1)
        second_a (np.ndarray): Shape (height, width, num_neurons_from_pipeline2)
        second_c (np.ndarray): Shape (num_frames, num_neurons_from_pipeline2)
        spatial_similarity_ceiling (float): All similarities for neuron "i" should below this value to declare it unmatched
        temporal_similarity_ceiling (float): All similarities for neuron "i" should be below this value to declare it unmatched
    Returns:
        unmatched_indices (np.ndarray): A boolean array of shape (num_neurons_from_pipeline1) indicating whether a neuron is unmatched
    """
    first_a_norm = first_a / np.linalg.norm(first_a, axis = (0, 1))
    first_b_norm = second_a / np.linalg.norm(second_a, axis = (0, 1))

    spatial_similarity_mat = np.einsum('hwt,hws->ts', first_a_norm, first_b_norm)

    first_c_norm = first_c / np.linalg.norm(first_c, axis = 0)
    second_c_norm = second_c / np.linalg.norm(second_c, axis=0)

    temporal_similarity_mat = first_c_norm.T @ second_c_norm

    max_spatial_sim = np.amax(spatial_similarity_mat, axis = 1)
    max_temporal_sim = np.amax(temporal_similarity_mat, axis = 1)

    misses = np.logical_and(max_spatial_sim < spatial_similarity_ceiling, max_temporal_sim < temporal_similarity_ceiling)
    return misses
