import scipy
import scipy.sparse
from typing import *

def extract_labels(outputs_roicat: tuple[dict]) -> list[list]:
    """
    Args:
        outputs_roicat (tuple[dict]): The standard tuple output from ROICaT tracking pipeline
    Returns:
        - list[list]: For each session, a list indicating the per-ROI labeling by ROICaT
    """
    return outputs_roicat[0]['clusters']['labels_bySession']

def extract_footprints(outputs_roicat: tuple[dict]) -> list[scipy.sparse.csr_matrix]:
    """
    Returns a list of aligned ROIs (spatial footprints) across the sessions. Useful for visualization + further analysis of spatial profiles

    Args:
        outputs_roicat (tuple[dict]): The standard tuple output from ROICaT tracking pipeline
    Returns:
        - list[scipy.sparse.csr_matrix]: For each session, a scipy.sparse.csr_matrix of shape (num_neurons, num_pixels)
    """
    return outputs_roicat[1]['aligner']['ROIs_aligned']