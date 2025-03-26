import torch
import numpy as np
from .strategies import MotionCorrectionStrategy
from .registration_arrays import RegistrationArray
from masknmf.arrays.array_interfaces import LazyFrameLoader
from tqdm import tqdm
from typing import *
from masknmf import display
import random

def compute_template(frames: LazyFrameLoader,
                     rigid_strategy: MotionCorrectionStrategy,
                     num_splits_per_iteration: int = 10,
                     num_frames_per_split: int = 200,
                     num_iterations_rigid: int = 3,
                     num_iterations_piecewise_rigid: int = 1,
                     pwrigid_strategy: Optional[MotionCorrectionStrategy] = None,
                     device: str = "cpu",
                     batch_size: int = 500) -> MotionCorrectionStrategy:
    """
    Iteratively estimates a stable template by refining over multiple correction passes.
    Note: updates the templates of rigid_strategy and piecewise_rigid strategy in-place

    Steps:
    1. Run rigid motion correction for `num_iterations_rigid` iterations.
    2. Use the refined template as input for piecewise rigid correction.
    3. Run piecewise rigid correction for `num_iterations_piecewise` iterations.

    Returns:
        MotionCorrectionStrategy: The updated MotionCorrectionStrategy object with the refined template
    """

    if num_iterations_rigid <= 0:
        raise ValueError(f"Must have at least one pass of rigid registration")

    # Step 1: Initial Template (Mean Image)
    frames_loaded = frames[:500]
    template = torch.from_numpy(np.median(frames_loaded, axis=0))
    rigid_strategy.template = template

    ## Prepare the template estimation pipeline by establishing the chunks of data to sample
    slice_list = compute_frame_chunks(frames.shape[0], num_frames_per_split)
    num_splits_to_sample = min(num_splits_per_iteration, len(slice_list))

    # Step 2: Rigid Motion Correction Stage

    for pass_iter_rigid in range(num_iterations_rigid):
        current_registration_array = RegistrationArray(frames,
                                                       rigid_strategy,
                                                       device = device,
                                                       batch_size = batch_size)

        slices_sampled = random.sample(slice_list, num_splits_to_sample)
        template_list = []
        for j in tqdm(range(num_splits_to_sample)):
            corrected_frames = current_registration_array.index_frames_tensor(slices_sampled[j])
            template = torch.mean(corrected_frames, dim=0)
            template_list.append(template)

        rigid_strategy.template = torch.median(torch.stack(template_list, dim=0), dim=0)[0]

    # Step 3: Piecewise Rigid Motion Correction Stage
    if pwrigid_strategy is not None:
        pwrigid_strategy.template = rigid_strategy.template
        for pass_iter_pwrigid in range(num_iterations_piecewise_rigid):
            current_registration_array = RegistrationArray(frames,
                                                           pwrigid_strategy,
                                                           device = device,
                                                           batch_size = batch_size)
            slices_sampled = random.sample(slice_list, num_splits_to_sample)
            template_list = []
            for j in tqdm(range(num_splits_to_sample)):
                corrected_frames = current_registration_array.index_frames_tensor(slices_sampled[j])
                template = torch.mean(corrected_frames, dim=0)
                template_list.append(template)

            # Update strategy template
            pwrigid_strategy.template = torch.median(torch.stack(template_list, dim=0), dim=0)[0]
        return pwrigid_strategy

    else:
        return rigid_strategy

def compute_frame_chunks(num_frames: int, frames_per_split: int) -> list:
    """
    Sets a partition of the frames into individual contiguous chunks of frames.

    Args:
        num_frames (int): The number of frames in the dataset
        frames_per_split (int): The number of frames in each chunk of data which we use to estimate a local template
    Returns:
        slice_list (list): A list of slices, each describing a start and end for a given chunk of data which can be used to
            refine the template estimate.
    """
    start_pts = list(range(0, num_frames, frames_per_split))
    if start_pts[-1] > num_frames - frames_per_split and start_pts[-1] > 0:
        start_pts[-1] = num_frames - frames_per_split

    slice_list = [slice(start_pts[i], min(num_frames, start_pts[i] + frames_per_split)) for i in range(len(start_pts))]
    return slice_list