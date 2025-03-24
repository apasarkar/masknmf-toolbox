import torch
from .strategies import RigidMotionCorrection, PiecewiseRigidMotionCorrection, MotionCorrectionStrategy
from .registration_arrays import RegistrationArray
from masknmf.arrays.array_interfaces import LazyFrameLoader
from tqdm import tqdm
from typing import *
from masknmf import display



def compute_template(frames: LazyFrameLoader,
                     rigid_strategy: MotionCorrectionStrategy,
                     num_iterations_rigid: int = 3,
                     num_iterations_piecewise: int = 1,
                     pwrigid_strategy: Optional[MotionCorrectionStrategy] = None):
    """
    Iteratively estimates a stable template by refining over multiple correction passes.

    Steps:
    1. Run rigid motion correction for `num_iterations_rigid` iterations.
    2. Use the refined template as input for piecewise rigid correction.
    3. Run piecewise rigid correction for `num_iterations_piecewise` iterations.

    Returns:
        RegistrationArray: The final motion-corrected dataset wrapper.
    """

    if num_iterations_rigid <= 0:
        raise ValueError(f"Must have at least one pass of rigid registration")


    # Step 1: Initial Template (Mean Image)
    frames_loaded = torch.from_numpy([frames[:500]])
    template = torch.median(frames_loaded, dim=0)

    # Step 2: Rigid Motion Correction Stage
    display("Running rigid registration strategy")
    for _ in range(num_iterations_rigid):

        corrected_frames = rigid_strategy.correct(frames_loaded)
        template = torch.mean(corrected_frames, dim=0)
        rigid_strategy.template = template  # Update strategy template

    # Step 3: Piecewise Rigid Motion Correction Stage
    if pwrigid_strategy is not None:
        display("Rigid registration complete. Running pwrigid strategy now")
        for _ in tqdm(range(num_iterations_piecewise)):
            corrected_frames = pwrigid_strategy.correct(frames_loaded)
            template = torch.mean(corrected_frames, dim=0)
            pwrigid_strategy.template = template  # Update strategy template
    else:
        return RegistrationArray(frames, rigid_strategy)

    # Return RegistrationArray with final template & piecewise correction
    return RegistrationArray(frames, pwrigid_strategy)