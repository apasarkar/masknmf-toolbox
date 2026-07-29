from .registration_methods import (
    apply_rigid_shifts,
    estimate_rigid_shifts,
    interpolate_to_border,
    register_frames_rigid,

    ## Pwrigid functions
    register_frames_pwrigid,
    pwrigid_shift_estimation_routine,
    apply_pwrigid_shifts
)
from .strategies import (
    DummyMotionCorrector,
    MotionCorrectionStrategy,
)

from .piecewise_rigid_strategy import PiecewiseRigidMotionCorrector, PiecewiseRigidRegistrationArray
from .rigid_strategy import RigidMotionCorrector, RigidRegistrationArray
from .gradient_strategy import GradientMotionCorrector, GradientRegistrationArray
from .registration_arrays import FilteredArray, OphysArray, BaseRegistrationArray
from .spatial_filters import (
    image_filter,
    gaussian_kernel,
    compute_highpass_filter_kernel,
)

from .moco_preprocessing import (
    compute_saturation_mask,
    dilate_saturation_mask,
    mask_inpainting_routine,
)

__all__ = [
    "FilteredArray",
    "OphysArray",
    "BaseRegistrationArray",
    "GradientMotionCorrector",
    "GradientRegistrationArray",
    "RigidRegistrationArray",
    "RigidMotionCorrector",
    "PiecewiseRigidMotionCorrector",
    "PiecewiseRigidRegistrationArray",
    "DummyMotionCorrector",
    "MotionCorrectionStrategy",
]
