"""
Labels, tooltips and tiers for pipeline config fields, and the column each config goes in, for the launcher's
Parameters popup.

Hints are keyed by config class, so every pipeline that takes a config shares its hints. A field without a hint
shows its own name at the normal tier. A config defined outside the three config modules goes in an Other column.
"""

from typing import Literal, Optional

import dataclasses

from masknmf.pipelines import scraper
from masknmf.pipelines.configs import compression_configs, demixing_configs, motion_correction_configs
from masknmf.pipelines.configs.compression_configs import CompressConfig, CompressDenoiseConfig
from masknmf.pipelines.configs.demixing_configs import (
    GlobalSignalFilterConfig,
    NMFConfig,
    SinglepassDemixingConfig,
    SpatialHighpassConfig,
    SuperpixelInitConfig,
)
from masknmf.pipelines.configs.motion_correction_configs import (
    GradientMotionCorrectionConfig,
    PiecewiseRigidMotionCorrectionConfig,
    RigidMotionCorrectionConfig,
    SpatialHighpassMotionCorrectionConfig,
)


TIERS = ("important", "normal", "advanced", "stable")

STAGE_REGISTRATION = "Registration"
STAGE_COMPRESSION = "Compression"
STAGE_DEMIXING = "Demixing"
STAGE_OTHER = "Other"
STAGES_ORDER = (STAGE_REGISTRATION, STAGE_COMPRESSION, STAGE_DEMIXING, STAGE_OTHER)
STAGE_BY_MODULE = {
    motion_correction_configs.__name__: STAGE_REGISTRATION,
    compression_configs.__name__: STAGE_COMPRESSION,
    demixing_configs.__name__: STAGE_DEMIXING,
}


@dataclasses.dataclass(frozen=True)
class Hint:
    """
    How the Parameters popup shows one field.

    Parameters
    ----------
    label : str
        Short name shown beside the widget.
    tooltip : str
        What the field does, shown on hover above its name and default.
    tier : str
        One of TIERS. Important fields get a boxed label; advanced and stable fields sit in folds.
    """

    label: str
    tooltip: str = ""
    tier: Literal["important", "normal", "advanced", "stable"] = "normal"

    def __post_init__(self):
        if self.tier not in TIERS:
            raise ValueError(f"tier {self.tier!r} of {self.label!r} is not one of {TIERS}")


HINT_MAX_SHIFTS = Hint(
    "Max shifts",
    "Largest shift allowed in px (y, x). Raise it if the shift traces look clipped at their extremes.",
    "important",
)

HINTS_COMPRESS = {
    "block_sizes": Hint(
        "Block sizes",
        "PMD patch size in px; about 2x the largest feature you expect (10-20 for dendrites and spines, 20+ for somata).",
        "important",
    ),
    "max_components": Hint("Max components", "Max PMD components per block."),
    "frame_range": Hint("Fit frames", "Number of frames the decomposition is fit on.", "advanced"),
    "sim_conf": Hint("Sim conf", tier="stable"),
    "max_consecutive_failures": Hint(
        "Max failures", "Maximum number of consecutive rejected components allowed per block.", "advanced"
    ),
    "spatial_avg_factor": Hint("Spatial avg", tier="stable"),
    "temporal_avg_factor": Hint("Temporal avg", tier="stable"),
    "compute_normalizer": Hint("Noise normalizer", tier="advanced"),
    "tile_device": Hint(
        "Tile device",
        "Where compression holds its tiles: cpu (host RAM), device (the compute device), or none to let the planner "
        "choose by read cost. device suits data already in RAM; none or cpu suits data read from disk.",
        "advanced",
    ),
}

HINTS: dict[type, dict[str, Hint]] = {
    RigidMotionCorrectionConfig: {"max_shifts": HINT_MAX_SHIFTS},
    PiecewiseRigidMotionCorrectionConfig: {
        "minimum_patch_sizes": Hint("Patch size", "Smallest (y, x) patch in px a local shift is estimated in."),
        "overlaps": Hint("Overlaps", "Overlap in px (y, x) between neighbouring patches."),
        "max_rigid_shifts": HINT_MAX_SHIFTS,
        "max_deviation_rigid": Hint("Max deviation", "Largest amount in px (y, x) a patch's shift may differ from the rigid shift."),
    },
    GradientMotionCorrectionConfig: {
        "num_frames_template": Hint("Template frames", "The first this many frames are averaged into the template."),
    },
    SpatialHighpassMotionCorrectionConfig: {
        "reference_highpass_sigma": Hint(
            "Highpass sigma",
            "Width in px of the highpass applied to the copy of the movie that shifts are estimated on; the shifts "
            "are applied to the raw frames.",
        ),
    },
    CompressConfig: HINTS_COMPRESS,
    CompressDenoiseConfig: {
        **HINTS_COMPRESS,
        "noise_variance_quantile": Hint("Noise quantile", tier="stable"),
        "num_epochs": Hint("Denoiser epochs", "Training epochs of the temporal denoiser.", "stable"),
        "filepath_denoiser": Hint(
            "Denoiser weights",
            "Weights from masknmf.compression.save_denoiser; training is skipped, so Denoiser epochs is unused.",
        ),
    },
    SpatialHighpassConfig: {
        "filter_sigma": Hint(
            "Highpass sigma",
            "Width in px of the spatial highpass applied to the compressed data the filtered passes initialize on.",
            "important",
        ),
    },
    GlobalSignalFilterConfig: {
        "rank": Hint("Global rank", "Number of global components subtracted from the compressed data.", "important"),
        "num_oversamples": Hint(
            "Oversamples", "Extra random vectors in the randomized SVD that finds the global components.", "advanced"
        ),
    },
    SinglepassDemixingConfig: {"InitConfig": Hint("Initialization"), "NMFConfig": Hint("NMF")},
    SuperpixelInitConfig: {
        "mad_correlation_threshold": Hint(
            "Correlation thr",
            "Superpixel seed threshold and the main signal-vs-noise knob: the temporal correlation neighbouring "
            "pixels need after MAD thresholding. Lower finds more, dimmer cells.",
            "important",
        ),
        "min_peak_distance": Hint("Min peak dist", "Minimum distance between adjacent peaks.", "stable"),
        "mad_threshold": Hint(
            "MAD thr", "Each pixel's trace is zeroed below its median plus this many MADs before correlating.", "stable"
        ),
        "residual_threshold": Hint(
            "Residual thr", "How far a superpixel's trace must lie from its neighbours' to count as pure; higher keeps fewer.", "stable"
        ),
        "patch_size": Hint("Patch size", "Patch (height, width) within which superpixels are compared to pick pure ones.", "stable"),
        "sign": Hint("Signal sign"),
        "carry_background": Hint("Carry background", "Keep the previous pass's background estimate.", "advanced"),
    },
    NMFConfig: {
        "maxiter": Hint("NMF iterations", "HALS iterations per pass."),
        "support_threshold": Hint(
            "Support thr", "Fraction of each footprint's peak correlation that sets the cutoff for its new support.", "advanced"
        ),
        "deletion_threshold": Hint(
            "Deletion thr", "Footprints whose residual correlation over their support falls below this are deleted.", "stable"
        ),
        "min_brightness": Hint("Min brightness", "Footprints dimmer than this at every time point are deleted.", "stable"),
        "ring_model_start_pt": Hint(
            "Ring start", "HALS iterations before the ring background model is fit; more than NMF iterations disables it.", "advanced"
        ),
        "ring_radius": Hint("Ring radius", "Radius of the ring background model.", "advanced"),
        "background_downsampling_factor": Hint(
            "Bg downsample", "Spatial downsampling in each dimension before the neuropil's temporal basis is estimated.", "stable"
        ),
        "merge_threshold": Hint(
            "Merge thr", "Cutoff for the thresholded correlation images whose overlap decides which signals merge.", "important"
        ),
        "merge_overlap_threshold": Hint(
            "Merge overlap",
            "Fraction of one footprint's thresholded correlation image that must overlap another's for the two to merge.",
            "advanced",
        ),
        "update_frequency": Hint(
            "Update freq", "Every this many HALS iterations, supports are updated and footprints deleted and merged.", "stable"
        ),
        "c_nonneg": Hint("Nonnegative traces", "Keep traces nonnegative; voltage imaging may need it off.", "advanced"),
        "denoise": Hint("Denoise traces", "Run a denoiser on the traces at each HALS iteration.", "advanced"),
        "plot_en": Hint("Debug plots", tier="advanced"),
        "reassign_background": Hint("Reassign background", tier="advanced"),
        "sign": Hint("Signal sign"),
        "max_speckle_size": Hint(
            "Max speckle px",
            "After the last HALS iteration, detached islands of at most this many px are removed from each footprint "
            "and the traces refit; none disables it.",
            "advanced",
        ),
        "speckle_connectivity": Hint("Speckle connectivity", "4 or 8: how islands are counted. Keep 8 for thin structures.", "advanced"),
        "speckle_rel_threshold": Hint(
            "Speckle rel thr",
            "Islands are also counted among pixels at or above this fraction of the footprint's peak, so a faint halo "
            "does not protect specks.",
            "advanced",
        ),
        "final_refit_radius": Hint(
            "Refit radius", "After the last HALS iteration, dilate every support by this many px with no correlation gate; 0 disables it.", "advanced"
        ),
        "final_refit_iters": Hint("Refit iterations", "Ungated HALS iterations on the dilated supports.", "advanced"),
    },
}

# the plain __init__ arguments of BasePipeline that the popup's Runtime row shows
HINTS_RUNTIME = {
    "frame_batch_size": Hint("Frame batch", "Frames read and processed per batch."),
    "device": Hint("Device"),
    "log_level": Hint("Log level"),
    "load_into_ram": Hint("Load into RAM", "Read the whole movie into RAM before processing."),
    "device_memory_gb": Hint("GPU memory (GB)", "The memory the compute device may use, in GB."),
    "host_memory_gb": Hint("Host memory (GB)", "The memory the host may use, in GB."),
}


def hint_for(cls_config: type, name: str) -> Optional[Hint]:
    """Returns the hint for a config field, or None when it has none."""
    return HINTS.get(cls_config, {}).get(name)


def tier_of(cls_config: type, name: str) -> str:
    """Returns a config field's tier; normal when it has no hint."""
    hint = hint_for(cls_config=cls_config, name=name)
    return "normal" if hint is None else hint.tier


def stage_of(section: scraper.Section) -> str:
    """Returns the column a section goes in, from the module defining its configs; Other for any other module."""
    stages = [STAGE_BY_MODULE.get(cls.__module__, STAGE_OTHER) for cls in section.configs_by_kind.values()]
    return min(stages, key=STAGES_ORDER.index, default=STAGE_OTHER)
