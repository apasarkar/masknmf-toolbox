import os
from dataclasses import asdict
from typing import *

import numpy as np
import torch

import masknmf
from masknmf.arrays import ArrayLike
from masknmf.compression import CompressDenoiseStrategy, CompressStrategy
from masknmf.compression.preprocessing import MaximinSplineDetrend
from masknmf.demixing import NoSignalsDetectedError
from masknmf.motion_correction import (
    BaseRegistrationArray,
    PiecewiseRigidMotionCorrector,
    RigidMotionCorrector,
)
from masknmf.pipelines.configs.compression_configs import (
    CompressConfig,
    CompressDenoiseConfig,
)
from masknmf.pipelines.configs.demixing_configs import (
    CustomInitConfig,
    MultipassDemixingConfig,
    NMFConfig,
    SinglepassDemixingConfig,
    SpatialHighpassConfig,
    SuperpixelInitConfig,
)
from masknmf.pipelines.configs.motion_correction_configs import (
    PiecewiseRigidMotionCorrectionConfig,
    RigidMotionCorrectionConfig,
)
from masknmf.utils import display, torch_select_device


def run_singlepass_demixing(
    demixing_obj: "masknmf.SignalDemixer",
    singlepass_config: SinglepassDemixingConfig,
) -> Union[None, "masknmf.SignalDemixer"]:
    """
    Initialize and demix one pass.

    Args:
        demixing_obj (SignalDemixer): The demixer to advance.
        singlepass_config (SinglepassDemixingConfig): Init and NMF parameters for this pass.

    Returns:
        SignalDemixer | None: The demixer, or None when initialization found no signals.
    """
    init_config = singlepass_config.InitConfig
    nmf_config = singlepass_config.NMFConfig

    try:
        demixing_obj.initialize_signals(**asdict(init_config))
    except NoSignalsDetectedError:
        return None
    else:
        demixing_obj.demix(**asdict(nmf_config))
        return demixing_obj


def resolve_device(device: Literal["auto", "cuda", "cpu"]) -> str | torch.device:
    """Turn an ``"auto"`` device specification into a concrete torch device."""
    return torch_select_device() if device == "auto" else device


def build_detrender(
    num_frames: int,
    frame_rate: float,
    window_seconds: float,
    knot_seconds: float,
    device: Literal["auto", "cuda", "cpu"] = "auto",
) -> MaximinSplineDetrend:
    """
    Build a spline detrender sized from the recording's duration.

    Args:
        num_frames (int): Frames in the recording.
        frame_rate (float): Acquisition rate, in Hz.
        window_seconds (float): Rolling max-min window, in seconds.
        knot_seconds (float): Seconds of recording per spline knot.
        device (str): Device the detrender runs on; ``"auto"`` selects one.

    Returns:
        MaximinSplineDetrend: The detrender.
    """
    recording_seconds = num_frames / frame_rate
    return MaximinSplineDetrend(
        num_frames=num_frames,
        num_knots=max(4, int(recording_seconds / knot_seconds)),
        window=int(window_seconds * frame_rate),
        sigma=max(2.0, 0.3 * frame_rate),
        device=resolve_device(device),
    )


def build_pixel_weighting(
    moco_data: np.ndarray | ArrayLike,
    exclude_border_radius: int = 0,
) -> np.ndarray:
    """
    Per-pixel weights that drop pixels motion correction pushed outside the field of view.

    Args:
        moco_data: The registered movie. A :class:`BaseRegistrationArray` contributes its
            shifts; anything else is weighted uniformly.
        exclude_border_radius (int): Additionally zero this many pixels at each edge.

    Returns:
        np.ndarray: A (height, width) weight image.
    """
    if isinstance(moco_data, BaseRegistrationArray):
        mask = masknmf.motion_correction.moco_preprocessing.construct_moco_template(
            moco_data.shifts.cpu().numpy(), moco_data.shape[1:]
        ).astype("float")
    else:
        mask = np.ones((moco_data.shape[1], moco_data.shape[2])).astype("float")

    if exclude_border_radius > 0:
        mask[:exclude_border_radius, :] = 0
        mask[:, :exclude_border_radius] = 0
        mask[-1 * exclude_border_radius:, :] = 0
        mask[:, -1 * exclude_border_radius:] = 0
    return mask


def register(
    data: np.ndarray | ArrayLike,
    config: RigidMotionCorrectionConfig
    | PiecewiseRigidMotionCorrectionConfig
    | Literal["skip"]
    | None = None,
    *,
    device: Literal["auto", "cuda", "cpu"] = "auto",
    batch_size: int = 300,
    outpath: Optional[str] = None,
) -> np.ndarray | ArrayLike:
    """
    Motion correct a movie.

    Args:
        data: The raw (frames, height, width) stack.
        config: Rigid or piecewise-rigid parameters. None uses
            :class:`RigidMotionCorrectionConfig` defaults; ``"skip"`` returns ``data`` unchanged.
        device (str): Device pytorch runs on.
        batch_size (int): Frames held on the GPU at a time.
        outpath (str | None): Where to export the registration array; skipped when None.

    Returns:
        The registered movie, or ``data`` itself when correction was skipped.
    """
    if config is None:
        moco_strategy = RigidMotionCorrector(
            **asdict(RigidMotionCorrectionConfig()), device=device, batch_size=batch_size
        )
    elif isinstance(config, RigidMotionCorrectionConfig):
        moco_strategy = RigidMotionCorrector(
            **asdict(config), device=device, batch_size=batch_size
        )
    elif isinstance(config, PiecewiseRigidMotionCorrectionConfig):
        moco_strategy = PiecewiseRigidMotionCorrector(
            **asdict(config), device=device, batch_size=batch_size
        )
    else:
        moco_strategy = None

    if isinstance(config, str):
        if config.lower() == "skip":
            display("Not Running Motion Correction")
            return data
        raise ValueError("Invalid MotionCorrectionConfig input")
    if moco_strategy is None:
        raise ValueError("Invalid MotionCorrectionConfig input")

    if moco_strategy.template is None:
        moco_strategy.compute_template(data)
    moco_data = moco_strategy.motion_correct(data)
    moco_data.output_device = moco_data.strategy.device
    if outpath is not None:
        moco_data.export(os.path.abspath(outpath))
    return moco_data


def compress(
    moco_data: np.ndarray | ArrayLike,
    config: CompressConfig | CompressDenoiseConfig | None = None,
    *,
    pixel_weighting: Optional[np.ndarray] = None,
    detrender: Optional[MaximinSplineDetrend] = None,
    device: Literal["auto", "cuda", "cpu"] = "auto",
    outpath: Optional[str] = None,
) -> "masknmf.PMDArray":
    """
    Compress a registered movie, optionally denoising as it goes.

    Args:
        moco_data: The registered (frames, height, width) stack.
        config: Compression parameters. None runs compression with denoising.
        pixel_weighting (np.ndarray | None): Per-pixel weights, multiplied into any weighting
            the config already carries. Typically from :func:`build_pixel_weighting`.
        detrender (MaximinSplineDetrend | None): Detrender applied before compression.
        device (str): Device pytorch runs on.
        outpath (str | None): Where to export the PMD array; skipped when None.

    Returns:
        PMDArray: The compressed movie.
    """
    display("Running Compression")
    if config is None:
        curr_config = CompressDenoiseConfig()
        curr_config.pixel_weighting = pixel_weighting
        compress_strategy = CompressDenoiseStrategy(device=device, **asdict(curr_config))
    elif isinstance(config, (CompressConfig, CompressDenoiseConfig)):
        curr_config = asdict(config)
        if config.pixel_weighting is not None and pixel_weighting is not None:
            curr_config["pixel_weighting"] = config.pixel_weighting * pixel_weighting
        elif pixel_weighting is not None:
            curr_config["pixel_weighting"] = pixel_weighting
        strategy_cls = (
            CompressStrategy if isinstance(config, CompressConfig) else CompressDenoiseStrategy
        )
        compress_strategy = strategy_cls(device=device, **curr_config)
    else:
        raise ValueError("Invalid compression config")

    if detrender is not None:
        compress_strategy.detrender = detrender

    compressed_results = compress_strategy.compress(moco_data)
    if outpath is not None:
        compressed_results.export(outpath)
    return compressed_results


def default_filtered_demixing_config(
    detrender: Optional[MaximinSplineDetrend] = None,
) -> MultipassDemixingConfig:
    """Two positive-signed passes over the high-pass filtered movie."""
    conf_list = []
    for corr_threshold in [0.8, 0.8]:
        conf_list.append(
            SinglepassDemixingConfig(
                SuperpixelInitConfig(
                    mad_correlation_threshold=corr_threshold,
                    detrender=detrender,
                    sign="positive",
                ),
                NMFConfig(
                    support_threshold=(0.95, corr_threshold),
                    ring_model_start_pt=None,
                    detrender=detrender,
                ),
            )
        )
    return MultipassDemixingConfig(conf_list)


def default_unfiltered_demixing_config(
    detrender: Optional[MaximinSplineDetrend] = None,
) -> MultipassDemixingConfig:
    """Three positive-signed passes over the unfiltered movie, with the ring model on."""
    conf_list = []
    for corr_threshold, support_threshold in [(0.8, 0.4), (0.8, 0.4), (0.8, 0.4)]:
        conf_list.append(
            SinglepassDemixingConfig(
                SuperpixelInitConfig(
                    mad_correlation_threshold=corr_threshold,
                    detrender=detrender,
                    sign="positive",
                ),
                NMFConfig(
                    support_threshold=(0.95, support_threshold),
                    ring_model_start_pt=0,
                    detrender=detrender,
                ),
            )
        )
    return MultipassDemixingConfig(conf_list)


def demix_two_phase(
    pmd_denoise: "masknmf.PMDArray",
    frame_rate: float,
    *,
    filtered_config: Optional[MultipassDemixingConfig] = None,
    unfiltered_config: Optional[MultipassDemixingConfig] = None,
    spatial_highpass_config: Optional[SpatialHighpassConfig] = None,
    device: Literal["auto", "cuda", "cpu"] = "auto",
    frame_batch_size: int = 300,
    num_frames: Optional[int] = None,
    outpath: Optional[str] = None,
) -> "masknmf.DemixingResults":
    """
    Demix a compressed movie, first on a spatially high-pass filtered copy and then on the
    unfiltered one seeded from those signals.

    Args:
        pmd_denoise (PMDArray): The compressed movie.
        frame_rate (float): Acquisition rate, in Hz, used to size the detrender.
        filtered_config (MultipassDemixingConfig | None): Passes over the filtered movie.
            None uses :func:`default_filtered_demixing_config`.
        unfiltered_config (MultipassDemixingConfig | None): Passes over the unfiltered movie.
            None uses :func:`default_unfiltered_demixing_config`. Its first pass is re-seeded
            from the filtered result regardless.
        spatial_highpass_config (SpatialHighpassConfig | None): The high-pass filter.
        device (str): Device pytorch runs on.
        frame_batch_size (int): Frames held on the GPU at a time.
        num_frames (int | None): Frames in the recording; taken from ``pmd_denoise`` when None.
        outpath (str | None): Where to export the results; skipped when None.

    Returns:
        DemixingResults: The demixed signals.

    Raises:
        ValueError: If no pass over either movie completed.
    """
    device = resolve_device(device)
    display("Running demixing analysis")

    if spatial_highpass_config is None:
        spatial_highpass_config = SpatialHighpassConfig()
    spatial_filt_pmd = masknmf.demixing.filters.spatial_filter_pmd(
        pmd_denoise,
        batch_size=frame_batch_size,
        filter_sigma=spatial_highpass_config.filter_sigma,
        device=device,
    )
    torch.cuda.empty_cache()

    highpass_pmd_demixer = masknmf.demixing.signal_demixer.SignalDemixer(
        spatial_filt_pmd, device=device, frame_batch_size=frame_batch_size
    )

    if num_frames is None:
        num_frames = pmd_denoise.shape[0]
    detrender = build_detrender(num_frames, frame_rate, 20.0, 20.0, device)

    if filtered_config is None:
        filtered_config = default_filtered_demixing_config(detrender)
    if unfiltered_config is None:
        unfiltered_config = default_unfiltered_demixing_config(detrender)

    curr_demix_results = None
    for k in range(len(filtered_config.DemixingConfigs)):
        highpass_pmd_demixer = run_singlepass_demixing(
            highpass_pmd_demixer, filtered_config.DemixingConfigs[k]
        )
        if highpass_pmd_demixer is not None:
            curr_demix_results = highpass_pmd_demixer.results
        else:
            if curr_demix_results is None:
                raise ValueError(
                    "The demixer did not identify any signals in the highpass filtered movie. "
                    "Lower thresholds or inspect data to resolve this issue."
                )
            break
        torch.cuda.empty_cache()

    ac_arr = curr_demix_results.ac_array
    custom_unfiltered_conf = SinglepassDemixingConfig(
        CustomInitConfig(ac_arr.export_a(), ac_arr.export_c(), c_nonneg=True),
        unfiltered_config.DemixingConfigs[0].NMFConfig,
    )

    unfiltered_pmd_demixer = masknmf.demixing.signal_demixer.SignalDemixer(
        pmd_denoise, device=device, frame_batch_size=frame_batch_size
    )

    latest_demix_results = None
    for k in range(len(unfiltered_config.DemixingConfigs)):
        conf = custom_unfiltered_conf if k == 0 else unfiltered_config.DemixingConfigs[k]
        unfiltered_pmd_demixer = run_singlepass_demixing(unfiltered_pmd_demixer, conf)
        if unfiltered_pmd_demixer is not None:
            latest_demix_results = unfiltered_pmd_demixer.results
        else:
            if latest_demix_results is None:
                raise ValueError(
                    "The unfiltered pmd demixer did not complete a full round of demixing."
                )
            break

    if outpath is not None:
        latest_demix_results.export(os.path.abspath(outpath))
    return latest_demix_results
