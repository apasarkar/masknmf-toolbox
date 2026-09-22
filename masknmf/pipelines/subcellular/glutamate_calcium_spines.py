from dataclasses import asdict
import masknmf
from masknmf.compression import CompressStrategy, CompressDenoiseStrategy
from masknmf.arrays import LazyFrameLoader, ArrayLike
from masknmf.motion_correction import BaseRegistrationArray, DummyMotionCorrector, RigidMotionCorrector, PiecewiseRigidMotionCorrector, OphysArray
from masknmf.utils import display, drop_group
from masknmf.utils._serialization import save_dict
from masknmf.demixing import NoSignalsDetectedError, DemixingError

from masknmf.compression.preprocessing import MaximinSplineDetrend

from masknmf.pipelines._base import BasePipeline
from masknmf.pipelines.stages import run_singlepass_demixing
from masknmf.pipelines.configs.motion_correction_configs import RigidMotionCorrectionConfig, PiecewiseRigidMotionCorrectionConfig
from masknmf.pipelines.configs.compression_configs import CompressConfig, CompressDenoiseConfig
from masknmf.pipelines.configs.demixing_configs import SuperpixelInitConfig, SinglepassDemixingConfig, NMFConfig
from masknmf.pipelines.configs.demixing_configs import NMFConfig, CustomInitConfig, SuperpixelInitConfig, SpatialHighpassConfig, SinglepassDemixingConfig, MultipassDemixingConfig
from pathlib import Path
from masknmf.utils import torch_select_device
from typing import *
import numpy as np
import os
from numbers import Integral
import torch
import cv2

DEFAULT_MOTION_CORRECTION_CONFIG = RigidMotionCorrectionConfig(max_shifts=(40, 40))
DEFAULT_COMPRESSION_CONFIG = CompressDenoiseConfig(block_sizes=(10, 10),
                                                   max_components=20,
                                                   sim_conf=5,
                                                   spatial_avg_factor=1,
                                                   temporal_avg_factor=10,
                                                   num_epochs=10)

DEFAULT_SPINE_SUPERPIXEL_CONFIG = SuperpixelInitConfig(residual_threshold=0.1,
                                                       sign="positive")
DEFAULT_SPINE_NMF_CONFIG = NMFConfig(maxiter=40,
                                     support_threshold=np.linspace(0.95, 0.7, 40).tolist(),
                                     ring_model_start_pt=41,
                                     min_brightness=0.0,
                                     merge_threshold=0.7,
                                     merge_overlap_threshold=0.6,
                                     update_frequency=4,
                                     c_nonneg=True,
                                     denoise=False,
                                     plot_en=False,
                                     reassign_background=False)

NMF_JUST_HALS = {'maxiter': 40,
                'deletion_threshold': 0.2,
                'ring_model_start_pt': 41,
                'min_brightness': 0.0,
                'update_frequency': 41,
                'c_nonneg': True}

RUN_GROUP = "SpinePipelineRun"
CALCIUM_PREFIX = "calcium/"
GLOBAL_PREFIX = "global/"
INDICATOR_SIGNS = ("positive", "negative")

def otsu_threshold(image):
    norm_image = (image - image.min()) / (image.max() - image.min()) * 255
    norm_image = norm_image.astype(np.uint8)
    # Apply Otsu's thresholding
    _, mask = cv2.threshold(norm_image, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return mask

def get_std_based_mask(stack):
    std_img = np.std(stack[:], axis = 0)
    mask = otsu_threshold(std_img)
    return mask


def _load_channel(channel: np.ndarray | ArrayLike | None,
                  exclude_initial_frames: int,
                  indicator_sign: Literal["positive", "negative"]) -> np.ndarray | None:
    """The kept frames of one channel in RAM, flipped about the mean image for a negative indicator."""
    if channel is None:
        return None
    movie = np.from_dlpack(channel[exclude_initial_frames:], device='cpu')
    if movie.ndim != 3:
        raise ValueError(f"Each channel should be (frames, height, width), got shape {movie.shape}")
    if indicator_sign == "negative":
        movie = OphysArray(movie, negative_indicator=True, include_mean=True, device="cpu")[:].numpy()
    return movie


class GlutamateCalciumSpinePipeline(BasePipeline):

    def __init__(self,
                 outpath: str | Path = "results.hdf5",
                 motion_correct_config: RigidMotionCorrectionConfig | None = None,
                 compress_config: CompressDenoiseConfig | None = None,
                 demixing_config: MultipassDemixingConfig | None = None,
                 frame_batch_size: int = 300,
                 device: Literal["auto", "cuda", "cpu"] = "auto"):
        """
        Args:
            outpath (str | Path): The one results file every stage writes into. The primary channel (glutamate
                when given, else calcium) takes the plain group names (``RigidRegistrationArray``, ``PMDArray``,
                ``DemixingResults``), so the viewers and ``masknmf view`` open it directly. Its whole-dendrite
                fit goes under ``global/``; in a two-channel run the calcium channel's groups go under
                ``calcium/`` and ``calcium/global/``. ``SpinePipelineRun`` records the kept frames, the mask
                and the indicator signs.
        """
        outpath = Path(outpath).expanduser().resolve()
        if outpath.is_dir():
            raise IsADirectoryError(f"outpath is a directory, expected an .hdf5 file path: {outpath}")
        self._outpath = outpath

        self.motion_correct_config = motion_correct_config
        self.compress_config = compress_config
        self.demixing_config = demixing_config
        self._frame_batch_size = frame_batch_size
        self._device = device

    @property
    def motion_correct_config(self) -> RigidMotionCorrectionConfig | None:
        return self._motion_correct_config

    @motion_correct_config.setter
    def motion_correct_config(self, updated_config: RigidMotionCorrectionConfig | None):
        if updated_config is None:
            self._motion_correct_config = DEFAULT_MOTION_CORRECTION_CONFIG
        else:
            self._motion_correct_config = updated_config

    @property
    def compress_config(self) -> CompressConfig | CompressDenoiseConfig | None:
        return self._compress_config

    @compress_config.setter
    def compress_config(self, updated_config: CompressDenoiseConfig | None):
        if updated_config is None:
            self._compress_config = DEFAULT_COMPRESSION_CONFIG
        else:
            self._compress_config = updated_config

    @property
    def demixing_config(self) -> MultipassDemixingConfig | None:
        return self._demixing_config

    @demixing_config.setter
    def demixing_config(self, updated_config: MultipassDemixingConfig | None):
        if updated_config is None:
            conf_list = []
            curr_demix_conf = SinglepassDemixingConfig(DEFAULT_SPINE_SUPERPIXEL_CONFIG, DEFAULT_SPINE_NMF_CONFIG)
            for _ in range(2):
                conf_list.append(curr_demix_conf)
            self._demixing_config = MultipassDemixingConfig(conf_list)
        else:
            if len(updated_config.DemixingConfigs) < 1:
                raise ValueError("Must have sufficient configs for at least one pass of NMF in demixing configs")
            self._demixing_config = updated_config

    @property
    def outpath(self) -> Path:
        return self._outpath

    @property
    def frame_batch_size(self) -> int:
        return self._frame_batch_size

    @property
    def device(self) -> Literal["auto", "cuda", "cpu"]:
        return self._device

    @property
    def config(self):
        return {'outpath': self.outpath,
                'motion_correct_config': self.motion_correct_config,
                'compress_config': self.compress_config,
                'demixing_config': self.demixing_config,
                'frame_batch_size': self.frame_batch_size,
                'device': self.device}


    def run(self,
            glutamate_channel: np.ndarray | ArrayLike | None,
            calcium_channel: np.ndarray | ArrayLike | None,
            exclude_initial_frames: int = 200,
            glutamate_indicator_sign: Literal["positive", "negative"] = "positive",
            calcium_indicator_sign: Literal["positive", "negative"] = "positive"):
        """
        This routine runs the pipeline for processing single-plane glutamate and calcium imaging videos.
        It can analyze joint calcium/glutamate recordings or just process a single channel of either glutamate or calcium data

        Key assumptions:
            - All datasets fully fit into the RAM of the computer (eventually can generalize)
            - The number of frames in both channels is the same (otherwise the joint registration is not as meaningful)
        Args:
            glutamate_channel (np.ndarray | ArrayLike | None): (frames, height, width)
            calcium_channel (np.ndarray | ArrayLike | None): (frames, height, width)
            exclude_initial_frames (int): Leading frames to drop before any processing.
            glutamate_indicator_sign (str): "negative" for an indicator that dims with activity (e.g. ASAP-family
                voltage sensors). That channel is flipped about its mean image before registration, as
                :class:`OphysArray` does, so activity reads as positive deflections everywhere downstream and the
                positive-signed detection and nonnegative traces apply unchanged.
            calcium_indicator_sign (str): The same, for the calcium channel.

        Returns:
            Path: The results file.
        """
        device = torch_select_device(self.device)
        for name, sign in (("glutamate_indicator_sign", glutamate_indicator_sign),
                           ("calcium_indicator_sign", calcium_indicator_sign)):
            if sign not in INDICATOR_SIGNS:
                raise ValueError(f"{name} must be one of {INDICATOR_SIGNS}, got {sign!r}")
        if not isinstance(exclude_initial_frames, Integral):
            raise ValueError("exclude_initial_frames should be a positive integer, 200 is likely to be a good default.")
        else:
            exclude_initial_frames = int(exclude_initial_frames)
            if exclude_initial_frames <= 0:
                exclude_initial_frames = 0

        if glutamate_channel is None and calcium_channel is None:
            raise ValueError("No functional data was provided, both channels are None")

        glu = _load_channel(glutamate_channel, exclude_initial_frames, glutamate_indicator_sign)
        calcium = _load_channel(calcium_channel, exclude_initial_frames, calcium_indicator_sign)

        if glu is not None and calcium is not None:
            if glu.shape != calcium.shape:
                raise ValueError("The shape of both channels should be identical")

        if calcium is not None:
            reference_input = calcium
        else:
            reference_input = glu

        # glutamate, when present, owns the plain group names; calcium moves under calcium/ beside it
        glu_prefix = ""
        calcium_prefix = CALCIUM_PREFIX if glu is not None else ""
        self.outpath.parent.mkdir(parents=True, exist_ok=True)
        for stale in (CALCIUM_PREFIX.rstrip("/"), GLOBAL_PREFIX.rstrip("/"), RUN_GROUP):
            drop_group(str(self.outpath), stale)
        outpath = str(self.outpath)

        pre_moco_strategy = masknmf.CompressStrategy(block_sizes=self.compress_config.block_sizes,
                                               max_components=self.compress_config.max_components,
                                               max_consecutive_failures=self.compress_config.max_consecutive_failures,
                                               temporal_avg_factor=2,
                                               spatial_avg_factor=4,
                                               frame_batch_size=self.frame_batch_size,
                                                device=device)
        pmd_pre_moco_reference = pre_moco_strategy.compress(reference_input)
        pmd_pre_moco_reference.to(device) #Move it to the accelerator

        corrector = RigidMotionCorrector(**asdict(self.motion_correct_config),
                                         device=device,
                                         batch_size=self.frame_batch_size)

        corrector.compute_template(pmd_pre_moco_reference)

        if glu is not None:
            glu_moco_array = corrector.motion_correct(reference_movie=pmd_pre_moco_reference,
                                                  target_movie=glu)
            glu_moco_array.export(outpath, prefix=glu_prefix)
            glu_moco_array_dense = glu_moco_array[:].cpu().numpy() #Loads it all into RAM
        else:
            glu_moco_array = None
            glu_moco_array_dense = None

        if calcium is not None:
            calcium_moco_array = corrector.motion_correct(reference_movie=pmd_pre_moco_reference,
                                                      target_movie=calcium)
            calcium_moco_array.export(outpath, prefix=calcium_prefix)
            calcium_moco_array_dense = calcium_moco_array[:].cpu().numpy() #Loads it all into RAM
        else:
            calcium_moco_array = None
            calcium_moco_array_dense = None

        if calcium_moco_array_dense is not None:
            cross_channel_mask = get_std_based_mask(calcium_moco_array_dense)
        else:
            cross_channel_mask = get_std_based_mask(glu_moco_array_dense)

        if glu_moco_array_dense is not None:
            glu_video =  glu_moco_array_dense * cross_channel_mask.astype(np.float32)[None, :, :]
        else:
            glu_video = None

        if calcium_moco_array_dense is not None:
            calcium_video = calcium_moco_array_dense * cross_channel_mask.astype(np.float32)[None, :, :]
        else:
            calcium_video = None

        compress_strat = masknmf.CompressDenoiseStrategy(**asdict(self.compress_config), device=device)

        if glu_video is not None:
            pmd_glu = compress_strat.compress(glu_video)
            pmd_glu.export(outpath, prefix=glu_prefix)
        else:
            pmd_glu = None

        if calcium_video is not None:
            pmd_ca = compress_strat.compress(calcium_video)
            pmd_ca.export(outpath, prefix=calcium_prefix)
        else:
            pmd_ca = None


        ## If the demixer has spines in the glutamate channel
        if pmd_glu is not None:
            glu_pmd_demixer = masknmf.demixing.signal_demixer.SignalDemixer(
                pmd_glu,
                device=device)

            for k in range(len(self.demixing_config.DemixingConfigs)):
                glu_pmd_demixer = run_singlepass_demixing(glu_pmd_demixer,
                                                           self.demixing_config.DemixingConfigs[k])
                if k == 0:
                    if glu_pmd_demixer is None:
                        raise ValueError("With this set of demixing parameters, the glu demixer did not find any signals")
                    else:
                        curr_results = glu_pmd_demixer.results
                else:
                    if glu_pmd_demixer is None:
                        break ## curr_results from previous round will return
                    else:
                        curr_results = glu_pmd_demixer.results
            glu_pmd_demixer_results = curr_results

            ## Now pull out "whole dendrite" events. Can refactor this to a helper function to keep the "run" function readable
            glu_pmd_demixer_global= masknmf.demixing.signal_demixer.SignalDemixer(
                pmd_glu,
                device=device
            )
            glu_pmd_demixer_global.initialize_signals(is_custom=True,
                                                      spatial_footprints=(cross_channel_mask[:,:, None].astype("float"))
                                                      )

            glu_pmd_demixer_global.demix(**NMF_JUST_HALS)

            glu_pmd_demixer_results.export(outpath, prefix=glu_prefix)
            glu_pmd_demixer_global.results.export(outpath, prefix=glu_prefix + GLOBAL_PREFIX)

            if pmd_ca is not None:
                ## Pull out the spatial/temporal footprints from the glutamate movie
                spatialfoot_spines_glu = glu_pmd_demixer_results.ac_array.export_a()  # spatial_footprints

                ca_pmd_demixer = masknmf.demixing.signal_demixer.SignalDemixer(
                    pmd_ca,
                    device=device)

                ca_pmd_demixer.initialize_signals(is_custom=True,
                                                  spatial_footprints=spatialfoot_spines_glu)
                ca_pmd_demixer.demix(**NMF_JUST_HALS)

                ## Pull out global events from the calcium data
                ca_pmd_demixer_global = masknmf.demixing.signal_demixer.SignalDemixer(
                    pmd_ca,
                    device=device)

                ca_pmd_demixer_global.initialize_signals(is_custom=True,
                                                         spatial_footprints=(
                                                             cross_channel_mask[:, :, None].astype("float"))
                                                         )

                ca_pmd_demixer_global.demix(**NMF_JUST_HALS)

                ca_pmd_demixer.results.export(outpath, prefix=calcium_prefix)
                ca_pmd_demixer_global.results.export(outpath, prefix=calcium_prefix + GLOBAL_PREFIX)


        else: #In this case there is only a calcium channel
            ca_pmd_demixer = masknmf.demixing.signal_demixer.SignalDemixer(
                pmd_ca,
                device=device)

            for k in range(len(self.demixing_config.DemixingConfigs)):
                ca_pmd_demixer = run_singlepass_demixing(ca_pmd_demixer,
                                                          self.demixing_config.DemixingConfigs[k])
                if k == 0:
                    if ca_pmd_demixer is None:
                        raise ValueError("With this set of demixing parameters, the calcium demixer did not find any signals")
                    else:
                        curr_results = ca_pmd_demixer.results
                else:
                    if ca_pmd_demixer is None:
                        break ## curr_results from previous round will return
                    else:
                        curr_results = ca_pmd_demixer.results
            ca_pmd_demixer_results = curr_results


            ## Now pull out "whole dendrite" events. Can refactor this to a helper function to keep the "run" function readable
            ca_pmd_demixer_global = masknmf.demixing.signal_demixer.SignalDemixer(
                pmd_ca,
                device=device
            )
            ca_pmd_demixer_global.initialize_signals(is_custom=True,
                                                     spatial_footprints=(cross_channel_mask[:, :, None].astype("float"))
                                                     )

            ca_pmd_demixer_global.demix(**NMF_JUST_HALS)

            ca_pmd_demixer_results.export(outpath, prefix=calcium_prefix)
            ca_pmd_demixer_global.results.export(outpath, prefix=calcium_prefix + GLOBAL_PREFIX)

        save_dict({"retained_frames": np.arange(exclude_initial_frames,
                                                exclude_initial_frames + reference_input.shape[0]),
                   "exclude_initial_frames": exclude_initial_frames,
                   "primary_channel": "glutamate" if glu is not None else "calcium",
                   "channels": np.array([name for name, movie in (("glutamate", glu), ("calcium", calcium))
                                         if movie is not None]),
                   "glutamate_indicator_sign": glutamate_indicator_sign,
                   "calcium_indicator_sign": calcium_indicator_sign,
                   "cross_channel_mask": cross_channel_mask.astype(np.uint8)},
                  filename=outpath, group=RUN_GROUP, exists_ok=True)
        display(f"Wrote {outpath}")
        return self.outpath



















