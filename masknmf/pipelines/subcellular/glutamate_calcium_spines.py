from dataclasses import asdict
import masknmf
from masknmf.compression import CompressStrategy, CompressDenoiseStrategy
from masknmf.arrays import LazyFrameLoader, ArrayLike
from masknmf.motion_correction import BaseRegistrationArray, DummyMotionCorrector, RigidMotionCorrector, PiecewiseRigidMotionCorrector
from masknmf.utils import display
from masknmf.demixing import NoSignalsDetectedError, DemixingError

from masknmf.compression.preprocessing import MaximinSplineDetrend

from masknmf.pipelines._base import BasePipeline
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
from datetime import datetime

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

def run_singlepass_demixing(demixing_obj: masknmf.SignalDemixer,
                            singlepass_config: SinglepassDemixingConfig) -> None | masknmf.SignalDemixer:
    init_config = singlepass_config.InitConfig
    nmf_config = singlepass_config.NMFConfig
    try:
        demixing_obj.initialize_signals(**asdict(init_config))
    except NoSignalsDetectedError:
        return None
    else:
        demixing_obj.demix(**asdict(nmf_config))
        return demixing_obj

class GlutamateCalciumSpinePipeline(BasePipeline):

    def __init__(self,
                 output_folder: str | Path | None = None,
                 motion_correct_config: RigidMotionCorrectionConfig | None = None,
                 compress_config: CompressDenoiseConfig | None = None,
                 demixing_config: MultipassDemixingConfig | None = None,
                 frame_batch_size: int = 300,
                 device: Literal["auto", "cuda", "cpu"] = "auto"):

        if output_folder is None:
            self._output_folder = None
        else:
            output_folder = Path(output_folder).expanduser().resolve()
            if output_folder.exists() and not output_folder.is_dir():
                raise NotADirectoryError(
                    f"output_folder exists and is not a directory: {output_folder}"
                )
            self._output_folder = output_folder

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
    def output_folder(self) -> Path | None:
        return self._output_folder

    @property
    def frame_batch_size(self) -> int:
        return self._frame_batch_size

    @property
    def device(self) -> Literal["auto", "cuda", "cpu"]:
        return self._device

    @property
    def config(self):
        return {'motion_correct_config': self.motion_correct_config,
                'compress_config': self.compress_config,
                'demixing_config': self.demixing_config,
                'frame_batch_size': self.frame_batch_size,
                'device': self.device}

    def create_run_folder(self) -> Path:
        base = Path.cwd() if self._output_folder is None else self._output_folder
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        candidate = base / f"{stamp}_glutamate_calcium_spine_results"
        suffix = 0
        while True:
            try:
                candidate.mkdir(parents=True, exist_ok=False)
                break
            except FileExistsError:
                suffix += 1
                candidate = base / f"{stamp}_glutamate_calcium_spine_results_{suffix}"
        return candidate


    def run(self,
            glutamate_channel: np.ndarray | ArrayLike | None,
            calcium_channel: np.ndarray | ArrayLike | None,
            exclude_initial_frames: int = 200):
        """
        This routine runs the pipeline for processing single-plane glutamate and calcium imaging videos.
        It can analyze joint calcium/glutamate recordings or just process a single channel of either glutamate or calcium data

        Key assumptions:
            - All datasets fully fit into the RAM of the computer (eventually can generalize)
            - The number of frames in both channels is the same (otherwise the joint registration is not as meaningful)
        Args:
            glutamate_channel (np.ndarray | ArrayLike | None):
            calcium_channel (np.ndarray | ArrayLike | None):
        """
        device = torch_select_device(self.device)
        final_output_folder = self.create_run_folder()
        if not isinstance(exclude_initial_frames, Integral):
            raise ValueError("exclude_initial_frames should be a positive integer, 200 is likely to be a good default.")
        else:
            exclude_initial_frames = int(exclude_initial_frames)
            if exclude_initial_frames <= 0:
                exclude_initial_frames = 0

        if glutamate_channel is None and calcium_channel is None:
            raise ValueError("No functional data was provided, both channels are None")

        if glutamate_channel is not None:
            glu = np.from_dlpack(glutamate_channel[exclude_initial_frames:], device='cpu') ##
        else:
            glu = None
        if calcium_channel is not None:
            calcium = np.from_dlpack(calcium_channel[exclude_initial_frames:], device='cpu')
        else:
            calcium = None

        if glu is not None and calcium is not None:
            if glu.shape != calcium.shape:
                raise ValueError("The shape of both channels should be identical")

        if calcium is not None:
            reference_input = calcium
        else:
            reference_input = glu

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
            glu_moco_array.export(os.path.join(final_output_folder, "glutamate_moco.hdf5"))
            glu_moco_array_dense = glu_moco_array[:].cpu().numpy() #Loads it all into RAM
        else:
            glu_moco_array = None
            glu_moco_array_dense = None

        if calcium is not None:
            calcium_moco_array = corrector.motion_correct(reference_movie=pmd_pre_moco_reference,
                                                      target_movie=calcium)
            calcium_moco_array.export(os.path.join(final_output_folder, "calcium_moco.hdf5"))
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
            pmd_glu.export(os.path.join(final_output_folder, "pmd_glutamate.hdf5"))
        else:
            pmd_glu = None

        if calcium_video is not None:
            pmd_ca = compress_strat.compress(calcium_video)
            pmd_ca.export(os.path.join(final_output_folder, "pmd_calcium.hdf5"))
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

            glu_pmd_demixer_results.export(os.path.join(final_output_folder, "glutamate_spine_demixing.hdf5"))
            glu_pmd_demixer_global.results.export(os.path.join(final_output_folder, "glutamate_global_activity_demixing.hdf5"))

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

                ca_pmd_demixer.results.export(os.path.join(final_output_folder, "calcium_spine_demixing.hdf5"))
                ca_pmd_demixer_global.results.export(
                    os.path.join(final_output_folder, "calcium_global_activity_demixing.hdf5"))


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

            ca_pmd_demixer_results.export(os.path.join(final_output_folder, "calcium_spine_demixing.hdf5"))
            ca_pmd_demixer_global.results.export(
                os.path.join(final_output_folder, "calcium_global_activity_demixing.hdf5"))



















