from dataclasses import asdict
import masknmf
from masknmf.compression import CompressStrategy, CompressDenoiseStrategy
from masknmf.arrays import LazyFrameLoader, ArrayLike
from masknmf.motion_correction import BaseRegistrationArray, DummyMotionCorrector, RigidMotionCorrector, PiecewiseRigidMotionCorrector
from masknmf.utils import display

from masknmf.compression.preprocessing import MaximinSplineDetrend

from masknmf.pipelines._base import BasePipeline
from masknmf.pipelines.configs.motion_correction_configs import RigidMotionCorrectionConfig, PiecewiseRigidMotionCorrectionConfig
from masknmf.pipelines.configs.compression_configs import CompressConfig, CompressDenoiseConfig
from masknmf.pipelines.configs.demixing_configs import NMFConfig, CustomInitConfig, SuperpixelInitConfig, SpatialHighpassConfig, SinglepassDemixingConfig, MultipassDemixingConfig, MultipassDemixingConfigs
from pathlib import Path
from typing import *
import numpy as np
import os
from numbers import Integral
import torch
import cv2
import h5py

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

class GlutamateCalciumSpinePipeline(BasePipeline):

    def __init__(self,
                 output_folder: str | Path | None = None,
                 motion_correct_config: RigidMotionCorrectionConfig | None = None,
                 compress_config: CompressDenoiseConfig | None = None,
                 demixing_config: MultipassDemixingConfigs | None = None,
                 frame_batch_size: int = 300,
                 device: Literal["auto", "cuda", "cpu"] = "auto"):
        super().__init__(output_folder=output_folder, frame_batch_size=frame_batch_size, device=device,
                         motion_correct_config=motion_correct_config, compress_config=compress_config,
                         demixing_config=demixing_config)

    @classmethod
    def default_configs(cls) -> dict:
        """
        Rigid motion correction allowing large shifts, compression with denoising in small blocks over temporally
        averaged frames, and two positive-signed demixing passes tuned for spines.
        """
        passes = []
        for _ in range(2):
            init_config = SuperpixelInitConfig(residual_threshold=0.1, sign="positive")
            # ring_model_start_pt past maxiter keeps the ring model off
            nmf_config = NMFConfig(support_threshold=(0.95, 0.7),
                                   ring_model_start_pt=41,
                                   min_brightness=0.0,
                                   merge_threshold=0.7,
                                   reassign_background=False)
            passes.append(SinglepassDemixingConfig(init_config, nmf_config))
        return {'motion_correct_config': RigidMotionCorrectionConfig(max_shifts=(40, 40)),
                'compress_config': CompressDenoiseConfig(block_sizes=(10, 10), temporal_avg_factor=10),
                'demixing_config': MultipassDemixingConfig(passes)}

    def run(self,
            glutamate_channel: np.ndarray | ArrayLike | None,
            calcium_channel: np.ndarray | ArrayLike | None,
            exclude_initial_frames: int = 200) -> Path:
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
        device = self.torch_device
        self.run_config = {"exclude_initial_frames": exclude_initial_frames}
        run_folder = self.create_run_folder()
        glu_path = os.path.join(run_folder, "results.glutamate.hdf5")
        ca_path = os.path.join(run_folder, "results.calcium.hdf5")
        display(f"Writing results to {run_folder}")
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
        retained_frames = np.arange(exclude_initial_frames, exclude_initial_frames + reference_input.shape[0])
        for channel, path in ((glu, glu_path), (calcium, ca_path)):
            if channel is not None:
                with h5py.File(path, "w") as f:
                    f["retained_frames"] = retained_frames

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
            glu_moco_array.export(glu_path)
            glu_moco_array_dense = glu_moco_array[:].cpu().numpy() #Loads it all into RAM
        else:
            glu_moco_array = None
            glu_moco_array_dense = None

        if calcium is not None:
            calcium_moco_array = corrector.motion_correct(reference_movie=pmd_pre_moco_reference,
                                                      target_movie=calcium)
            calcium_moco_array.export(ca_path)
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

        compress_strat = self.compress_strategy(self.compress_config)

        if glu_video is not None:
            pmd_glu = compress_strat.compress(glu_video)
            pmd_glu.export(glu_path)
        else:
            pmd_glu = None

        if calcium_video is not None:
            pmd_ca = compress_strat.compress(calcium_video)
            pmd_ca.export(ca_path)
        else:
            pmd_ca = None


        ## If the demixer has spines in the glutamate channel
        if pmd_glu is not None:
            glu_pmd_demixer = masknmf.demixing.signal_demixer.SignalDemixer(
                pmd_glu,
                device=device)

            glu_pmd_demixer_results = self.run_multipass(glu_pmd_demixer, self.demixing_config)

            ## Now pull out "whole dendrite" events. Can refactor this to a helper function to keep the "run" function readable
            glu_pmd_demixer_global= masknmf.demixing.signal_demixer.SignalDemixer(
                pmd_glu,
                device=device
            )
            glu_pmd_demixer_global.initialize_signals(is_custom=True,
                                                      spatial_footprints=(cross_channel_mask[:,:, None].astype("float"))
                                                      )

            glu_pmd_demixer_global.demix(**NMF_JUST_HALS)

            glu_pmd_demixer_results.export(glu_path)
            glu_pmd_demixer_global.results.export(glu_path, prefix="global")

            if pmd_ca is not None:
                ## Pull out the spatial/temporal footprints from the glutamate movie
                spatialfoot_spines_glu = glu_pmd_demixer_results.signals_array.export_spatial_demixed()  # spatial_footprints

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

                ca_pmd_demixer.results.export(ca_path)
                ca_pmd_demixer_global.results.export(ca_path, prefix="global")


        else: #In this case there is only a calcium channel
            ca_pmd_demixer = masknmf.demixing.signal_demixer.SignalDemixer(
                pmd_ca,
                device=device)

            ca_pmd_demixer_results = self.run_multipass(ca_pmd_demixer, self.demixing_config)


            ## Now pull out "whole dendrite" events. Can refactor this to a helper function to keep the "run" function readable
            ca_pmd_demixer_global = masknmf.demixing.signal_demixer.SignalDemixer(
                pmd_ca,
                device=device
            )
            ca_pmd_demixer_global.initialize_signals(is_custom=True,
                                                     spatial_footprints=(cross_channel_mask[:, :, None].astype("float"))
                                                     )

            ca_pmd_demixer_global.demix(**NMF_JUST_HALS)

            ca_pmd_demixer_results.export(ca_path)
            ca_pmd_demixer_global.results.export(ca_path, prefix="global")

        return run_folder



















