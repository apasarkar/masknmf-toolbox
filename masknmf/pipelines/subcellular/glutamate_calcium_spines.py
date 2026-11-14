from dataclasses import asdict
import masknmf
from masknmf.compression import CompressStrategy, CompressDenoiseStrategy
from masknmf.arrays import LazyFrameLoader, ArrayLike, OphysArray
from masknmf.motion_correction import BaseRegistrationArray, DummyMotionCorrector, RigidMotionCorrector, PiecewiseRigidMotionCorrector
from masknmf.utils import display

from masknmf.compression.preprocessing import MaximinSplineDetrend

from masknmf.pipelines._base import BasePipeline
from masknmf.pipelines.configs.motion_correction_configs import RigidMotionCorrectionConfig, PiecewiseRigidMotionCorrectionConfig, MotionCorrectionConfigs
from masknmf.pipelines.configs.compression_configs import CompressConfig, CompressDenoiseConfig, CompressionConfigs
from masknmf.pipelines.configs.demixing_configs import NMFConfig, CustomInitConfig, SuperpixelInitConfig, SpatialHighpassConfig, SinglepassDemixingConfig, MultipassDemixingConfig, MultipassDemixingConfigs
from pathlib import Path
from typing import *
import numpy as np
import os
from numbers import Integral
import torch
import cv2
import h5py
import logging

logger = logging.getLogger(__name__)

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
                 motion_correct_config: MotionCorrectionConfigs | Literal["skip"] | None = None,
                 compress_config: CompressionConfigs | Literal["skip"] | None = None,
                 demixing_config: MultipassDemixingConfigs | Literal["skip"] | None = None,
                 frame_batch_size: int = 300,
                 device: Literal["auto", "cuda", "cpu"] = "auto",
                 log_level: Literal["debug", "info", "warning"] = "info",
                 device_memory_gb: float | None = None,
                 host_memory_gb: float | None = None,
                 reference_temporal_avg_factor: int = 2,
                 reference_spatial_avg_factor: int = 4):
        """
        reference_temporal_avg_factor and reference_spatial_avg_factor set the averaging of the plain PMD compression
        of the reference channel that the registration template and shifts are estimated on.
        """
        # both channels are always read into RAM
        super().__init__(output_folder=output_folder, frame_batch_size=frame_batch_size, device=device,
                         log_level=log_level,
                         load_into_ram=True,
                         motion_correct_config=motion_correct_config, compress_config=compress_config,
                         demixing_config=demixing_config,
                         device_memory_gb=device_memory_gb,
                         host_memory_gb=host_memory_gb)
        self.reference_temporal_avg_factor = reference_temporal_avg_factor
        self.reference_spatial_avg_factor = reference_spatial_avg_factor

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

    @property
    def demixing_configs(self) -> MultipassDemixingConfigs | Literal["skip"]:
        """The spine passes; the whole-dendrite and calcium passes run NMF_JUST_HALS and are not listed."""
        return self.demixing_config

    def apply_stored_registration(self,
                                  movie: OphysArray,
                                  stored: type[BaseRegistrationArray] | None,
                                  results_path: str) -> np.ndarray:
        """
        movie registered with the registration resume copied into results_path, read into RAM. With stored None,
        an earlier run that skipped motion correction, it is movie as recorded.
        """
        if stored is None:
            return movie[:].cpu().numpy()
        logger.info(f"applying the stored {stored.__name__}")
        return stored.from_hdf5(results_path, input_movie=movie, device=self.device)[:].cpu().numpy()

    def run(self,
            glutamate_channel: np.ndarray | ArrayLike | None,
            calcium_channel: np.ndarray | ArrayLike | None,
            exclude_initial_frames: int = 200,
            good_frame_indicator: torch.Tensor | np.ndarray | None = None,
            glutamate_pixel_mask: np.ndarray | None = None,
            calcium_pixel_mask: np.ndarray | None = None,
            stop_after: Literal["registration", "compression", "demixing"] = "demixing",
            resume: str | Path | None = None) -> Path:
        """
        This routine runs the pipeline for processing single-plane glutamate and calcium imaging videos.
        It can analyze joint calcium/glutamate recordings or just process a single channel of either glutamate or calcium data

        Key assumptions:
            - All datasets fully fit into the RAM of the computer (eventually can generalize)
            - The number of frames in both channels is the same (otherwise the joint registration is not as meaningful)
        Args:
            glutamate_channel (np.ndarray | ArrayLike | None):
            calcium_channel (np.ndarray | ArrayLike | None):
            exclude_initial_frames (int): the first frames, marked bad on top of good_frame_indicator
            good_frame_indicator (torch.Tensor | np.ndarray | None): 1D bool over the raw frames, shared by both
                channels. Only good frames are registered, compressed and demixed; the demixing results are infilled
                with zeros at the bad frames so they keep the raw frame count. retained_frames holds the good ones
            glutamate_pixel_mask (np.ndarray | None): Boolean (height, width), True on the pixels to use. The rest
                of the glutamate channel are zeroed before compression, on top of the automatic SD mask; masknmf mask
                draws one
            calcium_pixel_mask (np.ndarray | None): The same for the calcium channel
            stop_after: the last stage to run: "registration" ends once both channels' shifts are written,
                "compression" once their compressions are, "demixing" runs everything. A demixing_config "skip" ends
                the run after compression too; a motion_correct_config "skip" compresses the channels as recorded
            resume: either channel's results file of an earlier run; the other channel's file is read from the same
                folder. Its registration and retained_frames are copied into the new run folder and applied to the
                channels given, so exclude_initial_frames and good_frame_indicator are not used, and the run goes
                on from compression. With compress_config "skip" its compression and mask are copied too and the run
                goes on from demixing, on the channels the earlier run holds; a channel given is then only used for
                the raw traces, which a channel left None goes without. The settings not given, all but stop_after,
                are that run's, from the config.json beside it.
        """
        reuse_compression = isinstance(self.compress_config, str)
        if reuse_compression and self.compress_config.lower() != "skip":
            raise ValueError("If compress_config is a string, it can only be `skip`")
        if reuse_compression and resume is None:
            raise ValueError('compress_config "skip" reuses a compression; resume names the run that holds it')
        if reuse_compression and stop_after != "demixing":
            raise ValueError('compress_config "skip" reuses a compression for its demixing; stop_after leaves nothing to run')
        if stop_after == "registration" and isinstance(self.motion_correct_config, str) and resume is None:
            raise ValueError('stop_after "registration" with motion_correct_config "skip" has nothing to write')
        # the earlier results file of each channel this run works on, checked before anything is written
        paths_resumed = {}
        if resume is not None:
            folder_resumed = Path(resume).expanduser().resolve().parent
            for name, channel in (("glutamate", glutamate_channel), ("calcium", calcium_channel)):
                path_resumed = folder_resumed / f"results.{name}.hdf5"
                if channel is not None or (reuse_compression and path_resumed.is_file()):
                    paths_resumed[name], _ = self.resume_source(path_resumed, reuse_compression)
        device = self.torch_device
        self.run_config = {"exclude_initial_frames": exclude_initial_frames, "stop_after": stop_after,
                           "resume": None if resume is None else str(resume)}
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

        if glutamate_channel is None and calcium_channel is None and len(paths_resumed) == 0:
            raise ValueError("No functional data was provided, both channels are None")
        if glutamate_channel is not None and calcium_channel is not None:
            if tuple(glutamate_channel.shape) != tuple(calcium_channel.shape):
                raise ValueError("The shape of both channels should be identical")

        movie_given = glutamate_channel if calcium_channel is None else calcium_channel
        if resume is not None:
            # the stored shifts fit the frames the earlier run kept
            with h5py.File(next(iter(paths_resumed.values())), "r") as f:
                retained_frames = f["retained_frames"][()]
                num_raw_frames = f["retained_frames"].attrs.get("num_raw_frames")
            if num_raw_frames is None and movie_given is None:
                raise ValueError(f"{resume} does not record its raw frame count; give a channel's movie")
            good_frames = np.zeros(movie_given.shape[0] if num_raw_frames is None else int(num_raw_frames), dtype=bool)
            good_frames[retained_frames] = True
        else:
            if good_frame_indicator is None:
                good_frames = np.ones(movie_given.shape[0], dtype=bool)
            else:
                good_frames = torch.as_tensor(good_frame_indicator).cpu().numpy().astype(bool)
            good_frames[:exclude_initial_frames] = False

        if glutamate_channel is not None:
            glu = self._construct_ophys_array(self.read_into_ram(glutamate_channel), indicator_polarity="positive",
                                              good_frame_indicator=good_frames)
        else:
            glu = None
        if calcium_channel is not None:
            calcium = self._construct_ophys_array(self.read_into_ram(calcium_channel), indicator_polarity="positive",
                                                  good_frame_indicator=good_frames)
        else:
            calcium = None

        if calcium is not None:
            reference_input = calcium
        else:
            reference_input = glu
        retained_frames = np.flatnonzero(good_frames)
        results_paths = {"glutamate": glu_path, "calcium": ca_path}
        for name, channel in (("glutamate", glu), ("calcium", calcium)):
            if channel is not None or name in paths_resumed:
                with h5py.File(results_paths[name], "w") as f:
                    f["retained_frames"] = retained_frames
                    f["retained_frames"].attrs["num_raw_frames"] = good_frames.shape[0]

        registers = resume is None and not isinstance(self.motion_correct_config, str)
        if resume is not None:
            stored = {name: self.resume(path_resumed, results_paths[name], reuse_compression)
                      for name, path_resumed in paths_resumed.items()}
            glu_moco_array_dense = None if glu is None else self.apply_stored_registration(
                glu, stored["glutamate"], glu_path)
            calcium_moco_array_dense = None if calcium is None else self.apply_stored_registration(
                calcium, stored["calcium"], ca_path)
        elif isinstance(self.motion_correct_config, str):
            # unregistered: the channels go to compression as recorded
            glu_moco_array_dense = None if glu is None else glu[:].cpu().numpy()
            calcium_moco_array_dense = None if calcium is None else calcium[:].cpu().numpy()
        else:
            pre_moco_strategy = masknmf.CompressStrategy(block_sizes=self.compress_config.block_sizes,
                                                   max_components=self.compress_config.max_components,
                                                   max_consecutive_failures=self.compress_config.max_consecutive_failures,
                                                   temporal_avg_factor=self.reference_temporal_avg_factor,
                                                   spatial_avg_factor=self.reference_spatial_avg_factor,
                                                   device=device,
                                                   device_memory_gb=self.device_memory_gb,
                                                   host_memory_gb=self.host_memory_gb)
            with self.step("reference compression"):
                pmd_pre_moco_reference = pre_moco_strategy.compress(reference_input)
                pmd_pre_moco_reference.to(device) #Move it to the accelerator

            if isinstance(self.motion_correct_config, PiecewiseRigidMotionCorrectionConfig):
                corrector = PiecewiseRigidMotionCorrector(**asdict(self.motion_correct_config),
                                                          device=device,
                                                          batch_size=self.frame_batch_size)
            else:
                corrector = RigidMotionCorrector(**asdict(self.motion_correct_config),
                                                 device=device,
                                                 batch_size=self.frame_batch_size)

            corrector.compute_template(pmd_pre_moco_reference)

        if not registers:
            pass
        elif glu is not None:
            with self.step("glutamate motion correction"):
                glu_moco_array = corrector.motion_correct(reference_movie=pmd_pre_moco_reference,
                                                          target_movie=glu)
                glu_moco_array.export(glu_path)
                glu_moco_array_dense = glu_moco_array[:].cpu().numpy() #Loads it all into RAM
        else:
            glu_moco_array = None
            glu_moco_array_dense = None

        if not registers:
            pass
        elif calcium is not None:
            with self.step("calcium motion correction"):
                calcium_moco_array = corrector.motion_correct(reference_movie=pmd_pre_moco_reference,
                                                              target_movie=calcium)
                calcium_moco_array.export(ca_path)
                calcium_moco_array_dense = calcium_moco_array[:].cpu().numpy() #Loads it all into RAM
        else:
            calcium_moco_array = None
            calcium_moco_array_dense = None

        if stop_after == "registration":
            return self.finish()

        if reuse_compression:
            with h5py.File(next(iter(paths_resumed.values())), "r") as f:
                if "cross_channel_mask" not in f:
                    raise ValueError(f"{resume} was written before the mask was saved; resume from its registration "
                                     f"with a compress_config instead")
                cross_channel_mask = f["cross_channel_mask"][()]
        elif calcium_moco_array_dense is not None:
            cross_channel_mask = get_std_based_mask(calcium_moco_array_dense)
        else:
            cross_channel_mask = get_std_based_mask(glu_moco_array_dense)
        for name in results_paths:
            if os.path.isfile(results_paths[name]):
                with h5py.File(results_paths[name], "a") as f:
                    f["cross_channel_mask"] = cross_channel_mask

        for name, pixel_mask in (("glutamate_pixel_mask", glutamate_pixel_mask), ("calcium_pixel_mask", calcium_pixel_mask)):
            if pixel_mask is not None and pixel_mask.shape != cross_channel_mask.shape:
                raise ValueError(f"{name} has shape {pixel_mask.shape}; the movie's frames are {cross_channel_mask.shape}")

        if glu_moco_array_dense is not None:
            glu_mask = cross_channel_mask if glutamate_pixel_mask is None else cross_channel_mask * glutamate_pixel_mask
            glu_video =  glu_moco_array_dense * glu_mask.astype(np.float32)[None, :, :]
        else:
            glu_video = None

        if calcium_moco_array_dense is not None:
            calcium_mask = cross_channel_mask if calcium_pixel_mask is None else cross_channel_mask * calcium_pixel_mask
            calcium_video = calcium_moco_array_dense * calcium_mask.astype(np.float32)[None, :, :]
        else:
            calcium_video = None

        if reuse_compression:
            pmd_glu = masknmf.CompressionArray.from_hdf5(glu_path) if "glutamate" in paths_resumed else None
            pmd_ca = masknmf.CompressionArray.from_hdf5(ca_path) if "calcium" in paths_resumed else None
        else:
            compress_strat = self.compress_strategy(self.compress_config)

            if glu_video is not None:
                with self.step("glutamate compression"):
                    pmd_glu = compress_strat.compress(glu_video)
                    pmd_glu.export(glu_path)
            else:
                pmd_glu = None

            if calcium_video is not None:
                with self.step("calcium compression"):
                    pmd_ca = compress_strat.compress(calcium_video)
                    pmd_ca.export(ca_path)
            else:
                pmd_ca = None

        if stop_after == "compression" or isinstance(self.demixing_config, str):
            return self.finish()

        ## If the demixer has spines in the glutamate channel
        if pmd_glu is not None:
            glu_pmd_demixer = masknmf.demixing.signal_demixer.SignalDemixer(
                pmd_glu,
                device=device)

            glu_pmd_demixer_results = self.run_multipass(glu_pmd_demixer, self.demixing_config, "glutamate spine demixing")
            # saved now so a failure in the steps after it still leaves the spines; written again with raw traces
            masknmf.DemixingResults.infill_bad_frames(glu_pmd_demixer_results, good_frame_indicator=good_frames).export(glu_path)

            ## Now pull out "whole dendrite" events. Can refactor this to a helper function to keep the "run" function readable
            glu_pmd_demixer_global= masknmf.demixing.signal_demixer.SignalDemixer(
                pmd_glu,
                device=device
            )
            with self.step("glutamate whole-dendrite demixing"):
                glu_pmd_demixer_global.initialize_signals(is_custom=True,
                                                          spatial_footprints=(cross_channel_mask[:,:, None].astype("float"))
                                                          )
                glu_pmd_demixer_global.demix(**NMF_JUST_HALS)

            glu_global_results = glu_pmd_demixer_global.results
            # a compression resumed without the channel's movie has no registered movie to estimate raw traces from
            if glu_video is not None:
                with self.step("glutamate raw traces"):
                    for results in (glu_pmd_demixer_results, glu_global_results):
                        results.temporal_demixed_raw = masknmf.demixing.estimate_temporal_demixed_raw(
                            results, glu_video, device=device, nonneg=True, frame_batch_size=self.frame_batch_size)
            masknmf.DemixingResults.infill_bad_frames(glu_pmd_demixer_results, good_frame_indicator=good_frames).export(glu_path, overwrite=True)
            masknmf.DemixingResults.infill_bad_frames(glu_global_results, good_frame_indicator=good_frames).export(glu_path, prefix="global")

            if pmd_ca is not None:
                ## Pull out the spatial/temporal footprints from the glutamate movie
                spatialfoot_spines_glu = glu_pmd_demixer_results.signals_array.export_spatial_demixed()  # spatial_footprints

                ca_pmd_demixer = masknmf.demixing.signal_demixer.SignalDemixer(
                    pmd_ca,
                    device=device)

                with self.step("calcium spine demixing"):
                    ca_pmd_demixer.initialize_signals(is_custom=True,
                                                      spatial_footprints=spatialfoot_spines_glu)
                    ca_pmd_demixer.demix(**NMF_JUST_HALS)

                ## Pull out global events from the calcium data
                ca_pmd_demixer_global = masknmf.demixing.signal_demixer.SignalDemixer(
                    pmd_ca,
                    device=device)

                with self.step("calcium whole-dendrite demixing"):
                    ca_pmd_demixer_global.initialize_signals(is_custom=True,
                                                             spatial_footprints=(
                                                                 cross_channel_mask[:, :, None].astype("float"))
                                                             )
                    ca_pmd_demixer_global.demix(**NMF_JUST_HALS)

                ca_results = ca_pmd_demixer.results
                ca_global_results = ca_pmd_demixer_global.results
                if calcium_video is not None:
                    with self.step("calcium raw traces"):
                        for results in (ca_results, ca_global_results):
                            results.temporal_demixed_raw = masknmf.demixing.estimate_temporal_demixed_raw(
                                results, calcium_video, device=device, nonneg=True, frame_batch_size=self.frame_batch_size)
                masknmf.DemixingResults.infill_bad_frames(ca_results, good_frame_indicator=good_frames).export(ca_path)
                masknmf.DemixingResults.infill_bad_frames(ca_global_results, good_frame_indicator=good_frames).export(ca_path, prefix="global")


        else: #In this case there is only a calcium channel
            ca_pmd_demixer = masknmf.demixing.signal_demixer.SignalDemixer(
                pmd_ca,
                device=device)

            ca_pmd_demixer_results = self.run_multipass(ca_pmd_demixer, self.demixing_config, "calcium spine demixing")
            # saved now so a failure in the steps after it still leaves the spines; written again with raw traces
            masknmf.DemixingResults.infill_bad_frames(ca_pmd_demixer_results, good_frame_indicator=good_frames).export(ca_path)


            ## Now pull out "whole dendrite" events. Can refactor this to a helper function to keep the "run" function readable
            ca_pmd_demixer_global = masknmf.demixing.signal_demixer.SignalDemixer(
                pmd_ca,
                device=device
            )
            with self.step("calcium whole-dendrite demixing"):
                ca_pmd_demixer_global.initialize_signals(is_custom=True,
                                                         spatial_footprints=(cross_channel_mask[:, :, None].astype("float"))
                                                         )
                ca_pmd_demixer_global.demix(**NMF_JUST_HALS)

            ca_global_results = ca_pmd_demixer_global.results
            if calcium_video is not None:
                with self.step("calcium raw traces"):
                    for results in (ca_pmd_demixer_results, ca_global_results):
                        results.temporal_demixed_raw = masknmf.demixing.estimate_temporal_demixed_raw(
                            results, calcium_video, device=device, nonneg=True, frame_batch_size=self.frame_batch_size)
            masknmf.DemixingResults.infill_bad_frames(ca_pmd_demixer_results, good_frame_indicator=good_frames).export(ca_path, overwrite=True)
            masknmf.DemixingResults.infill_bad_frames(ca_global_results, good_frame_indicator=good_frames).export(ca_path, prefix="global")

        return self.finish()



















