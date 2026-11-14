from dataclasses import asdict
from pathlib import Path
from typing import *

import numpy as np
import torch

from masknmf.arrays import ArrayLike, InterleavedChannelArray
from masknmf.compression import CompressionArray
from masknmf.hemocorrection import hemocorrect, interleaved_frame_positions
from masknmf.motion_correction import RigidMotionCorrector, RigidRegistrationArray
from masknmf.motion_correction.moco_preprocessing import construct_moco_template
from masknmf.pipelines._base import BasePipeline
from masknmf.pipelines.configs.compression_configs import CompressDenoiseConfig, CompressionConfigs
from masknmf.pipelines.configs.hemocorrection_configs import PixelwiseHemocorrectionConfig, HemocorrectionConfigs
from masknmf.pipelines.configs.motion_correction_configs import RigidMotionCorrectionConfig
from masknmf.utils import display


def widefield_compress_config() -> CompressDenoiseConfig:
    """Compression with the temporal denoiser, tuned for widefield fields of view (the January 2026 IBL settings, 30 epochs)."""
    return CompressDenoiseConfig(block_sizes=(100, 100),
                                 max_components=20,
                                 max_consecutive_failures=1,
                                 noise_variance_quantile=0.3,
                                 num_epochs=30)


class WidefieldHemocorrectionPipeline(BasePipeline):
    """
    Widefield calcium imaging with a hemodynamic reference channel interleaved in time (e.g. alternating 470 nm and
    405 nm excitation): shared rigid registration of both channels, compression with denoising of each channel, and
    pixelwise hemodynamic correction of the functional channel on the compressed data.

    Run folder contents:
        functional.hdf5  the functional channel's registration (when run) and CompressionArray
        reference.hdf5   the reference channel's registration (when run) and CompressionArray
        results.hdf5     the corrected functional CompressionArray and HemocorrectionResults (coefficient image,
                         projection diagnostics)
    """

    def __init__(self,
                 motion_correct_config: RigidMotionCorrectionConfig | Literal["skip"] | None = None,
                 compress_config: CompressionConfigs | Literal["skip"] | None = None,
                 reference_compress_config: CompressionConfigs | None = None,
                 hemocorrection_config: HemocorrectionConfigs | None = None,
                 output_folder: str | Path | None = None,
                 frame_batch_size: int = 300,
                 device: Literal["auto", "cuda", "cpu"] = "auto",
                 log_level: Literal["debug", "info", "warning"] = "info",
                 load_into_ram: bool = False,
                 device_memory_gb: float | None = None,
                 host_memory_gb: float | None = None,
                 ):
        """
        Args:
            motion_correct_config: Rigid registration; the shifts are estimated on one channel (run's
                registration_channel) and applied to both. "skip" compresses the channels as recorded, e.g. data
                registered elsewhere. None uses default_configs().
            compress_config: Compression of the functional channel. "skip" ends the run after registration.
                None uses default_configs().
            reference_compress_config: Compression of the reference channel. None uses default_configs(), the same
                settings as the functional channel's default.
            hemocorrection_config: Parameters of the pixelwise correction. None uses default_configs().
            output_folder: Each run writes to ``<output_folder>/<timestamp>_widefield-hemocorrection/``. None uses the
                working directory
            frame_batch_size (int): Frames loaded at a time for registration
            device (str): Which device pytorch runs on
            log_level (str): How much the run logs, to the console and to the run folder's .log file
            load_into_ram (bool): Read both channels into RAM first. Widefield sessions are often tens of GB per
                channel, so this is off by default and the channels are read lazily
            device_memory_gb (float | None): Memory, in gigabytes, the pipeline may use on the compute device. None
                detects the free device memory
            host_memory_gb (float | None): Host RAM, in gigabytes, the pipeline may use. None detects the free RAM
        """
        super().__init__(output_folder=output_folder,
                         frame_batch_size=frame_batch_size,
                         device=device,
                         log_level=log_level,
                         load_into_ram=load_into_ram,
                         motion_correct_config=motion_correct_config,
                         compress_config=compress_config,
                         reference_compress_config=reference_compress_config,
                         hemocorrection_config=hemocorrection_config,
                         device_memory_gb=device_memory_gb,
                         host_memory_gb=host_memory_gb)

    @classmethod
    def default_configs(cls) -> dict:
        """
        Rigid registration, compression with denoising (100 x 100 blocks, 30 epochs) of each channel and pixelwise
        correction.
        """
        return {"motion_correct_config": RigidMotionCorrectionConfig(),
                "compress_config": widefield_compress_config(),
                "reference_compress_config": widefield_compress_config(),
                "hemocorrection_config": PixelwiseHemocorrectionConfig()}

    def _register(self,
                  registration: ArrayLike,
                  other: ArrayLike,
                  paths: tuple[str, str]) -> tuple[ArrayLike, ArrayLike, np.ndarray]:
        """
        Estimate rigid shifts on registration, apply the same shifts to other, export each channel's registration to
        its file, and return both registered channels with the pixel weighting that zeroes pixels shifted in from
        outside the field of view.
        """
        corrector = RigidMotionCorrector(**asdict(self.motion_correct_config),
                                         device=self.device, batch_size=self.frame_batch_size)
        with self.step("motion correction"):
            if corrector.template is None:
                corrector.compute_template(registration)
            registered = corrector.motion_correct(reference_movie=registration)
            registered.output_device = torch.device("cpu")
            registered_other = RigidRegistrationArray(input_movie=other, shifts=registered.shifts,
                                                      strategy=registered.strategy)
            registered_other.output_device = torch.device("cpu")
            registered.export(paths[0])
            registered_other.export(paths[1])
        shift_mask = construct_moco_template(registered.shifts.cpu().numpy(), registered.shape[1:]).astype("float")
        return registered, registered_other, shift_mask

    def _compress(self, data: ArrayLike, config: CompressionConfigs, pixel_weighting: np.ndarray, name: str,
                  path: str) -> CompressionArray:
        strategy = self.compress_strategy(config, pixel_weighting)
        with self.step(f"{name} compression"):
            compressed = strategy.compress(data)
            compressed.export(path)
        torch.cuda.empty_cache()
        return compressed

    def run(self,
            data: np.ndarray | ArrayLike,
            functional_channel: int = 0,
            num_channels: int = 2,
            pixel_weighting: np.ndarray | None = None,
            frame_times: np.ndarray | None = None,
            frame_rate: float | None = None,
            registration_channel: Literal["functional", "reference"] = "functional",
            exclude_border_radius: int = 0,
            stop_after: Literal["registration", "compression", "hemocorrection"] = "hemocorrection") -> Path:
        """
        Register, compress and hemocorrect an interleaved widefield recording.

        Args:
            data: The interleaved recording: (cycles, channels, height, width), e.g. load_interleaved_binary of a
                wfield .bin, or (frames in acquisition order, height, width). Read lazily.
            functional_channel (int): The calcium-dependent channel; the reference is the other one. With more than
                two channels, the reference is channel (functional_channel + 1) % num_channels.
            num_channels (int): Channels per acquisition cycle.
            pixel_weighting: Shape (height, width), e.g. a brain mask (1 inside, 0 outside). Multiplied into the
                compression weighting of both channels. None weights every pixel.
            frame_times: Shape (cycles * num_channels,), the time of every raw frame in acquisition order. Used to
                align the reference to the functional frames and, without frame_rate, to derive the frame rate. None
                assumes evenly spaced frames and aligns by acquisition order.
            frame_rate (float | None): Functional channel frame rate in Hz, for the hemocorrection filters.
            registration_channel: The channel the rigid shifts are estimated on.
            exclude_border_radius (int): Also zero the weighting of this many pixels at each edge.
            stop_after: The last stage to run.
        Returns:
            Path: The run folder.
        """
        if not 0 <= functional_channel < num_channels:
            raise ValueError(f"functional_channel must be in [0, {num_channels}), got {functional_channel}")
        reference_channel = (functional_channel + 1) % num_channels
        ends_at_registration = stop_after == "registration" or isinstance(self.compress_config, str)
        if ends_at_registration and isinstance(self.motion_correct_config, str):
            raise ValueError('a run ending after registration with motion_correct_config "skip" has nothing to write')
        self.run_config = {"functional_channel": functional_channel, "num_channels": num_channels,
                           "reference_channel": reference_channel, "frame_rate": frame_rate,
                           "frame_times_given": frame_times is not None,
                           "pixel_weighting_given": pixel_weighting is not None,
                           "registration_channel": registration_channel,
                           "exclude_border_radius": exclude_border_radius, "stop_after": stop_after}
        results_path = self.results_path()
        folder = Path(results_path).parent
        paths = {"functional": str(folder / "functional.hdf5"), "reference": str(folder / "reference.hdf5")}

        functional = InterleavedChannelArray(data, channel=functional_channel, num_channels=num_channels)
        reference = InterleavedChannelArray(data, channel=reference_channel, num_channels=num_channels)
        display(f"functional channel {functional_channel}, reference channel {reference_channel}, "
                f"{functional.shape[0]} frames each of {functional.shape[1]} x {functional.shape[2]}")
        if self.load_into_ram:
            functional, reference = self.read_into_ram(functional), self.read_into_ram(reference)

        if isinstance(self.motion_correct_config, str):
            if self.motion_correct_config.lower() != "skip":
                raise ValueError("Invalid motion_correct_config")
            display("Not Running Motion Correction")
            shift_mask = np.ones(functional.shape[1:], dtype=float)
        elif registration_channel == "functional":
            functional, reference, shift_mask = self._register(functional, reference,
                                                               (paths["functional"], paths["reference"]))
        elif registration_channel == "reference":
            reference, functional, shift_mask = self._register(reference, functional,
                                                               (paths["reference"], paths["functional"]))
        else:
            raise ValueError(f'registration_channel must be "functional" or "reference", got {registration_channel!r}')
        if ends_at_registration:
            return self.finish()

        if exclude_border_radius > 0:
            shift_mask[:exclude_border_radius, :] = 0
            shift_mask[:, :exclude_border_radius] = 0
            shift_mask[-exclude_border_radius:, :] = 0
            shift_mask[:, -exclude_border_radius:] = 0
        weighting = shift_mask
        if pixel_weighting is not None:
            pixel_weighting = np.asarray(pixel_weighting, dtype=float)
            if pixel_weighting.shape != tuple(functional.shape[1:]):
                raise ValueError(f"pixel_weighting has shape {pixel_weighting.shape}, the field of view is "
                                 f"{tuple(functional.shape[1:])}")
            weighting = weighting * pixel_weighting

        compressed_functional = self._compress(functional, self.compress_config, weighting, "functional",
                                               paths["functional"])
        compressed_reference = self._compress(reference, self.reference_compress_config, weighting, "reference",
                                              paths["reference"])
        if stop_after == "compression":
            return self.finish()

        num_frames = functional.shape[0]
        if frame_times is not None:
            frame_times = np.asarray(frame_times, dtype=np.float64).ravel()
            if frame_times.size < num_frames * num_channels:
                raise ValueError(f"frame_times has {frame_times.size} entries; the recording has "
                                 f"{num_frames * num_channels} frames")
            times_functional = frame_times[functional_channel::num_channels][:num_frames]
            times_reference = frame_times[reference_channel::num_channels][:num_frames]
            if frame_rate is None:
                frame_rate = float(1.0 / np.median(np.diff(times_functional)))
        else:
            times_functional = interleaved_frame_positions(num_frames, functional_channel, num_channels)
            times_reference = interleaved_frame_positions(num_frames, reference_channel, num_channels)

        with self.step("hemocorrection"):
            corrected, results = hemocorrect(compressed_functional, compressed_reference,
                                             functional_frame_times=times_functional,
                                             reference_frame_times=times_reference,
                                             frame_rate=frame_rate,
                                             device=self.device,
                                             **asdict(self.hemocorrection_config))
            corrected.export(results_path)
            results.export(results_path)
        return self.finish()
