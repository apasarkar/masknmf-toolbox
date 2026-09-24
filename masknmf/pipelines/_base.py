from abc import ABC, abstractmethod
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
import os
import numpy as np
import torch
from typing import *

from masknmf.pipelines.scraper import slugify
from masknmf.arrays import ArrayLike
from masknmf.compression import CompressionArray, CompressStrategy, CompressDenoiseStrategy
from masknmf.compression.preprocessing import MaximinSplineDetrend
from masknmf.demixing import SignalDemixer, DemixingResults, NoSignalsDetectedError
from masknmf.motion_correction import BaseRegistrationArray, RigidMotionCorrector, PiecewiseRigidMotionCorrector
from masknmf.motion_correction.moco_preprocessing import construct_moco_template
from masknmf.pipelines.configs.motion_correction_configs import RigidMotionCorrectionConfig, PiecewiseRigidMotionCorrectionConfig
from masknmf.pipelines.configs.compression_configs import CompressConfig, CompressDenoiseConfig
from masknmf.pipelines.configs.demixing_configs import MultipassDemixingConfig
from masknmf.utils import display, has_group, drop_group, torch_select_device

class BasePipeline(ABC):
    def __init__(self,
                 output_folder: str | Path | None = None,
                 frame_batch_size: int = 300,
                 device: Literal["auto", "cuda", "cpu"] = "auto"):
        self._output_folder = output_folder
        self._frame_batch_size = frame_batch_size
        self._device = device

    @property
    @abstractmethod
    def config(self):
        pass

    @property
    def output_folder(self) -> str | Path | None:
        return self._output_folder

    @property
    def frame_batch_size(self) -> int:
        return self._frame_batch_size

    @property
    def device(self) -> Literal["auto", "cuda", "cpu"]:
        return self._device

    @property
    def torch_device(self) -> str:
        """``device`` with "auto" resolved to the device pytorch will use."""
        return torch_select_device(self.device)

    def create_run_folder(self) -> Path:
        """
        Make ``<output_folder>/<YYYYmmdd_HHMMSS>_<pipeline slug>/`` (the working directory when output_folder is None),
        adding a numeric suffix when a run started in the same second.
        """
        base = Path.cwd() if self.output_folder is None else Path(self.output_folder).expanduser().resolve()
        name = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{slugify(name_class=type(self).__name__)}"
        candidate = base / name
        suffix = 0
        while True:
            try:
                candidate.mkdir(parents=True, exist_ok=False)
                return candidate
            except FileExistsError:
                suffix += 1
                candidate = base / f"{name}_{suffix}"

    def results_path(self, resume: bool = False) -> str:
        """
        ``results.hdf5`` in a new run folder. With ``resume``, the one in output_folder itself, an earlier run
        folder whose compression is reused.
        """
        if not resume:
            path = os.path.join(self.create_run_folder(), "results.hdf5")
            display(f"Writing results to {path}")
            return path
        path = os.path.join(Path.cwd() if self.output_folder is None else self.output_folder, "results.hdf5")
        if not has_group(path, CompressionArray.__name__):
            raise ValueError(f"You specified that compression should be skipped but {path} holds no compression")
        return path

    def motion_correct(self,
                       data: np.ndarray | ArrayLike,
                       config: RigidMotionCorrectionConfig | PiecewiseRigidMotionCorrectionConfig | Literal["skip"] | None,
                       results_path: str,
                       exclude_border_radius: int = 0) -> tuple[np.ndarray | ArrayLike, np.ndarray]:
        """
        Register data with a rigid (the default when config is None) or piecewise rigid corrector and export it to
        results_path, or pass it through for "skip". Also returns a pixel weighting that is 0 where the shifts
        moved pixels in from outside the fov and on the outer exclude_border_radius pixels.
        """
        if isinstance(config, str):
            if config.lower() != "skip":
                raise ValueError("Invalid MotionCorrectionConfig input")
            display("Not Running Motion Correction")
            moco_data = data
        else:
            if config is None or isinstance(config, RigidMotionCorrectionConfig):
                moco_strategy = RigidMotionCorrector(**asdict(RigidMotionCorrectionConfig() if config is None else config),
                                                     device=self.device, batch_size=self.frame_batch_size)
            elif isinstance(config, PiecewiseRigidMotionCorrectionConfig):
                moco_strategy = PiecewiseRigidMotionCorrector(**asdict(config), device=self.device,
                                                              batch_size=self.frame_batch_size)
            else:
                raise ValueError("Invalid MotionCorrectionConfig input")
            if moco_strategy.template is None:
                moco_strategy.compute_template(data)
            moco_data = moco_strategy.motion_correct(data)
            moco_data.output_device = moco_data.strategy.device
            moco_data.export(results_path)

        if isinstance(moco_data, BaseRegistrationArray):
            shift_mask = construct_moco_template(moco_data.shifts.cpu().numpy(), moco_data.shape[1:]).astype("float")
        else:
            shift_mask = np.ones((moco_data.shape[1], moco_data.shape[2])).astype("float")
        if exclude_border_radius > 0:
            shift_mask[:exclude_border_radius, :] = 0
            shift_mask[:, :exclude_border_radius] = 0
            shift_mask[-1 * exclude_border_radius:, :] = 0
            shift_mask[:, -1 * exclude_border_radius:] = 0
        return moco_data, shift_mask

    def compress_strategy(self,
                          config: CompressConfig | CompressDenoiseConfig | None,
                          pixel_weighting: np.ndarray | None = None) -> CompressStrategy:
        """
        The strategy for config (CompressDenoiseConfig defaults when None) on this pipeline's device, with
        pixel_weighting multiplied into the config's own.
        """
        if config is None:
            config = CompressDenoiseConfig()
        kwargs = asdict(config)
        if pixel_weighting is not None:
            kwargs["pixel_weighting"] = pixel_weighting if config.pixel_weighting is None else config.pixel_weighting * pixel_weighting
        if isinstance(config, CompressConfig):
            return CompressStrategy(device=self.device, **kwargs)
        if isinstance(config, CompressDenoiseConfig):
            return CompressDenoiseStrategy(device=self.device, **kwargs)
        raise ValueError("Invalid compression config")

    def spline_detrender(self,
                         num_frames: int,
                         frame_rate: float,
                         window_seconds: float,
                         knot_seconds: float,
                         sigma_seconds: float) -> MaximinSplineDetrend:
        """A maximin spline detrender with its rolling window, knot spacing and smoothing given in seconds."""
        return MaximinSplineDetrend(num_frames=num_frames,
                                    num_knots=max(4, int(num_frames / frame_rate / knot_seconds)),
                                    window=int(window_seconds * frame_rate),
                                    sigma=max(2.0, sigma_seconds * frame_rate),
                                    device=self.torch_device)

    def run_multipass(self, demixer: SignalDemixer, config: MultipassDemixingConfig) -> DemixingResults:
        """
        Run the passes of config in order, stopping at the first that finds no signals, and return the results of
        the last pass that ran. Raises when the first pass finds none.
        """
        results = None
        for singlepass in config.DemixingConfigs:
            try:
                demixer.initialize_signals(**asdict(singlepass.InitConfig))
            except NoSignalsDetectedError:
                if results is None:
                    raise ValueError("The demixer did not identify any signals. Lower thresholds or inspect the data "
                                     "to resolve this issue.")
                break
            demixer.demix(**asdict(singlepass.NMFConfig))
            results = demixer.results
            torch.cuda.empty_cache()
        return results

    def drop_compression(self, results_path: str):
        """Remove the CompressionArray group once demixing is done; the demixing results carry the pmd."""
        display("Removing intermediates")
        drop_group(results_path, CompressionArray.__name__)

    @abstractmethod
    def run(self, **kwargs) -> Path:
        """Run the pipeline on its movie(s) and return the run folder it wrote to."""
        raise NotImplementedError
