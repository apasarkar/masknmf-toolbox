from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import asdict, replace
from datetime import datetime, timedelta
from pathlib import Path
import inspect
import json
import logging
import os
import threading
import time
import numpy as np
import torch
from typing import *

from masknmf._version import __version__
from masknmf.pipelines.scraper import slugify, config_json_value, config_from_json
from masknmf.arrays import ArrayLike
from masknmf.compression import CompressionArray, CompressStrategy, CompressDenoiseStrategy
from masknmf.compression.preprocessing import MaximinSplineDetrend
from masknmf.demixing import SignalDemixer, DemixingResults, NoSignalsDetectedError
from masknmf.motion_correction import BaseRegistrationArray, RigidMotionCorrector, PiecewiseRigidMotionCorrector
from masknmf.motion_correction.moco_preprocessing import construct_moco_template
from masknmf.pipelines.configs.motion_correction_configs import RigidMotionCorrectionConfig, PiecewiseRigidMotionCorrectionConfig
from masknmf.pipelines.configs.compression_configs import CompressConfig, CompressDenoiseConfig
from masknmf.pipelines.configs.demixing_configs import MultipassDemixingConfig, SinglepassDemixingConfig, SuperpixelInitConfig
from masknmf.utils import display, has_group, drop_group, torch_select_device

logger = logging.getLogger(__name__)


def heartbeat(name: str, start: float, seconds: float, stop: threading.Event):
    """Log every seconds that the step name, started at start (monotonic), is still running, until stop is set."""
    while not stop.wait(seconds):
        logger.info(f"{name} still running after {timedelta(seconds=round(time.monotonic() - start))}")


class BasePipeline(ABC):
    """
    Motion correction, compression (and optional demixing) of one session.
    """

    heartbeat_seconds = 600

    def __init__(self,
                 output_folder: str | Path | None = None,
                 frame_batch_size: int = 300,
                 device: Literal["auto", "cuda", "cpu"] = "auto",
                 log_level: Literal["debug", "info", "warning"] = "info",
                 load_into_ram: bool = False,
                 **configs):
        if output_folder is not None:
            output_folder = Path(output_folder).expanduser().resolve()
            if output_folder.exists() and not output_folder.is_dir():
                raise NotADirectoryError(f"output_folder exists and is not a directory: {output_folder}")
        self.output_folder = output_folder
        self.frame_batch_size = frame_batch_size
        self.device = device
        self.log_level = log_level
        logging.getLogger("masknmf").setLevel(log_level.upper())
        # the handler writing the run's log file, once log_to has opened one
        self.log_handler = None
        self.load_into_ram = load_into_ram
        # the folder the last create_run_folder made
        self.run_folder = None
        # the scalar arguments of the run in progress, saved to config.json beside the __init__ ones
        self.run_config = {}
        # the command line that started the run in progress, when the cli did
        self.command = None
        # the run in progress as config.json holds it: command, device, gpu, start, and once finish has run, its
        # end, seconds and status
        self.run_record = {}
        # the file each movie or array argument of the run in progress was read from, by argument name, as
        # config.json holds it: {"path": ..., "name": ...}; the cli fills it
        self.inputs = {}
        # what config.json keeps from the run whose results a resumed run reuses
        self.configs_reused = {}
        # the seconds, start and status of each step that ran in the run folder, by step name
        self.timings = {}
        defaults = self.default_configs()
        unknown = set(configs) - set(defaults)
        if len(unknown) > 0:
            raise TypeError(f"{type(self).__name__}.default_configs() has no entry for {', '.join(sorted(unknown))}")
        for name, default in defaults.items():
            setattr(self, name, default if configs.get(name) is None else configs[name])

    @classmethod
    @abstractmethod
    def default_configs(cls) -> dict:
        """
        The value each config argument of __init__ takes when it is None, keyed by argument name. Built fresh on every
        call, so callers may change what they get back. Gives only what differs from each config's own defaults, so
        changes to those carry through.
        """
        pass

    @classmethod
    def from_config(cls, path: str | Path, **overrides):
        """
        The pipeline a run folder's config.json, or ``masknmf params --json`` output, describes, each value built on
        default_configs(); overrides replace the file's values. The run arguments the file holds under "configs"
        (frame_rate and the like) are left for the caller to pass to run.
        """
        loaded = json.loads(Path(path).expanduser().read_text())
        if loaded.get("pipeline", cls.__name__) != cls.__name__:
            raise ValueError(f"{path} is for {loaded['pipeline']}, not {cls.__name__}")
        defaults = cls.default_configs()
        parameters = inspect.signature(cls.__init__).parameters
        values = {name: config_from_json(value=value, annotation=parameters[name].annotation, base=defaults.get(name))
                  for name, value in loaded["configs"].items() if name in parameters}
        return cls(**{**values, **overrides})

    @property
    def config(self) -> dict:
        """Every __init__ argument by name, as constructed."""
        return {name: getattr(self, name) for name in inspect.signature(type(self).__init__).parameters if name != "self"}

    @property
    def torch_device(self) -> str:
        """``device`` with "auto" resolved to the device pytorch will use."""
        return torch_select_device(self.device)

    def create_run_folder(self) -> Path:
        """
        Make ``<output_folder>/<YYYYmmdd_HHMMSS>_<pipeline slug>/`` (the working directory when output_folder is None),
        adding a numeric suffix when a run started in the same second, and write config.json in it.
        """
        base = Path.cwd() if self.output_folder is None else self.output_folder
        name = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{slugify(name_class=type(self).__name__)}"
        candidate = base / name
        suffix = 0
        while True:
            try:
                candidate.mkdir(parents=True, exist_ok=False)
                break
            except FileExistsError:
                suffix += 1
                candidate = base / f"{name}_{suffix}"
        self.run_folder = candidate
        self.configs_reused = {}
        self.timings = {}
        self.log_to(candidate)
        self.write_config()
        return candidate

    def write_config(self) -> Path:
        """
        Write ``config.json`` in the run folder: the masknmf version, the pipeline, the run record under "run", the
        files its run arguments came from under "inputs", its __init__ and run arguments under "configs" (those of a
        reused run's steps over this run's) and the steps that ran under "timings".
        """
        path = self.run_folder / "config.json"
        with open(path, "w") as f:
            json.dump({"masknmf_version": __version__, "pipeline": type(self).__name__, "run": self.run_record,
                       "inputs": self.inputs,
                       "configs": {**self.config, **self.run_config, **self.configs_reused}, "timings": self.timings},
                      f, indent=2, default=config_json_value)
        return path

    def log_to(self, folder: Path) -> Path:
        """
        Write the masknmf log to ``<folder>/<folder name>.log`` from here on, appending to the file an earlier run
        left there and closing the file of the run logged until now, start the run record (command, device, gpu and
        start time; finish ends it) and log this run's header line.
        """
        for handler in [h for h in logging.getLogger("masknmf").handlers if isinstance(h, logging.FileHandler)]:
            logging.getLogger("masknmf").removeHandler(handler)
            handler.close()
        path = folder / f"{folder.name}.log"
        self.log_handler = logging.FileHandler(path, encoding="utf-8")
        self.log_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
        logging.getLogger("masknmf").addHandler(self.log_handler)
        device = str(self.torch_device)
        self.run_record = {"command": self.command, "device": device,
                           "gpu": torch.cuda.get_device_name() if device.startswith("cuda") else None,
                           "started": datetime.now().isoformat(timespec="seconds"), "finished": None, "seconds": None,
                           "status": "running"}
        logger.info(f"masknmf {__version__} {type(self).__name__} on {device}, {self.log_level} log at {path}")
        return path

    @contextmanager
    def step(self, name: str):
        """
        Log that name starts, that it is still running every heartbeat_seconds and, once the block ends, how long it
        took (with its peak cuda memory on a cuda device), or that it failed and after how long, and record its start,
        seconds, status and peak under name in timings and config.json.
        """
        logger.info(name)
        started = datetime.now().isoformat(timespec="seconds")
        start = time.monotonic()
        cuda = str(self.torch_device).startswith("cuda")
        if cuda:
            torch.cuda.reset_peak_memory_stats()
        stop = threading.Event()
        threading.Thread(target=heartbeat, args=(name, start, self.heartbeat_seconds, stop), daemon=True).start()
        status = "failed"
        try:
            yield
            status = "done"
        finally:
            stop.set()
            seconds = time.monotonic() - start
            self.timings[name] = {"seconds": round(seconds, 1), "started": started, "status": status}
            peak = ""
            if cuda:
                self.timings[name]["peak_cuda_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
                peak = f", peak cuda {self.timings[name]['peak_cuda_gb']} GB"
            if self.run_folder is not None:
                self.write_config()
            if status == "done":
                logger.info(f"{name} done in {timedelta(seconds=round(seconds))}{peak}")
            else:
                logger.error(f"{name} failed after {timedelta(seconds=round(seconds))}{peak}")

    def finish(self, status: Literal["done", "failed"] = "done") -> Path:
        """Record in the run record and config.json that the run ended now with status and how long it took, and return the run folder."""
        finished = datetime.now()
        seconds = (finished - datetime.fromisoformat(self.run_record["started"])).total_seconds()
        self.run_record.update(finished=finished.isoformat(timespec="seconds"), seconds=round(seconds, 1), status=status)
        self.write_config()
        return self.run_folder

    def results_path(self, resume: bool = False) -> str:
        """
        ``results.hdf5`` in a new run folder. With ``resume``, the one in output_folder itself, an earlier run
        folder whose compression is reused; its config.json keeps that run's motion correction and compression.
        """
        if not resume:
            path = os.path.join(self.create_run_folder(), "results.hdf5")
            display(f"Writing results to {path}")
            return path
        folder = Path.cwd() if self.output_folder is None else self.output_folder
        path = os.path.join(folder, "results.hdf5")
        if not has_group(path, CompressionArray.__name__):
            raise ValueError(f"You specified that compression should be skipped but {path} holds no compression")
        self.run_folder = folder
        # the earlier run's motion correction and compression made the results reused now, so config.json keeps
        # their inputs, configs and timings and takes the rest from this run
        earlier = json.loads((folder / "config.json").read_text()) if (folder / "config.json").is_file() else {}
        self.inputs = {**earlier.get("inputs", {}), **self.inputs}
        self.configs_reused = {name: value for name, value in earlier.get("configs", {}).items()
                               if name in ("motion_correct_config", "compress_config", "exclude_border_radius")}
        self.timings = {name: timing for name, timing in earlier.get("timings", {}).items()
                        if name in ("motion correction", "compression")}
        self.log_to(folder)
        self.write_config()
        return path

    def read_into_ram(self, data: np.ndarray | ArrayLike) -> np.ndarray:
        """
        data as one numpy array, read frame_batch_size frames at a time so no second full copy is made. A numpy array
        comes back as it is.
        """
        if isinstance(data, np.ndarray):
            return data
        display(f"Loading {data.shape} into RAM")
        movie = None
        for start in range(0, data.shape[0], self.frame_batch_size):
            stop = min(start + self.frame_batch_size, data.shape[0])
            chunk = data[start:stop]
            if isinstance(chunk, torch.Tensor):
                chunk = chunk.cpu().numpy()
            if movie is None:
                movie = np.empty(data.shape, dtype=chunk.dtype.newbyteorder("="))
            movie[start:stop] = chunk
        return movie

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
            with self.step("motion correction"):
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
        The strategy for config (CompressDenoiseConfig defaults when None) on this pipeline's device and
        frame_batch_size, with pixel_weighting multiplied into the config's own.
        """
        if config is None:
            config = CompressDenoiseConfig()
        kwargs = asdict(config)
        if pixel_weighting is not None:
            kwargs["pixel_weighting"] = pixel_weighting if config.pixel_weighting is None else config.pixel_weighting * pixel_weighting
        if isinstance(config, CompressConfig):
            return CompressStrategy(device=self.device, frame_batch_size=self.frame_batch_size, **kwargs)
        if isinstance(config, CompressDenoiseConfig):
            return CompressDenoiseStrategy(device=self.device, frame_batch_size=self.frame_batch_size, **kwargs)
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

    def run_multipass(self, demixer: SignalDemixer, config: MultipassDemixingConfig,
                      name: str = "demixing") -> DemixingResults:
        """
        Run the passes of config in order as the steps "<name> pass i of n", stopping at the first that finds no
        signals, and return the results of the last pass that ran. Raises when the first pass finds none.
        """
        if len(config.DemixingConfigs) < 1:
            raise ValueError("Demixing needs at least one pass")
        results = None
        for i, singlepass in enumerate(config.DemixingConfigs):
            with self.step(f"{name} pass {i + 1} of {len(config.DemixingConfigs)}"):
                try:
                    demixer.initialize_signals(**asdict(singlepass.InitConfig))
                except NoSignalsDetectedError:
                    if results is None:
                        raise ValueError("The demixer did not identify any signals. Lower thresholds or inspect the data "
                                         "to resolve this issue.")
                    logger.info("no signals detected, keeping the previous pass")
                    break
                demixer.demix(**asdict(singlepass.NMFConfig))
                results = demixer.results
                logger.info(f"{results.spatial_demixed.shape[1]} signals")
                torch.cuda.empty_cache()
        return results

    def with_detrender(self, config: MultipassDemixingConfig, detrender: MaximinSplineDetrend) -> MultipassDemixingConfig:
        """A copy of config whose superpixel initializations and NMF steps without a detrender use detrender."""
        passes = []
        for singlepass in config.DemixingConfigs:
            init_config = singlepass.InitConfig
            if isinstance(init_config, SuperpixelInitConfig) and init_config.detrender is None:
                init_config = replace(init_config, detrender=detrender)
            nmf_config = singlepass.NMFConfig
            if nmf_config.detrender is None:
                nmf_config = replace(nmf_config, detrender=detrender)
            passes.append(SinglepassDemixingConfig(init_config, nmf_config))
        return MultipassDemixingConfig(passes)

    def drop_compression(self, results_path: str):
        """Remove the CompressionArray group once demixing is done; the demixing results carry the pmd."""
        display("Removing intermediates")
        drop_group(results_path, CompressionArray.__name__)

    @abstractmethod
    def run(self, **kwargs) -> Path:
        """Run the pipeline on its movie(s) and return the run folder it wrote to."""
        raise NotImplementedError
