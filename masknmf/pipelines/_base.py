from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import asdict, replace
from datetime import datetime, timedelta
from pathlib import Path
import functools
import inspect
import json
import logging
import os
import threading
import time
import h5py
import numpy as np
import torch
from typing import *
from functools import partial

from masknmf._version import __version__
from masknmf.pipelines.scraper import slugify, config_json_value, config_from_json
from masknmf.arrays import ArrayLike, OphysArray
from masknmf.compression import CompressionArray, CompressStrategy, CompressDenoiseStrategy
from masknmf.compression.preprocessing import MaximinSplineDetrend, geometric_knot_positions
from masknmf.demixing import SignalDemixer, DemixingResults, NoSignalsDetectedError
from masknmf.motion_correction import BaseRegistrationArray, RigidMotionCorrector, PiecewiseRigidMotionCorrector
from masknmf.motion_correction.moco_preprocessing import construct_moco_template
from masknmf.pipelines.configs.motion_correction_configs import RigidMotionCorrectionConfig, PiecewiseRigidMotionCorrectionConfig, MotionCorrectionFilterConfig, SpatialHighpassMotionCorrectionConfig
from masknmf.pipelines.configs.compression_configs import CompressConfig, CompressDenoiseConfig
from masknmf.pipelines.configs.demixing_configs import MultipassDemixingConfig, SinglepassDemixingConfig, SuperpixelInitConfig
from masknmf.utils import display, get_timestamp, TIMESTAMP_FORMAT, torch_select_device
from masknmf.io import REGISTRATION_ARRAYS, has_group, drop_group, create_run_folder, log_to, write_run_config
from masknmf.motion_correction.registration_arrays import FilteredArray
from masknmf.motion_correction.spatial_filters import compute_highpass_filter_kernel, image_filter

logger = logging.getLogger(__name__)
WIDTH_STAGE_LINE = 60



def heartbeat(name: str, start: float, seconds: float, stop: threading.Event):
    """Log every seconds that the step name, started at start (monotonic), is still running, until stop is set."""
    while not stop.wait(seconds):
        logger.info(f"{name} still running after {timedelta(seconds=round(time.monotonic() - start))}")


def _highpass_relu(frames: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    """Highpass frames with kernel, then clip at zero."""
    return torch.relu(image_filter(frames.to(kernel.device), kernel))


class RecordsFailure:
    """
    A pipeline's run that, when it raises after making its run folder, logs the error and records in config.json
    that the run failed, then lets the error through. BasePipeline wraps every subclass's run in one, so a run
    started from Python leaves the same record as one started from the command line. A run given resume takes
    the settings it was not given from the earlier run's config.json first.
    """

    def __init__(self, run):
        functools.update_wrapper(self, run)
        self.run = run

    def __get__(self, pipeline, owner=None):
        return self if pipeline is None else functools.partial(self, pipeline)

    def __call__(self, pipeline, *args, **kwargs):
        # a folder left from an earlier run of the same pipeline is not this run's
        pipeline.run_folder = None
        try:
            return self.run(pipeline, **pipeline.arguments_resumed(run=self.run, args=args, kwargs=kwargs))
        except BaseException:
            if pipeline.run_folder is not None:
                logger.exception(f"{type(pipeline).__name__} failed")
                pipeline.finish("failed")
            raise


class BasePipeline(ABC):
    """
    Motion correction, compression (and optional demixing) of one session.
    """

    heartbeat_seconds = 600

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if "run" in cls.__dict__:
            cls.run = RecordsFailure(cls.__dict__["run"])

    def __init__(self,
                 output_folder: str | Path | None = None,
                 frame_batch_size: int = 300,
                 device: Literal["auto", "cuda", "cpu"] = "auto",
                 log_level: Literal["debug", "info", "warning"] = "info",
                 load_into_ram: bool = False,
                 device_memory_gb: float | None = None,
                 host_memory_gb: float | None = None,
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
        # the config arguments given, which a resumed run keeps over the earlier run's
        self.configs_given = {name for name in defaults if configs.get(name) is not None}
        self.device_memory_gb = device_memory_gb
        self.host_memory_gb = host_memory_gb

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

    def arguments_resumed(self, run: Callable, args: tuple, kwargs: dict) -> dict:
        """
        The arguments of run by name. With a resume that has a config.json beside it, the run arguments not
        given and the config arguments of __init__ not given take that run's values; stop_after is this run's own.

        Args:
            run (Callable): The pipeline's run function, taking the pipeline first
            args (tuple): The positional arguments run was called with, after the pipeline
            kwargs (dict): The keyword arguments run was called with
        Returns:
            dict: Every argument run was given, and those taken from the earlier run, by name
        Raises:
            ValueError: If the config.json is another pipeline's
        """
        parameters_run = inspect.signature(run).parameters
        arguments = inspect.signature(run).bind_partial(self, *args, **kwargs).arguments
        del arguments[next(iter(parameters_run))]
        if arguments.get("resume") is None:
            return arguments
        filepath_config = Path(arguments["resume"]).expanduser().resolve().parent / "config.json"
        if not filepath_config.is_file():
            logger.info(f"no config.json beside {arguments['resume']}; the settings not given are the defaults")
            return arguments
        earlier = json.loads(filepath_config.read_text())
        if earlier.get("pipeline", type(self).__name__) != type(self).__name__:
            raise ValueError(f"{filepath_config} is for {earlier['pipeline']}, not {type(self).__name__}")
        values = earlier.get("configs", {})
        defaults = self.default_configs()
        parameters_init = inspect.signature(type(self).__init__).parameters
        for name in defaults:
            if name in values and name not in self.configs_given:
                setattr(self, name, config_from_json(value=values[name], annotation=parameters_init[name].annotation,
                                                     base=defaults[name]))
        for name in parameters_run:
            if name in values and name not in arguments and name != "stop_after":
                arguments[name] = values[name]
        logger.info(f"the settings not given are those of {filepath_config}")
        return arguments

    @property
    def device_memory_gb(self) -> float | None:
        """
        The allowed memory footprint of the computational device (gpu for e.g.) measured in gigabytes
        """
        return self._device_memory_gb

    @device_memory_gb.setter
    def device_memory_gb(self, new_memory: float | None):
        if new_memory is None or new_memory > 0:
            self._device_memory_gb = new_memory
        else:
            raise ValueError(f"Must provide a positive value to device_memory_gb setter or None, provided {new_memory}")

    @property
    def host_memory_gb(self) -> float | None:
        """
        The allowed memory footprint of the host system measured in gigabytes
        """
        return self._host_memory_gb

    @host_memory_gb.setter
    def host_memory_gb(self, new_memory: float | None):
        if new_memory is None or new_memory > 0:
            self._host_memory_gb = new_memory
        else:
            raise ValueError(f"Must provide a positive value to host_memory_gb setter or None, provided {new_memory}")

    @property
    def config(self) -> dict:
        """Every __init__ argument by name, as constructed."""
        return {name: getattr(self, name) for name in inspect.signature(type(self).__init__).parameters if name != "self"}

    @property
    def torch_device(self) -> str:
        """``device`` with "auto" resolved to the device pytorch will use."""
        return torch_select_device(self.device)

    def create_run_folder(self, base: Path | None = None) -> Path:
        """
        Make ``<base>/<yyyymmddTHHMMSS>_<pipeline slug>/``, base being output_folder unless given (the working
        directory when both are None), adding a numeric suffix when a run started in the same second, and write
        config.json in it.
        """
        if base is None:
            base = Path.cwd() if self.output_folder is None else self.output_folder
        candidate = create_run_folder(base, slugify(name_class=type(self).__name__))
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
        return write_run_config(self.run_folder, type(self).__name__, {**self.config, **self.run_config, **self.configs_reused},
                                run=self.run_record, inputs=self.inputs, timings=self.timings, default=config_json_value)

    def log_to(self, folder: Path) -> Path:
        """
        Write the masknmf log to ``<folder>/<folder name>.log`` from here on, appending to the file an earlier run
        left there and closing the file of the run logged until now, start the run record (command, device, gpu and
        start time; finish ends it) and log this run's header line.
        """
        self.log_handler = log_to(folder)
        path = Path(self.log_handler.baseFilename)
        device = str(self.torch_device)
        self.run_record = {"command": self.command, "device": device,
                           "gpu": torch.cuda.get_device_name() if device.startswith("cuda") else None,
                           "started": get_timestamp(), "finished": None, "seconds": None,
                           "status": "running"}
        gpu = f", {self.run_record['gpu']}" if self.run_record["gpu"] is not None else ""
        logger.info(f"masknmf {__version__} {type(self).__name__} on {device}{gpu}")
        logger.info(f"run folder {folder}" + ("" if self.log_level == "info" else f", {self.log_level} log"))
        return path

    @contextmanager
    def step(self, name: str):
        """
        Log that name starts, that it is still running every heartbeat_seconds and, once the block ends, how long it
        took (with its peak cuda memory on a cuda device), or that it failed and after how long, and record its start,
        seconds, status and peak under name in timings and config.json.
        """
        logger.info(f"starting {name} ".ljust(WIDTH_STAGE_LINE, "-"))
        started = get_timestamp()
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
        self.run_record.update(finished=finished.strftime(TIMESTAMP_FORMAT), seconds=round(seconds, 1), status=status)
        self.write_config()
        return self.run_folder

    def results_path(self, base: Path | None = None) -> str:
        """``results.hdf5`` in a new run folder made in base, or in output_folder unless base is given."""
        path = os.path.join(self.create_run_folder(base), "results.hdf5")
        logger.debug(f"writing results to {path}")
        return path

    def resume_source(self, resume: str | Path | None, reuse_compression: bool) -> tuple[Path | None, Path | None]:
        """
        The earlier results file a run resumes and the folder its new run folder goes in, checked before anything is
        written. With reuse_compression and no resume, as compress_config "skip" asks, it is the results.hdf5 in
        output_folder, an earlier run folder, and the new run folder goes beside that one.

        Args:
            resume (str | Path | None): The earlier results file, or None
            reuse_compression (bool): Whether its compression is reused, else only its registration
        Returns:
            Path | None: The earlier results file, or None for a run that resumes nothing
            Path | None: The folder the new run folder goes in, or None for output_folder
        Raises:
            FileNotFoundError: If the earlier results file is missing
            ValueError: If it holds no compression to reuse, or no registration when only that is reused
        """
        base = None
        if resume is None and reuse_compression:
            folder = Path.cwd() if self.output_folder is None else self.output_folder
            resume = folder / "results.hdf5"
            base = folder.parent
        if resume is None:
            return None, None
        path = Path(resume).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"no results file to resume from at {path}")
        if reuse_compression and not has_group(path, CompressionArray.__name__):
            raise ValueError(f"{path} holds no compression to reuse")
        if not reuse_compression and not any(has_group(path, name) for name in REGISTRATION_ARRAYS):
            raise ValueError(f"{path} holds no registration to resume from")
        return path, base

    def resume(self, resume: Path, results_path: str, reuse_compression: bool) -> type[BaseRegistrationArray] | None:
        """
        Copy the registration of an earlier results file, and its compression with reuse_compression, into this run's
        results_path; the earlier file is left as it is. config.json keeps the inputs, configs and timings of the run
        that made what is reused.

        Args:
            resume (Path): The earlier results file
            results_path (str): This run's results file
            reuse_compression (bool): Whether the compression is copied too
        Returns:
            type[BaseRegistrationArray] | None: The registration array class copied, or None when there was none
        """
        stored = next((c for c in REGISTRATION_ARRAYS.values() if has_group(resume, c.__name__)), None)
        names = [] if stored is None else [stored.__name__, stored._strategy_cls.__name__]
        if reuse_compression:
            names.append(CompressionArray.__name__)
        with h5py.File(resume, "r") as source, h5py.File(results_path, "a") as target:
            for name in names:
                source.copy(source[name], target, name=name)
        logger.info(f"reusing {', '.join(names)} from {resume}")
        filepath_config = resume.parent / "config.json"
        earlier = json.loads(filepath_config.read_text()) if filepath_config.is_file() else {}
        reused = ["motion_correct_config"] + (["compress_config"] if reuse_compression else [])
        steps = ["motion correction"] + (["compression"] if reuse_compression else [])
        self.inputs = {**earlier.get("inputs", {}), **self.inputs}
        self.configs_reused = {name: value for name, value in earlier.get("configs", {}).items() if name in reused}
        self.timings = {name: timing for name, timing in earlier.get("timings", {}).items() if name in steps}
        self.write_config()
        return stored

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
                       filter_config: MotionCorrectionFilterConfig | None = None,
                       exclude_border_radius: int = 0,
                       stored: type[BaseRegistrationArray] | None = None) -> tuple[np.ndarray | ArrayLike, np.ndarray]:
        """
        Register data with a rigid (the default when config is None) or piecewise rigid corrector and export it to
        results_path, or pass it through for "skip". With stored, the registration array class resume copied into
        results_path, that registration is applied to data instead. Also returns a pixel weighting that is 0 where
        the shifts moved pixels in from outside the fov and on the outer exclude_border_radius pixels.
        """
        if stored is not None:
            logger.info(f"applying the stored {stored.__name__}")
            moco_data = stored.from_hdf5(results_path, input_movie=data, device=self.device)
            moco_data.output_device = torch.device('cpu')
        elif isinstance(config, str):
            if config.lower() != "skip":
                raise ValueError("Invalid MotionCorrectionConfig input")
            display("Not Running Motion Correction")
            moco_data = data
        else:
            if config is None or isinstance(config, RigidMotionCorrectionConfig):
                moco_strategy = RigidMotionCorrector(**asdict(RigidMotionCorrectionConfig() if config is None else config),
                                                     device=self.device, batch_size=self.frame_batch_size)
            elif isinstance(config, PiecewiseRigidMotionCorrectionConfig):
                kwargs = asdict(config)
                moco_strategy = PiecewiseRigidMotionCorrector(**kwargs, device=self.device,
                                                              batch_size=self.frame_batch_size)
            else:
                raise ValueError("Invalid MotionCorrectionConfig input")
            with self.step("motion correction"):
                if filter_config is None:
                    reference_data = data
                else:
                    # Define a filtered version of the movie, estimate shifts by motion correcting that movie, apply to original movie
                    reference_data = self._filtered_reference(data, filter_config)
                if moco_strategy.template is None:
                    moco_strategy.compute_template(reference_data)
                moco_data = moco_strategy.motion_correct(reference_data, target_movie=data)
                moco_data.output_device = torch.device('cpu')
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
    
    def _filtered_reference(self, data: ArrayLike, config: MotionCorrectionFilterConfig) -> FilteredArray:
        """Define a lazy transformation of the input dataset that we use to compute per-frame shifts, and we
        apply this to the original dataset
        """
        if isinstance(config, SpatialHighpassMotionCorrectionConfig):
            "Here we use a highpassed movie (box - gaussian kernel) with kernel parameter sigma. clipped at 0"
            sigma = config.reference_highpass_sigma
            kernel = compute_highpass_filter_kernel([sigma, sigma]).to(self.torch_device)
            return FilteredArray(data, partial(_highpass_relu, kernel=kernel),
                                 batching=self.frame_batch_size, device=self.torch_device)
        else:
            raise ValueError(f"Invalid config type for filtered motion correction, type {type(config)} and it is {config}")
    
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
            return CompressStrategy(device=self.device, **kwargs)
        if isinstance(config, CompressDenoiseConfig):
            return CompressDenoiseStrategy(device=self.device, **kwargs)
        raise ValueError("Invalid compression config")

    def spline_detrender(self,
                         num_frames: int,
                         frame_rate: float,
                         window_seconds: float,
                         knot_seconds: float,
                         sigma_seconds: float,
                         geometric_first_seconds: float | None = None,
                         geometric_growth: float = 1.5,
                         edge_mode: Literal["mirror", "line"] = "mirror") -> MaximinSplineDetrend:
        """
        A maximin spline detrender with its rolling window, knot spacing and smoothing given in seconds.

        By default the knots are uniform, knot_seconds apart. With geometric_first_seconds set, they are dense at
        the start instead: the first knot geometric_first_seconds in, each gap geometric_growth times the last,
        capped at knot_seconds. Use this when photobleaching is fast at the start of the recording.
        """
        if int(window_seconds * frame_rate) < 3:
            raise ValueError(f"at {frame_rate} fps this pipeline's {window_seconds} s detrending window is under 3 frames; "
                             f"it needs at least {3 / window_seconds:g} fps")
        knot_positions = None
        num_knots = max(4, int(num_frames / frame_rate / knot_seconds))
        if geometric_first_seconds is not None:
            positions = geometric_knot_positions(num_frames, frame_rate,
                                                 first_seconds=geometric_first_seconds,
                                                 growth=geometric_growth,
                                                 max_spacing_seconds=knot_seconds)
            if positions:  # a recording too short for any knot keeps the uniform ones
                knot_positions, num_knots = positions, len(positions)
        return MaximinSplineDetrend(num_frames=num_frames,
                                    num_knots=num_knots,
                                    knot_positions=knot_positions,
                                    window=int(window_seconds * frame_rate),
                                    sigma=max(2.0, sigma_seconds * frame_rate),
                                    device=self.torch_device,
                                    edge_mode=edge_mode)

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
                initialized = demixer.results.spatial_demixed.shape[1]
                demixer.demix(**asdict(singlepass.NMFConfig))
                results = demixer.results
                logger.info(f"{initialized} signals initialized, {results.spatial_demixed.shape[1]} kept")
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

    def _construct_ophys_array(self,
                               data: ArrayLike,
                               indicator_polarity: Literal["positive", "negative"],
                               good_frame_indicator: torch.Tensor | np.ndarray | list | None):
        if indicator_polarity.lower() not in {"positive", "negative"}:
            raise ValueError("indicator polarity can only be positive or negative")
        ophys_output = OphysArray(data,
                                  negative_indicator = True if indicator_polarity.lower() == "negative" else False,
                                  include_mean = True,
                                  device=self.torch_device,
                                  good_frames = good_frame_indicator)
        return ophys_output

    @property
    def demixing_configs(self) -> None | MultipassDemixingConfig | list[SinglepassDemixingConfig | MultipassDemixingConfig]:
        return None

    def demixing_passes(self) -> list[SinglepassDemixingConfig]:
        """Every single demixing pass of demixing_configs, in order; "skip" entries are left out."""
        configs = self.demixing_configs
        if configs is None:
            raise NotImplementedError(f"{type(self).__name__} does not list its demixing configs")
        if isinstance(configs, (SinglepassDemixingConfig, MultipassDemixingConfig)):
            configs = [configs]
        passes = []
        for config in configs:
            if isinstance(config, SinglepassDemixingConfig):
                passes.append(config)
            elif isinstance(config, MultipassDemixingConfig):
                passes.extend(config.DemixingConfigs)
        return passes

    @property
    def min_brightness(self) -> float | list[float | None] | None:
        """The min brightness every demixing pass uses, or each pass's in the order they run when they differ."""
        values = [p.NMFConfig.min_brightness for p in self.demixing_passes()]
        return values[0] if len(set(values)) == 1 else values

    @min_brightness.setter
    def min_brightness(self, value: float | tuple[float, ...] | None):
        """
        Set the min brightness of the demixing passes: one value for every pass, or one per pass in the order they
        run (a list in Python, comma separated on the command line). 0 or None turns the check off.
        """
        passes = self.demixing_passes()
        values = list(value) if isinstance(value, (list, tuple)) else [value]
        if len(values) == 1:
            values = values * len(passes)
        if len(values) != len(passes):
            raise ValueError(f"{len(values)} min brightness values for {len(passes)} demixing passes")
        if any(v is not None and v < 0 for v in values):
            raise ValueError("min_brightness cannot be negative")
        for p, v in zip(passes, values):
            p.NMFConfig.min_brightness = v
        # a resumed run replaces every config not in configs_given with the earlier run's; mark the ones changed here
        configs_changed = self.demixing_configs if isinstance(self.demixing_configs, list) else [self.demixing_configs]
        for name in self.default_configs():
            if any(getattr(self, name) is config for config in configs_changed if not isinstance(config, str)):
                self.configs_given.add(name)



    @abstractmethod
    def run(self, **kwargs) -> Path:
        """Run the pipeline on its movie(s) and return the run folder it wrote to."""
        raise NotImplementedError
