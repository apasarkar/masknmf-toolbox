"""Masknmf results files and run folders: the stages a file holds, the raw movie, what the viewers show of it, and
the run folder a run writes them in."""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import h5py
import numpy as np
import torch

from masknmf.arrays import ArrayLike, Hdf5Array, TiffArray, TiffSeriesLoader
from masknmf.compression import CompressionArray
from masknmf.demixing import DemixingResults
from masknmf.demixing._base_results import BaseResults
from masknmf.motion_correction import BaseRegistrationArray, GradientRegistrationArray, PiecewiseRigidRegistrationArray, RigidRegistrationArray
from masknmf._version import __version__
from masknmf.utils import get_timestamp, TIMESTAMP_FORMAT

__all__ = [
    "SUFFIXES_HDF5",
    "SUFFIXES_TIFF",
    "REGISTRATION_ARRAYS",
    "group_names_registration",
    "group_name_compression",
    "group_name_demixing",
    "load_movie",
    "movie_beside",
    "stage_groups",
    "OpenedResults",
    "has_group",
    "drop_group",
    "create_run_folder",
    "log_to",
    "write_run_config",
]

SUFFIXES_TIFF = (".tif", ".tiff")
SUFFIXES_HDF5 = (".h5", ".hdf5")
# the registration arrays by the hdf5 group name each is stored under
REGISTRATION_ARRAYS = {
    cls.__name__: cls for cls in (RigidRegistrationArray, PiecewiseRigidRegistrationArray, GradientRegistrationArray)
}


def group_names_registration() -> tuple[str, ...]:
    """The hdf5 group names a registration stage can be stored under."""
    return tuple(REGISTRATION_ARRAYS)


def group_name_compression() -> str:
    """The hdf5 group name the compression stage is stored under."""
    return CompressionArray.__name__


def group_name_demixing() -> str:
    """The hdf5 group name the demixing stage is stored under."""
    return DemixingResults.__name__


def load_movie(path: str | Path, dataset: Optional[str] = None):
    """
    A raw movie as a lazy (frames, height, width) array: a tiff file, a directory of tiff files, or an hdf5 file
    with the movie under ``dataset``.
    """
    path = Path(path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"no such file or directory: {path}")
    if path.is_dir():
        tiffs = sorted(str(p) for p in path.iterdir() if p.suffix.lower() in SUFFIXES_TIFF)
        if len(tiffs) == 0:
            raise ValueError(f"no tiff files in {path}")
        return TiffArray(tiffs[0]) if len(tiffs) == 1 else TiffSeriesLoader(tiffs)
    suffix = path.suffix.lower()
    if suffix in SUFFIXES_TIFF:
        return TiffArray(str(path))
    if suffix in SUFFIXES_HDF5:
        if dataset is None:
            raise ValueError(f"{path.name} is hdf5; name the dataset holding the movie")
        return Hdf5Array(str(path), dataset)
    raise ValueError(f"masknmf cannot read {path.name}; expected a .tif/.tiff file, a directory of them, or a .h5/.hdf5 file")


def movie_beside(path: str | Path) -> Optional[Path]:
    """The movie beside a results file: the path of the one .tif in its folder, None when there is none or several."""
    folder = Path(path).parent
    tiffs = sorted(p for ext in ("*.tif", "*.tiff") for p in folder.glob(ext))
    return tiffs[0] if len(tiffs) == 1 else None


def stage_groups(path: str | Path) -> list[str]:
    """The masknmf stage groups a results file holds, in pipeline order, then demixing results under a prefix."""
    path = Path(path)
    if path.suffix.lower() not in SUFFIXES_HDF5:
        raise ValueError(f"{path.name} is not a .h5/.hdf5 results file")
    if not path.is_file():
        raise FileNotFoundError(f"no such file: {path}")
    with h5py.File(path, "r") as f:
        names = [name for name in (*group_names_registration(), group_name_compression(), group_name_demixing()) if name in f]
        if len(names) > 0:
            names += [
                f"{key}/{group_name_demixing()}"
                for key in f
                if isinstance(f[key], h5py.Group) and group_name_demixing() in f[key]
            ]
    return names


@dataclass
class OpenedResults:
    """
    Results and the movies that go with them: the raw movie, the registered one, the registration shifts and the
    template the registration aligned to, each None when there is none. ``path`` is the file the results were read
    from, ``raw_source`` and ``shifts_source`` the files the raw movie and shifts were read from (None when given as
    data or taken from the results), and ``skipped`` says what was found but did not line up with the results.
    """

    results: DemixingResults | CompressionArray | BaseRegistrationArray | BaseResults
    path: Optional[Path] = None
    raw: Optional[ArrayLike] = None
    registered: Optional[ArrayLike] = None
    shifts: Optional[np.ndarray] = None
    template: Optional[np.ndarray] = None
    raw_source: Optional[Path] = None
    shifts_source: Optional[Path] = None
    skipped: list[str] = field(default_factory=list)

    @classmethod
    def open(cls, path: str | Path, raw=None, prefix: str = "", device: str = "cpu") -> "OpenedResults":
        """
        A results file, any stage: its DemixingResults (under ``prefix`` when given), else its CompressionArray, else
        its registration replayed on the raw movie. ``raw`` (an array or a movie path) is skipped when it does not
        line up with the results, as when a viewer moves on to another movie's results; the rest is found as
        :meth:`resolve` finds it.
        """
        path = Path(path)
        groups = stage_groups(path)
        if len(groups) == 0:
            raise ValueError(f"{path} holds no masknmf results")
        name_demixing = f"{prefix}/{group_name_demixing()}" if prefix else group_name_demixing()
        if prefix and name_demixing not in groups:
            raise ValueError(f"{path} holds no {name_demixing}")
        raw_source = Path(raw) if isinstance(raw, (str, Path)) else None
        if raw_source is not None:
            raw = load_movie(raw_source)
        skipped = []
        if name_demixing in groups or group_name_compression() in groups:
            if name_demixing in groups:
                results = DemixingResults.from_hdf5(path, prefix=prefix, device=device)
            else:
                results = CompressionArray.from_hdf5(path)
            shape = tuple(int(v) for v in results.shape)
            if raw is not None and tuple(raw.shape) != shape:
                # a raw movie the pipeline trimmed (the glutamate pipeline drops its first frames) no longer lines up
                skipped.append(f"raw movie is {tuple(raw.shape)}, the results {shape}; not shown")
                raw, raw_source = None, None
            opened = cls.resolve(results, path, raw=raw, device=device)
        else:
            name_registration = next(n for n in group_names_registration() if n in groups)
            with h5py.File(path, "r") as f:
                frames = f[name_registration]["shifts"].shape[0]
            if raw is not None and raw.shape[0] != frames:
                skipped.append(f"raw movie has {raw.shape[0]} frames, the {name_registration} {frames}; not shown")
                raw, raw_source = None, None
            if raw is None:
                raw_source = movie_beside(path)
                raw = None if raw_source is None else TiffArray(str(raw_source))
            if raw is not None and raw.shape[0] != frames:
                skipped.append(f"skipping {raw_source}: {raw.shape[0]} frames, the {name_registration} {frames}")
                raw, raw_source = None, None
            if raw is None:
                raise ValueError(f"{path} holds only {name_registration}; showing it needs the raw movie the run registered")
            opened = cls.resolve(REGISTRATION_ARRAYS[name_registration].from_hdf5(path, input_movie=raw, device=device), path)
        if raw_source is not None:
            opened.raw_source = raw_source
        opened.skipped[:0] = skipped
        return opened

    @classmethod
    def resolve(cls, results, path: str | Path | None = None, raw=None, registered=None, shifts=None,
                device: Optional[str] = None) -> "OpenedResults":
        """
        ``results`` with the movies that go with them, each the one given, else the results' own, else found from
        ``path``: the one .tif in its folder for the raw movie, and the registration the file holds for the shifts
        and template, replayed on the raw movie for the registered one. Given ones that do not line up with the
        results raise; found ones are skipped. ``raw`` is an array or a movie path, ``shifts`` an array or the path
        of a results file holding a registration. ``device`` is where a registration found in the file is replayed,
        the one it was stored with when None.
        """
        path = None if path is None else Path(path)
        shape = tuple(int(v) for v in results.shape)
        is_registration = isinstance(results, BaseRegistrationArray)
        skipped = []
        if registered is not None and is_registration:
            raise ValueError("registered= goes with compression or demixing results; a registration array is the registered movie")
        if registered is not None and tuple(registered.shape) != shape:
            raise ValueError(f"registered movie has shape {tuple(registered.shape)}, the results have {shape}")

        raw_source = Path(raw) if isinstance(raw, (str, Path)) else None
        if raw_source is not None:
            raw = load_movie(raw_source)
        if raw is None and is_registration:
            raw = results.input_movie
        if raw is None and isinstance(results, BaseResults):
            raw = results.raw_array
        if raw is not None and tuple(raw.shape) != shape:
            raise ValueError(f"raw movie has shape {tuple(raw.shape)}, the results have {shape}")
        if raw is None and path is not None:
            raw_source = movie_beside(path)
            raw = None if raw_source is None else TiffArray(str(raw_source))
            if raw is not None and tuple(raw.shape) != shape:
                skipped.append(f"skipping {raw_source}: shape {tuple(raw.shape)} does not match the results")
                raw, raw_source = None, None

        name_registration = None
        if path is not None and not is_registration:
            with h5py.File(path, "r") as f:
                name_registration = next((n for n in group_names_registration() if n in f), None)
        if registered is None and isinstance(results, BaseResults):
            registered = results.registered_array
        if registered is None and raw is not None and name_registration is not None:
            with h5py.File(path, "r") as f:
                frames = f[name_registration]["shifts"].shape[0]
            if frames == shape[0]:
                registered = REGISTRATION_ARRAYS[name_registration].from_hdf5(path, input_movie=raw, device=device)
            else:
                skipped.append(f"the {name_registration} in {path} has {frames} frames, the movie {shape[0]}; shifts not applied")

        shifts_source = Path(shifts) if isinstance(shifts, (str, Path)) else None
        found_shifts = False
        if shifts is None and (is_registration or isinstance(results, BaseResults)):
            shifts = results.shifts
        if shifts is None and name_registration is not None:
            shifts_source, found_shifts = path, True
        template = None
        if is_registration:
            template = results.strategy.template
        elif isinstance(registered, BaseRegistrationArray):
            template = registered.strategy.template
        if shifts_source is not None:
            with h5py.File(shifts_source, "r") as f:
                name = next((n for n in group_names_registration() if n in f), None)
                if name is None:
                    raise ValueError(f"{shifts_source} holds no registration array")
                shifts = f[name]["shifts"][()]
                strategy = REGISTRATION_ARRAYS[name]._strategy_cls.__name__
                if template is None and strategy in f and "template" in f[strategy]:
                    template = f[strategy]["template"][()]
        if isinstance(shifts, torch.Tensor):
            shifts = shifts.cpu().numpy()
        if shifts is not None:
            shifts = np.asarray(shifts, np.float32)
            if shifts.shape[0] != shape[0]:
                if not found_shifts:
                    raise ValueError(f"{shifts.shape[0]} shifts for {shape[0]} frames")
                skipped.append(f"skipping {shifts_source}: {shifts.shape[0]} shifts for {shape[0]} frames")
                shifts, shifts_source, template = None, None, None
        if isinstance(template, torch.Tensor):
            template = template.cpu().numpy()
        if template is not None:
            template = np.asarray(template, np.float32)
        return cls(results=results, path=path, raw=raw, registered=registered, shifts=shifts, template=template,
                   raw_source=raw_source, shifts_source=shifts_source, skipped=skipped)


def has_group(filename: str | Path, group: str) -> bool:
    """Whether ``filename`` is an hdf5 file that holds ``group``."""
    if not Path(filename).is_file():
        return False
    with h5py.File(filename, "r") as f:
        return group in f


def drop_group(filename: str | Path, group: str):
    """Remove ``group`` from an hdf5 file by rewriting it without the group, so the space comes back; a no-op when absent."""
    if not has_group(filename, group):
        return
    filename = Path(filename)
    packed = filename.with_name(f"{filename.name}.repack")
    with h5py.File(filename, "r") as src, h5py.File(packed, "w") as dst:
        for name in src:
            if name != group:
                src.copy(name, dst)
        dst.attrs.update(src.attrs)
    packed.replace(filename)


def create_run_folder(base: str | Path, name: str) -> Path:
    """Make ``<base>/<yyyymmddTHHMMSS>_<name>/``, adding a numeric suffix when a run started in the same second."""
    stem = f"{get_timestamp()}_{name}"
    candidate = Path(base) / stem
    suffix = 0
    while True:
        try:
            candidate.mkdir(parents=True, exist_ok=False)
            return candidate
        except FileExistsError:
            suffix += 1
            candidate = Path(base) / f"{stem}_{suffix}"


def log_to(folder: str | Path) -> logging.FileHandler:
    """
    Write the masknmf log to ``<folder>/<folder name>.log`` from here on, appending to the file an earlier run left
    there and closing the file logged to until now.
    """
    for handler in [h for h in logging.getLogger("masknmf").handlers if isinstance(h, logging.FileHandler)]:
        logging.getLogger("masknmf").removeHandler(handler)
        handler.close()
    folder = Path(folder)
    handler = logging.FileHandler(folder / f"{folder.name}.log", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s", datefmt=TIMESTAMP_FORMAT))
    logging.getLogger("masknmf").addHandler(handler)
    return handler


def write_run_config(folder: str | Path,
                     pipeline: str,
                     configs: dict,
                     run: Optional[dict] = None,
                     inputs: Optional[dict] = None,
                     timings: Optional[dict] = None,
                     default=str) -> Path:
    """
    Write ``config.json`` in a run folder: the masknmf version, what made the run (a pipeline class name, or any name
    for a run made by hand), the run record (command, device, start, end, status), the files the run read by
    argument name ({"path": ..., "name": ...}), the configs it ran with and the steps that ran. default writes what
    json cannot itself.
    """
    path = Path(folder) / "config.json"
    with open(path, "w") as f:
        json.dump({"masknmf_version": __version__, "pipeline": pipeline, "run": run or {}, "inputs": inputs or {},
                   "configs": configs, "timings": timings or {}}, f, indent=2, default=default)
    return path
