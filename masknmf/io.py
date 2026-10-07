"""Reading and writing masknmf results files and run folders."""

import itertools
import json
import logging
from dataclasses import dataclass, field, replace
from pathlib import Path

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
    """Group names a registration can be stored under."""
    return tuple(REGISTRATION_ARRAYS)


def group_name_compression() -> str:
    """Group name of the compression."""
    return CompressionArray.__name__


def group_name_demixing() -> str:
    """Group name of the demixing results."""
    return DemixingResults.__name__


def load_movie(path: str | Path, dataset: str | None = None) -> ArrayLike:
    """
    Open a movie as a lazy (frames, height, width) array.

    Parameters
    ----------
    path : str | Path
        A .tif/.tiff file, a folder of them (read in name order as one movie), or a .h5/.hdf5 file.
    dataset : str | None
        The dataset holding the movie in an hdf5 file.

    Returns
    -------
    ArrayLike

    Raises
    ------
    FileNotFoundError
        If ``path`` does not exist.
    ValueError
        If ``path`` is not a format masknmf reads, a folder holds no tiffs, or an hdf5 file is given without ``dataset``.
    """
    path = Path(path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"no such file or directory: {path}")
    if path.is_dir():
        tiffs = sorted(str(p) for p in path.iterdir() if p.suffix.lower() in SUFFIXES_TIFF)
        if not tiffs:
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


def movie_beside(path: str | Path) -> Path | None:
    """The only tiff in the folder of ``path``, or None when there are none or several."""
    tiffs = [p for p in Path(path).parent.iterdir() if p.suffix.lower() in SUFFIXES_TIFF]
    return tiffs[0] if len(tiffs) == 1 else None


def stage_groups(path: str | Path) -> list[str]:
    """
    List the masknmf stages a results file holds.

    Parameters
    ----------
    path : str | Path
        A .h5/.hdf5 file.

    Returns
    -------
    list[str]
        Group names in pipeline order (registration, compression, demixing), then any ``<prefix>/DemixingResults``.
        Empty for an hdf5 file masknmf did not write.

    Raises
    ------
    ValueError
        If ``path`` is not .h5/.hdf5.
    FileNotFoundError
        If ``path`` does not exist.
    """
    path = Path(path)
    if path.suffix.lower() not in SUFFIXES_HDF5:
        raise ValueError(f"{path.name} is not a .h5/.hdf5 results file")
    if not path.is_file():
        raise FileNotFoundError(f"no such file: {path}")
    demixing = group_name_demixing()
    with h5py.File(path, "r") as f:
        names = [name for name in (*group_names_registration(), group_name_compression(), demixing) if name in f]
        if names:
            names += [f"{key}/{demixing}" for key in f if isinstance(f[key], h5py.Group) and demixing in f[key]]
    return names


@dataclass
class OpenedResults:
    """
    Results together with the movies that go with them.

    Build one with :meth:`open` (from a file) or :meth:`resolve` (from results in memory).

    Attributes
    ----------
    results : DemixingResults | CompressionArray | BaseRegistrationArray | BaseResults
        The results, of whichever stage.
    path : Path | None
        The file the results were read from.
    raw : ArrayLike | None
        The raw movie.
    registered : ArrayLike | None
        The registration replayed on ``raw``; None when ``results`` is itself the registration.
    shifts : np.ndarray | None
        Registration shifts, float32, (frames, 2) or (frames, height blocks, width blocks, 2) for piecewise rigid.
    template : np.ndarray | None
        The template the registration aligned to, float32.
    raw_source, shifts_source : Path | None
        The files ``raw`` and ``shifts`` were read from; None when given as data or taken from ``results``.
    skipped : list[str]
        What was found but left out because it did not line up with the results.
    """

    results: DemixingResults | CompressionArray | BaseRegistrationArray | BaseResults
    path: Path | None = None
    raw: ArrayLike | None = None
    registered: ArrayLike | None = None
    shifts: np.ndarray | None = None
    template: np.ndarray | None = None
    raw_source: Path | None = None
    shifts_source: Path | None = None
    skipped: list[str] = field(default_factory=list)

    @classmethod
    def open(cls,
             path: str | Path,
             raw: ArrayLike | np.ndarray | str | Path | None = None,
             prefix: str = "",
             device: str = "cpu") -> "OpenedResults":
        """
        Open a results file of any stage.

        The results are the file's DemixingResults, else its CompressionArray, else its registration replayed on the
        raw movie. A ``raw`` that does not line up with them is skipped, not an error, so a viewer can keep its movie
        when moving on to another recording's results. Everything else is found as in :meth:`resolve`.

        Parameters
        ----------
        path : str | Path
            The results file.
        raw : ArrayLike | np.ndarray | str | Path | None
            The raw movie, or a path :func:`load_movie` reads.
        prefix : str
            Read the DemixingResults under ``<prefix>/``, e.g. "global".
        device : str
            Device for the DemixingResults and for replaying the registration.

        Returns
        -------
        OpenedResults

        Raises
        ------
        ValueError
            If the file holds no masknmf results, none under ``prefix``, or only a registration and no raw movie that
            lines up with it.
        """
        path = Path(path)
        groups = stage_groups(path)
        if not groups:
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
            shape = tuple(int(n) for n in results.shape)
            if raw is not None and tuple(raw.shape) != shape:
                # e.g. the glutamate pipeline drops a movie's first frames
                skipped.append(f"raw movie is {tuple(raw.shape)}, the results {shape}; not shown")
                raw = raw_source = None
            opened = cls.resolve(results, path, raw=raw, device=device)
        else:
            stored = next(name for name in group_names_registration() if name in groups)
            with h5py.File(path, "r") as f:
                frames = f[stored]["shifts"].shape[0]
            if raw is not None and raw.shape[0] != frames:
                skipped.append(f"raw movie has {raw.shape[0]} frames, the {stored} {frames}; not shown")
                raw = raw_source = None
            if raw is None and (beside := movie_beside(path)) is not None:
                movie = TiffArray(str(beside))
                if movie.shape[0] == frames:
                    raw, raw_source = movie, beside
                else:
                    skipped.append(f"skipping {beside}: {movie.shape[0]} frames, the {stored} {frames}")
            if raw is None:
                raise ValueError(f"{path} holds only {stored}; showing it needs the raw movie the run registered")
            opened = cls.resolve(REGISTRATION_ARRAYS[stored].from_hdf5(path, input_movie=raw, device=device), path)
        return replace(opened, raw_source=raw_source or opened.raw_source, skipped=skipped + opened.skipped)

    @classmethod
    def resolve(cls,
                results: DemixingResults | CompressionArray | BaseRegistrationArray | BaseResults,
                path: str | Path | None = None,
                raw: ArrayLike | np.ndarray | str | Path | None = None,
                registered: ArrayLike | None = None,
                shifts: np.ndarray | torch.Tensor | str | Path | None = None,
                device: str | None = None) -> "OpenedResults":
        """
        Gather the movies that go with results already in memory.

        Each of ``raw``, ``registered`` and ``shifts`` is the one given, else the one ``results`` carries, else one
        found from ``path``: the only tiff in its folder for the raw movie, and the registration in the file for the
        shifts, the template and (replayed on the raw movie) the registered movie. Given ones must line up with
        ``results``; found ones that do not are skipped.

        Parameters
        ----------
        results : DemixingResults | CompressionArray | BaseRegistrationArray | BaseResults
            The results.
        path : str | Path | None
            The file ``results`` were read from; nothing is looked for without it.
        raw : ArrayLike | np.ndarray | str | Path | None
            The raw movie, or a path :func:`load_movie` reads.
        registered : ArrayLike | None
            The registered movie, for compression or demixing results.
        shifts : np.ndarray | torch.Tensor | str | Path | None
            Registration shifts, or a results file holding a registration.
        device : str | None
            Device for replaying the registration in ``path``; its stored device when None.

        Returns
        -------
        OpenedResults

        Raises
        ------
        ValueError
            If a given movie or shifts do not match the results' frames or shape, ``registered`` is given with a
            registration array, or a ``shifts`` file holds no registration.
        """
        path = Path(path) if path is not None else None
        shape = tuple(int(n) for n in results.shape)
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
        if raw is None and path is not None and (beside := movie_beside(path)) is not None:
            movie = TiffArray(str(beside))
            if tuple(movie.shape) == shape:
                raw, raw_source = movie, beside
            else:
                skipped.append(f"skipping {beside}: shape {tuple(movie.shape)} does not match the results")

        stored = None
        if path is not None and not is_registration:
            with h5py.File(path, "r") as f:
                stored = next((name for name in group_names_registration() if name in f), None)
                frames = f[stored]["shifts"].shape[0] if stored is not None else None
        if registered is None and isinstance(results, BaseResults):
            registered = results.registered_array
        if registered is None and raw is not None and stored is not None:
            if frames == shape[0]:
                registered = REGISTRATION_ARRAYS[stored].from_hdf5(path, input_movie=raw, device=device)
            else:
                skipped.append(f"the {stored} in {path} has {frames} frames, the movie {shape[0]}; shifts not applied")

        shifts_source = Path(shifts) if isinstance(shifts, (str, Path)) else None
        found_shifts = False
        if shifts is None and (is_registration or isinstance(results, BaseResults)):
            shifts = results.shifts
        if shifts is None and stored is not None:
            shifts_source, found_shifts = path, True
        if is_registration:
            template = results.strategy.template
        elif isinstance(registered, BaseRegistrationArray):
            template = registered.strategy.template
        else:
            template = None
        if shifts_source is not None:
            with h5py.File(shifts_source, "r") as f:
                name = next((name for name in group_names_registration() if name in f), None)
                if name is None:
                    raise ValueError(f"{shifts_source} holds no registration array")
                shifts = f[name]["shifts"][()]
                strategy = f.get(REGISTRATION_ARRAYS[name]._strategy_cls.__name__)
                if template is None and strategy is not None and "template" in strategy:
                    template = strategy["template"][()]
        if shifts is not None:
            shifts = np.asarray(shifts.cpu() if isinstance(shifts, torch.Tensor) else shifts, np.float32)
            if shifts.shape[0] != shape[0]:
                if not found_shifts:
                    raise ValueError(f"{shifts.shape[0]} shifts for {shape[0]} frames")
                skipped.append(f"skipping {shifts_source}: {shifts.shape[0]} shifts for {shape[0]} frames")
                shifts = shifts_source = template = None
        if template is not None:
            template = np.asarray(template.cpu() if isinstance(template, torch.Tensor) else template, np.float32)
        return cls(results=results, path=path, raw=raw, registered=registered, shifts=shifts, template=template,
                   raw_source=raw_source, shifts_source=shifts_source, skipped=skipped)


def has_group(path: str | Path, group: str) -> bool:
    """Whether ``path`` is an hdf5 file holding ``group``."""
    if not Path(path).is_file():
        return False
    with h5py.File(path, "r") as f:
        return group in f


def drop_group(path: str | Path, group: str) -> None:
    """
    Remove a group from an hdf5 file.

    The file is rewritten without the group, so its space is freed; on a large file this takes as long as copying
    it. Does nothing when the group is absent.

    Parameters
    ----------
    path : str | Path
        The hdf5 file.
    group : str
        The top-level group to remove.
    """
    if not has_group(path, group):
        return
    path = Path(path)
    packed = path.with_name(f"{path.name}.repack")
    with h5py.File(path, "r") as src, h5py.File(packed, "w") as dst:
        for name in src:
            if name != group:
                src.copy(name, dst)
        dst.attrs.update(src.attrs)
    packed.replace(path)


def create_run_folder(base: str | Path, name: str) -> Path:
    """
    Make a new run folder, ``<base>/<yyyymmddTHHMMSS>_<name>``.

    A run started in the same second as another gets ``_1``, ``_2``, ... appended.

    Parameters
    ----------
    base : str | Path
        The folder to make it in; made too when missing.
    name : str
        What follows the timestamp, e.g. the pipeline.

    Returns
    -------
    Path
    """
    stem = f"{get_timestamp()}_{name}"
    for suffix in itertools.count():
        folder = Path(base) / (stem if suffix == 0 else f"{stem}_{suffix}")
        try:
            folder.mkdir(parents=True)
            return folder
        except FileExistsError:
            continue


def log_to(folder: str | Path) -> logging.FileHandler:
    """
    Send the masknmf log to ``<folder>/<folder name>.log``.

    Appends when the file exists. The log file written until now, if any, is closed.

    Parameters
    ----------
    folder : str | Path
        The run folder.

    Returns
    -------
    logging.FileHandler
        The handler added to the "masknmf" logger.
    """
    logger = logging.getLogger("masknmf")
    for handler in [h for h in logger.handlers if isinstance(h, logging.FileHandler)]:
        logger.removeHandler(handler)
        handler.close()
    folder = Path(folder)
    handler = logging.FileHandler(folder / f"{folder.name}.log", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s", datefmt=TIMESTAMP_FORMAT))
    logger.addHandler(handler)
    return handler


def write_run_config(folder: str | Path,
                     pipeline: str,
                     configs: dict,
                     run: dict | None = None,
                     inputs: dict | None = None,
                     timings: dict | None = None,
                     default=str) -> Path:
    """
    Write a run's ``config.json``.

    Parameters
    ----------
    folder : str | Path
        The run folder.
    pipeline : str
        What made the run: a pipeline class name, or any name for a run put together by hand. ``masknmf run
        --config`` and the launcher only reload runs named after a pipeline.
    configs : dict
        The arguments the run used, by name.
    run : dict | None
        The run record: command, device, start, end, status.
    inputs : dict | None
        The files the run read, by argument name, each ``{"path": ..., "name": ...}``.
    timings : dict | None
        Each step's record (seconds, start, status), by step name.
    default : callable
        Turns what json cannot write into something it can; ``masknmf.pipelines.scraper.config_json_value`` keeps
        config dataclasses readable by ``--config``.

    Returns
    -------
    Path
        The file written.
    """
    path = Path(folder) / "config.json"
    record = {"masknmf_version": __version__, "pipeline": pipeline, "run": run or {}, "inputs": inputs or {},
              "configs": configs, "timings": timings or {}}
    path.write_text(json.dumps(record, indent=2, default=default))
    return path
