"""Opening masknmf results files: the stages a file holds, the raw movie, and what the viewers show of it."""

import os
from pathlib import Path
from typing import Optional

import h5py

from masknmf.arrays import Hdf5Array, TiffArray, TiffSeriesLoader
from masknmf.compression import CompressionArray
from masknmf.demixing import DemixingResults
from masknmf.motion_correction import GradientRegistrationArray, PiecewiseRigidRegistrationArray, RigidRegistrationArray
from masknmf.utils import display

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
    "open_results",
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


def load_movie(path: str | os.PathLike, dataset: Optional[str] = None):
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


def movie_beside(path: str | os.PathLike) -> Optional[Path]:
    """The movie beside a results file: the path of the one .tif in its folder, None when there is none or several."""
    folder = Path(path).parent
    tiffs = sorted(p for ext in ("*.tif", "*.tiff") for p in folder.glob(ext))
    return tiffs[0] if len(tiffs) == 1 else None


def stage_groups(path: str | os.PathLike) -> list[str]:
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


def open_results(path: str | os.PathLike, prefix: str = "", device: str = "cpu", raw=None):
    """
    What a results file has to show, as (results, raw, registered).

    ``results`` is the file's DemixingResults (under ``prefix`` when given), else its CompressionArray, else its
    registration replayed on the raw movie. The raw movie is ``raw`` (an array or a path) when it lines up with
    the results, else the one .tif beside the file when that does; a movie that lines up with neither is reported
    and left out. ``registered`` is the file's registration replayed on the raw movie when the file also holds a
    later stage. A file holding only a registration needs the movie it registered.
    """
    path = str(path)
    groups = stage_groups(path)
    if len(groups) == 0:
        raise ValueError(f"{path} holds no masknmf results")
    name_demixing = f"{prefix}/{group_name_demixing()}" if prefix else group_name_demixing()
    if prefix and name_demixing not in groups:
        raise ValueError(f"{path} holds no {name_demixing}")
    name_registration = next((n for n in group_names_registration() if n in groups), None)
    frames_registered = None
    if name_registration is not None:
        with h5py.File(path, "r") as f:
            frames_registered = f[name_registration]["shifts"].shape[0]
    if name_demixing in groups:
        results = DemixingResults.from_hdf5(path, prefix=prefix, device=device)
    elif group_name_compression() in groups:
        results = CompressionArray.from_hdf5(path)
    else:
        results = None
    shape = None if results is None else tuple(int(v) for v in results.shape)
    if isinstance(raw, (str, os.PathLike)):
        raw = load_movie(raw)
    beside = movie_beside(path)
    movie = None
    for candidate in (raw, None if beside is None else TiffArray(str(beside))):
        if candidate is None:
            continue
        if shape is not None and tuple(candidate.shape) != shape:
            # a raw movie the pipeline trimmed (the glutamate pipeline drops its first frames) no longer lines up
            display(f"raw movie is {tuple(candidate.shape)}, the results {shape}; not shown")
            continue
        if shape is None and candidate.shape[0] != frames_registered:
            display(f"raw movie has {candidate.shape[0]} frames, the {name_registration} {frames_registered}; not shown")
            continue
        movie = candidate
        break
    registered = None
    if movie is not None and name_registration is not None:
        if movie.shape[0] == frames_registered:
            registered = REGISTRATION_ARRAYS[name_registration].from_hdf5(path, input_movie=movie)
        else:
            display(f"the {name_registration} in {path} has {frames_registered} frames, the movie {movie.shape[0]}; shifts not applied")
    if results is None:
        if registered is None:
            raise ValueError(f"{path} holds only {name_registration}; showing it needs the raw movie the run registered")
        return registered, movie, None
    return results, movie, registered
