from __future__ import annotations

from pathlib import Path
from typing import *

import h5py

from masknmf.arrays.array_interfaces import LazyFrameLoader
from masknmf.arrays.data_loaders import Hdf5Array, TiffArray, TiffSeriesLoader

TIFF_SUFFIXES = (".tif", ".tiff")
HDF5_SUFFIXES = (".h5", ".hdf5")

SUPPORTED = "a .tif/.tiff file, a directory of them, or a .h5/.hdf5 file"


def find_dataset(path: str | Path) -> str:
    """
    Name of the sole 3D dataset in an hdf5 file.

    Args:
        path (str | Path): Path to the hdf5 file.

    Returns:
        str: The dataset's name, for use as ``Hdf5Array``'s ``field``.

    Raises:
        ValueError: If the file holds no 3D dataset, or more than one.
    """
    found: list[str] = []

    def visit(name, obj):
        if isinstance(obj, h5py.Dataset) and obj.ndim == 3:
            found.append(name)

    with h5py.File(path, "r") as f:
        f.visititems(visit)

    if not found:
        raise ValueError(f"{path} holds no 3D dataset to read as a movie")
    if len(found) > 1:
        listed = ", ".join(sorted(found))
        raise ValueError(
            f"{path} holds more than one 3D dataset ({listed}); name one explicitly"
        )
    return found[0]


def imread(
    path: str | Path,
    dataset: str | None = None,
    memmap: bool = False,
) -> LazyFrameLoader:
    """
    Open an imaging movie as a lazy array, picking the loader from the path.

    ================================  ==========================
    Path                              Loader
    ================================  ==========================
    ``.tif`` / ``.tiff`` file         :class:`TiffArray`
    directory holding tiff files      :class:`TiffSeriesLoader`
    ``.h5`` / ``.hdf5`` file          :class:`Hdf5Array`
    ================================  ==========================

    Args:
        path (str | Path): File or directory to open.
        dataset (str | None): For hdf5 input, the dataset holding the movie. When omitted,
            the file's sole 3D dataset is used; a file with several is an error.
        memmap (bool): Memory-map tiff input rather than reading pages on demand.

    Returns:
        LazyFrameLoader: A (frames, height, width) lazy array.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        ValueError: If the path is not one masknmf can read, if a tiff directory is empty,
            or if an hdf5 dataset cannot be resolved.
    """
    path = Path(path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"no such file or directory: {path}")

    if path.is_dir():
        files = sorted(
            p for p in path.iterdir() if p.suffix.lower() in TIFF_SUFFIXES
        )
        if not files:
            raise ValueError(f"no tiff files in {path}")
        if len(files) == 1:
            return TiffArray(str(files[0]), memmap=memmap)
        return TiffSeriesLoader([str(p) for p in files], memmap=memmap)

    suffix = path.suffix.lower()
    if suffix in TIFF_SUFFIXES:
        return TiffArray(str(path), memmap=memmap)
    if suffix in HDF5_SUFFIXES:
        field = dataset if dataset is not None else find_dataset(path)
        return Hdf5Array(str(path), field)

    raise ValueError(f"masknmf cannot read {path.name}; expected {SUPPORTED}")
