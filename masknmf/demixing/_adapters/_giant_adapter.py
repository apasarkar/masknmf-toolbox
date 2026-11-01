"""
GIAnT results as a masknmf results object. GIAnT (Xie et al. 2026, "GIAnT: a Glutamate Imaging Analysis Toolbox")
corrects motion with StripRegistration, then finds and extracts sources with SILo (Source Identification by Activity
Localization). Two writers are read, one trial each.

GIAnT-MATLAB writes a run folder: ``motion_correction/<stem>_REGISTERED_RAW.tif``, one page per frame and channel with
the channels interleaved, ``motion_correction/<stem>_ALIGNMENTDATA.h5`` and ``source_extraction/experiment_summary.h5``,
whose ``profiles`` are dense images on the registered canvas. MATLAB writes HDF5 column-major (``row_major`` 0), so
h5py reads every dataset with the axes of GIAnT-MATLAB's README reversed.

subcell, GIAnT's Python port, writes ``experiment.zarr``: one canvas for the summary and one registered movie per
trial, (height, width, frames * channels) with the channels interleaved in time; footprints are (selected pixels,
sources), row i being the i-th True pixel of ``summary/sources/sel_pix`` in C order.

SILo fits each subproblem (a connected group of sources and their pixels) as
    registered / lambda = footprints @ denoised + B
with footprints summing to one per source. Neither writer saves B, the per-pixel baseline, so there is no background
or residual here. GIAnT-MATLAB multiplies its traces back by lambda (its ``photonScale``), so they are in the units of
the registered movie; subcell's are in units of lambda, which it does not save.
"""
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch

from masknmf.arrays.array_interfaces import ArrayLike, LazyFrameLoader
from masknmf.demixing._base_results import BaseResults
from masknmf.demixing.demixing_arrays import SignalsArray
from masknmf.demixing.demixing_utils import scipy_sparse_to_torch
from masknmf.utils import SparseCOOTensor

STORE_NAME = "experiment.zarr"
MATLAB_SUMMARY = Path("source_extraction") / "experiment_summary.h5"


def giant_store_and_trial(trial_dir: str | os.PathLike) -> tuple[Path, str]:
    """Splits ``<store>/trials/<trial>`` into the store and the trial's name."""
    trial_dir = Path(trial_dir)
    return trial_dir.parents[1], trial_dir.name


class GiantRegisteredArray(LazyFrameLoader):
    """One channel of a trial's registered movie, (frames, height, width); NaN where no recorded pixel landed."""

    def __init__(self, registered_raw, num_channels: int, channel: int, planes_last: bool):
        """
        Args:
            registered_raw (zarr.Array | np.memmap): the registered movie, frames * num_channels planes
            num_channels (int): channels interleaved along the planes
            channel (int): the channel to read, 0-based
            planes_last (bool): True for (height, width, planes), subcell's layout; False for (planes, height, width)
        """
        num_planes = registered_raw.shape[2 if planes_last else 0]
        if num_planes % num_channels != 0:
            raise ValueError(f"the registered movie has {num_planes} planes, not a multiple of {num_channels} channels")
        self._registered_raw = registered_raw
        self._num_channels = num_channels
        self._channel = channel
        self._planes_last = planes_last

    @property
    def dtype(self):
        return np.float32

    @property
    def shape(self) -> tuple[int, int, int]:
        if self._planes_last:
            height, width, planes = self._registered_raw.shape
        else:
            planes, height, width = self._registered_raw.shape
        return planes // self._num_channels, height, width

    @property
    def ndim(self) -> int:
        return 3

    def _compute_at_indices(self, indices) -> np.ndarray:
        frame_ids = np.atleast_1d(np.arange(self.shape[0])[indices])
        plane_ids = frame_ids * self._num_channels + self._channel
        if self._planes_last:
            planes = np.moveaxis(self._registered_raw.oindex[:, :, plane_ids], -1, 0)
        else:
            planes = self._registered_raw[plane_ids]
        return np.ascontiguousarray(planes, dtype=np.float32)


@dataclass
class GiantFields:
    """One trial of GIAnT results as either writer saves it; censored frames are 0 in both traces."""

    registered: GiantRegisteredArray
    frame_time: float
    footprints: sp.coo_matrix  ## (height * width, sources), pixel row * width + col
    fov_shape: tuple[int, int]
    denoised: np.ndarray  ## (frames, sources)
    least_squares: np.ndarray  ## (frames, sources)
    shifts: np.ndarray  ## (frames, 2), (row, col)
    discard_frames: np.ndarray  ## (frames,)


def read_subcell_trial(trial_dir: str | os.PathLike) -> GiantFields:
    """Reads one trial of subcell's ``experiment.zarr``, ``<run>/experiment.zarr/trials/trial_NNN``."""
    import zarr

    store_dir, trial_name = giant_store_and_trial(trial_dir)
    store = zarr.open_group(str(store_dir), mode="r")
    trial = store[f"trials/{trial_name}"]
    extraction = trial["extraction"]
    alignment = trial["alignment"]

    sel_pix = np.asarray(store["summary/sources/sel_pix"][:], bool)
    registered_raw = trial["registered_raw"]
    if tuple(registered_raw.shape[:2]) != sel_pix.shape:
        raise ValueError(f"{trial_name}'s canvas is {registered_raw.shape[:2]}, the sources' is {sel_pix.shape}; "
                         "trials registered to different canvases are not supported")
    num_channels = int(alignment.attrs["num_channels"])
    activity_channel = int(store["summary"].attrs["params"]["activity_channel"])  # 1-based
    registered = GiantRegisteredArray(registered_raw, num_channels, activity_channel - 1, planes_last=True)

    footprints = np.nan_to_num(np.asarray(extraction["footprints"][:], np.float32))  # NaN outside a source's subproblem
    # censored frames are NaN in the store; 0 keeps every trace's peak (torch max propagates NaN) and the signals movie finite
    denoised = np.nan_to_num(np.asarray(extraction["denoised"][:], np.float32))
    ls = np.nan_to_num(np.asarray(extraction["ls"][:], np.float32))
    num_frames = registered.shape[0]
    if footprints.shape[0] != sel_pix.sum() or denoised.shape != (footprints.shape[1], num_frames):
        raise ValueError(f"footprints {footprints.shape}, denoised {denoised.shape} do not fit {int(sel_pix.sum())} "
                         f"selected pixels and {num_frames} frames")

    height, width = sel_pix.shape
    rows, cols = np.nonzero(sel_pix)
    pixel_ids, source_ids = np.nonzero(footprints)
    spatial = sp.coo_matrix((footprints[pixel_ids, source_ids], (rows[pixel_ids] * width + cols[pixel_ids], source_ids)),
                            shape=(height * width, footprints.shape[1]))
    motion = np.stack([alignment["motion_r"][:], alignment["motion_c"][:]], axis=-1).astype(np.float32)
    return GiantFields(
        registered=registered,
        frame_time=float(alignment.attrs["frame_time"]),
        footprints=spatial,
        fov_shape=(height, width),
        denoised=np.ascontiguousarray(denoised.T),
        least_squares=np.ascontiguousarray(ls.T),
        # StripRegistration samples each frame at canvas + motion, so the content moves by -motion
        shifts=-motion,
        discard_frames=np.asarray(extraction["discard_frames"][:], bool),
    )


def read_scalar(dataset) -> float:
    """The one value of a MATLAB scalar, saved as a (1, 1) dataset."""
    return np.asarray(dataset[()]).item()


def read_giant_matlab_run(run_dir: str | os.PathLike) -> GiantFields:
    """Reads a single-trial GIAnT-MATLAB run folder, the ``savedr`` of its ``trial_table.h5``."""
    import h5py
    import tifffile

    run_dir = Path(run_dir)
    filepaths_registered = sorted((run_dir / "motion_correction").glob("*_REGISTERED_RAW.tif"))
    if len(filepaths_registered) != 1:
        raise ValueError(f"{run_dir / 'motion_correction'} holds {len(filepaths_registered)} registered trials; "
                         "only single-trial GIAnT-MATLAB runs are supported")
    filepath_registered = filepaths_registered[0]
    filepath_alignment = filepath_registered.with_name(filepath_registered.name.replace("_REGISTERED_RAW.tif", "_ALIGNMENTDATA.h5"))
    with h5py.File(filepath_alignment, "r") as alignment:
        num_channels = int(read_scalar(alignment["numChannels"]))
        frame_time = float(read_scalar(alignment["frametime"]))

    with h5py.File(run_dir / MATLAB_SUMMARY, "r") as summary:
        if int(read_scalar(summary["row_major"])) != 0:
            raise ValueError(f"{run_dir / MATLAB_SUMMARY} is row-major; only MATLAB's column-major files are supported")
        activity_channel = int(read_scalar(summary["params/activityChannel"]))  # 1-based
        path = summary["Path1"]
        ## NOTE: SILo holds images transposed (its rows are the tif's columns), so the reversed axes come out as the
        ## tif's (height, width, fastz, sources); its coords' y and x are swapped the same way
        profiles = np.asarray(path["sources/spatial/profiles"][:], np.float32)
        # censored frames are NaN in the file; 0 keeps every trace's peak (torch max propagates NaN) and the signals movie finite
        denoised = np.nan_to_num(np.asarray(path["sources/temporal/dF_denoised"][:, activity_channel - 1, :], np.float32))
        ls = np.nan_to_num(np.asarray(path["sources/temporal/dF_ls"][:, activity_channel - 1, :], np.float32))
        motion = np.stack([path["frame_info/offlineYshifts"][:].ravel(), path["frame_info/offlineXshifts"][:].ravel()],
                          axis=-1).astype(np.float32)
        discard_frames = np.asarray(path["frame_info/discard_frames"][:]).ravel().astype(bool)

    registered = GiantRegisteredArray(tifffile.memmap(filepath_registered), num_channels, activity_channel - 1,
                                      planes_last=False)
    footprints = profiles[:, :, 0, :]  ## (height, width, sources)
    num_frames, height, width = registered.shape
    if footprints.shape[:2] != (height, width) or denoised.shape != (num_frames, footprints.shape[2]):
        raise ValueError(f"profiles {footprints.shape}, dF_denoised {denoised.shape} do not fit the registered movie "
                         f"{registered.shape}")
    return GiantFields(
        registered=registered,
        frame_time=frame_time,
        footprints=sp.coo_matrix(footprints.reshape(height * width, -1)),
        fov_shape=(height, width),
        denoised=denoised,
        least_squares=ls,
        # StripRegistration samples each frame at canvas + motion, so the content moves by -motion
        shifts=-motion,
        discard_frames=discard_frames,
    )


class GiantResults(BaseResults):

    pipeline = "giant"
    source = "a GIAnT-MATLAB run folder, or a subcell run folder, its experiment.zarr or one of its trials"

    @classmethod
    def expand(cls, path: Path) -> list[Path]:
        """A GIAnT-MATLAB run folder, or the subcell trials with extraction results under the path given."""
        path = Path(path).expanduser()
        if path.name == MATLAB_SUMMARY.name:
            path = path.parents[1]
        if (path / MATLAB_SUMMARY).is_file():
            return [path]
        if (path / STORE_NAME).is_dir():
            path = path / STORE_NAME
        if path.parent.name == "trials":
            store, trials = path.parents[1], [path]
        else:
            store, trials = path, sorted((path / "trials").glob("trial_*"))
        if not (store / "summary" / "sources" / "sel_pix").is_dir():
            return []
        return [trial for trial in trials if (trial / "extraction" / "footprints").is_dir()]

    @classmethod
    def matches(cls, path: Path) -> bool:
        return len(cls.expand(path)) > 0

    @classmethod
    def describe(cls, path: Path) -> str:
        expanded = cls.expand(path)[0]
        if (expanded / MATLAB_SUMMARY).is_file():
            import h5py

            with h5py.File(expanded / MATLAB_SUMMARY, "r") as summary:
                num_frames, _, num_sources = summary["Path1/sources/temporal/dF_ls"].shape
            return f"GIAnT-MATLAB: {num_sources} sources, {num_frames} frames"
        import zarr

        store, trial = giant_store_and_trial(expanded)
        num_sources, num_frames = zarr.open_group(str(store), mode="r")[f"trials/{trial}/extraction/denoised"].shape
        return f"GIAnT (subcell): {num_sources} sources, {num_frames} frames, {trial}"

    @classmethod
    def from_path(cls, path: Path, raw: ArrayLike | str | Path | None = None, device: str = "cpu") -> tuple["GiantResults", ArrayLike | None, list[str]]:
        """raw is shown only when it has the shape of the registered canvas; the scan itself is smaller."""
        from masknmf.io import load_movie

        results = cls(cls.expand(path)[0], device=device)
        if isinstance(raw, (str, Path)):
            raw = load_movie(raw)
        skipped = []
        if raw is not None and tuple(raw.shape) != tuple(results.shape):
            skipped.append(f"raw movie shape {tuple(raw.shape)} does not match the registered canvas {tuple(results.shape)}; not used")
            raw = None
        return results, raw, skipped

    def __init__(self, path: str | os.PathLike, device: str | torch.device = "cpu"):
        """
        Args:
            path (str | os.PathLike): a GIAnT-MATLAB run folder, or a subcell trial, ``<run>/experiment.zarr/trials/trial_NNN``
            device (str | torch.device): where the footprints and traces live
        """
        if (Path(path) / MATLAB_SUMMARY).is_file():
            fields = read_giant_matlab_run(path)
        else:
            fields = read_subcell_trial(path)
        self._registered_array = fields.registered
        self._frame_time = fields.frame_time
        self._fov_shape = fields.fov_shape
        self._spatial_demixed = scipy_sparse_to_torch(fields.footprints).coalesce()
        self._temporal_demixed = torch.from_numpy(fields.denoised)
        self._temporal_demixed_raw = torch.from_numpy(fields.least_squares)
        self._signals_array = SignalsArray.from_tensors(self._fov_shape, self._spatial_demixed, self._temporal_demixed)
        self._shifts = fields.shifts
        self._discard_frames = fields.discard_frames

        self._device = torch.device("cpu")
        self.to(device)

    def to(self, new_device: str | torch.device):
        updated_device = torch.device(new_device)
        self._spatial_demixed = self._spatial_demixed.to(updated_device)
        self._temporal_demixed = self._temporal_demixed.to(updated_device)
        self._temporal_demixed_raw = self._temporal_demixed_raw.to(updated_device)
        self._signals_array.to(updated_device)
        self._device = updated_device

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def shape(self) -> tuple[int, int, int]:
        return self._registered_array.shape

    @property
    def frame_rate(self) -> float:
        return 1.0 / self._frame_time

    @property
    def discard_frames(self) -> np.ndarray:
        """(frames,), True where GIAnT censored the frame for motion; the traces are 0 there."""
        return self._discard_frames

    @property
    def spatial_demixed(self) -> SparseCOOTensor:
        """Footprints, (pixels, sources), pixel row * width + col; each sums to one, or to zero when NMF emptied it."""
        return self._spatial_demixed

    @property
    def temporal_demixed(self) -> torch.Tensor:
        """``denoised`` (subcell) or ``dF_denoised`` (GIAnT-MATLAB), (frames, sources): the events convolved with the decay kernel."""
        return self._temporal_demixed

    @property
    def temporal_demixed_raw(self) -> torch.Tensor:
        """``ls`` (subcell) or ``dF_ls`` (GIAnT-MATLAB), (frames, sources): the least-squares fit of the footprints to the registered movie minus B."""
        return self._temporal_demixed_raw

    @property
    def signals_array(self) -> SignalsArray:
        return self._signals_array

    @property
    def registered_array(self) -> GiantRegisteredArray:
        return self._registered_array

    @property
    def shifts(self) -> np.ndarray:
        """(frames, 2), the (row, col) shift applied to each frame."""
        return self._shifts

    @property
    def fluctuating_background_array(self) -> None:
        """None: neither writer saves the per-pixel baseline B."""
        return None

    @property
    def residual_array(self) -> None:
        """None: the residual needs B, which neither writer saves."""
        return None

    @property
    def fluctuating_background_roi_averages(self) -> None:
        return None

    @property
    def residual_roi_averages(self) -> None:
        return None
