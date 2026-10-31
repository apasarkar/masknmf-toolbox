"""
GIAnT results as a masknmf results object, read from the ``experiment.zarr`` that subcell, GIAnT's Python port,
writes. GIAnT (Xie et al. 2026, "GIAnT: a Glutamate Imaging Analysis Toolbox") corrects motion with StripRegistration,
then finds and extracts sources with SILo (Source Identification by Activity Localization).

SILo fits each subproblem (a connected group of sources and their pixels) as
    registered / lambda = footprints @ denoised + B
with footprints summing to one per source. B, the per-pixel baseline, and lambda, the subproblem's noise scale, are
not saved, so there is no background or residual here and the signals are in units of lambda.

The store keeps one canvas for the summary and one registered movie per trial, (height, width, frames * channels)
with the channels interleaved in time; footprints are (selected pixels, sources), row i being the i-th True pixel of
``summary/sources/sel_pix`` in C order.
"""
import os
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


def giant_store_and_trial(trial_dir: str | os.PathLike) -> tuple[Path, str]:
    """Splits ``<store>/trials/<trial>`` into the store and the trial's name."""
    trial_dir = Path(trial_dir)
    return trial_dir.parents[1], trial_dir.name


class GiantRegisteredArray(LazyFrameLoader):
    """One channel of a trial's registered movie, (frames, height, width); NaN where no recorded pixel landed."""

    def __init__(self, registered_raw, num_channels: int, channel: int):
        """
        Args:
            registered_raw (zarr.Array): the trial's ``registered_raw``, (height, width, frames * num_channels)
            num_channels (int): channels interleaved along the stored time axis
            channel (int): the channel to read, 0-based
        """
        if registered_raw.shape[2] % num_channels != 0:
            raise ValueError(f"registered_raw has {registered_raw.shape[2]} planes, not a multiple of {num_channels} channels")
        self._registered_raw = registered_raw
        self._num_channels = num_channels
        self._channel = channel

    @property
    def dtype(self):
        return np.float32

    @property
    def shape(self) -> tuple[int, int, int]:
        height, width, planes = self._registered_raw.shape
        return planes // self._num_channels, height, width

    @property
    def ndim(self) -> int:
        return 3

    def _compute_at_indices(self, indices) -> np.ndarray:
        frame_ids = np.atleast_1d(np.arange(self.shape[0])[indices])
        planes = self._registered_raw.oindex[:, :, frame_ids * self._num_channels + self._channel]
        return np.ascontiguousarray(np.moveaxis(planes, -1, 0), dtype=np.float32)


class GiantResults(BaseResults):

    pipeline = "giant"
    source = "a GIAnT run folder from subcell, its experiment.zarr, or one of its trials"

    @classmethod
    def expand(cls, path: Path) -> list[Path]:
        """The trials with extraction results in a run folder or its experiment.zarr, or the trial folder given."""
        path = Path(path).expanduser()
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
        import zarr

        store, trial = giant_store_and_trial(cls.expand(path)[0])
        num_sources, num_frames = zarr.open_group(str(store), mode="r")[f"trials/{trial}/extraction/denoised"].shape
        return f"GIAnT: {num_sources} sources, {num_frames} frames, {trial}"

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

    def __init__(self, trial_dir: str | os.PathLike, device: str | torch.device = "cpu"):
        """
        Args:
            trial_dir (str | os.PathLike): ``<run>/experiment.zarr/trials/trial_NNN``
            device (str | torch.device): where the footprints and traces live
        """
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
        self._registered_array = GiantRegisteredArray(registered_raw, num_channels, activity_channel - 1)
        self._frame_time = float(alignment.attrs["frame_time"])

        footprints = np.nan_to_num(np.asarray(extraction["footprints"][:], np.float32))  # NaN outside a source's subproblem
        discard_frames = np.asarray(extraction["discard_frames"][:], bool)
        # censored frames are NaN in the store; 0 keeps every trace's peak (torch max propagates NaN) and the signals movie finite
        denoised = np.nan_to_num(np.asarray(extraction["denoised"][:], np.float32))
        ls = np.nan_to_num(np.asarray(extraction["ls"][:], np.float32))
        num_frames = self._registered_array.shape[0]
        if footprints.shape[0] != sel_pix.sum() or denoised.shape != (footprints.shape[1], num_frames):
            raise ValueError(f"footprints {footprints.shape}, denoised {denoised.shape} do not fit {int(sel_pix.sum())} "
                             f"selected pixels and {num_frames} frames")

        height, width = sel_pix.shape
        rows, cols = np.nonzero(sel_pix)
        pixel_ids, source_ids = np.nonzero(footprints)
        spatial = sp.coo_matrix((footprints[pixel_ids, source_ids], (rows[pixel_ids] * width + cols[pixel_ids], source_ids)),
                                shape=(height * width, footprints.shape[1]))
        self._fov_shape = (height, width)
        self._spatial_demixed = scipy_sparse_to_torch(spatial).coalesce()
        self._temporal_demixed = torch.from_numpy(np.ascontiguousarray(denoised.T))
        self._temporal_demixed_raw = torch.from_numpy(np.ascontiguousarray(ls.T))
        self._signals_array = SignalsArray.from_tensors(self._fov_shape, self._spatial_demixed, self._temporal_demixed)
        motion = np.stack([alignment["motion_r"][:], alignment["motion_c"][:]], axis=-1).astype(np.float32)
        # StripRegistration samples each frame at canvas + motion, so the content moves by -motion
        self._shifts = -motion
        self._discard_frames = discard_frames

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
        """Footprints, (pixels, sources), pixel row * width + col; each sums to one."""
        return self._spatial_demixed

    @property
    def temporal_demixed(self) -> torch.Tensor:
        """``denoised``, (frames, sources): the events convolved with the decay kernel, in units of lambda."""
        return self._temporal_demixed

    @property
    def temporal_demixed_raw(self) -> torch.Tensor:
        """``ls``, (frames, sources): the least-squares fit of the footprints to the registered movie minus B."""
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
        """None: the store holds no per-pixel baseline B."""
        return None

    @property
    def residual_array(self) -> None:
        """None: the residual needs B and lambda, which the store does not hold."""
        return None

    @property
    def fluctuating_background_roi_averages(self) -> None:
        return None

    @property
    def residual_roi_averages(self) -> None:
        return None
