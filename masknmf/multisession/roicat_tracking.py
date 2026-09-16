from masknmf.demixing.demixing_results import DemixingResults
import torch
import os
import sys
from typing import *
from pathlib import Path
import scipy
import scipy.sparse
from roicat.data_importing import Data_roicat
from roicat.pipelines import pipeline_tracking
from roicat.util import get_default_parameters
from roicat import helpers
from roicat.util import RichFile_ROICaT
import json
import datetime

import warnings
from typing import Callable, Optional, Sequence

import numpy as np
import scipy.sparse


class RoicatDataAdapter(Data_roicat):

    def __init__(self,
                 mean_img_list: List[np.ndarray],
                 spatial_fp_list: List[scipy.sparse.coo_matrix],
                 session_files: tuple[str | tuple[str, ...]],
                 um_per_pixel: float = 1.2,
                 roi_image_dims: tuple[int, int] = (36, 36),
                 highpass_sigma: Optional[int] = 3,
                 ):
        """
        Notes: um_per_pixel is by default set to 1.2 since this is what is used for IBL 2p mesoscope recordings
        Generic interface for doing multi-session tracking with any analysis pipeline
        Args:
            mean_img_list (List[np.ndarray]): List of mean images from each imaging session. Each image should have same dimensions.
            spatial_fp_list (List[np.ndarray]): List of spatial footprint arrays, one for each session. Each individual array has shape (num_rois, num_pixels).
                Each spatial footprint is flattened into a row of this array in "C" order.
            session_files (tuple[str] | tuple[tuple[str, ...]]): A tuple whose length is equal to the number of sessions. Each element contains filepath data for one session.
            um_per_pixel (float): Describes the resolution of the imaging
            roi_image_dims (tuple[int, int]): Each ROI is spatially cropped for purposes of feature extraction in the ROICat pipeline. This specifies the crop dimensions.
            highpass_sigma (int): We highpass filter the mean image to define an "enhanced" mean image (this is what s2p does) for use in the tracking pipeline.
        """

        super().__init__()
        self.um_per_pixel = um_per_pixel
        self._highpass_sigma = highpass_sigma
        self._mean_img_list = mean_img_list
        self.set_FOVHeightWidth(int(mean_img_list[0].shape[0]), int(mean_img_list[0].shape[1]))
        self.set_fov_imgs_from_mean_imgs()
        self.set_spatialFootprints(spatial_fp_list, self.um_per_pixel)
        self.transform_spatialFootprints_to_ROIImages(out_height_width=roi_image_dims)
        if session_files is not None:
            if len(self._mean_img_list) != len(session_files):
                raise ValueError(f"You provided {len(session_files)} file paths, but there seem to be {len(self._mean_img_list)} sessions. Need to provide one file path "
                                 "(or tuple of file paths) per session. ")

        self._session_files = session_files

    @property
    def session_files(self) -> tuple[str | tuple[str, ...]]:
        """
        Serves as useful metadata to assess where the demixing results files come from. Allows users to match session id to the actual file from which it came
        This does not get updated if the file gets moved to another location after object instantiation.
        """
        return self._session_files

    def set_fov_imgs_from_mean_imgs(self):
        fov_list = self._filter_and_normalize_mean_img()
        return self.set_FOV_images(fov_list)

    def _filter_and_normalize_mean_img(self):
        """
        This pipeline convolves each image with a
        """
        if self._highpass_sigma is None:
            return self._mean_img_list
        else:
            """
            Spatially high-pass filter each image and normalize the data between 0 and 1
            """
            radius = int(torch.ceil(torch.tensor(2 * self._highpass_sigma)).item())
            size = 2 * radius + 1
            coords = torch.arange(-radius, radius + 1, dtype=torch.float32)
            yy, xx = torch.meshgrid(coords, coords, indexing='ij')

            # 2D Gaussian
            kernel = torch.exp(-(xx ** 2 + yy ** 2) / (2 * self._highpass_sigma ** 2))

            # Normalize so sum = 1
            kernel /= kernel.sum()
            kernel *= -1

            kernel[radius, radius] += 1

            # Reshape for conv2d: (out_ch, in_ch, H, W)
            kernel = kernel.unsqueeze(0).unsqueeze(0)

            new_list = []
            for k in range(len(self._mean_img_list)):
                curr_mean_img = torch.from_numpy(self._mean_img_list[k]).float()

                image = curr_mean_img.unsqueeze(0).unsqueeze(0)
                image = torch.nn.functional.pad(image,
                                                pad=(radius, radius, radius, radius), mode="reflect")

                # Convolve
                output = torch.nn.functional.conv2d(image, kernel, padding=0).squeeze(0).squeeze(0).cpu()

                # Normalize + clip
                p1 = torch.quantile(output, 0.01)
                p99 = torch.quantile(output, 0.99)
                x_clipped = torch.clamp(output, min=p1, max=p99)
                x_norm = (x_clipped - p1) / (p99 - p1)

                x_norm = x_norm.numpy()
                new_list.append(x_norm)
            return new_list

    @classmethod
    def from_masknmf(cls,
                     demixing_result_files: list[str | Path],
                     **kwargs):
        """
        Constructs the ROICaT data adapter using a list of masknmf demixing result hdf5 files
        Args:
            demixing_result_files (list[str | Path]): A list of file paths, one per session, point to masknmf demixing result .hdf5 files.
        """
        spatial_footprint_list = []
        mean_img_list = []
        files_list = []
        for fname in demixing_result_files:
            files_list.append(os.path.abspath(fname))
            dmr = DemixingResults.from_hdf5(fname)
            footprint = extract_masknmf_spatial_footprints(dmr)
            mean_img = extract_masknmf_mean_img(dmr)
            spatial_footprint_list.append(footprint)
            mean_img_list.append(mean_img)

        return cls(mean_img_list,
                   spatial_footprint_list,
                   tuple(files_list),
                   **kwargs)

    @classmethod
    def _from_suite2p(cls,
                      ops_list: list[str | Path],
                      stat_list: list[str | Path],
                      **kwargs):
        spatial_footprint_list = []
        mean_img_list = []
        files_list = []
        for ops_file, stat_file in zip(ops_list, stat_list):
            ops_abspath = os.path.abspath(ops_file)
            stat_abspath = os.path.abspath(stat_file)
            files_list.append((ops_abspath, stat_abspath))
            ops = np.load(ops_abspath, allow_pickle=True).item()
            stat = np.load(stat_abspath, allow_pickle=True)
            footprint = extract_suite2p_spatial_footprints(ops, stat)
            mean_img = extract_suite2p_mean_img(ops)
            spatial_footprint_list.append(footprint)
            mean_img_list.append(mean_img)

        return cls(mean_img_list,
                   spatial_footprint_list,
                   tuple(files_list),
                   **kwargs)


def extract_masknmf_spatial_footprints(dr: DemixingResults):
    """
    Given a masknmf demixingresults object, extracts the spatial footprints in a format needed for ROICaT cross-session matching
    """
    a = dr.ac_array.a.cpu().t().coalesce()  # Shape (num_neurons, num_pixels)
    row, col = a.indices()
    vals = a.values().clone()

    row_sum = torch.zeros(a.shape[0], device=a.device)
    row_sum.scatter_reduce_(0, row, vals, reduce="sum")
    per_value_divisors = row_sum[row]
    vals /= per_value_divisors
    vals = torch.nan_to_num(vals, nan=0.0, posinf=0.0, neginf=0.0)

    row = row.cpu().numpy()
    col = col.cpu().numpy()
    vals = vals.cpu().numpy()

    shape = a.shape
    curr_csr_scipy = scipy.sparse.coo_matrix((vals, (row, col)), shape=shape).tocsr()
    return curr_csr_scipy


def extract_masknmf_mean_img(dr: DemixingResults):
    return dr.pmd_array.mean_img.cpu().numpy()


def extract_suite2p_spatial_footprints(
        ops: np.ndarray,
        stat: np.ndarray,
) -> scipy.sparse.csr_matrix:
    """
    From the suite2p/ROICaT repos
    Returns:
        (scipy.sparse.csr_matrix):
            spatialFootprints (scipy.sparse.csr_matrix):
                Sparse array of shape *(n_roi, frame_height * frame_width)*
                containing the spatial footprints of the ROIs.
    """
    height, width = ops['Ly'], ops['Lx']
    ## Add some code here to infer the height/width of the data from the ops file
    dtype = None
    isInt = np.issubdtype(dtype, np.integer)

    rois_to_stack = []

    for jj, roi in enumerate(stat):
        lam = np.array(roi['lam'], ndmin=1)
        dtype = lam.dtype
        if isInt:
            lam = dtype(lam / lam.sum() * np.iinfo(dtype).max)
        else:
            lam = lam / lam.sum()
        ypix = np.array(roi['ypix'], dtype=np.uint64, ndmin=1)
        xpix = np.array(roi['xpix'], dtype=np.uint64, ndmin=1)

        tmp_roi = scipy.sparse.csr_matrix((lam, (ypix, xpix)), shape=(height, width), dtype=dtype)
        rois_to_stack.append(tmp_roi.reshape(1, -1))

    return scipy.sparse.vstack(rois_to_stack).tocsr()


def extract_suite2p_mean_img(ops) -> np.ndarray:
    mean_img = ops['meanImg']
    return mean_img



class RoicatTracker:

    def __init__(self, params: None | dict = None):
        self._params = params if params is not None else get_default_parameters(pipeline='tracking')

    @property
    def params(self) -> dict:
        return self._params

    def run_tracking(self,
                     multisession_data: RoicatDataAdapter | Data_roicat):
        if not multisession_data.check_completeness()['tracking']:
            raise ValueError("input data does not have all necessary properties to run the tracking code")
        tracked_outputs = pipeline_tracking(self.params, custom_data=multisession_data)

        return RoicatTrackingResults(results=tracked_outputs[0],
                                     run_data=tracked_outputs[1],
                                     session_files=multisession_data.session_files,
                                     params=tracked_outputs[2])



_STEM_RESULTS = '.tracking.results_all.'
_STEM_RUN_DATA = '.tracking.run_data.'
_EXT_WRITE = 'richfile.zip'
_SUFFIX_PARAMS = '.tracking.params.yaml'
_SUFFIX_SESSIONS = '.tracking.masknmf_sessions.json'

class RoicatTrackingResults:
    """
    Wrapper for the Roicat tracking results

    The raw ROICaT outputs are always made directly available (``results``, ``run_data``,
    ``params``). Everything is derived from this and is provided for downstream access.

    Conventions to note:
      - Spatial footprints are exposed as ``(num_pixels, num_rois)`` sparse
        arrays, one per session.
      - Cluster ids are the integers ``[0, ..., num_clusters - 1]``. ROICaT
        squeezes labels to a contiguous range, so no remapping is needed.
        ``-1`` denotes an ROI that was not assigned to any cluster.
      - Neuron indices are always *session-local*.
    """

    def __init__(self,
                 results: dict,
                 run_data: dict,
                 session_files: tuple[str],
                 params: dict | None = None,
                 ):
        if results is None:
            raise ValueError("results is required; RoicatTrackingResults cannot be constructed without it.")
        if run_data is None:
            raise ValueError("run_data is required; RoicatTrackingResults cannot be constructed without it.")
        if session_files is None:
            raise ValueError("session_files is required; RoicatTrackingResults cannot be constructed without it.")

        self._results = results
        self._run_data = run_data
        self._params = params

        self._labels_by_session = [
            np.asarray(l, dtype=np.int64)
            for l in self._results["clusters"]["labels_bySession"]
        ]

        self.session_files = session_files

        self._n_roi_per_session = np.array(
            [len(l) for l in self.labels_by_session], dtype=np.int64
        )

        if len(self._labels_by_session) == 0:
            raise ValueError("Tracking results contain no sessions.")

        all_labels = np.concatenate(self._labels_by_session)
        if all_labels.size > 0 and all_labels.max() >= 0:
            self._num_clusters = int(all_labels.max()) + 1
        else:
            self._num_clusters = 0

        # counts[c, s] = number of ROIs from session s assigned to cluster c
        self._counts = np.zeros(
            (self._num_clusters, len(self._labels_by_session)), dtype=np.int32
        )
        for s, lab in enumerate(self._labels_by_session):
            valid = lab[lab >= 0]
            np.add.at(self._counts, (valid, s), 1)

        self._validate()

        # Lazily built caches
        self._aligned_rois = None
        self._raw_rois = None

    def _validate(self) -> None:
        rois_raw = self._results["ROIs"]["ROIs_raw"]
        if len(rois_raw) != self.num_sessions:
            raise ValueError(
                f"Got {len(rois_raw)} sessions of footprints but "
                f"{self.num_sessions} sessions of labels."
            )
        for s, (fp, n) in enumerate(zip(rois_raw, self._n_roi_per_session)):
            if fp.shape[0] != n:
                raise ValueError(
                    f"Session {s}: {fp.shape[0]} footprints but {n} labels."
                )

        if self._num_clusters and self._counts.max() > 1:
            bad = np.argwhere(self._counts > 1)
            warnings.warn(
                f"{len(bad)} (cluster, session) pairs contain more than one ROI "
                f"from the same session; e.g. cluster {bad[0][0]} in session "
                f"{bad[0][1]}. This usually indicates a merge/split in the "
                "upstream segmentation.",
                stacklevel=2,
            )

    ### Raw outputs from ROICaT

    @property
    def results(self) -> dict:
        """The ROICaT tracking outputs, exactly as generated by the pipeline."""
        return self._results

    @property
    def run_data(self) -> dict:
        """
        The ``__dict__`` of every object used in the ROICaT pipeline.

        This is raw internal state and is not stable across ROICaT versions;
        prefer the derived properties on this class where they exist.
        """
        return self._run_data

    @property
    def params(self) -> dict:
        """Parameters actually used by the pipeline (user params merged with defaults)."""
        return self._params

    @property
    def session_files(self) -> tuple[str]:
        """A list of .hdf5 file paths, one for each session, that contain serialized masknmf.DemixingResults objects"""
        return self._session_files

    @staticmethod
    def _abspath_entry(entry: str | tuple[str, ...] | list[str]) -> str | tuple[str, ...]:
        if isinstance(entry, (tuple, list)):
            return tuple(os.path.abspath(x) for x in entry)
        return os.path.abspath(entry)

    @session_files.setter
    def session_files(self, new_files: tuple[str | tuple[str, ...], ...]):
        if len(new_files) != self.num_sessions:
            raise ValueError(
                f"The new set of files has length {len(new_files)} but the number of "
                f"sessions is {self.num_sessions}"
            )
        self._session_files = tuple(self._abspath_entry(f) for f in new_files)

    @property
    def num_sessions(self) -> int:
        return len(self._labels_by_session)

    @property
    def num_clusters(self) -> int:
        """Cluster ids are the integers ``[0, ..., num_clusters - 1]``."""
        return self._num_clusters

    @property
    def num_roi_per_session(self) -> np.ndarray:
        """Shape ``(num_sessions,)``; number of ROIs in each session."""
        return self._n_roi_per_session

    @property
    def num_roi_total(self) -> int:
        return int(self._n_roi_per_session.sum())

    @property
    def frame_height(self) -> int:
        return int(self._results["ROIs"]["frame_height"])

    @property
    def frame_width(self) -> int:
        return int(self._results["ROIs"]["frame_width"])

    @property
    def frame_shape(self) -> tuple[int, int]:
        return (self.frame_height, self.frame_width)

    ## Membership utils
    @property
    def labels_by_session(self) -> list[np.ndarray]:
        """
        A list of arrays, one per session. The k-th array has length equal to
        the number of ROIs in session k. Each value is the cluster id that ROI
        belongs to, or -1 if it belongs to no cluster.
        """
        return self._labels_by_session

    @property
    def counts_per_label(self) -> np.ndarray:
        """
        Shape ``(num_clusters, num_sessions)``. ``counts_per_label[c, s]`` is
        the number of ROIs from session s assigned to cluster c.
        """
        return self._counts

    @property
    def presence(self) -> np.ndarray:
        """
        Boolean array of shape ``(num_clusters, num_sessions)``.
        ``presence[c, s]`` is True if cluster c appears in session s.
        """
        return self._counts > 0

    @property
    def num_sessions_per_cluster(self) -> np.ndarray:
        """Shape ``(num_clusters,)``; in how many sessions each cluster appears."""
        return self.presence.sum(axis=1)

    def find_cluster(self, session: int, neuron: int) -> int:
        """Cluster id for a given ROI, or -1 if it was not clustered."""
        return int(self._labels_by_session[session][neuron])

    def find_members(self, cluster_id: int) -> dict[int, np.ndarray]:
        """Cluster id -> {session index: array of session-local neuron indices}."""
        return {
            s: np.flatnonzero(lab == cluster_id)
            for s, lab in enumerate(self._labels_by_session)
        }

    def unclustered(self) -> dict[int, np.ndarray]:
        """{session index: array of session-local indices of unclustered ROIs}."""
        return {
            s: np.flatnonzero(lab < 0)
            for s, lab in enumerate(self._labels_by_session)
        }

    ### For selecting clusters

    def select(self, predicate: Callable[[int], bool]) -> np.ndarray:
        """
        Cluster ids for which ``predicate(cluster_id)`` is True.

        The predicate is evaluated once per cluster. For vectorizable criteria
        it is usually faster to index ``presence`` or ``cluster_silhouette``
        directly and call ``np.flatnonzero`` yourself.
        """
        return np.array(
            [c for c in range(self.num_clusters) if predicate(c)], dtype=np.int64
        )

    def select_by_min_sessions(self, min_sessions: int) -> np.ndarray:
        """Cluster ids appearing in at least ``min_sessions`` sessions."""
        return np.flatnonzero(self.num_sessions_per_cluster >= min_sessions)

    def select_by_sessions(
        self,
        required: Optional[Sequence[int]] = None,
        forbidden: Optional[Sequence[int]] = None,
    ) -> np.ndarray:
        """
        Cluster ids present in every session in ``required`` and absent from
        every session in ``forbidden``.

        To accept clusters spanning all sessions except session 2::

            keep = res.select_by_sessions(
                required=[s for s in range(res.num_sessions) if s != 2]
            )
        """
        mask = np.ones(self.num_clusters, dtype=bool)
        if required is not None and len(required):
            mask &= self.presence[:, list(required)].all(axis=1)
        if forbidden is not None and len(forbidden):
            mask &= ~self.presence[:, list(forbidden)].any(axis=1)
        return np.flatnonzero(mask)

    @staticmethod
    def _to_pixels_by_rois(footprints) -> list[scipy.sparse.spmatrix]:
        """ROICaT stores (num_rois, num_pixels); transpose to (num_pixels, num_rois)."""
        return [scipy.sparse.csc_matrix(fp.T) for fp in footprints]

    @property
    def raw_rois(self) -> list[scipy.sparse.spmatrix]:
        """
        Unaligned spatial footprints as passed into the tracker: a list of
        ``(num_pixels, num_rois)`` sparse arrays, one per session. Pixels are
        flattened in C order relative to ``frame_shape``.
        """
        if self._raw_rois is None:
            self._raw_rois = self._to_pixels_by_rois(self._results["ROIs"]["ROIs_raw"])
        return self._raw_rois

    @property
    def aligned_rois(self) -> list[scipy.sparse.spmatrix]:
        """
        Spatial footprints after cross-session FOV registration, as a list of
        ``(num_pixels, num_rois)`` sparse arrays, one per session.
        """
        if self._aligned_rois is None:
            self._aligned_rois = self._to_pixels_by_rois(
                self._results["ROIs"]["ROIs_aligned"]
            )
        return self._aligned_rois

    def cluster_footprints(
        self, cluster_id: int, aligned: bool = True
    ) -> dict[int, np.ndarray]:
        """
        Dense images of every ROI in a cluster.

        Returns {session index: array of shape (num_rois_in_session, height, width)},
        omitting sessions in which the cluster does not appear.
        """
        rois = self.aligned_rois if aligned else self.raw_rois
        height, width = self.frame_shape
        out = {}
        for s, idx in self.find_members(cluster_id).items():
            if len(idx) == 0:
                continue
            block = np.asarray(rois[s][:, idx].todense()).T
            out[s] = block.reshape(len(idx), height, width)
        return out

    @property
    def aligned_fov_images(self) -> Optional[list[np.ndarray]]:
        """
        Session FOV images under the final alignment. Thi s is the transform applied to the spatial footprints
        before embedding and matching. Nonrigid
        if nonrigid registration ran (it composes on top of the geometric fit),
        otherwise geometric. None if aligner state is unavailable.
        """
        aligner = self._run_data.get("aligner", {})
        ims = aligner.get("ims_registered_nonrigid")
        if ims is None:
            ims = aligner.get("ims_registered_geo")
        return ims

    @property
    def alignment_was_nonrigid(self) -> bool:
        """Whether the final alignment included a nonrigid stage, useful as check"""
        return self._run_data.get("aligner", {}).get("ims_registered_nonrigid") is not None

    def roi_projection(self, session: int, aligned: bool = True) -> np.ndarray:
        """Max-intensity projection of all ROIs in a session, shape ``frame_shape``."""
        rois = self.aligned_rois if aligned else self.raw_rois
        return np.asarray(rois[session].max(axis=1).todense()).reshape(self.frame_shape)

    def __repr__(self) -> str:
        n_clustered = int(sum((l >= 0).sum() for l in self._labels_by_session))
        frac = n_clustered / self.num_roi_total if self.num_roi_total else 0.0
        return (
            f"{type(self).__name__}("
            f"num_sessions={self.num_sessions}, "
            f"num_clusters={self.num_clusters}, "
            f"num_roi_total={self.num_roi_total}, "
            f"frac_clustered={frac:.2f})"
        )

    ## Serialization code

    def to_roicat_dir(self,
                      dir_save: str | Path,
                      prefix_name_save=None):

        dir_save = Path(dir_save).resolve()
        dir_save.mkdir(parents=True, exist_ok=True)
        name = prefix_name_save or datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        RichFile_ROICaT(
            path=str(dir_save / f'{name}{_STEM_RESULTS}{_EXT_WRITE}'), backend='zip'
        ).save(obj=self._results, overwrite=True)
        RichFile_ROICaT(
            path=str(dir_save / f'{name}{_STEM_RUN_DATA}{_EXT_WRITE}'), backend='zip'
        ).save(obj=self.run_data, overwrite=True)


        if self._params is not None:
            ## This guarantees everything we write out is supported natively by pyyaml
            helpers.yaml_save(
                obj=json.loads(json.dumps(self._params, default=str)),
                filepath=str(dir_save / f'{name}{_SUFFIX_PARAMS}'),
            )

        (dir_save / f'{name}{_SUFFIX_SESSIONS}').write_text(
            json.dumps(self._session_files, indent=4),
            encoding='utf-8',
        )

        return dir_save

    @classmethod
    def from_roicat_dir(cls,
                        dir_load: str | Path,
                        prefix_name_save: str | None =None,
                        session_files=None):

        dir_load = Path(dir_load)
        pat = f'{prefix_name_save or "*"}{_STEM_RESULTS}*'
        hits = sorted(dir_load.glob(pat))
        if len(hits) != 1:
            raise ValueError(
                f"Expected exactly one results_all in {dir_load}, found {len(hits)}."
                + (" Pass prefix_name_save to disambiguate." if hits else "")
            )
        p_results = hits[0]
        name, ext = p_results.name.split(_STEM_RESULTS, 1) #maxsplit = 1 address case where prefix has identical to STEM

        p_run = dir_load / f'{name}{_STEM_RUN_DATA}{ext}'
        p_params = dir_load / f'{name}{_SUFFIX_PARAMS}'
        p_sessions = dir_load / f'{name}{_SUFFIX_SESSIONS}'

        if not p_run.exists():
            raise FileNotFoundError(f"run_data file {p_run} does not exist")

        params = helpers.yaml_load(str(p_params)) if p_params.exists() else None

        if session_files is None and p_sessions.exists():
            session_files = tuple(json.loads(p_sessions.read_text(encoding='utf-8')))

        return cls(
            results=RichFile_ROICaT(path=str(p_results), backend='auto').load(),
            run_data=RichFile_ROICaT(path=str(p_run), backend='auto').load(),
            session_files=session_files,
            params=params,
        )







