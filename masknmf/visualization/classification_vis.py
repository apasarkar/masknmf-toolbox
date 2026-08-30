from typing import *
import os
import threading
import time
import numpy as np
import fastplotlib as fpl
from imgui_bundle import imgui

from masknmf.visualization.imgui.movie_player import MoviePlayer
from masknmf.visualization.summary_widget import SummaryImageViewer

_LABEL_COLORS = (
    (0.12, 0.47, 0.71), (1.00, 0.50, 0.05), (0.17, 0.63, 0.17),
    (0.84, 0.15, 0.16), (0.58, 0.40, 0.74), (0.55, 0.34, 0.29),
    (0.89, 0.47, 0.76), (0.50, 0.50, 0.50), (0.74, 0.74, 0.13),
    (0.09, 0.75, 0.81),
)
_LABEL_KEYS = (
    imgui.Key._1, imgui.Key._2, imgui.Key._3, imgui.Key._4, imgui.Key._5,
    imgui.Key._6, imgui.Key._7, imgui.Key._8, imgui.Key._9,
)
_COLUMNS = ("id", "label", "pred", "area", "peak", "snr", "skew")


class ClassificationVis:
    """
    Label ROI images (e.g. from ROICat) one ROI at a time.
    """

    def __init__(
        self,
        roi_images: np.ndarray,
        label_names: Sequence[str] = (),
        class_labels: Optional[np.ndarray] = None,
        save_path: Optional[str] = None,
    ):
        """
        roi_images: (num_rois, Y, X) spatial footprint images.
        save_path: .npz file; label names and per-ROI labels are written on every
        change and restored from it on launch unless overridden by arguments.
        """
        self._save_path = str(save_path) if save_path is not None else None
        self._error: Optional[str] = None
        self._new_label = ""
        self._loading: Optional[str] = None
        self._load_result = None
        self._load_error: Optional[str] = None
        self._roicat_input = None
        self._session_sizes: Optional[tuple[int, ...]] = None
        self._save_files: Optional[list[str]] = None  # hdf5 mode: labels live in-file
        self._classifier = None
        self._classifier_path = ""
        self._clf_status: Optional[str] = None
        self._clf_done: Optional[str] = None
        self._clf_started = 0.0
        self._clf_thread: Optional[threading.Thread] = None
        self._clf_result = None
        self._clf_error: Optional[str] = None

        if self._save_path is not None and os.path.exists(self._save_path):
            saved = np.load(self._save_path)
            if not label_names:
                label_names = [str(n) for n in saved["label_names"]]
            if class_labels is None:
                saved_labels = saved["class_labels"]
                # a pre-load placeholder won't match a previous run's labels; those
                # are restored in _poll_load once the real ROI images arrive
                if saved_labels.shape[0] == np.asarray(roi_images).shape[0]:
                    class_labels = saved_labels

        self._label_names = tuple(label_names)

        self._show_bg = True
        self._show_mask = True
        self._bg_alpha = 0.5
        self._roi_alpha = 1.0
        self._advance_on_label = True
        self._keybinds_open = False
        self._dmrs: Optional[list] = None
        self._peak_frames: Optional[np.ndarray] = None
        self._movie_player = MoviePlayer()
        self._bg_movie = False
        self._movie_range: Optional[tuple] = None

        self._figure = fpl.Figure(size=(1200, 900))
        self._summary = SummaryImageViewer(self._figure)
        self._bg = None
        self._fg = None
        self.set_roi_images(roi_images, class_labels)

        self._figure.add_imgui_window(
            self._draw_panel, location="top", size=185, title="Classification"
        )
        self._figure.add_imgui_window(
            self._draw_table, location="right", size=360, title="ROIs"
        )

    @classmethod
    def from_masknmf(
        cls,
        demixing_result_files: Sequence[str],
        label_names: Sequence[str] = (),
        save_path: Optional[str] = None,
        roi_image_dims: tuple[int, int] = (36, 36),
        **adapter_kwargs,
    ) -> "ClassificationVis":
        """
        Label ROIs from masknmf demixing result .hdf5 files (one per session).

        The GUI opens immediately on an empty placeholder while ROICaT builds the
        centered ROI images (RoicatDataAdapter.from_masknmf) in a background thread,
        then swaps in the real data. Labels, label names, and ROI masks are stored
        inside each session's hdf5 (DemixingResults/class_labels, label_names,
        roi_masks) — no extra files. Pass save_path for an additional npz copy.
        """
        placeholder = np.zeros((1, *roi_image_dims), dtype=np.float32)
        vis = cls(placeholder, label_names=label_names, save_path=save_path)
        vis.load_masknmf(demixing_result_files, roi_image_dims=roi_image_dims, **adapter_kwargs)
        return vis

    @classmethod
    def from_classifier(
        cls,
        classifier,
        label_names: Sequence[str] = (),
        save_path: Optional[str] = None,
    ) -> "ClassificationVis":
        """
        Label the ROIs already held by a RoicatClassifier's data_adapter; the
        train button hands the labels back to that same classifier. Labels persist
        into the adapter's session .hdf5 files when it was built from masknmf results.
        Labels already set on the classifier are shown as the starting point.
        """
        adapter = classifier.data_adapter
        if adapter is None:
            raise ValueError("classifier has no data_adapter to label")
        imgs = np.concatenate([np.asarray(s) for s in adapter.ROI_images], axis=0)
        sizes = [len(s) for s in adapter.ROI_images]
        files = [
            str(f)
            for f in (adapter.session_files or ())
            if isinstance(f, (str, os.PathLike)) and str(f).endswith((".h5", ".hdf5"))
        ]
        if len(files) != len(sizes):
            files = None
        saved_labels, saved_names = cls._read_saved_labels(files) if files else (None, None)
        if not label_names and saved_names:
            label_names = saved_names
        if classifier.labels is not None:
            flat = [str(l) for s in classifier.labels for l in s]
            label_names = list(dict.fromkeys([*label_names, *flat]))
            saved_labels = np.array([label_names.index(l) for l in flat], dtype=np.int64)
        if saved_labels is not None and saved_labels.shape[0] != imgs.shape[0]:
            saved_labels = None

        vis = cls(np.zeros((1, *imgs.shape[1:]), dtype=np.float32), label_names=label_names, save_path=save_path)
        vis._save_files = files
        vis.set_roi_images(
            imgs,
            saved_labels,
            session_sizes=sizes,
            fov_images=adapter.FOV_images,
            centroids=adapter.centroids,
        )
        vis._roicat_input = adapter
        vis._classifier = classifier
        try:
            vis._save_masks_to_hdf5()
        except OSError as e:
            vis._error = f"mask save failed: {e}"
        return vis

    def set_roi_images(
        self,
        roi_images: np.ndarray,
        class_labels: Optional[np.ndarray] = None,
        session_sizes: Optional[Sequence[int]] = None,
        fov_images: Optional[Sequence[np.ndarray]] = None,
        centroids: Optional[Sequence[np.ndarray]] = None,
        bg_sources: Optional[dict] = None,
        roi_stats: Optional[dict] = None,
    ):
        """
        Replace the ROI image stack; derived stats, filters, and view are rebuilt.

        fov_images/centroids: per-session FOV images and (n_roi, 2) y/x centroids.
        When given, the background shows the FOV cropped around the current ROI
        (its surrounding context) instead of a static max projection.
        bg_sources: {name: per-session image list} of alternative FOV backgrounds.
        roi_stats: {"snr": (num_rois,), "skew": (num_rois,)} per-ROI trace stats.
        """
        roi_images = np.asarray(roi_images, dtype=np.float32)
        if roi_images.ndim != 3:
            raise ValueError(f"roi_images must be (num_rois, Y, X), got {roi_images.shape}")
        num_rois = roi_images.shape[0]
        if session_sizes is not None and sum(session_sizes) != num_rois:
            raise ValueError(
                f"session_sizes sums to {sum(session_sizes)}, but there are {num_rois} ROIs"
            )
        self._roi_images = roi_images
        self._session_sizes = tuple(int(s) for s in session_sizes) if session_sizes else None

        if fov_images is not None and centroids is not None:
            self._fov_images = [np.asarray(f, dtype=np.float32) for f in fov_images]
            self._centroids = np.concatenate([np.asarray(c) for c in centroids], axis=0)
            if self._centroids.shape[0] != num_rois:
                raise ValueError(
                    f"centroids cover {self._centroids.shape[0]} ROIs, expected {num_rois}"
                )
            sizes = self._session_sizes if self._session_sizes is not None else (num_rois,)
            self._session_of = np.repeat(np.arange(len(sizes)), sizes)
            self._bg_sources = {
                name: [np.asarray(i, dtype=np.float32) for i in imgs_]
                for name, imgs_ in (bg_sources or {}).items()
            }
            if not self._bg_sources:
                self._bg_sources = {"mean img (enhanced)": self._fov_images}
            self._bg_source_names = list(self._bg_sources)
            self._bg_source_idx = 0
            self._fov_images = self._bg_sources[self._bg_source_names[0]]
        else:
            self._fov_images = None
            self._centroids = None
            self._session_of = None
            self._bg_sources = None
            self._bg_source_names = []
            self._bg_source_idx = 0

        roi_stats = roi_stats or {}
        self._snr = np.asarray(roi_stats.get("snr", np.zeros(num_rois)), dtype=np.float32)
        self._skew = np.asarray(roi_stats.get("skew", np.zeros(num_rois)), dtype=np.float32)
        if self._snr.shape[0] != num_rois or self._skew.shape[0] != num_rois:
            raise ValueError("roi_stats arrays must have one entry per ROI")

        if class_labels is None:
            class_labels = np.full((num_rois,), -1, dtype=np.int64)
        else:
            class_labels = np.asarray(class_labels).astype(np.int64)
            if class_labels.shape[0] != num_rois:
                raise ValueError(
                    f"class_labels has {class_labels.shape[0]} entries, expected {num_rois}"
                )
        self._class_labels = class_labels

        # labels restored from disk can reference classes the current name set
        # doesn't have; extend it so every label index stays displayable
        max_label = int(class_labels.max(initial=-1))
        if max_label >= len(self._label_names):
            self._label_names = (
                *self._label_names,
                *(f"class{i}" for i in range(len(self._label_names), max_label + 1)),
            )

        flat = roi_images.reshape(num_rois, -1)
        self._peak = flat.max(axis=1)
        self._area = np.count_nonzero(flat, axis=1).astype(np.int64)
        self._pred = np.full(num_rois, -1, dtype=np.int64)  # classifier prediction per ROI
        self._probs = np.full(num_rois, np.nan, dtype=np.float32)  # its confidence

        self._mip = roi_images.max(axis=0)

        self._filter_label = -2  # -2 all, -1 unlabeled, >=0 label index
        self._area_range = (0, int(self._area.max(initial=0)))
        self._sort_column = 0
        self._sort_ascending = True
        self._order = np.arange(num_rois)
        self._pos = 0
        self._scroll_to_current = True

        subplot = self._figure[0, 0]
        for graphic in (self._bg, self._fg):
            if graphic is not None:
                subplot.delete_graphic(graphic)
        self._bg = subplot.add_image(
            self._mip, cmap="gray", name="background", alpha_mode="blend"
        )
        self._bg.vmin = 0.0
        self._bg.vmax = float(self._mip.max()) or 1.0
        self._fg = subplot.add_image(
            np.zeros((*roi_images.shape[1:], 4), dtype=np.float32),
            name="roi",
            alpha_mode="blend",
        )
        subplot.auto_scale()
        self._apply_overlay()
        self._show_current()

    def load_masknmf(self, demixing_result_files: Sequence[str], **adapter_kwargs):
        """Build a RoicatDataAdapter from demixing .hdf5 files in a background thread"""
        files = [str(f) for f in demixing_result_files]
        self._save_files = files
        self._loading = f"ROICaT: building ROI images from {len(files)} session(s)..."
        self._load_result = None
        self._load_error = None

        def work():
            try:
                import torch
                from masknmf.demixing.demixing_results import DemixingResults
                from masknmf.multisession.roicat_tracking import (
                    RoicatDataAdapter,
                    extract_masknmf_mean_img,
                    extract_masknmf_spatial_footprints,
                )

                bg_sources: dict[str, list] = {}
                snr, skew, peak_frames = [], [], []
                mean_imgs, footprints, dmrs = [], [], []

                def add_bg(name, img):
                    bg_sources.setdefault(name, []).append(np.asarray(img, dtype=np.float32))

                for fname in files:
                    dmr = DemixingResults.from_hdf5(fname)
                    dmrs.append(dmr)
                    footprints.append(extract_masknmf_spatial_footprints(dmr))
                    mean_img = extract_masknmf_mean_img(dmr)
                    mean_imgs.append(mean_img)

                    def reduce_stack(stack):
                        n = stack.shape[0]
                        running_max = None
                        running_sum = None
                        for start in range(0, n, 32):
                            batch = stack.getitem_tensor(slice(start, min(start + 32, n)))
                            bmax = batch.amax(dim=0)
                            bsum = batch.sum(dim=0)
                            running_max = (
                                bmax if running_max is None else torch.maximum(running_max, bmax)
                            )
                            running_sum = bsum if running_sum is None else running_sum + bsum
                        return running_max.cpu().numpy(), (running_sum / n).cpu().numpy()

                    # first source added is the default background
                    resid_imgs = dmr.residual_correlation_images
                    if resid_imgs is not None:
                        rmax, _ = reduce_stack(resid_imgs)
                        add_bg("resid corr img (max proj)", rmax)
                    std_imgs = dmr.standard_correlation_images
                    if std_imgs is not None:
                        smax, smean = reduce_stack(std_imgs)
                        add_bg("corr img (max proj)", smax)
                        add_bg("corr img (mean)", smean)
                    if dmr.global_residual_correlation_image is not None:
                        add_bg(
                            "resid corr img (global)",
                            dmr.global_residual_correlation_image.cpu().numpy(),
                        )

                    c = dmr.c.cpu().numpy()  # (num_frames, num_rois)
                    med = np.median(c, axis=0)
                    mad = np.median(np.abs(c - med), axis=0) * 1.4826
                    snr.append((c.max(axis=0) - med) / np.where(mad == 0, 1, mad))
                    mean = c.mean(axis=0)
                    std = c.std(axis=0)
                    skew.append(((c - mean) ** 3).mean(axis=0) / np.where(std == 0, 1, std) ** 3)
                    peak_frames.append(c.argmax(axis=0))

                adapter = RoicatDataAdapter(
                    mean_imgs,
                    footprints,
                    tuple(os.path.abspath(f) for f in files),
                    **adapter_kwargs,
                )
                saved_labels, saved_names = self._read_saved_labels(files)

                self._load_result = {
                    "adapter": adapter,
                    # only offer a source if every session's file has it
                    "bg_sources": {
                        k: v for k, v in bg_sources.items() if len(v) == len(files)
                    },
                    "stats": {
                        "snr": np.concatenate(snr).astype(np.float32),
                        "skew": np.concatenate(skew).astype(np.float32),
                    },
                    "saved_labels": saved_labels,
                    "saved_names": saved_names,
                    "dmrs": dmrs,
                    "peak_frames": np.concatenate(peak_frames).astype(np.int64),
                }
            except Exception as e:
                self._load_error = f"load failed: {type(e).__name__}: {e}"

        threading.Thread(target=work, daemon=True).start()

    def _poll_load(self):
        if self._loading is None:
            return
        if self._load_error is not None:
            self._error = self._load_error
            self._loading = None
        elif self._load_result is not None:
            result = self._load_result
            adapter = result["adapter"]
            self._load_result = None
            imgs = np.concatenate([np.asarray(s) for s in adapter.ROI_images], axis=0)
            sizes = [len(s) for s in adapter.ROI_images]
            labels = result["saved_labels"]
            if labels is not None and labels.shape[0] != imgs.shape[0]:
                labels = None
            if not self._label_names and result["saved_names"]:
                self._label_names = tuple(result["saved_names"])
            self.set_roi_images(
                imgs,
                labels,
                session_sizes=sizes,
                fov_images=adapter.FOV_images,
                centroids=adapter.centroids,
                bg_sources=result["bg_sources"],
                roi_stats=result["stats"],
            )
            self._roicat_input = adapter
            self._dmrs = result["dmrs"]
            self._peak_frames = result["peak_frames"]
            self._loading = None
            try:
                self._save_masks_to_hdf5()
            except OSError as e:
                self._error = f"mask save failed: {e}"

    @property
    def current(self) -> Optional[int]:
        """ROI id at the current position in the sorted/filtered view"""
        if len(self._order) == 0:
            return None
        return int(self._order[self._pos])

    def step(self, delta: int):
        """Move through the current view by delta ROIs"""
        if len(self._order):
            self._pos = int(np.clip(self._pos + delta, 0, len(self._order) - 1))
            self._show_current()

    def goto(self, roi: int):
        """Jump to an ROI if it is in the current view"""
        hits = np.flatnonzero(self._order == roi)
        if len(hits):
            self._pos = int(hits[0])
            self._show_current()

    def label(self, roi_ids: Sequence[int], label_index: int | Sequence[int]):
        """Assign a class label (or one per ROI) to the given ROIs; -1 clears"""
        self._class_labels[list(roi_ids)] = label_index
        self._autosave()
        if self._filter_label != -2:
            self._rebuild_order()
        self._show_current()

    def label_current(self, label_index: int):
        """Label the current ROI and advance to the next one in view"""
        if self.current is None:
            return
        size_before = len(self._order)
        self.label([self.current], label_index)
        # if the filter dropped the labeled ROI, pos already sits on its successor
        if self._advance_on_label and len(self._order) == size_before:
            self.step(1)

    def add_label(self, name: str):
        """Add a new class name to the label set"""
        if name and name not in self._label_names:
            self._label_names = (*self._label_names, name)
            self._autosave()

    def remove_label(self, index: int):
        """Delete a class: its ROIs become unlabeled, higher labels shift down"""
        if not 0 <= index < len(self._label_names):
            return
        for arr in (self._class_labels, self._pred):
            arr[arr == index] = -1
            arr[arr > index] -= 1
        self._label_names = tuple(
            n for i, n in enumerate(self._label_names) if i != index
        )
        if self._filter_label == index:
            self._filter_label = -2
        elif self._filter_label > index:
            self._filter_label -= 1
        self._autosave()
        self._rebuild_order()

    def save(self, path: Optional[str] = None):
        """
        Persist labels: into each session's hdf5 (DemixingResults/class_labels +
        label_names) when launched from demixing results, and/or to an npz at path
        """
        if path is None and self._save_files is not None:
            self._save_labels_to_hdf5()
        path = path if path is not None else self._save_path
        if path is None:
            return
        data = dict(label_names=np.array(self._label_names), class_labels=self._class_labels)
        if self._session_sizes is not None:
            data["session_sizes"] = np.array(self._session_sizes)
        np.savez(path, **data)

    @staticmethod
    def _read_saved_labels(files: Sequence[str]):
        """(concatenated class_labels or None, label_names or None) stored in the session hdf5s"""
        import h5py

        labels, names = [], None
        for fname in files:
            with h5py.File(fname, "r") as f:
                g = f.get("DemixingResults")
                if g is not None and "class_labels" in g:
                    labels.append(g["class_labels"][()])
                if names is None and g is not None and "label_names" in g:
                    names = [n.decode() for n in g["label_names"][()]]
        return (np.concatenate(labels) if len(labels) == len(files) else None), names

    @staticmethod
    def _write_dataset(group, key: str, data: np.ndarray):
        # overwrite in place when possible so the hdf5 doesn't grow with each write
        if key in group and group[key].shape == data.shape and group[key].dtype == data.dtype:
            group[key][...] = data
        else:
            if key in group:
                del group[key]
            group.create_dataset(key, data=data)

    def _save_labels_to_hdf5(self):
        import h5py

        names = np.array([n.encode() for n in self._label_names])
        for fname, labels in zip(self._save_files, self.class_labels_by_session):
            with h5py.File(fname, "r+") as f:
                g = f.require_group("DemixingResults")
                self._write_dataset(g, "class_labels", labels.astype(np.int64))
                self._write_dataset(g, "label_names", names)
                self._write_dataset(
                    g, "labels_complete", np.bool_((labels >= 0).all())
                )

    def _save_masks_to_hdf5(self):
        if self._save_files is None:
            return
        import h5py

        start = 0
        for fname, n in zip(self._save_files, self._session_sizes or (len(self._roi_images),)):
            with h5py.File(fname, "r+") as f:
                g = f.require_group("DemixingResults")
                self._write_dataset(g, "roi_masks", self._roi_images[start : start + n])
            start += n

    def _autosave(self):
        # don't clobber a previous session's labels with placeholder state mid-load
        if (self._save_path is None and self._save_files is None) or self._loading is not None:
            return
        try:
            self.save()
            self._error = None
        except OSError as e:
            self._error = f"save failed: {e}"

    def _rebuild_order(self):
        current = self.current
        mask = (self._area >= self._area_range[0]) & (self._area <= self._area_range[1])
        if self._filter_label >= -1:
            mask &= self._class_labels == self._filter_label
        idx = np.flatnonzero(mask)
        if self._sort_column:
            keys = (self._class_labels, self._probs, self._area, self._peak, self._snr, self._skew)
            idx = idx[np.argsort(keys[self._sort_column - 1][idx], kind="stable")]
        if not self._sort_ascending:
            idx = idx[::-1]
        self._order = idx
        hits = np.flatnonzero(self._order == current) if current is not None else ()
        if len(hits):
            self._pos = int(hits[0])
        else:
            self._pos = int(min(self._pos, max(len(self._order) - 1, 0)))
        self._show_current()

    def _label_color(self, label_index: int) -> tuple:
        return _LABEL_COLORS[label_index % len(_LABEL_COLORS)]

    def _crop_origin(self, roi: int) -> tuple[int, int]:
        """FOV coordinate of the ROI crop's top-left corner (roicat's convention)"""
        h, w = self._roi_images.shape[1:]
        cy, cx = self._centroids[roi]
        return int(cy) - int(np.ceil(h / 2)), int(cx) - int(np.ceil(w / 2))

    def _context_crop(self, roi: int) -> np.ndarray:
        """FOV image cropped around the ROI, aligned with roicat's centered ROI images"""
        fov = self._fov_images[self._session_of[roi]]
        h, w = self._roi_images.shape[1:]
        top, left = self._crop_origin(roi)
        crop = np.zeros((h, w), dtype=np.float32)
        y0, y1 = max(top, 0), min(top + h, fov.shape[0])
        x0, x1 = max(left, 0), min(left + w, fov.shape[1])
        if y1 > y0 and x1 > x0:
            crop[y0 - top : y1 - top, x0 - left : x1 - left] = fov[y0:y1, x0:x1]
        return crop

    def _update_movie_bg(self):
        roi = self.current
        if roi is None or self._movie_player.movie is None:
            return
        top, left = self._crop_origin(roi)
        crop = self._movie_player.frame((top, left, *self._roi_images.shape[1:]))
        if self._movie_range is None:
            # fixed at the ROI's peak frame so playback doesn't flicker
            self._movie_range = (float(crop.min()), float(crop.max()) or 1.0)
        self._bg.data = crop
        self._bg.vmin, self._bg.vmax = self._movie_range

    def _show_current(self):
        roi = self.current
        rgba = np.zeros((*self._roi_images.shape[1:], 4), dtype=np.float32)
        title = "no ROIs in view"
        if roi is not None:
            label = int(self._class_labels[roi])
            color = self._label_color(label) if label >= 0 else (1.0, 1.0, 1.0)
            rgba[..., :3] = color
            rgba[..., 3] = self._roi_images[roi] / (self._peak[roi] or 1.0)
            name = self._label_names[label] if label >= 0 else "unlabeled"
            title = f"ROI {roi}  [{name}]  ({self._pos + 1}/{len(self._order)})"
            if self._fov_images is not None:
                if self._bg_movie and self._dmrs is not None:
                    sess = int(self._session_of[roi])
                    self._movie_player.set_movie(self._dmrs[sess].ac_array)
                    if self._peak_frames is not None:
                        self._movie_player.jump_to(int(self._peak_frames[roi]))
                    self._movie_range = None
                    self._update_movie_bg()
                else:
                    crop = self._context_crop(roi)
                    self._bg.data = crop
                    self._bg.vmin = float(crop.min())
                    self._bg.vmax = float(crop.max()) or 1.0
                top, left = self._crop_origin(roi)
                self._summary.set_highlight((top, left, *self._roi_images.shape[1:]))
                if self._summary.is_open and self._bg_sources is not None:
                    sess = int(self._session_of[roi])
                    self._summary.set_images(
                        {name: imgs[sess] for name, imgs in self._bg_sources.items()}
                    )
                    if self._dmrs is not None:
                        self._summary.set_movies(
                            {"demixed movie": self._dmrs[sess].ac_array}
                        )
        if roi is None:
            self._summary.set_highlight(None)
        self._fg.data = rgba
        self._figure[0, 0].title = title
        self._scroll_to_current = True

    def _apply_overlay(self):
        self._bg.visible = self._show_bg
        self._bg.alpha = self._bg_alpha
        self._fg.visible = self._show_mask
        self._fg.alpha = self._roi_alpha

    def goto_next_unlabeled(self):
        """Jump to the next unlabeled ROI in the current view (wraps around)"""
        labels = self._class_labels[self._order]
        hits = np.flatnonzero(labels < 0)
        if not len(hits):
            return
        after = hits[hits > self._pos]
        self._pos = int(after[0] if len(after) else hits[0])
        self._show_current()

    def _step_group(self, direction: int):
        """Cycle to the first ROI in view of the next/previous label class"""
        if self.current is None:
            return
        labels = self._class_labels[self._order]
        values = np.unique(labels)
        if len(values) < 2:
            return
        current = int(self._class_labels[self.current])
        i = int(np.flatnonzero(values == current)[0])
        target = values[(i + direction) % len(values)]
        self._pos = int(np.flatnonzero(labels == target)[0])
        self._show_current()

    def _handle_keys(self):
        io = imgui.get_io()
        if io.want_text_input:
            return
        stride = 10 if io.key_shift else 1
        if imgui.is_key_pressed(imgui.Key.up_arrow):
            self.step(-stride)
        if imgui.is_key_pressed(imgui.Key.down_arrow):
            self.step(stride)
        if imgui.is_key_pressed(imgui.Key.left_arrow):
            self._step_group(-1)
        if imgui.is_key_pressed(imgui.Key.right_arrow):
            self._step_group(1)
        if imgui.is_key_pressed(imgui.Key.b, False):
            self._show_bg = not self._show_bg
            self._apply_overlay()
        if imgui.is_key_pressed(imgui.Key.u, False):
            self.goto_next_unlabeled()
        if imgui.is_key_pressed(imgui.Key._0, False):
            self.label_current(-1)
        if imgui.is_key_pressed(imgui.Key.m, False):
            self._show_mask = not self._show_mask
            self._apply_overlay()
        for i, key in enumerate(_LABEL_KEYS[: len(self._label_names)]):
            if imgui.is_key_pressed(key, False):
                self.label_current(i)

    def _open_full_fov(self):
        if self._bg_sources is not None:
            roi = self.current
            sess = (
                int(self._session_of[roi])
                if roi is not None and self._session_of is not None
                else 0
            )
            images = {name: imgs[sess] for name, imgs in self._bg_sources.items()}
            selected = self._bg_source_names[self._bg_source_idx]
        else:
            images = {"mask MIP": self._mip}
            selected = "mask MIP"
        self._summary.set_images(images, selected=selected)
        if self._dmrs is not None and self._session_of is not None and self.current is not None:
            sess = int(self._session_of[self.current])
            self._summary.set_movies({"demixed movie": self._dmrs[sess].ac_array})
            if self._peak_frames is not None:
                self._summary.player.jump_to(int(self._peak_frames[self.current]))
        else:
            self._summary.set_movies({})
        if self._fov_images is not None and self.current is not None:
            top, left = self._crop_origin(self.current)
            self._summary.set_highlight((top, left, *self._roi_images.shape[1:]))
        else:
            self._summary.set_highlight(None)
        self._summary.open()

    @property
    def classifier(self):
        """RoicatClassifier over the loaded sessions (built lazily once the adapter exists)"""
        if self._classifier is None and self._roicat_input is not None:
            from masknmf.classification import RoicatClassifier

            self._classifier = RoicatClassifier(data_adapter=self._roicat_input)
        return self._classifier

    @property
    def classifier_path(self) -> str:
        """Where train() saves the classifier package and classify() loads it from"""
        return self._classifier_path

    @classifier_path.setter
    def classifier_path(self, path: str):
        self._classifier_path = str(path)

    @property
    def labels_complete(self) -> bool:
        return len(self._class_labels) > 0 and bool((self._class_labels >= 0).all())

    def _default_classifier_path(self) -> str:
        base = os.path.dirname(self._save_files[0]) if self._save_files else os.getcwd()
        return os.path.join(base, "classifier")

    def _adapter_missing(self) -> Optional[str]:
        if self._roicat_input is not None:
            return None
        if self._loading is not None:
            return "ROI images are still loading"
        return "no ROICaT data (load demixing .hdf5 files or use from_classifier)"

    @property
    def _clf_busy(self) -> bool:
        return self._clf_thread is not None and self._clf_thread.is_alive()

    def _run_clf(self, status: str, work: Callable):
        self._clf_status = status
        self._clf_started = time.perf_counter()
        self._clf_result = None
        self._clf_error = None

        def target():
            try:
                self._clf_result = work()
            except Exception as e:
                self._clf_error = f"{type(e).__name__}: {e}"

        self._clf_thread = threading.Thread(target=target, daemon=True)
        self._clf_thread.start()

    def train(self, classifier_path: Optional[str] = None):
        """
        Hand the labels to the RoicatClassifier, train it in a background thread
        and save the package to classifier_path (default: <first session dir>/classifier).
        Every ROI must be labeled first.
        """
        if self._clf_busy:
            return
        if not self.labels_complete:
            self._error = "label every ROI before training"
            return
        if self._adapter_missing():
            self._error = f"cannot train: {self._adapter_missing()}"
            return
        if classifier_path:
            self._classifier_path = str(classifier_path)
        if not self._classifier_path:
            self._classifier_path = self._default_classifier_path()
        path = self._classifier_path
        names = [[self._label_names[i] for i in s] for s in self.class_labels_by_session]
        clf = self.classifier

        def work():
            clf.labels = names
            clf.train(num_workers=0)
            return "trained", clf.save(path)

        self._run_clf("training: ROInet embedding + logistic regression...", work)

    def classify(self, classifier=None):
        """
        Predict a label for every ROI in a background thread using the classifier
        trained here, a RoicatClassifier, or a .roicat_classifier path (default:
        classifier_path). Predictions fill the pred column; ROIs that are still
        unlabeled take the prediction, existing labels are kept.
        """
        if self._clf_busy:
            return
        if self._adapter_missing():
            self._error = f"cannot classify: {self._adapter_missing()}"
            return
        path = None
        if classifier is None:
            if self._classifier is not None and self._classifier.classifier is not None:
                classifier = self._classifier
            else:
                path = self._classifier_path
        elif isinstance(classifier, (str, os.PathLike)):
            path = str(classifier)
        if path is not None and not os.path.isfile(path):
            self._error = f"classifier file not found: {path or '(set classifier path)'}"
            return
        adapter = self._roicat_input

        def work():
            from masknmf.classification import RoicatClassifier

            clf = RoicatClassifier.from_file(path) if path is not None else classifier
            _, names, probs = clf.classify(adapter)
            return "classified", names, probs

        self._run_clf("classifying...", work)

    def _apply_predictions(self, names_by_session, probs_by_session) -> int:
        """Store predictions, label the unlabeled ROIs with them; returns how many were labeled"""
        flat = [str(n) for s in names_by_session for n in s]
        if len(flat) != len(self._class_labels):
            raise ValueError(f"got {len(flat)} predictions for {len(self._class_labels)} ROIs")
        for n in dict.fromkeys(flat):
            if n not in self._label_names:
                self._label_names = (*self._label_names, n)
        index = {n: i for i, n in enumerate(self._label_names)}
        self._pred = np.array([index[n] for n in flat], dtype=np.int64)
        self._probs = np.concatenate([p.max(axis=1) for p in probs_by_session]).astype(np.float32)
        fill = np.flatnonzero(self._class_labels < 0)
        self.label(fill, self._pred[fill])
        self._rebuild_order()
        return len(fill)

    def _poll_classifier(self):
        if self._clf_status is None:
            return
        if self._clf_error is not None:
            self._error = f"{self._clf_status.split(':')[0]} failed: {self._clf_error}"
            self._clf_status = None
        elif self._clf_result is not None:
            result, self._clf_result = self._clf_result, None
            self._clf_status = None
            self._error = None
            if result[0] == "trained":
                self._classifier_path = result[1]
                self._clf_done = f"saved classifier to {result[1]}"
            else:
                try:
                    n = self._apply_predictions(result[1], result[2])
                except ValueError as e:
                    self._error = f"classify failed: {e}"
                    return
                self._clf_done = (
                    f"predicted {len(self._pred)} ROIs, {n} unlabeled ones took the prediction "
                    "(existing labels kept, see the pred column)"
                )

    def _draw_classifier_row(self):
        imgui.set_next_item_width(320)
        _, self._classifier_path = imgui.input_text_with_hint(
            "##clf-path", "classifier path (.roicat_classifier)", self._classifier_path
        )
        if imgui.is_item_hovered():
            imgui.set_tooltip(
                "train saves the classifier here, classify loads it from here\n"
                "(default: <first session dir>/classifier.roicat_classifier)"
            )
        imgui.same_line(0, 6)
        busy = self._clf_busy
        can_train = self.labels_complete and self.classifier is not None and not busy
        imgui.begin_disabled(not can_train)
        if imgui.button("train"):
            self.train()
        imgui.end_disabled()
        if imgui.is_item_hovered(imgui.HoveredFlags_.allow_when_disabled):
            imgui.set_tooltip(
                "train a ROICaT classifier on the labeled ROIs and save it"
                if can_train or busy
                else "label every ROI first (training needs complete labels)"
            )
        imgui.same_line(0, 6)
        has_trained = self._classifier is not None and self._classifier.classifier is not None
        can_classify = (
            self._roicat_input is not None
            and not busy
            and (has_trained or bool(self._classifier_path))
        )
        imgui.begin_disabled(not can_classify)
        if imgui.button("classify"):
            self.classify()
        imgui.end_disabled()
        if imgui.is_item_hovered(imgui.HoveredFlags_.allow_when_disabled):
            imgui.set_tooltip(
                "predict every ROI: unlabeled ROIs take the prediction, existing labels are kept\n"
                + (
                    "uses the classifier trained in this session"
                    if has_trained
                    else "loads the classifier at the path"
                )
            )
        imgui.same_line(0, 16)
        if self._clf_status is not None:
            imgui.text_colored(
                imgui.ImVec4(1.0, 0.8, 0.2, 1.0),
                f"{self._clf_status} {time.perf_counter() - self._clf_started:.0f}s",
            )
        elif self._clf_done is not None:
            imgui.text_disabled(self._clf_done)

    def _draw_save_note(self):
        if self._save_path is None and self._save_files is None:
            imgui.text_disabled("autosave off — labels are kept in memory only")
            return
        imgui.text("Accessing masks in output file")
        imgui.same_line(0, 4)
        imgui.text_disabled("(?)")
        if imgui.is_item_hovered():
            imgui.begin_tooltip()
            if self._save_files is not None:
                imgui.text_colored(
                    imgui.ImVec4(0.55, 0.75, 1.0, 1.0),
                    "import h5py\n"
                    "\n"
                    'with h5py.File(r"path/to/demixing_results.hdf5", "r") as f:\n'
                    '    g = f["DemixingResults"]\n'
                    '    names  = [n.decode() for n in g["label_names"][()]]\n'
                    '    labels = g["class_labels"][()]  # (num_rois,) int64; -1 = unlabeled\n'
                    '    masks  = g["roi_masks"][()]     # (num_rois, Y, X) float32\n'
                    "\n"
                    "# ROICaT: one label vector per session file\n"
                    "roicat_input.set_class_labels(labels=[labels_session0, ...])",
                )
            else:
                imgui.text_colored(
                    imgui.ImVec4(0.55, 0.75, 1.0, 1.0),
                    "import numpy as np\n"
                    "\n"
                    'data = np.load(r"path/to/labels.npz")\n'
                    'names  = data["label_names"]   # class names; row index = label value\n'
                    'labels = data["class_labels"]  # (num_rois,) int64; -1 = unlabeled',
                )
            imgui.end_tooltip()
        imgui.same_line(0, 20)
        imgui.text_disabled("Autosaved")
        if imgui.is_item_hovered():
            imgui.set_tooltip("Labels saved automatically")

    def _draw_panel(self):
        self._poll_load()
        self._poll_classifier()
        self._handle_keys()

        if self._loading is not None:
            imgui.text_colored(imgui.ImVec4(1.0, 0.8, 0.2, 1.0), self._loading)

        if imgui.button("prev"):
            self.step(-1)
        imgui.same_line(0, 5)
        if imgui.button("next"):
            self.step(1)
        imgui.same_line(0, 10)
        imgui.set_next_item_width(200)
        changed, pos = imgui.slider_int("##pos", self._pos, 0, max(len(self._order) - 1, 0))
        if changed and len(self._order):
            self._pos = pos
            self._show_current()
        imgui.same_line(0, 30)
        if imgui.button("Open full FOV"):
            self._open_full_fov()
        imgui.same_line(0, 30)
        self._draw_save_note()
        imgui.same_line(max(imgui.get_window_width() - 90, 0))
        if imgui.button("keybinds"):
            self._keybinds_open = True
        if self._error is not None:
            imgui.same_line(0, 30)
            imgui.text(self._error)

        changed_bg, self._show_bg = imgui.checkbox("##bg-on", self._show_bg)
        if changed_bg:
            self._apply_overlay()
        imgui.same_line(0, 6)
        sources = self._bg_source_names if self._bg_sources is not None else ["mask MIP"]
        options = [*sources] + (["demixed movie"] if self._dmrs is not None else [])
        current = (
            len(sources)
            if self._bg_movie
            else (self._bg_source_idx if self._bg_sources is not None else 0)
        )
        imgui.set_next_item_width(180)
        changed_src, idx = imgui.combo("##bg-source", current, options)
        if changed_src:
            self._bg_movie = idx >= len(sources)
            if not self._bg_movie and self._bg_sources is not None:
                self._bg_source_idx = idx
                self._fov_images = self._bg_sources[sources[idx]]
            self._show_current()
        imgui.same_line(0, 6)
        imgui.text("bg image")
        imgui.same_line(0, 4)
        imgui.text_disabled("(b)")
        imgui.same_line(0, 20)
        imgui.set_next_item_width(75)
        changed_bga, self._bg_alpha = imgui.slider_float(
            "bg opacity", self._bg_alpha, 0.0, 1.0
        )
        imgui.same_line(0, 30)
        changed_mask, self._show_mask = imgui.checkbox("mask", self._show_mask)
        imgui.same_line(0, 4)
        imgui.text_disabled("(m)")
        imgui.same_line(0, 20)
        imgui.set_next_item_width(75)
        changed_fga, self._roi_alpha = imgui.slider_float(
            "roi opacity", self._roi_alpha, 0.0, 1.0
        )
        if changed_bga or changed_mask or changed_fga:
            self._apply_overlay()
        if self._bg_movie:
            if self._movie_player.draw():
                self._update_movie_bg()

        self._draw_progress()
        imgui.same_line(0, 24)
        imgui.set_next_item_width(120)
        entered, self._new_label = imgui.input_text_with_hint(
            "##new-label",
            "new label",
            self._new_label,
            imgui.InputTextFlags_.enter_returns_true,
        )
        imgui.same_line(0, 5)
        if (imgui.button("add") or entered) and self._new_label.strip():
            self.add_label(self._new_label.strip())
            self._new_label = ""
        imgui.same_line(0, 5)
        if imgui.button("del"):
            imgui.open_popup("##del-labels")
        if imgui.begin_popup("##del-labels"):
            if not self._label_names:
                imgui.text_disabled("no labels")
            remove = None
            for i, name in enumerate(self._label_names):
                if imgui.small_button(f"x##del{i}"):
                    remove = i
                imgui.same_line(0, 8)
                count = int((self._class_labels == i).sum())
                imgui.text_colored(
                    imgui.ImVec4(*self._label_color(i), 1.0), f"{name} ({count})"
                )
            if remove is not None:
                self.remove_label(remove)
            imgui.end_popup()

        for i, name in enumerate(self._label_names):
            imgui.same_line(0, 10)
            count = int((self._class_labels == i).sum())
            imgui.push_style_color(imgui.Col_.button, imgui.ImVec4(*self._label_color(i), 0.5))
            if imgui.button(f"{name} ({count})##label{i}"):
                self.label_current(i)
            imgui.pop_style_color()
            if i < len(_LABEL_KEYS):
                imgui.same_line(0, 4)
                imgui.text_disabled(f"({i + 1})")
        if self._label_names:
            imgui.same_line(0, 10)
            if imgui.button("unlabel"):
                self.label_current(-1)
            imgui.same_line(0, 4)
            imgui.text_disabled("(0)")
            imgui.same_line(0, 10)
            imgui.push_style_color(imgui.Col_.button, imgui.ImVec4(0.75, 0.15, 0.15, 0.8))
            imgui.push_style_color(
                imgui.Col_.button_hovered, imgui.ImVec4(0.90, 0.20, 0.20, 1.0)
            )
            if imgui.button("unlabel all"):
                self.label(range(len(self._class_labels)), -1)
            imgui.pop_style_color(2)

        self._draw_classifier_row()
        self._summary.draw()
        self._draw_keybinds_popup()

    def _draw_progress(self):
        """Per-session labeling progress; training requires every ROI labeled"""
        done_color = imgui.ImVec4(0.35, 0.9, 0.35, 1.0)
        todo_color = imgui.ImVec4(1.0, 0.75, 0.25, 1.0)
        labeled = self._class_labels >= 0
        done, total = int(labeled.sum()), len(labeled)
        imgui.text_colored(
            done_color if done == total else todo_color, f"labeled {done}/{total}"
        )
        if self._session_sizes is not None and len(self._session_sizes) > 1:
            start = 0
            for k, n in enumerate(self._session_sizes):
                session_done = int(labeled[start : start + n].sum())
                imgui.same_line(0, 10)
                imgui.text_colored(
                    done_color if session_done == n else todo_color,
                    f"s{k}: {session_done}/{n}",
                )
                start += n
        if done < total:
            imgui.same_line(0, 12)
            if imgui.button("next unlabeled"):
                self.goto_next_unlabeled()
            imgui.same_line(0, 4)
            imgui.text_disabled("(u)")

    _KEYBINDS = (
        ("up / down", "previous / next ROI"),
        ("shift + up / down", "jump 10 ROIs"),
        ("left / right", "previous / next label group"),
        ("1-9", "assign label"),
        ("0", "clear label"),
        ("u", "jump to next unlabeled ROI"),
        ("m", "toggle mask overlay"),
        ("b", "toggle background"),
    )

    def _draw_keybinds_popup(self):
        if not self._keybinds_open:
            return
        em = imgui.get_font_size()
        imgui.set_next_window_pos(
            imgui.get_main_viewport().get_center(),
            imgui.Cond_.appearing,
            pivot=imgui.ImVec2(0.5, 0.5),
        )
        opened, self._keybinds_open = imgui.begin(
            "Keybinds###keybinds",
            self._keybinds_open,
            flags=imgui.WindowFlags_.no_saved_settings
            | imgui.WindowFlags_.always_auto_resize,
        )
        if opened:
            flags = imgui.TableFlags_.row_bg | imgui.TableFlags_.borders_inner_h
            if imgui.begin_table("##keybinds-table", 2, flags):
                imgui.table_setup_column(
                    "key", imgui.TableColumnFlags_.width_fixed, 10 * em
                )
                imgui.table_setup_column("action")
                for key, action in self._KEYBINDS:
                    imgui.table_next_row()
                    imgui.table_next_column()
                    imgui.text_colored(imgui.ImVec4(1.0, 0.85, 0.4, 1.0), key)
                    imgui.table_next_column()
                    imgui.text(action)
                imgui.end_table()
        imgui.end()

    def _draw_table(self):
        names = ("all", "unlabeled", *self._label_names)
        imgui.set_next_item_width(-1)
        changed, sel = imgui.combo("##filter", self._filter_label + 2, list(names))
        if changed:
            self._filter_label = sel - 2
            self._rebuild_order()
        imgui.set_next_item_width(-1)
        changed, lo, hi = imgui.drag_int_range2(
            "##area",
            self._area_range[0],
            self._area_range[1],
            1,
            0,
            int(self._area.max(initial=0)),
            "area >= %d",
            "area <= %d",
        )
        if changed:
            self._area_range = (lo, hi)
            self._rebuild_order()
        imgui.text(f"{len(self._order)}/{len(self._roi_images)} in view")

        flags = (
            imgui.TableFlags_.sortable
            | imgui.TableFlags_.row_bg
            | imgui.TableFlags_.resizable
            | imgui.TableFlags_.scroll_y
        )
        avail = imgui.get_content_region_avail()
        if not imgui.begin_table("rois", len(_COLUMNS), flags, imgui.ImVec2(0, avail.y)):
            return
        imgui.table_setup_scroll_freeze(0, 1)
        imgui.table_setup_column(_COLUMNS[0], imgui.TableColumnFlags_.default_sort)
        for name in _COLUMNS[1:]:
            if name == "pred":
                imgui.table_setup_column(
                    name, imgui.TableColumnFlags_.width_fixed, 7.5 * imgui.get_font_size()
                )
            else:
                imgui.table_setup_column(name)
        imgui.table_headers_row()

        specs = imgui.table_get_sort_specs()
        if specs is not None and specs.specs_dirty:
            if specs.specs_count > 0:
                self._sort_column = int(specs.specs.column_index)
                self._sort_ascending = (
                    specs.specs.sort_direction == imgui.SortDirection.ascending
                )
            specs.specs_dirty = False
            self._rebuild_order()

        clipper = imgui.ListClipper()
        clipper.begin(len(self._order))
        if self._scroll_to_current:
            clipper.include_item_by_index(self._pos)
        while clipper.step():
            for row in range(clipper.display_start, clipper.display_end):
                roi = int(self._order[row])
                label = int(self._class_labels[roi])
                imgui.table_next_row()
                imgui.table_next_column()
                clicked, _ = imgui.selectable(
                    f"{roi}##row{row}",
                    row == self._pos,
                    imgui.SelectableFlags_.span_all_columns,
                )
                if clicked:
                    self._pos = row
                    self._show_current()
                if row == self._pos and self._scroll_to_current:
                    imgui.set_scroll_here_y(0.5)
                    self._scroll_to_current = False
                imgui.table_next_column()
                if label >= 0:
                    imgui.text_colored(
                        imgui.ImVec4(*self._label_color(label), 1.0), self._label_names[label]
                    )
                else:
                    imgui.text("-")
                imgui.table_next_column()
                pred = int(self._pred[roi])
                if pred >= 0:
                    disagree = label >= 0 and pred != label
                    imgui.text_colored(
                        imgui.ImVec4(1.0, 0.4, 0.4, 1.0) if disagree else imgui.ImVec4(0.75, 0.75, 0.75, 1.0),
                        f"{self._label_names[pred]} {self._probs[roi]:.2f}",
                    )
                imgui.table_next_column()
                imgui.text(f"{self._area[roi]}")
                imgui.table_next_column()
                imgui.text(f"{self._peak[roi]:.3g}")
                imgui.table_next_column()
                imgui.text(f"{self._snr[roi]:.1f}")
                imgui.table_next_column()
                imgui.text(f"{self._skew[roi]:.2f}")
        imgui.end_table()

    @property
    def class_labels(self) -> np.ndarray:
        """(num_rois,) label indices into label_names, -1 = unlabeled"""
        return self._class_labels

    @property
    def class_labels_by_session(self) -> list[np.ndarray]:
        """
        Labels split per session — the shape RoicatDataAdapter.set_class_labels
        expects for its `labels` argument
        """
        if self._session_sizes is None:
            return [self._class_labels]
        return np.split(self._class_labels, np.cumsum(self._session_sizes)[:-1])

    @property
    def session_sizes(self) -> Optional[tuple[int, ...]]:
        """Number of ROIs per session, when loaded from demixing hdf5 files"""
        return self._session_sizes

    @property
    def roicat_input(self):
        """The RoicatDataAdapter built by load_masknmf, once loading completes"""
        return self._roicat_input

    @property
    def label_names(self) -> tuple[str, ...]:
        return self._label_names

    @property
    def roi_images(self) -> np.ndarray:
        return self._roi_images

    @property
    def figure(self) -> fpl.Figure:
        return self._figure

    def show(self):
        return self._figure.show()

    def _ipython_display_(self):
        from IPython.display import display

        display(self.show())


def main(argv=None):
    import argparse

    parser = argparse.ArgumentParser(description="Label ROI images one ROI at a time")
    parser.add_argument(
        "paths",
        nargs="+",
        help=".npy file with a (num_rois, Y, X) array, or one or more masknmf "
        "demixing_results .hdf5 files (ROI images are built with ROICaT in the background)",
    )
    parser.add_argument(
        "--labels",
        default=None,
        help="comma-separated class names, e.g. soma,dendrite,junk",
    )
    parser.add_argument(
        "--save",
        default=None,
        help="optional labels npz path; hdf5 inputs store labels in-file by default "
        "(npy default: <path>.labels.npz)",
    )
    parser.add_argument(
        "--classifier",
        default=None,
        help="classifier .roicat_classifier path: train saves here, classify loads from here",
    )
    args = parser.parse_args(argv)

    label_names = args.labels.split(",") if args.labels else ()
    if args.paths[0].endswith((".h5", ".hdf5")):
        vis = ClassificationVis.from_masknmf(
            args.paths, label_names=label_names, save_path=args.save
        )
    else:
        if len(args.paths) != 1:
            parser.error("expected exactly one .npy file")
        save_path = args.save if args.save else f"{args.paths[0]}.labels.npz"
        vis = ClassificationVis(np.load(args.paths[0]), label_names=label_names, save_path=save_path)
    if args.classifier:
        vis.classifier_path = args.classifier
    vis.show()
    fpl.loop.run()


if __name__ == "__main__":
    main()
