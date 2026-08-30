import os
from typing import Optional, Sequence

import fastplotlib as fpl
import numpy as np
from imgui_bundle import imgui

from masknmf.visualization.imgui import (
    FILTER_ALL,
    AsyncLoad,
    LabelSet,
    LabelStore,
    MoviePlayer,
    OverlayPair,
    RoiOrder,
    SummaryImageViewer,
    UNLABEL_ALL,
    UNLABELED,
    context_crop,
    crop_origin,
    draw_filter_row,
    draw_keybinds_popup,
    draw_label_buttons,
    draw_label_editor,
    draw_progress,
    draw_roi_table,
    footprint_rgba,
)
from masknmf.visualization.imgui import theme
from masknmf.visualization.imgui.store import HDF5_GROUP

_COLUMNS = ("id", "label", "area", "peak", "snr", "skew")
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


def _stack_max_mean(stack, batch: int = 32) -> tuple:
    import torch

    n = stack.shape[0]
    running_max = running_sum = None
    for start in range(0, n, batch):
        b = stack.getitem_tensor(slice(start, min(start + batch, n)))
        bmax, bsum = b.amax(dim=0), b.sum(dim=0)
        running_max = bmax if running_max is None else torch.maximum(running_max, bmax)
        running_sum = bsum if running_sum is None else running_sum + bsum
    return running_max.cpu().numpy(), (running_sum / n).cpu().numpy()


def _trace_stats(c: np.ndarray) -> tuple:
    """(snr, skew, peak_frame) per column of a (frames, rois) trace matrix."""
    med = np.median(c, axis=0)
    mad = np.median(np.abs(c - med), axis=0) * 1.4826
    snr = (c.max(axis=0) - med) / np.where(mad == 0, 1, mad)
    std = c.std(axis=0)
    skew = ((c - c.mean(axis=0)) ** 3).mean(axis=0) / np.where(std == 0, 1, std) ** 3
    return snr, skew, c.argmax(axis=0)


def _load_sessions(files: Sequence[str], **adapter_kwargs) -> dict:
    """Build the RoicatDataAdapter plus backgrounds, stats and saved labels for each file."""
    import h5py
    from masknmf.demixing.demixing_results import DemixingResults
    from masknmf.multisession.roicat_tracking import (
        RoicatDataAdapter,
        extract_masknmf_mean_img,
        extract_masknmf_spatial_footprints,
    )

    bg_sources: dict[str, list] = {}

    def add_bg(name, img):
        bg_sources.setdefault(name, []).append(np.asarray(img, dtype=np.float32))

    snr, skew, peak_frames, saved_labels = [], [], [], []
    saved_names = None
    mean_imgs, footprints, dmrs = [], [], []
    for fname in files:
        dmr = DemixingResults.from_hdf5(fname)
        dmrs.append(dmr)
        footprints.append(extract_masknmf_spatial_footprints(dmr))
        mean_imgs.append(extract_masknmf_mean_img(dmr))

        # first source added is the default background
        if dmr.residual_correlation_images is not None:
            add_bg("resid corr img (max proj)", _stack_max_mean(dmr.residual_correlation_images)[0])
        if dmr.standard_correlation_images is not None:
            smax, smean = _stack_max_mean(dmr.standard_correlation_images)
            add_bg("corr img (max proj)", smax)
            add_bg("corr img (mean)", smean)
        if dmr.global_residual_correlation_image is not None:
            add_bg("resid corr img (global)", dmr.global_residual_correlation_image.cpu().numpy())

        with h5py.File(fname, "r") as f:
            g = f[HDF5_GROUP]
            if "class_labels" in g:
                saved_labels.append(g["class_labels"][()])
            if saved_names is None and "label_names" in g:
                saved_names = [n.decode() for n in g["label_names"][()]]

        s, k, p = _trace_stats(dmr.c.cpu().numpy())
        snr.append(s)
        skew.append(k)
        peak_frames.append(p)

    adapter = RoicatDataAdapter(
        mean_imgs, footprints, tuple(os.path.abspath(f) for f in files), **adapter_kwargs
    )
    return {
        "adapter": adapter,
        # only offer a source if every session's file has it
        "bg_sources": {k: v for k, v in bg_sources.items() if len(v) == len(files)},
        "stats": {
            "snr": np.concatenate(snr).astype(np.float32),
            "skew": np.concatenate(skew).astype(np.float32),
        },
        "saved_labels": np.concatenate(saved_labels) if len(saved_labels) == len(files) else None,
        "saved_names": saved_names,
        "dmrs": dmrs,
        "peak_frames": np.concatenate(peak_frames).astype(np.int64),
    }


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
        self._store = LabelStore(npz_path=save_path)
        self._load = AsyncLoad()
        self._error: Optional[str] = None
        self._new_label = ""
        self._roicat_input = None

        saved = self._store.load(np.asarray(roi_images).shape[0])
        if saved is not None:
            if not label_names:
                label_names = saved["label_names"]
            if class_labels is None:
                class_labels = saved.get("class_labels")
        self._labels = LabelSet(0, label_names)

        self._advance_on_label = True
        self._keybinds_open = False
        self._scroll_to_current = True
        self._dmrs: Optional[list] = None
        self._peak_frames: Optional[np.ndarray] = None
        self._movie_player = MoviePlayer()
        self._bg_movie = False
        self._movie_range: Optional[tuple] = None

        self._figure = fpl.Figure(size=(1200, 900))
        self._summary = SummaryImageViewer(self._figure)
        self._overlay: Optional[OverlayPair] = None
        self.set_roi_images(roi_images, class_labels)

        self._figure.add_imgui_window(
            self._draw_panel, location="top", size=150, title="Classification"
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
        self._store.session_sizes = self._session_sizes

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

        self._labels = LabelSet(num_rois, self._labels.names, class_labels)

        flat = roi_images.reshape(num_rois, -1)
        self._peak = flat.max(axis=1)
        self._area = np.count_nonzero(flat, axis=1).astype(np.int64)
        self._mip = roi_images.max(axis=0)

        self._order = RoiOrder(
            {"area": self._area, "peak": self._peak, "snr": self._snr, "skew": self._skew},
            self._labels.labels,
            num_rois,
        )
        self._order.set_range_column("area")
        self._formatters = {
            "area": lambda i: f"{self._area[i]}",
            "peak": lambda i: f"{self._peak[i]:.3g}",
            "snr": lambda i: f"{self._snr[i]:.1f}",
            "skew": lambda i: f"{self._skew[i]:.2f}",
        }

        subplot = self._figure[0, 0]
        if self._overlay is not None:
            self._overlay.remove()
        self._overlay = OverlayPair(subplot, roi_images.shape[1:])
        self._overlay.set_background(self._mip, (0.0, float(self._mip.max())))
        subplot.auto_scale()
        self._overlay.apply()
        self._show_current()

    def load_masknmf(self, demixing_result_files: Sequence[str], **adapter_kwargs):
        """Build a RoicatDataAdapter from demixing .hdf5 files in a background thread"""
        files = [str(f) for f in demixing_result_files]
        self._store.hdf5_files = files
        self._load.start(
            lambda: _load_sessions(files, **adapter_kwargs),
            f"ROICaT: building ROI images from {len(files)} session(s)...",
        )

    def _poll_load(self):
        if not self._load.busy:
            return
        result = self._load.poll()
        if self._load.error is not None:
            self._error = f"load failed: {self._load.error}"
            return
        if result is None:
            return
        adapter = result["adapter"]
        imgs = np.concatenate([np.asarray(s) for s in adapter.ROI_images], axis=0)
        labels = result["saved_labels"]
        if labels is not None and labels.shape[0] != imgs.shape[0]:
            labels = None
        if not self._labels.names and result["saved_names"]:
            self._labels = LabelSet(0, result["saved_names"])
        self.set_roi_images(
            imgs,
            labels,
            session_sizes=[len(s) for s in adapter.ROI_images],
            fov_images=adapter.FOV_images,
            centroids=adapter.centroids,
            bg_sources=result["bg_sources"],
            roi_stats=result["stats"],
        )
        self._roicat_input = adapter
        self._dmrs = result["dmrs"]
        self._peak_frames = result["peak_frames"]
        if not self._store.save(self._labels.names, self._labels.labels, masks=self._roi_images):
            self._error = self._store.error

    @property
    def current(self) -> Optional[int]:
        """ROI id at the current position in the sorted/filtered view"""
        return self._order.current

    def step(self, delta: int):
        """Move through the current view by delta ROIs"""
        if self._order.step(delta):
            self._show_current()

    def goto(self, roi: int):
        """Jump to an ROI if it is in the current view"""
        if self._order.goto(roi):
            self._show_current()

    def goto_next_unlabeled(self):
        """Jump to the next unlabeled ROI in the current view (wraps around)"""
        if self._order.next_unlabeled():
            self._show_current()

    def label(self, roi_ids: Sequence[int], label_index: int):
        """Assign a class label to the given ROIs; -1 clears"""
        self._labels.assign(roi_ids, label_index)
        self._autosave()
        if self._order.filter_label != FILTER_ALL:
            self._order.rebuild()
        self._show_current()

    def label_current(self, label_index: int):
        """Label the current ROI and advance to the next one in view"""
        if self.current is None:
            return
        size_before = len(self._order.order)
        self.label([self.current], label_index)
        # if the filter dropped the labeled ROI, pos already sits on its successor
        if self._advance_on_label and len(self._order.order) == size_before:
            self.step(1)

    def add_label(self, name: str):
        """Add a new class name to the label set"""
        if self._labels.add(name):
            self._autosave()

    def remove_label(self, index: int):
        """Delete a class: its ROIs become unlabeled, higher labels shift down"""
        names = self._labels.names
        if self._labels.remove(index):
            self._names_changed(names)

    def _names_changed(self, names_before: tuple):
        # keep the table filter on the same class name, or drop it if that class is gone
        f = self._order.filter_label
        if f >= 0:
            names = self._labels.names
            self._order.filter_label = (
                names.index(names_before[f]) if names_before[f] in names else FILTER_ALL
            )
        self._autosave()
        self._order.rebuild()
        self._show_current()

    def save(self, path: Optional[str] = None):
        """
        Persist labels: into each session's hdf5 (DemixingResults/class_labels +
        label_names) when launched from demixing results, and/or to an npz at path
        """
        store = self._store
        if path is not None:
            store = LabelStore(npz_path=path, session_sizes=self._session_sizes)
        if not store.save(self._labels.names, self._labels.labels):
            raise OSError(store.error)

    def _autosave(self):
        # don't clobber a previous session's labels with placeholder state mid-load
        if not self._store.enabled or self._load.busy:
            return
        self._store.save(self._labels.names, self._labels.labels)
        self._error = self._store.error

    def _crop_origin(self, roi: int) -> tuple[int, int]:
        return crop_origin(self._centroids[roi], self._roi_images.shape[1:])

    def _context_crop(self, roi: int) -> np.ndarray:
        return context_crop(
            self._fov_images[self._session_of[roi]],
            self._centroids[roi],
            self._roi_images.shape[1:],
        )

    def _update_movie_bg(self):
        roi = self.current
        if roi is None or self._movie_player.movie is None:
            return
        top, left = self._crop_origin(roi)
        crop = self._movie_player.frame((top, left, *self._roi_images.shape[1:]))
        if self._movie_range is None:
            # fixed at the ROI's peak frame so playback doesn't flicker
            self._movie_range = (float(crop.min()), float(crop.max()))
        self._overlay.set_background(crop, self._movie_range)

    def _show_current(self):
        roi = self.current
        rgba = np.zeros((*self._roi_images.shape[1:], 4), dtype=np.float32)
        title = "no ROIs in view"
        if roi is not None:
            label = int(self._labels.labels[roi])
            color = self._labels.color(label) if label >= 0 else (1.0, 1.0, 1.0)
            rgba = footprint_rgba(self._roi_images[roi], color, self._peak[roi])
            title = f"ROI {roi}  [{self._labels.name_of(roi)}]  ({self._order.pos + 1}/{len(self._order.order)})"
            if self._fov_images is not None:
                if self._bg_movie and self._dmrs is not None:
                    sess = int(self._session_of[roi])
                    self._movie_player.set_movie(self._dmrs[sess].ac_array)
                    if self._peak_frames is not None:
                        self._movie_player.jump_to(int(self._peak_frames[roi]))
                    self._movie_range = None
                    self._update_movie_bg()
                else:
                    self._overlay.set_background(self._context_crop(roi))
                top, left = self._crop_origin(roi)
                self._summary.set_highlight((top, left, *self._roi_images.shape[1:]))
                if self._summary.is_open and self._bg_sources is not None:
                    sess = int(self._session_of[roi])
                    self._summary.set_images(
                        {name: imgs[sess] for name, imgs in self._bg_sources.items()}
                    )
                    if self._dmrs is not None:
                        self._summary.set_movies({"demixed movie": self._dmrs[sess].ac_array})
        if roi is None:
            self._summary.set_highlight(None)
        self._overlay.set_overlay(rgba)
        self._figure[0, 0].title = title
        self._scroll_to_current = True

    def _step_group(self, direction: int):
        if self._order.step_group(direction):
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
            self._overlay.show_bg = not self._overlay.show_bg
            self._overlay.apply()
        if imgui.is_key_pressed(imgui.Key.m, False):
            self._overlay.show_fg = not self._overlay.show_fg
            self._overlay.apply()
        if imgui.is_key_pressed(imgui.Key.u, False):
            self.goto_next_unlabeled()
        picked = self._labels.hotkey_pressed()
        if picked is not None:
            self.label_current(picked)

    def _open_full_fov(self):
        if self._bg_sources is not None:
            roi = self.current
            sess = int(self._session_of[roi]) if roi is not None and self._session_of is not None else 0
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

    def _draw_save_note(self):
        if not self._store.enabled:
            imgui.text_disabled("autosave off — labels are kept in memory only")
            return
        imgui.text("Accessing masks in output file")
        imgui.same_line(0, 4)
        imgui.text_disabled("(?)")
        if imgui.is_item_hovered():
            imgui.begin_tooltip()
            if self._store.hdf5_files is not None:
                imgui.text_colored(
                    theme.CODE,
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
                    theme.CODE,
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
        self._handle_keys()
        overlay = self._overlay

        if self._load.busy:
            imgui.text_colored(theme.WARN, self._load.status)

        if imgui.button("prev"):
            self.step(-1)
        imgui.same_line(0, 5)
        if imgui.button("next"):
            self.step(1)
        imgui.same_line(0, 10)
        imgui.set_next_item_width(200)
        changed, pos = imgui.slider_int("##pos", self._order.pos, 0, max(len(self._order.order) - 1, 0))
        if changed and len(self._order.order):
            self._order.pos = pos
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

        changed_bg, overlay.show_bg = imgui.checkbox("##bg-on", overlay.show_bg)
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
        changed_bga, overlay.bg_alpha = imgui.slider_float("bg opacity", overlay.bg_alpha, 0.0, 1.0)
        imgui.same_line(0, 30)
        changed_mask, overlay.show_fg = imgui.checkbox("mask", overlay.show_fg)
        imgui.same_line(0, 4)
        imgui.text_disabled("(m)")
        imgui.same_line(0, 20)
        imgui.set_next_item_width(75)
        changed_fga, overlay.fg_alpha = imgui.slider_float("roi opacity", overlay.fg_alpha, 0.0, 1.0)
        if changed_bg or changed_bga or changed_mask or changed_fga:
            overlay.apply()
        if self._bg_movie and self._movie_player.draw():
            self._update_movie_bg()

        if draw_progress(self._labels.labels, self._session_sizes):
            self.goto_next_unlabeled()
        imgui.same_line(0, 24)
        names = self._labels.names
        self._new_label, changed = draw_label_editor(self._labels, self._new_label)
        if changed:
            self._names_changed(names)
        picked = draw_label_buttons(self._labels)
        if picked == UNLABEL_ALL:
            self.label(range(len(self._labels.labels)), UNLABELED)
        elif picked is not None:
            self.label_current(picked)

        self._summary.draw()
        self._keybinds_open = draw_keybinds_popup(_KEYBINDS, self._keybinds_open)

    def _draw_table(self):
        if draw_filter_row(self._order, self._labels):
            self._show_current()
        pos = self._order.pos
        self._scroll_to_current = draw_roi_table(
            self._order, self._labels, _COLUMNS, self._formatters, self._scroll_to_current
        )
        if self._order.pos != pos:
            self._show_current()

    @property
    def class_labels(self) -> np.ndarray:
        """(num_rois,) label indices into label_names, -1 = unlabeled"""
        return self._labels.labels

    @property
    def class_labels_by_session(self) -> list[np.ndarray]:
        """
        Labels split per session — the shape RoicatDataAdapter.set_class_labels
        expects for its `labels` argument
        """
        return self._store.split(self._labels.labels)

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
        return self._labels.names

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
    vis.show()
    fpl.loop.run()


if __name__ == "__main__":
    main()
