from typing import *
import os
import threading
import numpy as np
import fastplotlib as fpl
from imgui_bundle import imgui

try:
    from imgui_data_loader import ButtonSpec, FileDialog, FileDialogConfig, FileType, PickKind
except ImportError:
    FileDialog = None

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
_COLUMNS = ("id", "label", "area", "peak", "snr", "skew")


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

        if FileDialog is not None:
            self._label_dialog = FileDialog(FileDialogConfig(close_on_select=False))
            self._label_open_spec = ButtonSpec(
                "load labels",
                PickKind.OPEN_FILE,
                filetypes=[
                    FileType("Label sets", "*.json *.txt *.npz *.h5 *.hdf5"),
                    FileType("All Files", "*"),
                ],
            )
        else:
            self._label_dialog = None

        self._show_bg = True
        self._show_mask = True
        self._bg_alpha = 0.5
        self._roi_alpha = 1.0
        self._advance_on_label = True

        self._figure = fpl.Figure(size=(1200, 900))
        self._bg = None
        self._fg = None
        self.set_roi_images(roi_images, class_labels)

        self._figure.add_imgui_window(
            self._draw_panel, location="top", size=120, title="Classification"
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
        then swaps in the real data. Labels are kept per session so they can be fed
        back with roicat_input.set_class_labels(labels=vis.class_labels_by_session).
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

        if fov_images is not None and centroids is not None:
            self._fov_images = [np.asarray(f, dtype=np.float32) for f in fov_images]
            self._centroids = np.concatenate([np.asarray(c) for c in centroids], axis=0)
            if self._centroids.shape[0] != num_rois:
                raise ValueError(
                    f"centroids cover {self._centroids.shape[0]} ROIs, expected {num_rois}"
                )
            sizes = self._session_sizes if self._session_sizes is not None else (num_rois,)
            self._session_of = np.repeat(np.arange(len(sizes)), sizes)
            self._bg_sources = {"enhanced mean": self._fov_images}
            for name, imgs_ in (bg_sources or {}).items():
                self._bg_sources[name] = [np.asarray(i, dtype=np.float32) for i in imgs_]
            self._bg_source_names = list(self._bg_sources)
            self._bg_source_idx = 0
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
        self._loading = f"ROICaT: building ROI images from {len(files)} session(s)..."
        self._load_result = None
        self._load_error = None

        def work():
            try:
                import h5py
                from masknmf.multisession import RoicatDataAdapter

                adapter = RoicatDataAdapter.from_masknmf(files, **adapter_kwargs)

                extra_imgs = {
                    "mean": "DemixingResults/mean_img",
                    "variance": "DemixingResults/var_img",
                    "resid corr": "DemixingResults/global_residual_correlation_image",
                }
                bg_sources: dict[str, list] = {}
                snr, skew = [], []
                for fname in files:
                    with h5py.File(fname, "r") as f:
                        for label, key in extra_imgs.items():
                            if key in f:
                                bg_sources.setdefault(label, []).append(
                                    f[key][()].astype(np.float32)
                                )
                        c = f["DemixingResults/c"][()]  # (num_frames, num_rois)
                    med = np.median(c, axis=0)
                    mad = np.median(np.abs(c - med), axis=0) * 1.4826
                    snr.append((c.max(axis=0) - med) / np.where(mad == 0, 1, mad))
                    mean = c.mean(axis=0)
                    std = c.std(axis=0)
                    skew.append(((c - mean) ** 3).mean(axis=0) / np.where(std == 0, 1, std) ** 3)

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
            labels = None
            if self._save_path is not None and os.path.exists(self._save_path):
                saved_labels = np.load(self._save_path)["class_labels"]
                if saved_labels.shape[0] == imgs.shape[0]:
                    labels = saved_labels
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
            self._loading = None
            if self._masks_path is not None:
                try:
                    np.save(self._masks_path, self._roi_images)
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

    def label(self, roi_ids: Sequence[int], label_index: int):
        """Assign a class label to the given ROIs; -1 clears"""
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

    def load_label_names(self, path: str):
        """
        Merge a label-name set from a file: .json (list of names), .npz (a previous
        save), .h5/.hdf5 (DemixingResults/label_names), or plain text
        (comma/newline separated)
        """
        path = str(path)
        if path.endswith(".json"):
            import json

            with open(path) as f:
                names = json.load(f)
        elif path.endswith(".npz"):
            names = [str(n) for n in np.load(path)["label_names"]]
        elif path.endswith((".h5", ".hdf5")):
            import h5py

            with h5py.File(path, "r") as f:
                names = [n.decode() for n in f["DemixingResults/label_names"][()]]
        else:
            with open(path) as f:
                names = [t.strip() for t in f.read().replace(",", "\n").splitlines()]
        for name in names:
            self.add_label(str(name).strip())

    def save(self, path: Optional[str] = None):
        """Write label names and per-ROI labels to an npz file"""
        path = path if path is not None else self._save_path
        data = dict(label_names=np.array(self._label_names), class_labels=self._class_labels)
        if self._session_sizes is not None:
            data["session_sizes"] = np.array(self._session_sizes)
        np.savez(path, **data)

    def _autosave(self):
        # don't clobber a previous session's labels with placeholder state mid-load
        if self._save_path is None or self._loading is not None:
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
            keys = (self._class_labels, self._area, self._peak, self._snr, self._skew)
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

    def _context_crop(self, roi: int) -> np.ndarray:
        """FOV image cropped around the ROI, aligned with roicat's centered ROI images"""
        fov = self._fov_images[self._session_of[roi]]
        h, w = self._roi_images.shape[1:]
        # roicat places out pixel (0, 0) at FOV coordinate centroid - ceil(size / 2)
        cy, cx = self._centroids[roi]
        top = int(cy) - int(np.ceil(h / 2))
        left = int(cx) - int(np.ceil(w / 2))
        crop = np.zeros((h, w), dtype=np.float32)
        y0, y1 = max(top, 0), min(top + h, fov.shape[0])
        x0, x1 = max(left, 0), min(left + w, fov.shape[1])
        if y1 > y0 and x1 > x0:
            crop[y0 - top : y1 - top, x0 - left : x1 - left] = fov[y0:y1, x0:x1]
        return crop

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
                crop = self._context_crop(roi)
                self._bg.data = crop
                self._bg.vmin = float(crop.min())
                self._bg.vmax = float(crop.max()) or 1.0
        self._fg.data = rgba
        self._figure[0, 0].title = title
        self._scroll_to_current = True

    def _apply_overlay(self):
        self._bg.visible = self._show_bg
        self._bg.alpha = self._bg_alpha
        self._fg.visible = self._show_mask
        self._fg.alpha = self._roi_alpha

    def _handle_keys(self):
        if imgui.get_io().want_text_input:
            return
        if imgui.is_key_pressed(imgui.Key.left_arrow):
            self.step(-1)
        if imgui.is_key_pressed(imgui.Key.right_arrow) or imgui.is_key_pressed(imgui.Key.space):
            self.step(1)
        if imgui.is_key_pressed(imgui.Key._0, False):
            self.label_current(-1)
        if imgui.is_key_pressed(imgui.Key.m, False):
            self._show_mask = not self._show_mask
            self._apply_overlay()
        for i, key in enumerate(_LABEL_KEYS[: len(self._label_names)]):
            if imgui.is_key_pressed(key, False):
                self.label_current(i)

    def _draw_save_note(self):
        if self._save_path is None:
            imgui.text_disabled("autosave off — labels are kept in memory only")
            return
        imgui.text_disabled("(?)")
        if imgui.is_item_hovered():
            imgui.begin_tooltip()
            imgui.push_text_wrap_pos(560)
            imgui.text(
                "The source data file is opened read-only and is never modified by "
                "this tool. Labels are written to the .npz file on every change; the "
                "ROI mask stack is written once per launch."
            )
            imgui.separator()
            imgui.text("Access the results in Python:")
            imgui.text_colored(
                imgui.ImVec4(0.55, 0.75, 1.0, 1.0),
                "\n"
                "import numpy as np\n"
                "\n"
                f'data = np.load(r"{self._save_path}")\n'
                'names  = data["label_names"]   # class names; row index = label value\n'
                'labels = data["class_labels"]  # (num_rois,) int64; -1 = unlabeled\n'
                f'masks  = np.load(r"{self._masks_path}")  # (num_rois, Y, X)\n'
                "\n"
                "# continue a ROICaT classification pipeline\n"
                'sizes = data["session_sizes"]  # ROIs per session\n'
                "per_session = np.split(labels, np.cumsum(sizes)[:-1])\n"
                "roicat_input.set_class_labels(labels=per_session)\n",
            )
            imgui.separator()
            imgui.text(
                "Shortcuts: left/right or space = step, 1-9 = assign label, "
                "0 = clear label, m = toggle mask overlay."
            )
            imgui.pop_text_wrap_pos()
            imgui.end_tooltip()
        imgui.same_line(0, 5)
        imgui.text(f"source data unchanged; labels & masks -> {self._save_path}")

    def _draw_panel(self):
        self._poll_load()
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
        _, self._advance_on_label = imgui.checkbox("advance on label", self._advance_on_label)
        imgui.same_line(0, 30)
        self._draw_save_note()
        if self._error is not None:
            imgui.same_line(0, 30)
            imgui.text(self._error)

        bg_name = "FOV background" if self._fov_images is not None else "MIP background"
        changed_bg, self._show_bg = imgui.checkbox(bg_name, self._show_bg)
        imgui.same_line(0, 20)
        imgui.set_next_item_width(150)
        changed_bga, self._bg_alpha = imgui.slider_float(
            "bg opacity", self._bg_alpha, 0.0, 1.0
        )
        if self._bg_sources is not None and len(self._bg_source_names) > 1:
            imgui.same_line(0, 20)
            imgui.set_next_item_width(150)
            changed_src, idx = imgui.combo(
                "##bg-source", self._bg_source_idx, self._bg_source_names
            )
            if changed_src:
                self._bg_source_idx = idx
                self._fov_images = self._bg_sources[self._bg_source_names[idx]]
                self._show_current()
        imgui.same_line(0, 30)
        changed_mask, self._show_mask = imgui.checkbox("m:mask", self._show_mask)
        imgui.same_line(0, 20)
        imgui.set_next_item_width(150)
        changed_fga, self._roi_alpha = imgui.slider_float(
            "roi opacity", self._roi_alpha, 0.0, 1.0
        )
        if changed_bg or changed_bga or changed_mask or changed_fga:
            self._apply_overlay()

        if self._label_dialog is not None:
            if imgui.button("load labels"):
                self._label_dialog.pick(self._label_open_spec)
            self._label_dialog.poll()
            result = self._label_dialog.take_result()
            if result:
                try:
                    self.load_label_names(result.path)
                    self._error = None
                except (OSError, KeyError, ValueError) as e:
                    self._error = f"label load failed: {e}"
            imgui.same_line(0, 10)
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

        for i, name in enumerate(self._label_names):
            imgui.same_line(0, 10)
            count = int((self._class_labels == i).sum())
            key_hint = f"{i + 1}:" if i < len(_LABEL_KEYS) else ""
            imgui.push_style_color(imgui.Col_.button, imgui.ImVec4(*self._label_color(i), 0.5))
            if imgui.button(f"{key_hint}{name} ({count})"):
                self.label_current(i)
            imgui.pop_style_color()
        if self._label_names:
            imgui.same_line(0, 10)
            if imgui.button("0:unlabel"):
                self.label_current(-1)

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
    def _masks_path(self) -> Optional[str]:
        """Sidecar .npy next to the labels npz holding the ROI mask stack"""
        if self._save_path is None:
            return None
        base = self._save_path[:-4] if self._save_path.endswith(".npz") else self._save_path
        return f"{base}.masks.npy"

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
        help="labels npz path (default: <first path>.labels.npz)",
    )
    args = parser.parse_args(argv)

    label_names = args.labels.split(",") if args.labels else ()
    save_path = args.save if args.save else f"{args.paths[0]}.labels.npz"
    if args.paths[0].endswith((".h5", ".hdf5")):
        vis = ClassificationVis.from_masknmf(
            args.paths, label_names=label_names, save_path=save_path
        )
    else:
        if len(args.paths) != 1:
            parser.error("expected exactly one .npy file")
        vis = ClassificationVis(np.load(args.paths[0]), label_names=label_names, save_path=save_path)
    vis.show()
    fpl.loop.run()


if __name__ == "__main__":
    main()
