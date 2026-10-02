"""
SummaryImageViewer - floating imgui popup for browsing full-FOV summary images
(mean/variance/correlation images) with pan, zoom, colormap, contrast control,
pixel-value overlay, and a hover readout. Adapted from mbo_utilities'
summary_image widget; renders through fastplotlib's wgpu imgui backend.
"""

from typing import *
import numpy as np
import wgpu
import fastplotlib as fpl
from cmap import Colormap
from imgui_bundle import imgui

from masknmf.visualization.imgui.layout import is_notebook_canvas
from masknmf.visualization.imgui.movie_player import MoviePlayer
from masknmf.visualization.imgui.options import OPTIONS
from masknmf.visualization.imgui.theme import opaque_popups

_CMAPS = ("gray", "viridis", "magma", "inferno", "turbo")
_CONTRAST_MODES = ("full", "auto", "manual")
_CONTRAST_AUTO = 1
_CONTRAST_MANUAL = 2

_PIXEL_VALUES_MIN_ZOOM = 16.0
_PIXEL_VALUES_MAX_CELLS = 10_000
# screen pixels between the images of a row
_ROW_GAP = 2.0


def _data_range(arr: np.ndarray) -> tuple[float, float]:
    a = np.asarray(arr, dtype=np.float32)
    lo = float(np.nanmin(a))
    hi = float(np.nanmax(a))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        hi = lo + 1.0
    return lo, hi


def _auto_range(arr: np.ndarray) -> tuple[float, float]:
    a = np.asarray(arr, dtype=np.float32)
    lo = float(np.nanpercentile(a, 1.0))
    hi = float(np.nanpercentile(a, 99.0))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        hi = lo + 1.0
    return lo, hi


def _to_rgba(arr: np.ndarray, cmap_name: str, lo: float, hi: float) -> np.ndarray:
    a = np.asarray(arr, dtype=np.float32)
    span = max(hi - lo, 1e-12)
    n = np.clip((a - lo) / span, 0.0, 1.0)
    # an (H, W, 3) image keeps its own colors: contrast applies, the colormap does not
    if n.ndim == 3:
        n = np.concatenate([n, np.ones_like(n[..., :1])], axis=2)
        return np.ascontiguousarray((n * 255).astype(np.uint8))
    rgba = (Colormap(cmap_name)(n) * 255).astype(np.uint8)
    return np.ascontiguousarray(rgba)


def _blend(rgba: np.ndarray, overlay: Optional[np.ndarray]) -> np.ndarray:
    """Alpha-composite a float (H, W, 4) overlay onto a uint8 RGBA image"""
    if overlay is None or overlay.shape[:2] != rgba.shape[:2]:
        return rgba
    alpha = overlay[..., 3:4]
    base = rgba[..., :3].astype(np.float32) / 255.0
    rgba[..., :3] = ((base * (1.0 - alpha) + overlay[..., :3] * alpha) * 255).astype(np.uint8)
    return rgba


def _format_value(v: float, dtype) -> str:
    if np.issubdtype(dtype, np.integer):
        return f"{int(v)}"
    av = abs(v)
    if av != 0 and (av < 0.01 or av >= 10000):
        return f"{v:.1e}"
    return f"{v:.2f}"


class _GpuImage:
    """Owns one wgpu texture + its imgui registration for a single image."""

    def __init__(self, backend, arr: np.ndarray, cmap: str, lo: float, hi: float, overlay=None):
        self.backend = backend
        self.arr = arr
        self.cmap = cmap
        self.lo = lo
        self.hi = hi
        self.overlay = overlay
        self.h, self.w = arr.shape[:2]
        self._texture = None
        self._view = None
        self.ref = None
        self.rgba: Optional[np.ndarray] = None
        self._upload()

    def _upload(self):
        self.rgba = _blend(_to_rgba(self.arr, self.cmap, self.lo, self.hi), self.overlay)
        device = self.backend._device
        self._texture = device.create_texture(
            size=(self.w, self.h, 1),
            format=wgpu.TextureFormat.rgba8unorm,
            usage=wgpu.TextureUsage.TEXTURE_BINDING | wgpu.TextureUsage.COPY_DST,
        )
        device.queue.write_texture(
            {"texture": self._texture, "mip_level": 0, "origin": (0, 0, 0)},
            self.rgba.tobytes(),
            {"offset": 0, "bytes_per_row": self.w * 4, "rows_per_image": self.h},
            (self.w, self.h, 1),
        )
        self._view = self._texture.create_view()
        self.ref = self.backend.register_texture(self._view)

    def ensure(self, arr: np.ndarray, cmap: str, lo: float, hi: float, overlay=None):
        if (
            arr is self.arr
            and cmap == self.cmap
            and lo == self.lo
            and hi == self.hi
            and overlay is self.overlay
        ):
            return
        same_shape = arr.shape[:2] == (self.h, self.w)
        self.arr = arr
        self.cmap = cmap
        self.lo = lo
        self.hi = hi
        self.overlay = overlay
        if same_shape and self._texture is not None:
            # rewrite pixels in place (movie frames, contrast changes)
            self.rgba = _blend(_to_rgba(arr, cmap, lo, hi), overlay)
            self.backend._device.queue.write_texture(
                {"texture": self._texture, "mip_level": 0, "origin": (0, 0, 0)},
                self.rgba.tobytes(),
                {"offset": 0, "bytes_per_row": self.w * 4, "rows_per_image": self.h},
                (self.w, self.h, 1),
            )
        else:
            self.destroy()
            self.h, self.w = arr.shape[:2]
            self._upload()

    def destroy(self):
        if self.ref is not None:
            try:
                self.backend.unregister_texture(self.ref)
            except Exception:
                pass
            self.ref = None
        self._view = None
        if self._texture is not None:
            try:
                self._texture.destroy()
            except Exception:
                pass
            self._texture = None


class SummaryImageViewer:
    """
    Viewer over a {name: 2D or (H, W, 3) rgb array} image set. Call open() to show and
    draw() every imgui frame of ``figure`` (it is a no-op while closed). It is a popup inside ``figure``, or
    with File > Options' separate window on, an OS window of its own; a notebook always gets the popup.

    A name may hold a row instead, a list of (caption, image) pairs of one shape: its images are drawn side by
    side, ``_ROW_GAP`` pixels apart, under one zoom and pan. The window opens in the row's shape and resizes
    freely: the row is refitted to it, whole and centered, so resizing never cuts an image off.
    """

    def __init__(self, figure, images: Optional[dict] = None, title: str = "Full FOV"):
        self._figure = figure
        self._window: Optional[fpl.Figure] = None  # the figure of its own, while it is a separate window
        figure.canvas.add_event_handler(self._on_figure_close, "close")
        self._title = title
        self._images: dict = images or {}
        self._movies: dict = {}
        self._movie_frame: Optional[np.ndarray] = None
        self._movie_key: Optional[str] = None
        self._movie_range: dict = {}
        self.player = MoviePlayer()
        self._selected = 0
        self._popup_open = False
        self._cmap_idx = 0
        self._contrast_mode = _CONTRAST_AUTO
        self._zoom = 1.0
        self._pan_x = 0.0
        self._pan_y = 0.0
        self._needs_fit = True
        self._cell_height: Optional[float] = None  # how tall a row's images were on screen last frame
        self._show_pixel_values = False
        self._highlight: Optional[tuple] = None  # (y0, x0, h, w) in image coords
        self._overlay: Optional[np.ndarray] = None  # float (H, W, 4) drawn over the image
        self._index: Optional[int] = None  # which plane of a stacked image to show
        self._planes: dict = {}  # (stack, index, plane) last read for each source
        self._auto_cache: dict = {}
        self._gpu: dict = {}
        self._manual_lo: dict = {}
        self._manual_hi: dict = {}
        self._hist_cache: dict = {}

    def set_images(self, images: dict, selected: Optional[str] = None, index: Optional[int] = None):
        """
        Replace the image set; caches are dropped for images that changed.

        A value may be a (N, H, W) stack instead of an image, in which case
        ``index`` picks the plane. Only the plane actually on screen is read, so
        a stack that computes its planes on demand costs nothing until shown.
        A value may also be a row: a list of (caption, image) pairs drawn side by side.
        """
        for (key, column), gpu in list(self._gpu.items()):
            value = images.get(key)
            if isinstance(value, list):
                value = value[column][1] if column < len(value) else None
            if value is not gpu.arr:
                gpu.destroy()
                del self._gpu[key, column]
                self._hist_cache.pop(key, None)
        self._images = dict(images)
        self._index = index
        keys = list(self._images) + list(self._movies)
        if selected in keys:
            self._selected = keys.index(selected)
        elif self._selected >= len(keys):
            self._selected = 0

    def set_movies(self, movies: dict):
        """Lazy (T, H, W) arrays offered in the selector after the static images"""
        self._movies = dict(movies)
        self._movie_frame = None
        self._movie_key = None
        self._movie_range.clear()

    def open(self):
        self._popup_open = True
        self._reset_view()

    @property
    def is_open(self) -> bool:
        return self._popup_open and (self._window is None or not self._window.canvas.get_closed())

    def _columns(self, key: str) -> list:
        """What a source shows, as (caption, image) pairs: a row's own, or its one image without a caption"""
        value = self._images[key]
        if isinstance(value, list):
            return value
        return [("", self._image(key))]

    def _image(self, key: str) -> np.ndarray:
        """The image for a source, reading the plane out of a stack the first time it is shown"""
        arr = self._images[key]
        if getattr(arr, "ndim", 2) != 3 or arr.shape[-1] == 3:
            return arr
        i = 0 if self._index is None else int(self._index)
        cached = self._planes.get(key)
        if cached is None or cached[0] is not arr or cached[1] != i:
            cached = (arr, i, np.asarray(arr[i], dtype=np.float32).reshape(arr.shape[1:]))
            self._planes[key] = cached
        return cached[2]

    def set_overlay(self, overlay: Optional[np.ndarray]):
        """Composite a float (H, W, 4) RGBA layer over the image, e.g. ROI masks"""
        self._overlay = overlay

    def set_highlight(self, rect: Optional[tuple]):
        """Outline a region of the image: (y0, x0, height, width), or None"""
        self._highlight = rect

    def _backend(self):
        # textures are registered with the imgui backend of the figure that draws them
        figure = self._figure if self._window is None else self._window
        try:
            return figure.imgui_renderer.backend
        except AttributeError:
            return None

    def _reset_view(self):
        self._zoom = 1.0
        self._pan_x = 0.0
        self._pan_y = 0.0
        self._needs_fit = True

    def _get_range(self, key: str, arr: np.ndarray, column: int = 0) -> tuple[float, float]:
        if self._contrast_mode == _CONTRAST_AUTO:
            if key in self._movies:
                # fixed per movie so playback doesn't flicker
                if key not in self._movie_range:
                    self._movie_range[key] = _auto_range(arr)
                return self._movie_range[key]
            # each image of a row has its own auto range; the manual one below is the row's
            cached = self._auto_cache.get((key, column))
            if cached is None or cached[0] is not arr:
                cached = (arr, _auto_range(arr))
                self._auto_cache[key, column] = cached
            return cached[1]
        if self._contrast_mode == _CONTRAST_MANUAL:
            lo = self._manual_lo.get(key)
            hi = self._manual_hi.get(key)
            if lo is None or hi is None:
                lo, hi = _auto_range(arr)
                self._manual_lo[key] = lo
                self._manual_hi[key] = hi
            if hi <= lo:
                hi = lo + 1e-6
            return lo, hi
        return _data_range(arr)

    def _ensure_gpu(self, key: str, arr: np.ndarray, column: int = 0) -> Optional[_GpuImage]:
        backend = self._backend()
        if backend is None:
            return None
        cmap = _CMAPS[self._cmap_idx]
        lo, hi = self._get_range(key, arr, column)
        gpu = self._gpu.get((key, column))
        if gpu is None:
            gpu = _GpuImage(backend, arr, cmap, lo, hi, self._overlay)
            self._gpu[key, column] = gpu
        else:
            gpu.ensure(arr, cmap, lo, hi, self._overlay)
        return gpu

    def _get_histogram(self, key: str, arr: np.ndarray) -> np.ndarray:
        h = self._hist_cache.get(key)
        if h is not None:
            return h
        a = np.asarray(arr, dtype=np.float32)
        finite = a[np.isfinite(a)]
        if finite.size == 0:
            h = np.zeros(128, dtype=np.float32)
        else:
            counts, _ = np.histogram(finite, bins=128)
            h = counts.astype(np.float32)
        self._hist_cache[key] = h
        return h

    def _draw_toolbar(self, keys: list) -> str:
        # as wide as the longest name, so none is cut off in the preview
        longest = max((imgui.calc_text_size(str(k)).x for k in keys), default=0.0)
        imgui.set_next_item_width(max(180, longest + imgui.get_frame_height() + 2 * imgui.get_style().frame_padding.x))
        changed, idx = imgui.combo("image", self._selected, list(keys))
        if changed:
            self._selected = idx
            self._reset_view()
        imgui.same_line()
        imgui.set_next_item_width(100)
        _, self._cmap_idx = imgui.combo("cmap", self._cmap_idx, list(_CMAPS))
        imgui.same_line()
        imgui.set_next_item_width(100)
        _, self._contrast_mode = imgui.combo(
            "contrast", self._contrast_mode, list(_CONTRAST_MODES)
        )
        imgui.same_line()
        if imgui.button("reset"):
            self._reset_view()
        imgui.same_line()
        _, self._show_pixel_values = imgui.checkbox(
            "pixel values", self._show_pixel_values
        )
        return keys[self._selected]

    def _draw_contrast_panel(self, key: str, arr: np.ndarray):
        if self._contrast_mode != _CONTRAST_MANUAL:
            return
        data_lo, data_hi = _data_range(arr)
        lo = self._manual_lo.get(key)
        hi = self._manual_hi.get(key)
        if lo is None or hi is None:
            lo, hi = _auto_range(arr)
            self._manual_lo[key] = lo
            self._manual_hi[key] = hi

        bins = self._get_histogram(key, arr)
        if imgui.begin_child(
            "##levels", imgui.ImVec2(0, 32), child_flags=imgui.ChildFlags_.borders
        ):
            avail_w = max(imgui.get_content_region_avail().x, 100.0)
            hist_w = avail_w * 0.32
            slider_w = (avail_w - hist_w - 28) * 0.5
            imgui.plot_histogram("##hist", bins, graph_size=imgui.ImVec2(hist_w, 22))
            imgui.same_line()
            imgui.set_next_item_width(slider_w)
            ch_lo, new_lo = imgui.slider_float("min", lo, data_lo, data_hi, "%.4g")
            imgui.same_line()
            imgui.set_next_item_width(slider_w)
            ch_hi, new_hi = imgui.slider_float("max", hi, data_lo, data_hi, "%.4g")
            if ch_lo:
                self._manual_lo[key] = min(new_lo, hi - 1e-6)
            if ch_hi:
                self._manual_hi[key] = max(new_hi, lo + 1e-6)
        imgui.end_child()

    def _draw_pixel_values(self, draw_list, arr, canvas_pos, canvas_size, gpu):
        if self._zoom < _PIXEL_VALUES_MIN_ZOOM:
            return
        x0 = max(0, int(np.floor(-self._pan_x / self._zoom)))
        y0 = max(0, int(np.floor(-self._pan_y / self._zoom)))
        x1 = min(gpu.w, int(np.ceil((canvas_size.x - self._pan_x) / self._zoom)) + 1)
        y1 = min(gpu.h, int(np.ceil((canvas_size.y - self._pan_y) / self._zoom)) + 1)
        if x1 <= x0 or y1 <= y0 or (x1 - x0) * (y1 - y0) > _PIXEL_VALUES_MAX_CELLS:
            return

        rgba = gpu.rgba
        luma = (
            0.299 * rgba[..., 0].astype(np.float32)
            + 0.587 * rgba[..., 1].astype(np.float32)
            + 0.114 * rgba[..., 2].astype(np.float32)
        )
        white = imgui.color_convert_float4_to_u32(imgui.ImVec4(1.0, 1.0, 1.0, 1.0))
        black = imgui.color_convert_float4_to_u32(imgui.ImVec4(0.0, 0.0, 0.0, 1.0))
        z = self._zoom
        for y in range(y0, y1):
            sy = canvas_pos.y + self._pan_y + y * z + z * 0.5
            for x in range(x0, x1):
                sx = canvas_pos.x + self._pan_x + x * z + z * 0.5
                txt = _format_value(float(np.max(arr[y, x])), arr.dtype)
                color = black if luma[y, x] > 140 else white
                draw_list.add_text(
                    imgui.ImVec2(sx - len(txt) * 3.0, sy - 6.5), color, txt
                )

    def draw(self):
        """From the figure's imgui frame: draw the popup, or keep the separate window in step with File > Options."""
        if self._window is not None and self._window.canvas.get_closed():
            self._popup_open = False
        separate = self._popup_open and OPTIONS.separate_image_window and not is_notebook_canvas(self._figure)
        if separate and self._window is None:
            self.cleanup()
            self._reset_view()
            size = (900, 950)
            keys = list(self._images) + list(self._movies)
            row = self._images.get(keys[self._selected]) if self._selected < len(keys) else None
            if isinstance(row, list):
                # a row opens in its own shape, about 1400 wide at most
                h, w = row[0][1].shape[:2]
                cell = float(np.clip(1400 / len(row) * h / w, 300, 800))
                size = (round(len(row) * cell * w / h), round(cell) + 70)
            # a new figure makes its imgui context current; this frame needs its own back
            context = imgui.get_current_context()
            self._window = fpl.Figure(size=size, canvas_kwargs={"title": self._title, "max_fps": 60.0})
            self._window[0, 0].toolbar = False
            self._window.add_imgui_window(
                self._draw_viewer,
                extent=(0.0, 1.0, 0.0, 1.0),
                window_flags=imgui.WindowFlags_.no_decoration | imgui.WindowFlags_.no_background | imgui.WindowFlags_.no_inputs,
            )
            self._window.show()
            imgui.set_current_context(context)
        elif not separate and self._window is not None:
            self.cleanup()
            self._reset_view()
        if self._window is None:
            self._draw_viewer()

    def _draw_viewer(self):
        if not self._popup_open or not (self._images or self._movies):
            return
        keys = list(self._images) + list(self._movies)
        if self._selected >= len(keys):
            self._selected = 0

        viewport = imgui.get_main_viewport()
        flags = imgui.WindowFlags_.no_saved_settings
        if self._window is not None:
            # the separate window's own imgui frame: fill it, the OS window carries the title and close button
            opaque_popups()
            imgui.set_next_window_pos(viewport.pos, imgui.Cond_.always)
            imgui.set_next_window_size(viewport.size, imgui.Cond_.always)
            flags |= imgui.WindowFlags_.no_decoration | imgui.WindowFlags_.no_move | imgui.WindowFlags_.no_resize
        else:
            em = imgui.get_font_size()
            w = min(52.0 * em, viewport.size.x * 0.92)
            h = min(56.0 * em, viewport.size.y * 0.92)
            row = self._images.get(keys[self._selected])
            if isinstance(row, list):
                # a row opens no wider than the viewer, so its height follows from the images' shape
                w = viewport.size.x * 0.92
                h = min(h, w / len(row) * row[0][1].shape[0] / row[0][1].shape[1] + 7.0 * em)
            imgui.set_next_window_size(imgui.ImVec2(w, h), imgui.Cond_.first_use_ever)
            imgui.set_next_window_pos(
                viewport.get_center(), imgui.Cond_.first_use_ever, pivot=imgui.ImVec2(0.5, 0.5)
            )

        background = imgui.get_style().color_(imgui.Col_.window_bg)
        imgui.push_style_color(imgui.Col_.window_bg, imgui.ImVec4(background.x, background.y, background.z, 1.0))
        opened, self._popup_open = imgui.begin(
            f"{self._title}###summary_image_popup",
            self._popup_open,
            flags=flags,
        )
        imgui.pop_style_color()
        if not opened:
            imgui.end()
            return

        key = self._draw_toolbar(keys)
        if key in self._movies:
            self.player.set_movie(self._movies[key])
            frame_changed = self.player.draw(slider_width=260.0)
            if frame_changed or self._movie_frame is None or key != self._movie_key:
                self._movie_frame = self.player.frame()
                self._movie_key = key
            columns = [("", self._movie_frame)]
        else:
            columns = self._columns(key)
        arr = columns[0][1]
        self._draw_contrast_panel(key, arr)

        gpus = [self._ensure_gpu(key, image, column) for column, (_, image) in enumerate(columns)]
        if gpus[0] is None:
            imgui.text_colored(
                imgui.ImVec4(1.0, 0.3, 0.3, 1.0), "GPU backend unavailable"
            )
            imgui.end()
            return

        h, w = gpus[0].h, gpus[0].w
        imgui.begin_child(
            "##canvas",
            imgui.ImVec2(0, -28),
            child_flags=0,
            window_flags=imgui.WindowFlags_.no_scrollbar
            | imgui.WindowFlags_.no_scroll_with_mouse,
        )
        canvas_pos = imgui.get_cursor_screen_pos()
        canvas_size = imgui.get_content_region_avail()
        cw = max(canvas_size.x, 1.0)
        ch = max(canvas_size.y, 1.0)

        # every image is drawn in a cell: the whole canvas for one image, a row's cells side by side
        cell_w, cell_h, left = cw, ch, canvas_pos.x
        if isinstance(self._images.get(key), list):
            gaps = _ROW_GAP * (len(columns) - 1)
            # the whole row always fits the canvas: as tall as it, or as wide when the window is too narrow for that
            fit = min(ch / h, (cw - gaps) / (len(columns) * w))
            cell_w, cell_h = w * fit, h * fit
            left = canvas_pos.x + (cw - len(columns) * cell_w - gaps) * 0.5
            if self._cell_height is not None and not self._needs_fit:
                # a resized window takes the zoom and pan along: a fitted image stays fitted
                self._zoom *= cell_h / self._cell_height
                self._pan_x *= cell_h / self._cell_height
                self._pan_y *= cell_h / self._cell_height
            self._cell_height = cell_h
        top = canvas_pos.y + (ch - cell_h) * 0.5

        if self._needs_fit:
            self._zoom = float(min(cell_w / w, cell_h / h)) if w > 0 and h > 0 else 1.0
            self._pan_x = (cell_w - w * self._zoom) * 0.5
            self._pan_y = (cell_h - h * self._zoom) * 0.5
            self._needs_fit = False

        io = imgui.get_io()
        hovered = None
        for column in range(len(columns)):
            x = left + column * (cell_w + _ROW_GAP)
            imgui.set_cursor_screen_pos(imgui.ImVec2(x, top))
            imgui.invisible_button(f"##pan_capture{column}", imgui.ImVec2(cell_w, cell_h))
            if imgui.is_item_active():
                self._pan_x += io.mouse_delta.x
                self._pan_y += io.mouse_delta.y
            if imgui.is_item_hovered():
                hovered = column
            if imgui.is_item_hovered() and io.mouse_wheel != 0.0:
                mx = io.mouse_pos.x - x
                my = io.mouse_pos.y - top
                old = self._zoom
                self._zoom = float(np.clip(old * (1.1**io.mouse_wheel), 0.05, 64.0))
                scale = self._zoom / old
                self._pan_x = mx - (mx - self._pan_x) * scale
                self._pan_y = my - (my - self._pan_y) * scale

        readout = ""
        draw_list = imgui.get_window_draw_list()
        for column, ((caption, image), gpu) in enumerate(zip(columns, gpus)):
            x = left + column * (cell_w + _ROW_GAP)
            img_min = imgui.ImVec2(x + self._pan_x, top + self._pan_y)
            img_max = imgui.ImVec2(img_min.x + w * self._zoom, img_min.y + h * self._zoom)
            draw_list.push_clip_rect(imgui.ImVec2(x, top), imgui.ImVec2(x + cell_w, top + cell_h), True)
            draw_list.add_image(gpu.ref, img_min, img_max)
            if self._highlight is not None:
                y0, x0, hh, ww = self._highlight
                p0 = imgui.ImVec2(img_min.x + x0 * self._zoom, img_min.y + y0 * self._zoom)
                p1 = imgui.ImVec2(p0.x + ww * self._zoom, p0.y + hh * self._zoom)
                box = imgui.color_convert_float4_to_u32(imgui.ImVec4(1.0, 0.9, 0.2, 0.9))
                draw_list.add_rect(p0, p1, box, 0.0, 2.0)
            if self._show_pixel_values:
                self._draw_pixel_values(draw_list, image, imgui.ImVec2(x, top), imgui.ImVec2(cell_w, cell_h), gpu)
            if caption:
                size = imgui.calc_text_size(caption)
                shade = imgui.color_convert_float4_to_u32(imgui.ImVec4(0.0, 0.0, 0.0, 0.6))
                draw_list.add_rect_filled(imgui.ImVec2(x, top), imgui.ImVec2(x + size.x + 10, top + size.y + 6), shade)
                draw_list.add_text(imgui.ImVec2(x + 5, top + 3), imgui.color_convert_float4_to_u32(imgui.ImVec4(1, 1, 1, 1)), caption)
            draw_list.pop_clip_rect()
            if column == hovered:
                # the footer reads the image under the pointer
                arr = image
                px = int((io.mouse_pos.x - img_min.x) / max(self._zoom, 1e-6))
                py = int((io.mouse_pos.y - img_min.y) / max(self._zoom, 1e-6))
                if 0 <= px < w and 0 <= py < h:
                    readout = f"{caption}  px ({py}, {px}) = ".lstrip() + ", ".join(f"{v:.4g}" for v in np.atleast_1d(arr[py, px]))
        imgui.end_child()

        amin, amax = _data_range(arr)
        footer = f"{h}x{w}  {arr.dtype}  range [{amin:.4g}, {amax:.4g}]  zoom {self._zoom:.2f}x"
        if readout:
            footer = f"{readout}    |    {footer}"
        imgui.text(footer)
        imgui.end()

    def cleanup(self):
        for gpu in self._gpu.values():
            gpu.destroy()
        self._gpu.clear()
        self._hist_cache.clear()
        if self._window is not None:
            self._window.canvas.close()
            self._window = None

    def _on_figure_close(self, event):
        self.cleanup()
