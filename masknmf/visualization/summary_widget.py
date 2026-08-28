"""
SummaryImageViewer - floating imgui popup for browsing full-FOV summary images
(mean/variance/correlation images) with pan, zoom, colormap, contrast control,
pixel-value overlay, and a hover readout. Adapted from mbo_utilities'
summary_image widget; renders through fastplotlib's wgpu imgui backend.
"""

from typing import *
import numpy as np
import wgpu
from cmap import Colormap
from imgui_bundle import imgui

from masknmf.visualization.imgui.movie_player import MoviePlayer

_CMAPS = ("gray", "viridis", "magma", "inferno", "turbo")
_CONTRAST_MODES = ("full", "auto", "manual")
_CONTRAST_AUTO = 1
_CONTRAST_MANUAL = 2

_PIXEL_VALUES_MIN_ZOOM = 16.0
_PIXEL_VALUES_MAX_CELLS = 10_000


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
    rgba = (Colormap(cmap_name)(n) * 255).astype(np.uint8)
    return np.ascontiguousarray(rgba)


def _format_value(v: float, dtype) -> str:
    if np.issubdtype(dtype, np.integer):
        return f"{int(v)}"
    av = abs(v)
    if av != 0 and (av < 0.01 or av >= 10000):
        return f"{v:.1e}"
    return f"{v:.2f}"


class _GpuImage:
    """Owns one wgpu texture + its imgui registration for a single image."""

    def __init__(self, backend, arr: np.ndarray, cmap: str, lo: float, hi: float):
        self.backend = backend
        self.arr = arr
        self.cmap = cmap
        self.lo = lo
        self.hi = hi
        self.h, self.w = arr.shape[:2]
        self._texture = None
        self._view = None
        self.ref = None
        self.rgba: Optional[np.ndarray] = None
        self._upload()

    def _upload(self):
        self.rgba = _to_rgba(self.arr, self.cmap, self.lo, self.hi)
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

    def ensure(self, arr: np.ndarray, cmap: str, lo: float, hi: float):
        if arr is self.arr and cmap == self.cmap and lo == self.lo and hi == self.hi:
            return
        same_shape = arr.shape[:2] == (self.h, self.w)
        self.arr = arr
        self.cmap = cmap
        self.lo = lo
        self.hi = hi
        if same_shape and self._texture is not None:
            # rewrite pixels in place (movie frames, contrast changes)
            self.rgba = _to_rgba(arr, cmap, lo, hi)
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
    Popup viewer over a {name: 2D array} image set. Call open() to show and
    draw() every imgui frame (it is a no-op while closed).
    """

    def __init__(self, figure, images: Optional[dict] = None):
        self._figure = figure
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
        self._show_pixel_values = False
        self._highlight: Optional[tuple] = None  # (y0, x0, h, w) in image coords
        self._gpu: dict = {}
        self._manual_lo: dict = {}
        self._manual_hi: dict = {}
        self._hist_cache: dict = {}

    def set_images(self, images: dict, selected: Optional[str] = None):
        """Replace the image set; caches are dropped for images that changed"""
        for key, gpu in list(self._gpu.items()):
            if images.get(key) is not gpu.arr:
                gpu.destroy()
                del self._gpu[key]
                self._hist_cache.pop(key, None)
        self._images = dict(images)
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
        return self._popup_open

    def set_highlight(self, rect: Optional[tuple]):
        """Outline a region of the image: (y0, x0, height, width), or None"""
        self._highlight = rect

    def _backend(self):
        try:
            return self._figure.imgui_renderer.backend
        except AttributeError:
            return None

    def _reset_view(self):
        self._zoom = 1.0
        self._pan_x = 0.0
        self._pan_y = 0.0
        self._needs_fit = True

    def _get_range(self, key: str, arr: np.ndarray) -> tuple[float, float]:
        if self._contrast_mode == _CONTRAST_AUTO:
            if key in self._movies:
                # fixed per movie so playback doesn't flicker
                if key not in self._movie_range:
                    self._movie_range[key] = _auto_range(arr)
                return self._movie_range[key]
            return _auto_range(arr)
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

    def _ensure_gpu(self, key: str, arr: np.ndarray) -> Optional[_GpuImage]:
        backend = self._backend()
        if backend is None:
            return None
        cmap = _CMAPS[self._cmap_idx]
        lo, hi = self._get_range(key, arr)
        gpu = self._gpu.get(key)
        if gpu is None:
            gpu = _GpuImage(backend, arr, cmap, lo, hi)
            self._gpu[key] = gpu
        else:
            gpu.ensure(arr, cmap, lo, hi)
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
        imgui.set_next_item_width(180)
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
                txt = _format_value(float(arr[y, x]), arr.dtype)
                color = black if luma[y, x] > 140 else white
                draw_list.add_text(
                    imgui.ImVec2(sx - len(txt) * 3.0, sy - 6.5), color, txt
                )

    def draw(self):
        if not self._popup_open or not (self._images or self._movies):
            return
        keys = list(self._images) + list(self._movies)
        if self._selected >= len(keys):
            self._selected = 0

        viewport = imgui.get_main_viewport()
        em = imgui.get_font_size()
        w = min(52.0 * em, viewport.size.x * 0.92)
        h = min(56.0 * em, viewport.size.y * 0.92)
        imgui.set_next_window_size(imgui.ImVec2(w, h), imgui.Cond_.first_use_ever)
        imgui.set_next_window_pos(
            viewport.get_center(), imgui.Cond_.first_use_ever, pivot=imgui.ImVec2(0.5, 0.5)
        )

        opened, self._popup_open = imgui.begin(
            "Full FOV###summary_image_popup",
            self._popup_open,
            flags=imgui.WindowFlags_.no_saved_settings,
        )
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
            arr = self._movie_frame
        else:
            arr = self._images[key]
        self._draw_contrast_panel(key, arr)

        gpu = self._ensure_gpu(key, arr)
        if gpu is None:
            imgui.text_colored(
                imgui.ImVec4(1.0, 0.3, 0.3, 1.0), "GPU backend unavailable"
            )
            imgui.end()
            return

        h, w = gpu.h, gpu.w
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

        if self._needs_fit:
            self._zoom = float(min(cw / w, ch / h)) if w > 0 and h > 0 else 1.0
            self._pan_x = (cw - w * self._zoom) * 0.5
            self._pan_y = (ch - h * self._zoom) * 0.5
            self._needs_fit = False

        imgui.invisible_button("##pan_capture", imgui.ImVec2(cw, ch))
        io = imgui.get_io()
        if imgui.is_item_active():
            self._pan_x += io.mouse_delta.x
            self._pan_y += io.mouse_delta.y
        if imgui.is_item_hovered() and io.mouse_wheel != 0.0:
            mx = io.mouse_pos.x - canvas_pos.x
            my = io.mouse_pos.y - canvas_pos.y
            old = self._zoom
            self._zoom = float(np.clip(old * (1.1**io.mouse_wheel), 0.05, 64.0))
            scale = self._zoom / old
            self._pan_x = mx - (mx - self._pan_x) * scale
            self._pan_y = my - (my - self._pan_y) * scale

        img_min = imgui.ImVec2(canvas_pos.x + self._pan_x, canvas_pos.y + self._pan_y)
        img_max = imgui.ImVec2(img_min.x + w * self._zoom, img_min.y + h * self._zoom)
        clip_max = imgui.ImVec2(canvas_pos.x + cw, canvas_pos.y + ch)
        draw_list = imgui.get_window_draw_list()
        draw_list.push_clip_rect(canvas_pos, clip_max, True)
        draw_list.add_image(gpu.ref, img_min, img_max)
        if self._highlight is not None:
            y0, x0, hh, ww = self._highlight
            p0 = imgui.ImVec2(img_min.x + x0 * self._zoom, img_min.y + y0 * self._zoom)
            p1 = imgui.ImVec2(p0.x + ww * self._zoom, p0.y + hh * self._zoom)
            box = imgui.color_convert_float4_to_u32(imgui.ImVec4(1.0, 0.9, 0.2, 0.9))
            draw_list.add_rect(p0, p1, box, 0.0, 2.0)
        if self._show_pixel_values:
            self._draw_pixel_values(draw_list, arr, canvas_pos, canvas_size, gpu)
        draw_list.pop_clip_rect()

        readout = ""
        if imgui.is_item_hovered():
            px = int((io.mouse_pos.x - img_min.x) / max(self._zoom, 1e-6))
            py = int((io.mouse_pos.y - img_min.y) / max(self._zoom, 1e-6))
            if 0 <= px < w and 0 <= py < h:
                readout = f"px ({py}, {px}) = {float(arr[py, px]):.4g}"
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
