from typing import Callable, Optional

import numpy as np
import wgpu
from cmap import Colormap
from imgui_bundle import imgui

from masknmf.visualization.imgui.movie_player import MoviePlayer

CMAPS = ("gray", "viridis", "magma", "inferno", "turbo")
CONTRAST_MODES = ("full", "auto", "manual")
CONTRAST_FULL = 0
CONTRAST_AUTO = 1
CONTRAST_MANUAL = 2

PIXEL_VALUES_MIN_ZOOM = 16.0
PIXEL_VALUES_MAX_CELLS = 10_000


def data_range(arr: np.ndarray) -> tuple:
    a = np.asarray(arr, dtype=np.float32)
    lo, hi = float(np.nanmin(a)), float(np.nanmax(a))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        hi = lo + 1.0
    return lo, hi


def auto_range(arr: np.ndarray) -> tuple:
    a = np.asarray(arr, dtype=np.float32)
    lo, hi = float(np.nanpercentile(a, 1.0)), float(np.nanpercentile(a, 99.0))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        hi = lo + 1.0
    return lo, hi


def to_rgba(arr: np.ndarray, cmap_name: str, lo: float, hi: float) -> np.ndarray:
    a = np.asarray(arr, dtype=np.float32)
    n = np.clip((a - lo) / max(hi - lo, 1e-12), 0.0, 1.0)
    return np.ascontiguousarray((Colormap(cmap_name)(n) * 255).astype(np.uint8))


def format_value(v: float, dtype) -> str:
    if np.issubdtype(dtype, np.integer):
        return f"{int(v)}"
    av = abs(v)
    if av != 0 and (av < 0.01 or av >= 10000):
        return f"{v:.1e}"
    return f"{v:.2f}"


class GpuImage:
    """One wgpu texture plus its imgui registration."""

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
        self.rgba = to_rgba(self.arr, self.cmap, self.lo, self.hi)
        device = self.backend._device
        self._texture = device.create_texture(
            size=(self.w, self.h, 1),
            format=wgpu.TextureFormat.rgba8unorm,
            usage=wgpu.TextureUsage.TEXTURE_BINDING | wgpu.TextureUsage.COPY_DST,
        )
        self._write()
        self._view = self._texture.create_view()
        self.ref = self.backend.register_texture(self._view)

    def _write(self):
        self.backend._device.queue.write_texture(
            {"texture": self._texture, "mip_level": 0, "origin": (0, 0, 0)},
            self.rgba.tobytes(),
            {"offset": 0, "bytes_per_row": self.w * 4, "rows_per_image": self.h},
            (self.w, self.h, 1),
        )

    def ensure(self, arr: np.ndarray, cmap: str, lo: float, hi: float):
        if arr is self.arr and cmap == self.cmap and lo == self.lo and hi == self.hi:
            return
        same_shape = arr.shape[:2] == (self.h, self.w)
        self.arr, self.cmap, self.lo, self.hi = arr, cmap, lo, hi
        if same_shape and self._texture is not None:
            self.rgba = to_rgba(arr, cmap, lo, hi)
            self._write()
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
    Popup over a {name: 2D array} image set, with optional lazy movies.

    Call open() to show and draw() every imgui frame; it is a no-op while
    closed. Host-specific extras attach through the hooks: roi_provider draws
    contour overlays, on_export receives (key, array), extra_toolbar adds
    controls to the top row.
    """

    def __init__(
        self,
        figure=None,
        images: Optional[dict] = None,
        backend=None,
        title: str = "Full FOV",
        roi_provider: Optional[Callable] = None,
        on_export: Optional[Callable] = None,
        extra_toolbar: Optional[Callable] = None,
    ):
        self._figure = figure
        self._explicit_backend = backend
        self._title = title
        self.roi_provider = roi_provider
        self.on_export = on_export
        self.extra_toolbar = extra_toolbar

        self._images = dict(images or {})
        self._movies: dict = {}
        self._movie_frame = None
        self._movie_key = None
        self._movie_range: dict = {}
        self.player = MoviePlayer()

        self._selected = 0
        self._popup_open = False
        self._cmap_idx = 0
        self._contrast_mode = CONTRAST_AUTO
        self._zoom = 1.0
        self._pan_x = 0.0
        self._pan_y = 0.0
        self._needs_fit = True
        self._show_pixel_values = False
        self._show_rois = False
        self._highlight = None
        self._gpu: dict = {}
        self._manual_lo: dict = {}
        self._manual_hi: dict = {}
        self._hist_cache: dict = {}

    @property
    def is_open(self) -> bool:
        return self._popup_open

    @property
    def images(self) -> dict:
        return self._images

    def open(self):
        self._popup_open = True
        self._reset_view()

    def close(self):
        self._popup_open = False

    def set_images(self, images: dict, selected: Optional[str] = None):
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
        self._movies = dict(movies)
        self._movie_frame = None
        self._movie_key = None
        self._movie_range.clear()

    def set_highlight(self, rect: Optional[tuple]):
        """Outline (y0, x0, height, width) in image coords, or None."""
        self._highlight = rect

    def set_cmap(self, name: str):
        if name in CMAPS:
            self._cmap_idx = CMAPS.index(name)

    def _backend(self):
        if self._explicit_backend is not None:
            return self._explicit_backend
        try:
            return self._figure.imgui_renderer.backend
        except AttributeError:
            return None

    def _reset_view(self):
        self._zoom = 1.0
        self._pan_x = 0.0
        self._pan_y = 0.0
        self._needs_fit = True

    def _get_range(self, key: str, arr: np.ndarray) -> tuple:
        if self._contrast_mode == CONTRAST_AUTO:
            if key in self._movies:
                # fixed per movie so playback doesn't flicker
                if key not in self._movie_range:
                    self._movie_range[key] = auto_range(arr)
                return self._movie_range[key]
            return auto_range(arr)
        if self._contrast_mode == CONTRAST_MANUAL:
            lo, hi = self._manual_lo.get(key), self._manual_hi.get(key)
            if lo is None or hi is None:
                lo, hi = auto_range(arr)
                self._manual_lo[key], self._manual_hi[key] = lo, hi
            return lo, (hi if hi > lo else lo + 1e-6)
        return data_range(arr)

    def _ensure_gpu(self, key: str, arr: np.ndarray) -> Optional[GpuImage]:
        backend = self._backend()
        if backend is None:
            return None
        cmap = CMAPS[self._cmap_idx]
        lo, hi = self._get_range(key, arr)
        gpu = self._gpu.get(key)
        if gpu is None:
            gpu = GpuImage(backend, arr, cmap, lo, hi)
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
        _, self._cmap_idx = imgui.combo("cmap", self._cmap_idx, list(CMAPS))
        imgui.same_line()
        imgui.set_next_item_width(100)
        _, self._contrast_mode = imgui.combo(
            "contrast", self._contrast_mode, list(CONTRAST_MODES)
        )
        imgui.same_line()
        if imgui.button("reset"):
            self._reset_view()
        imgui.same_line()
        _, self._show_pixel_values = imgui.checkbox("pixel values", self._show_pixel_values)
        if self.roi_provider is not None:
            imgui.same_line()
            _, self._show_rois = imgui.checkbox("rois", self._show_rois)
        if self.on_export is not None:
            imgui.same_line()
            if imgui.button("export"):
                self.on_export(keys[self._selected], self._current_array(keys))
        if self.extra_toolbar is not None:
            self.extra_toolbar(self)
        return keys[self._selected]

    def _current_array(self, keys: list):
        key = keys[self._selected]
        return self._movie_frame if key in self._movies else self._images[key]

    def _draw_contrast_panel(self, key: str, arr: np.ndarray):
        if self._contrast_mode != CONTRAST_MANUAL:
            return
        lo_limit, hi_limit = data_range(arr)
        lo, hi = self._manual_lo.get(key), self._manual_hi.get(key)
        if lo is None or hi is None:
            lo, hi = auto_range(arr)
            self._manual_lo[key], self._manual_hi[key] = lo, hi

        bins = self._get_histogram(key, arr)
        if imgui.begin_child("##levels", imgui.ImVec2(0, 32), child_flags=imgui.ChildFlags_.borders):
            avail_w = max(imgui.get_content_region_avail().x, 100.0)
            hist_w = avail_w * 0.32
            slider_w = (avail_w - hist_w - 28) * 0.5
            imgui.plot_histogram("##hist", bins, graph_size=imgui.ImVec2(hist_w, 22))
            imgui.same_line()
            imgui.set_next_item_width(slider_w)
            ch_lo, new_lo = imgui.slider_float("min", lo, lo_limit, hi_limit, "%.4g")
            imgui.same_line()
            imgui.set_next_item_width(slider_w)
            ch_hi, new_hi = imgui.slider_float("max", hi, lo_limit, hi_limit, "%.4g")
            if ch_lo:
                self._manual_lo[key] = min(new_lo, hi - 1e-6)
            if ch_hi:
                self._manual_hi[key] = max(new_hi, lo + 1e-6)
        imgui.end_child()

    def _draw_pixel_values(self, draw_list, arr, canvas_pos, canvas_size, gpu):
        if self._zoom < PIXEL_VALUES_MIN_ZOOM:
            return
        x0 = max(0, int(np.floor(-self._pan_x / self._zoom)))
        y0 = max(0, int(np.floor(-self._pan_y / self._zoom)))
        x1 = min(gpu.w, int(np.ceil((canvas_size.x - self._pan_x) / self._zoom)) + 1)
        y1 = min(gpu.h, int(np.ceil((canvas_size.y - self._pan_y) / self._zoom)) + 1)
        if x1 <= x0 or y1 <= y0 or (x1 - x0) * (y1 - y0) > PIXEL_VALUES_MAX_CELLS:
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
                txt = format_value(float(arr[y, x]), arr.dtype)
                color = black if luma[y, x] > 140 else white
                draw_list.add_text(imgui.ImVec2(sx - len(txt) * 3.0, sy - 6.5), color, txt)

    def _draw_rois(self, draw_list, img_min):
        contours = self.roi_provider()
        if not contours:
            return
        color = imgui.color_convert_float4_to_u32(imgui.ImVec4(0.2, 1.0, 0.4, 0.9))
        z = self._zoom
        for contour in contours:
            pts = np.asarray(contour)
            if pts.ndim != 2 or pts.shape[0] < 2:
                continue
            for i in range(len(pts)):
                y0, x0 = pts[i]
                y1, x1 = pts[(i + 1) % len(pts)]
                draw_list.add_line(
                    imgui.ImVec2(img_min.x + x0 * z, img_min.y + y0 * z),
                    imgui.ImVec2(img_min.x + x1 * z, img_min.y + y1 * z),
                    color, 1.0,
                )

    def draw(self):
        if not self._popup_open or not (self._images or self._movies):
            return
        keys = list(self._images) + list(self._movies)
        if self._selected >= len(keys):
            self._selected = 0

        viewport = imgui.get_main_viewport()
        em = imgui.get_font_size()
        imgui.set_next_window_size(
            imgui.ImVec2(min(52.0 * em, viewport.size.x * 0.92),
                         min(56.0 * em, viewport.size.y * 0.92)),
            imgui.Cond_.first_use_ever,
        )
        imgui.set_next_window_pos(
            viewport.get_center(), imgui.Cond_.first_use_ever, pivot=imgui.ImVec2(0.5, 0.5)
        )
        opened, self._popup_open = imgui.begin(
            f"{self._title}###summary_image_popup",
            self._popup_open,
            flags=imgui.WindowFlags_.no_saved_settings,
        )
        if not opened:
            imgui.end()
            return

        key = self._draw_toolbar(keys)
        if key in self._movies:
            self.player.set_movie(self._movies[key])
            changed = self.player.draw(slider_width=260.0)
            if changed or self._movie_frame is None or key != self._movie_key:
                self._movie_frame = self.player.frame()
                self._movie_key = key
            arr = self._movie_frame
        else:
            arr = self._images[key]
        self._draw_contrast_panel(key, arr)

        gpu = self._ensure_gpu(key, arr)
        if gpu is None:
            imgui.text_colored(imgui.ImVec4(1.0, 0.3, 0.3, 1.0), "GPU backend unavailable")
            imgui.end()
            return

        h, w = gpu.h, gpu.w
        imgui.begin_child(
            "##canvas", imgui.ImVec2(0, -28), child_flags=0,
            window_flags=imgui.WindowFlags_.no_scrollbar | imgui.WindowFlags_.no_scroll_with_mouse,
        )
        canvas_pos = imgui.get_cursor_screen_pos()
        canvas_size = imgui.get_content_region_avail()
        cw, ch = max(canvas_size.x, 1.0), max(canvas_size.y, 1.0)

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
            mx, my = io.mouse_pos.x - canvas_pos.x, io.mouse_pos.y - canvas_pos.y
            old = self._zoom
            self._zoom = float(np.clip(old * (1.1**io.mouse_wheel), 0.05, 64.0))
            scale = self._zoom / old
            self._pan_x = mx - (mx - self._pan_x) * scale
            self._pan_y = my - (my - self._pan_y) * scale

        img_min = imgui.ImVec2(canvas_pos.x + self._pan_x, canvas_pos.y + self._pan_y)
        img_max = imgui.ImVec2(img_min.x + w * self._zoom, img_min.y + h * self._zoom)
        draw_list = imgui.get_window_draw_list()
        draw_list.push_clip_rect(canvas_pos, imgui.ImVec2(canvas_pos.x + cw, canvas_pos.y + ch), True)
        draw_list.add_image(gpu.ref, img_min, img_max)
        if self._highlight is not None:
            y0, x0, hh, ww = self._highlight
            p0 = imgui.ImVec2(img_min.x + x0 * self._zoom, img_min.y + y0 * self._zoom)
            p1 = imgui.ImVec2(p0.x + ww * self._zoom, p0.y + hh * self._zoom)
            box = imgui.color_convert_float4_to_u32(imgui.ImVec4(1.0, 0.9, 0.2, 0.9))
            draw_list.add_rect(p0, p1, box, 0.0, 2.0)
        if self._show_rois and self.roi_provider is not None:
            self._draw_rois(draw_list, img_min)
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

        amin, amax = data_range(arr)
        footer = f"{h}x{w}  {arr.dtype}  range [{amin:.4g}, {amax:.4g}]  zoom {self._zoom:.2f}x"
        imgui.text(f"{readout}    |    {footer}" if readout else footer)
        imgui.end()

    def cleanup(self):
        for gpu in self._gpu.values():
            gpu.destroy()
        self._gpu.clear()
        self._hist_cache.clear()
