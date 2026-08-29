import time
from typing import *
import numpy as np
from imgui_bundle import imgui


class MoviePlayer:
    """
    Imgui transport bar (play/pause, frame slider, fps) over a lazy (T, H, W)
    array such as ACArray or PMDArray. Any object with .shape and [t] /
    [t, rows, cols] indexing works, and only the frame on screen is read.

    Unlike CheckboxWindow this is not an ImguiWindow: it owns no window and no
    graphic, so draw() goes inline in a panel the host already has, whether
    that is an EdgeWindow.update(), the callback handed to
    figure.add_imgui_window(), or a bare imgui.begin()/end() block. It never
    touches the figure itself. draw() returns True on the frames where t
    moved; on those, pull frame() and assign it to your ImageGraphic.data.
    Set vmin/vmax once when the movie changes rather than per frame, or the
    contrast chases each frame's range and playback flickers.

    set_movie() swaps the source and clamps the playhead into the new range;
    jump_to() seeks, e.g. to an ROI's peak frame. Pass region=(top, left, h, w)
    to frame() to read only a crop, zero-padded where it runs past the FOV.
    Playback advances off the wall clock inside draw(), so it only runs while
    the figure is rendering and fps is independent of the render rate.
    """

    def __init__(self, movie=None, fps: float = 30.0):
        self._movie = movie
        self._t = 0
        self._playing = False
        self._fps = fps
        self._clock: Optional[tuple] = None  # (wall time, frame) at play/seek

    @property
    def movie(self):
        return self._movie

    @property
    def n_frames(self) -> int:
        return 0 if self._movie is None else int(self._movie.shape[0])

    @property
    def t(self) -> int:
        return self._t

    def set_movie(self, movie):
        if movie is not self._movie:
            self._movie = movie
            self._t = min(self._t, max(self.n_frames - 1, 0))

    def jump_to(self, t: int):
        self._t = int(np.clip(t, 0, max(self.n_frames - 1, 0)))
        self._clock = (time.perf_counter(), float(self._t))

    def frame(self, region: Optional[tuple] = None) -> Optional[np.ndarray]:
        """
        Current frame as a 2D array; region=(top, left, h, w) crops lazily,
        zero-padded where it extends past the FOV
        """
        if self._movie is None:
            return None
        if region is None:
            img = np.asarray(self._movie[self._t])
            return img.reshape(img.shape[-2:])
        top, left, h, w = region
        fov_h, fov_w = self._movie.shape[1:3]
        out = np.zeros((h, w), dtype=np.float32)
        y0, y1 = max(top, 0), min(top + h, fov_h)
        x0, x1 = max(left, 0), min(left + w, fov_w)
        if y1 > y0 and x1 > x0:
            img = np.asarray(self._movie[self._t, slice(y0, y1), slice(x0, x1)])
            out[y0 - top : y1 - top, x0 - left : x1 - left] = img.reshape(
                y1 - y0, x1 - x0
            )
        return out

    def draw(self, slider_width: float = 260.0) -> bool:
        """Transport controls; returns True if the current frame changed"""
        if self._movie is None:
            return False
        changed = False
        if imgui.button("pause##movie" if self._playing else "play##movie"):
            self._playing = not self._playing
            self._clock = (time.perf_counter(), float(self._t))
        imgui.same_line(0, 8)
        imgui.set_next_item_width(slider_width)
        slid, t = imgui.slider_int(
            f"##movie-frame", self._t, 0, max(self.n_frames - 1, 0)
        )
        if slid:
            self._t = t
            self._clock = (time.perf_counter(), float(t))
            changed = True
        imgui.same_line(0, 4)
        imgui.text(f"{self._t}")
        imgui.same_line(0, 12)
        imgui.set_next_item_width(60)
        _, self._fps = imgui.drag_float("fps", self._fps, 1.0, 1.0, 240.0, "%.0f")
        if self._playing:
            t0, f0 = self._clock
            new_t = int(f0 + (time.perf_counter() - t0) * self._fps) % max(
                self.n_frames, 1
            )
            if new_t != self._t:
                self._t = new_t
                changed = True
        return changed
