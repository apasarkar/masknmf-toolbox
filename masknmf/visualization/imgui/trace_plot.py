"""Stacked implot trace panels with a shared time axis and a draggable playhead."""

from typing import Callable, Optional, Sequence

import numpy as np
from fastplotlib.ui import ImguiWindow
from imgui_bundle import imgui, implot

from masknmf.visualization.imgui.layout import HANDLE_THICKNESS, draw_edge_handle
from masknmf.visualization.imgui.theme import em

_CURSOR_COLOR = imgui.ImVec4(1.0, 1.0, 1.0, 0.7)


def _line_colormap(rgb) -> int:
    """A single-color colormap for one line; this implot build has no per-line color argument."""
    key = tuple(int(round(float(v) * 255)) for v in rgb)
    name = "masknmf_line_{}_{}_{}".format(*key)
    index = implot.get_colormap_index(name)
    if index < 0:
        color = (key[0] / 255.0, key[1] / 255.0, key[2] / 255.0, 1.0)
        index = implot.add_colormap(name, np.array([color, color], np.float32))
    return int(index)


class TracePlot:
    """
    One panel per name, stacked with a linked time axis. A panel holds lines of
    ``(label, trace, rgb)`` with rgb in 0-1 or None for the default color.
    """

    def __init__(self, panels: Sequence[str], num_frames: int, frame_timings=None, link_y: bool = False):
        self._panels = tuple(panels)
        self._lines = {name: [] for name in self._panels}
        self._frames = np.arange(num_frames, dtype=np.float32)
        self._time = None
        if frame_timings is not None and not np.array_equal(frame_timings, self._frames):
            self._time = np.asarray(frame_timings, np.float32)
        self._link_y = link_y
        self._use_time = False
        self._autofit = True
        self._fit = True
        self._window = None
        self._on_frame: Optional[Callable] = None
        self.frame = 0

    @property
    def panels(self) -> tuple:
        return self._panels

    @property
    def x(self) -> np.ndarray:
        """Sample positions on the axis currently shown: frames, or timings when selected."""
        return self._time if self._use_time and self._time is not None else self._frames

    def set(self, panel: str, lines: Sequence[tuple]):
        """Replace a panel's lines; the axes refit on the next draw."""
        stored = []
        for label, trace, rgb in lines:
            trace = np.ascontiguousarray(trace, np.float32)
            if trace.shape != self._frames.shape:
                raise ValueError(f"trace has {trace.shape[0]} samples, the plot has {len(self._frames)} frames")
            stored.append((str(label), trace, rgb))
        self._lines[panel] = stored
        self._fit = True

    def clear(self):
        for name in self._panels:
            self._lines[name] = []

    def dock(self, figure, size: int = 320, title: str = "traces", on_frame: Optional[Callable] = None) -> ImguiWindow:
        """A resizable window along the top of ``figure``; ``on_frame`` gets the frame the playhead is dragged to."""
        self._on_frame = on_frame
        self._window = ImguiWindow(update_call=self._draw_dock)
        figure.add_imgui_window(self._window, location="top", size=size, title=title)
        return self._window

    def link(self, indices, dim: str = "time"):
        """Follow and drive a fastplotlib ReferenceIndex (``ndw.indices``) on ``dim``, in its reference units."""
        ref = self._time if self._time is not None else self._frames

        def follow(current):
            self.frame = int(np.clip(np.searchsorted(ref, current[dim]), 0, len(ref) - 1))

        indices.add_event_handler(follow)
        self._on_frame = lambda k: indices.set_dim_index(dim, float(ref[k]))

    def _draw_dock(self):
        moved = self.draw(reserve=HANDLE_THICKNESS)
        imgui.dummy(imgui.ImVec2(1, HANDLE_THICKNESS))
        draw_edge_handle(self._window)
        if moved is not None and self._on_frame is not None:
            self._on_frame(moved)

    def draw(self, reserve: float = 0.0) -> Optional[int]:
        """Options row and the stacked panels filling the window but ``reserve`` px; returns the frame when the playhead was dragged."""
        if implot.get_current_context() is None:
            implot.create_context()
        fit = self._draw_options()
        height = max(imgui.get_content_region_avail().y - reserve, em(4))
        flags = implot.SubplotFlags_.link_all_x
        if self._link_y:
            flags |= implot.SubplotFlags_.link_all_y
        if not implot.begin_subplots("##traces", len(self._panels), 1, imgui.ImVec2(-1, height), flags):
            return None
        moved = None
        try:
            for i, name in enumerate(self._panels):
                got = self._draw_panel(name, fit, last=i == len(self._panels) - 1)
                moved = got if got is not None else moved
        finally:
            implot.end_subplots()
        return moved

    def _draw_options(self) -> bool:
        changed, self._autofit = imgui.checkbox("autofit", self._autofit)
        force = changed and self._autofit
        imgui.same_line(0, em(0.4))
        force |= imgui.button("fit")
        if self._time is not None:
            imgui.same_line(0, em(0.6))
            imgui.set_next_item_width(em(6))
            changed, index = imgui.combo("##x_unit", int(self._use_time), ["frames", "time"])
            if changed:
                self._use_time = bool(index)
                force = True
        imgui.same_line(0, em(0.8))
        imgui.text_disabled(f"frame {self.frame}")
        fit = force or (self._fit and self._autofit)
        self._fit = False
        return fit

    def _draw_panel(self, name: str, fit: bool, last: bool) -> Optional[int]:
        if fit:
            implot.set_next_axes_to_fit()
        lines = self._lines[name]
        flags = implot.Flags_.no_title
        if len(lines) <= 1:
            flags |= implot.Flags_.no_legend
        if not implot.begin_plot(name, imgui.ImVec2(0, 0), flags):
            return None
        try:
            x_flags = implot.AxisFlags_.none if last else implot.AxisFlags_.no_tick_labels
            y_flags = implot.AxisFlags_.lock if imgui.get_io().key_shift else implot.AxisFlags_.none
            x_label = ("time" if self._use_time else "frame") if last else ""
            implot.setup_axes(x_label, name, x_flags, y_flags)
            # above the plot, so a panel with a legend keeps the same width as the others
            implot.setup_legend(implot.Location_.north, implot.LegendFlags_.outside | implot.LegendFlags_.horizontal)
            xs = self.x
            for label, trace, rgb in lines:
                if rgb is not None:
                    implot.push_colormap(_line_colormap(rgb))
                implot.plot_line(label, xs, trace)
                if rgb is not None:
                    implot.pop_colormap()
            moved, at = implot.drag_line_x(0, float(xs[self.frame]), _CURSOR_COLOR, 1.5)[:2]
            if moved:
                self.frame = int(np.clip(np.searchsorted(xs, at), 0, len(xs) - 1))
                return self.frame
        finally:
            implot.end_plot()
        return None
