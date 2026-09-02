"""
Traces are pulled per ROI from every movie behind the view, on a background
thread, and stored against the ROI's uid, so deleting an ROI never remaps
anyone else's traces.

A demixed signal's traces come from the ROI averages the
demixer already computed. The trace panel plots the selection (or whatever is
checked in the trace table) as percent dF/F, with the playhead bound to the
viewer's frame.

Drawn ROIs export as spatial footprints, alone or appended to the existing
signals for a seeded re-demix.
"""

from typing import Optional, Sequence, Tuple
import os
import queue
import threading
import warnings

import cv2
import numpy as np
import fastplotlib as fpl
from imgui_bundle import (
    imgui,
    implot,
    icons_fontawesome_6 as fa,
    portable_file_dialogs as pfd,
)
from fastplotlib.widgets.nd_widget._index import ReferenceIndex

from masknmf.compression import PMDArray
from masknmf.demixing.demixing_results import DemixingResults
from masknmf.visualization.imgui import (
    HANDLE_THICKNESS,
    draw_edge_handle,
    UNLABEL_ALL,
    UNLABELED,
    LabelSet,
    RoiOrder,
    RowAction,
    StrokeDrawer,
    close_figure,
    draw_category_filter,
    draw_keybinds_popup,
    draw_label_buttons,
    draw_label_editor,
    draw_label_filter,
    draw_progress,
    draw_range_filter,
    draw_roi_table,
    resolve_time_reference,
)
from fastplotlib.ui import ImguiWindow

from masknmf.visualization.imgui.files import PathPrompt, draw_path_prompt
from masknmf.visualization.imgui.theme import (
    THEME,
    card,
    danger_button,
    em,
    set_tooltip,
    toggle_button,
    to_vec4,
)
from masknmf.visualization.rois import (
    SELECTED_ALPHA,
    FootprintSet,
    RoiLabelStore,
    feather_mask,
    feathered_rgba,
)
from masknmf.visualization.traces import (
    TraceSet,
    baseline,
    display_trace,
    make_entry,
    roi_trace,
    trace_stats,
)

__all__ = ["SignalSelectionVis", "main", "resolve_device"]

CPU_WARNING = (
    "running on cpu: one frame of the pmd movie takes around 250 ms on a 512x512 "
    "field of view, and twice that for the residual movie, against about 2 ms on "
    "cuda. Pass --device cuda (or device='cuda') on a machine with a gpu."
)
CPU_BANNER = "on cpu: movie frames take ~0.25 s each"

# free pixels a stroke needs to enclose before it becomes an ROI
MIN_ROI_PIXELS = 9
SIGNAL_SET_NAME = "demixed"
DEFAULT_LABEL_NAMES = ("cell", "not cell")

_NPZ_FILTERS = ["NumPy archive", "*.npz", "All files", "*"]
_ROI_COLUMNS = ("id", "label", "source", "area")

# Trace table columns: (name, stretch weight, hidden by default). The order is
# the sort-key order in _sorted_trace_rows, so only ever append.
_TRACE_COLUMNS = (
    ("roi", 1.4, False),
    ("movie", 1.2, False),
    ("frames", 1.0, True),
    ("mean", 1.0, False),
    ("peak", 1.0, True),
    ("snr", 1.0, False),
)

# edge window sizes in px; all three are draggable once the figure is up
_ROI_PANEL_WIDTH = 380
_CONTROLS_WIDTH = 360
# starting plot height in px; the drag strip under the plot changes it
_TRACE_PLOT_HEIGHT = 240

_EXTRACT_LABEL = "extract traces"
_EXTRACT_TOOLTIP = (
    "Trace pmd and residual\n"
    "Averages the ROI's pixels in each movie, frame by frame, reading only its "
    "bounding box and weighting edge pixels down.\n"
    "It runs on a background thread and the traces arrive in the Traces panel; "
    "a demixed signal is instant, since the demixer already averaged it."
)

_TRACE_ICON = fa.ICON_FA_CHART_LINE
_REMOVE_ICON = fa.ICON_FA_XMARK
_CURSOR_COLOR = imgui.ImVec4(1.0, 0.85, 0.3, 0.9)
# how far one movie's trace color is lightened past the previous movie's
_MOVIE_FADE = 0.5

_KEYBINDS = (
    ("a", "arm / disarm drawing"),
    ("esc", "stop drawing, else empty the group"),
    ("ctrl+z", "undo the last drawn ROI"),
    ("delete", "delete the selected drawn ROI"),
    ("up / down", "previous / next ROI in the table (shift: by 10)"),
    ("left / right", "cycle the FOV image"),
    ("u", "next unlabeled ROI"),
    ("f", "center the selection; labeling then advances"),
    ("1-9", "label the selection, then advance"),
    ("0", "clear its label"),
    ("t", "trace the selected ROI"),
    ("b", "toggle the drawn overlay"),
    ("d", "toggle the demixed overlay"),
    ("k", "show these keybinds"),
    ("shift+scroll", "zoom the trace plot's time axis only"),
    ("click", "select what is under the cursor (drawing off)"),
    ("ctrl+click", "toggle an ROI in the group"),
    ("shift+click", "extend the group to a table row"),
)


def _enable_histogram(nd) -> None:
    """
    Turn a source's colorbar on, working around fastplotlib.

    ``NDImageProcessor.compute_histogram`` recomputes before flipping its own
    flag, and ``_recompute_histogram`` returns early while that flag is still
    False, so a source built with ``compute_histogram=False`` can never get a
    histogram back through the public setter and its colorbar never appears.
    Setting the flag first makes the recompute run.
    """
    processor = nd.processor
    if not processor.compute_histogram:
        processor._compute_histogram = True
        processor._recompute_histogram()
    nd.compute_histogram = True


def _as_numpy(img) -> np.ndarray:
    if hasattr(img, "cpu"):
        return img.cpu().numpy()
    return np.asarray(img)


def _cuda_available() -> bool:
    import torch

    return torch.cuda.is_available()


def resolve_device(device: Optional[str] = None) -> str:
    """
    The device to run on: the one asked for when it exists, else cpu.

    Warns whenever the answer is cpu. The warnings module prints it once per
    process, so a caller that resolves before building the viewer does not get
    it twice.

    Args:
        device (Optional[str]): 'cpu', 'cuda', or None to take a gpu when there
            is one
    """
    if device is not None and device.startswith("cuda") and not _cuda_available():
        warnings.warn(f"{device} is not available, falling back to cpu")
        device = None
    if device is None:
        device = "cuda" if _cuda_available() else "cpu"
    if device == "cpu":
        warnings.warn(CPU_WARNING)
    return device


def _line_colormap(rgb) -> int:
    """
    A registered single-color colormap for one trace line.

    This implot build has no per-line color argument, so a line takes its color
    from the pushed colormap. Looked up by name so a recreated context
    re-registers rather than duplicating.
    """
    key = tuple(int(round(float(v) * 255)) for v in rgb)
    name = "masknmf_line_{}_{}_{}".format(*key)
    index = implot.get_colormap_index(name)
    if index < 0:
        color = (key[0] / 255.0, key[1] / 255.0, key[2] / 255.0, 1.0)
        index = implot.add_colormap(name, np.array([color, color], np.float32))
    return int(index)


def _cleared_note(cleared) -> str:
    """Status tail naming the filters a selection had to drop to show itself."""
    if not cleared:
        return ""
    return f" · cleared the {', '.join(cleared)} filter" + (
        "s" if len(cleared) > 1 else ""
    )


class SignalSelectionVis:
    """
    Draw, label and trace signals over a PMD movie, with the demixed signals
    shown alongside when demixing results are given.

    Args:
        results (DemixingResults | PMDArray): demixing results, which add the
            residual movie and the demixed footprints, or just a PMD array
        frame_timings (Optional[np.ndarray]): per-frame timestamps
        ref_range (Optional[dict]): reference range for the time axis
        summary_images (Optional[dict]): {name: (fov dim1, fov dim2) array}
            projections to offer; defaults to the PMD mean image
        show_signals (bool): initial state of the demixed footprint overlay
        label_names (Sequence[str]): class names the ROIs can be labeled with
        device (Optional[str]): 'cpu' or 'cuda'; None takes a gpu when there is
            one, which is what makes the movies stream
        size (tuple): figure size in pixels. The panels are sized in pixels too,
            so a figure much under 1200 wide leaves the FOV a narrow strip.
        figure_kwargs: passed on to the figure, e.g. ``canvas="jupyter"``

    In a notebook, construct and call ``show()`` in the same cell; the canvas is
    picked up from the kernel, so nothing else is needed and ``fpl.loop.run()``
    must not be called. ``close()`` takes the figure back down.
    """

    def __init__(
        self,
        results: DemixingResults | PMDArray,
        frame_timings: Optional[np.ndarray] = None,
        ref_range: Optional[dict] = None,
        summary_images: Optional[dict] = None,
        show_signals: bool = True,
        label_names: Sequence[str] = DEFAULT_LABEL_NAMES,
        device: Optional[str] = None,
        size: Tuple[int, int] = (1700, 950),
        **figure_kwargs,
    ):
        self._device = resolve_device(device)
        self._results = results
        results.to(self._device)

        if isinstance(results, DemixingResults):
            self._pmd_array = results.pmd_array
            self._residual_array = results.residual_array
            self._ac_array = results.ac_array
        else:
            self._pmd_array = results
            self._residual_array = None
            self._ac_array = None

        self._shape = tuple(int(v) for v in self._pmd_array.shape)
        self._ny, self._nx = self._shape[1], self._shape[2]

        ref_range, frame_timings = resolve_time_reference(
            self._shape[0], frame_timings, ref_range
        )
        self._frame_timings = np.asarray(frame_timings)
        self._x_frames = np.arange(self._shape[0], dtype=np.float32)
        self._x_time = self._frame_timings.astype(np.float32)
        self._x_unit = "frames"

        self._movies = {"pmd": self._pmd_array}
        if self._residual_array is not None:
            self._movies["residual"] = self._residual_array
        self._traces = {name: TraceSet(name) for name in self._movies}
        self._signal_entries: dict = {}

        self._build_figure(summary_images, ref_range, size, figure_kwargs)

        self._store = RoiLabelStore(self._ny, self._nx, min_pixels=MIN_ROI_PIXELS)
        self._signals = None
        if self._ac_array is not None:
            self._signals = FootprintSet.from_sparse(
                SIGNAL_SET_NAME, self._ac_array.a, (self._ny, self._nx)
            )
            self._signals.visible = show_signals

        self._selected = -1
        self._selected_signal: Optional[int] = None
        self._buffer: list = []
        self._group_color = (1.0, 0.8, 0.2)
        self._feathers: dict = {}
        self._rows: list = []
        self._row_index: dict = {}
        self._promoted: dict = {}
        self._classes = LabelSet(0, label_names)
        self._order = RoiOrder(
            {"source": np.zeros(0, np.int64)}, self._classes.labels, 0
        )
        self._order.category_column = "source"

        self._show_masks = True
        self._opacity = 0.45
        self._signal_opacity = 0.5
        self._follow = False
        self._scroll_to_selection = False
        self._status = "press draw to start"
        self._new_label = ""
        self._note = ""
        self._pending_delete: Optional[tuple] = None
        self._file_dialog = None
        self._export = PathPrompt(
            "Export ROIs",
            os.path.join(os.getcwd(), "rois.npz"),
            action="save",
            hint="written by this process",
        )
        self._keybinds_open = False

        self._trace_sel: set = set()
        self._trace_sort = (0, True)
        self._trace_plot_height = float(_TRACE_PLOT_HEIGHT)
        self._trace_window = None
        self._roi_window = None
        # the trace dock sizes itself to its content until the user drags it
        self._trace_manual = False
        self._trace_stats: dict = {}
        self._trace_display: dict = {}
        self._dff = True
        self._autofit = True
        self._force_fit = False
        self._trace_fit = True
        self._plot_key = None
        self._trace_results: queue.Queue = queue.Queue()
        self._trace_threads: list = []

        self._build_overlays()
        self._drawer = StrokeDrawer(self._fov_subplot, self._on_stroke, self._pick)

        figure = self._ndw_fov.figure
        # an ImguiWindow instance rather than a bare callback, because
        # add_imgui_window hands the callback form back instead of the window
        # and the panel needs the window to size itself
        self._trace_window = ImguiWindow(update_call=self._draw_trace_panel)
        figure.add_imgui_window(
            self._trace_window,
            location="top",
            size=_TRACE_PLOT_HEIGHT,
            title="Traces",
        )
        self._roi_window = ImguiWindow(update_call=self._draw_roi_panel)
        figure.add_imgui_window(
            self._roi_window, location="left", size=_ROI_PANEL_WIDTH, title="ROIs"
        )
        figure.add_imgui_window(
            self._draw_controls,
            location="right",
            size=_CONTROLS_WIDTH,
            title="Signal Selection",
        )

        for subplot in figure:
            subplot.tooltip.enabled = False
            subplot.toolbar = False

        self._resync()
        self.refresh_overlays()
        self._show_first_row()

    def _show_first_row(self):
        """
        Open on the first ROI the table lists, with its traces already pulled,
        so the plot has something in it before the user clicks anything.
        """
        if not len(self._order.order):
            return
        item = int(self._order.order[0])
        self.select_row(item)
        si, k = self._rows[item]
        if si >= 0:
            self.collect_signal_traces(k)
        else:
            self.trace_rois([k])

    # ------------------------------------------------------------------
    # construction
    # ------------------------------------------------------------------

    def _build_figure(self, summary_images, ref_range, size, figure_kwargs):
        """The FOV widget and one NDImage per selectable source."""
        if summary_images is None:
            summary_images = {"mean img": self._pmd_array.mean_img}
        summary_images = {name: _as_numpy(img) for name, img in summary_images.items()}

        self._ndw_fov = fpl.NDWidget(
            ref_range, shape=(1, 1), names=["fov"], size=size, **figure_kwargs
        )
        self._reference_index = self._ndw_fov.indices
        self._fov_subplot = self._ndw_fov.figure["fov"]

        movies = {"pmd movie": self._pmd_array}
        if self._residual_array is not None:
            movies["residual movie"] = self._residual_array
        # the movies open first, the summary images sit at the end of the list
        self._source_names = list(movies) + list(summary_images)
        self._source_idx = 0
        first = self._source_names[0]

        movie_index_mapping = {"time": self._frame_timings}
        self._nd_images = {}
        for name, movie in movies.items():
            self._nd_images[name] = self._ndw_fov["fov"].add_nd_image(
                movie,
                ["time", "m", "n"],
                ["m", "n"],
                slider_dim_transforms=movie_index_mapping.copy(),
                compute_histogram=name == first,
                name=name,
            )
        for name, img in summary_images.items():
            self._nd_images[name] = self._ndw_fov["fov"].add_nd_image(
                img,
                ["m", "n"],
                ["m", "n"],
                compute_histogram=False,
                name=name,
            )
        for name, nd in self._nd_images.items():
            if name != first:
                nd.graphic.visible = False
                nd.pause = True
        self._fov_subplot.title = first

    def _build_overlays(self):
        """Two blended RGBA images over the FOV: drawn masks and demixed ones."""
        blank = np.zeros((self._ny, self._nx, 4), np.uint8)
        self._overlay = self._fov_subplot.add_image(
            blank, name="drawn_rois", alpha_mode="blend", offset=(0, 0, 1)
        )
        self._signal_overlay = self._fov_subplot.add_image(
            blank.copy(), name="demixed_signals", alpha_mode="blend", offset=(0, 0, 1.5)
        )
        for overlay in (self._overlay, self._signal_overlay):
            # literal RGBA bytes: auto-ranging the all-zero start saturates to white
            overlay.vmin, overlay.vmax = 0, 255
            for tile in overlay.world_object.children:
                tile.material.pick_write = False
        self._signal_overlay.visible = (
            self._signals is not None and self._signals.visible
        )

    # ------------------------------------------------------------------
    # public state
    # ------------------------------------------------------------------

    @property
    def results(self) -> DemixingResults | PMDArray:
        return self._results

    @property
    def pmd_array(self) -> PMDArray:
        return self._pmd_array

    @property
    def residual_array(self):
        return self._residual_array

    @property
    def device(self) -> str:
        return self._device

    @property
    def reference_index(self) -> ReferenceIndex:
        return self._reference_index

    @property
    def fov_widget(self) -> fpl.NDWidget:
        return self._ndw_fov

    @property
    def store(self) -> RoiLabelStore:
        return self._store

    @property
    def signals(self) -> Optional[FootprintSet]:
        """The demixed footprints, or None without demixing results."""
        return self._signals

    @property
    def n_rois(self) -> int:
        return len(self._store)

    @property
    def drawing(self) -> bool:
        return self._drawer.armed

    @property
    def label_names(self) -> tuple:
        return self._classes.names

    @property
    def roi_masks(self) -> np.ndarray:
        """Drawn ROIs as a ``(fov dim1, fov dim2, num_rois)`` binary stack"""
        return self._store.masks()

    @property
    def roi_labels(self) -> np.ndarray:
        """Class label per drawn ROI (-1 unlabeled), aligned with roi_masks"""
        return np.array([r.class_index for r in self._store.rois], dtype=np.int64)

    @property
    def export_path(self) -> str:
        """Where the export popup writes; set it to put the .npz beside the data."""
        return self._export.path

    @export_path.setter
    def export_path(self, path: str):
        self._export.path = str(path)

    def current_frame(self) -> int:
        index = int(np.searchsorted(self._frame_timings, self.reference_index["time"]))
        return min(index, self._shape[0] - 1)

    def set_frame(self, frame: int):
        index = int(np.clip(frame, 0, self._shape[0] - 1))
        self._ndw_fov.indices = {"time": float(self._frame_timings[index])}

    def show(self):
        """Show the one figure everything lives on."""
        return self._ndw_fov.show()

    def close(self):
        """Take the stroke handlers off the renderer and close the figure."""
        self._drawer.close()
        close_figure(self._ndw_fov.figure)

    # ------------------------------------------------------------------
    # export
    # ------------------------------------------------------------------

    def combined_footprints(self) -> np.ndarray:
        """
        The demixed spatial footprints with the drawn masks appended, shape
        (fov dim1, fov dim2, num_signals + num_rois). Passing this to
        ``SignalDemixer.initialize_signals(is_custom=True)`` re-demixes with the
        drawn ROIs added to the existing signals.
        """
        if self._ac_array is None:
            raise ValueError("combined footprints need demixing results")
        return np.concatenate([self._ac_array.export_a(), self.roi_masks], axis=-1)

    def export_rois(self, path: str) -> str:
        """
        Save the drawn ROIs to an .npz file.

        ``spatial_footprints`` is the (fov dim1, fov dim2, num_rois) mask stack
        for a custom demixing initialization, ``class_labels`` / ``label_names``
        carry the labels, and with demixing results loaded
        ``spatial_footprints_combined`` appends the drawn ROIs to the existing
        signals.
        """
        masks = self.roi_masks
        if masks.shape[-1] == 0:
            raise ValueError("no rois have been drawn")
        path = str(path)
        if not path.endswith(".npz"):
            path += ".npz"
        data = dict(
            spatial_footprints=masks,
            class_labels=self.roi_labels,
            label_names=np.array(self._classes.names),
        )
        if self._ac_array is not None:
            data["spatial_footprints_combined"] = np.concatenate(
                [self._ac_array.export_a(), masks], axis=-1
            )
        np.savez_compressed(path, **data)
        return path

    # ------------------------------------------------------------------
    # rows, labels and the table order
    # ------------------------------------------------------------------

    def _resync(self):
        """
        Rebuild the combined rows, the label set and the table order from the
        store and the demixed set. Drawn rows come first, so a table id below
        ``n_rois`` is a store index; promoted rows are recomputed from each
        record's source string.
        """
        rois = self._store.rois
        self._promoted = {}
        for i, record in enumerate(rois):
            name, _, row = record.source.rpartition(":")
            if name and row.isdigit():
                self._promoted[int(row)] = i
        self._rows = [(-1, i) for i in range(len(rois))]
        sources = [0] * len(rois)
        areas = [record.area for record in rois]
        if self._signals is not None and self._signals.visible:
            for k in range(len(self._signals)):
                self._rows.append((0, k))
                sources.append(1)
                areas.append(self._signals.area(k))
        self._row_index = {pair: row for row, pair in enumerate(self._rows)}
        if (
            self._selected_signal is not None
            and (0, self._selected_signal) not in self._row_index
        ):
            self._selected_signal = None
        self._buffer = [pair for pair in self._buffer if pair in self._row_index]

        labels = np.full(len(self._rows), UNLABELED, np.int64)
        labels[: len(rois)] = [record.class_index for record in rois]
        if self._signals is not None:
            for row in range(len(rois), len(self._rows)):
                labels[row] = self._signals.classes.get(self._rows[row][1], UNLABELED)
        self._classes = LabelSet(len(self._rows), self._classes.names, labels)
        self._order.columns = {
            "source": np.asarray(sources, np.int64),
            "area": np.asarray(areas, np.int64),
        }
        self._order.labels = self._classes.labels
        self._order.n_items = len(self._rows)
        if self._order.range_column is None:
            self._order.set_range_column("area")
        else:
            self._order.refresh_range()
        self._order.rebuild()

    def _sync_classes_to_model(self):
        """Push the label set back onto the store records and the demixed set."""
        for record, class_index in zip(self._store.rois, self._classes.labels):
            record.class_index = int(class_index)
        if self._signals is None:
            return
        for row in range(self.n_rois, len(self._rows)):
            k = self._rows[row][1]
            class_index = int(self._classes.labels[row])
            if class_index == UNLABELED:
                self._signals.classes.pop(k, None)
            else:
                self._signals.classes[k] = class_index

    def _format_source(self, row: int) -> str:
        si, k = self._rows[row]
        if si < 0:
            return "drawn"
        return (
            f"{SIGNAL_SET_NAME} · promoted" if k in self._promoted else SIGNAL_SET_NAME
        )

    def _format_area(self, row: int) -> str:
        si, k = self._rows[row]
        return f"{self._store.rois[k].area if si < 0 else self._signals.area(k)}"

    @property
    def _formatters(self) -> dict:
        return {"source": self._format_source, "area": self._format_area}

    # ------------------------------------------------------------------
    # drawing
    # ------------------------------------------------------------------

    def set_drawing(self, on: bool):
        """Arm or disarm stroke drawing; the pan binding is lifted while armed."""
        if on == self._drawer.armed:
            return
        self._drawer.arm(on)
        self._status = (
            "drag a closed stroke around a cell" if on else f"{self.n_rois} ROIs"
        )

    def _on_stroke(self, stroke):
        # runs inside a renderer pointer event, where a raise would vanish into
        # the event loop and could leave a stored ROI undrawn
        try:
            self.add_roi(stroke)
        except Exception as error:
            self._status = f"stroke failed: {type(error).__name__}: {error}"
            self._resync()
            self.refresh_overlays()

    def add_roi(self, stroke) -> Optional[int]:
        """Fill a closed stroke and store the enclosed free pixels as an ROI."""
        if len(stroke) < 3:
            self._status = "stroke too short"
            return None
        points = np.round(np.asarray(stroke, np.float32)).astype(np.int32)
        points[:, 0] = points[:, 0].clip(0, self._nx - 1)
        points[:, 1] = points[:, 1].clip(0, self._ny - 1)
        filled = np.zeros((self._ny, self._nx), np.uint8)
        cv2.fillPoly(filled, [points], 1)
        index = self._store.add_roi(filled.astype(bool))
        if index is None:
            self._status = f"under {MIN_ROI_PIXELS} free px, not added"
            return None
        self._resync()
        self.select_roi(index)
        return index

    def delete_roi(self, index: int):
        """Delete one drawn ROI and prune the traces that were keyed to it."""
        if not 0 <= index < self.n_rois:
            return
        self._store.delete_roi(index)
        uids = [record.uid for record in self._store.rois]
        for trace_set in self._traces.values():
            trace_set.prune(uids)
        self._traces_changed()
        self._resync()
        self.select_roi(min(index, self.n_rois - 1) if self.n_rois else -1)

    def clear_rois(self):
        self._store.clear()
        for trace_set in self._traces.values():
            trace_set.data.clear()
        self._traces_changed()
        self._resync()
        self.select_roi(-1)

    def promote_signal(self, k: int) -> Optional[int]:
        """Copy one demixed footprint into the drawn store so it can be exported."""
        if self._signals is None or k in self._promoted:
            return None
        ypix, xpix, _lam = self._signals.footprints[k]
        mask = np.zeros((self._ny, self._nx), bool)
        mask[ypix, xpix] = True
        index = self._store.add_roi(mask, source=f"{SIGNAL_SET_NAME}:{k}")
        if index is None:
            self._status = f"signal {k} has no free pixels left"
            return None
        class_index = self._signals.classes.get(k, UNLABELED)
        if class_index != UNLABELED:
            self._store.set_class(index, class_index)
        self._resync()
        self.select_roi(index)
        return index

    # ------------------------------------------------------------------
    # selection
    # ------------------------------------------------------------------

    def _pick(self, row: int, col: int, mods: frozenset = frozenset()):
        """
        Select what the click shows: a visible demixed footprint first, since
        that overlay draws on top, else the drawn ROI under the cursor.
        """
        hit = None
        if self._signals is not None and self._signals.visible:
            if 0 <= row < self._ny and 0 <= col < self._nx:
                k = int(self._signals.pick_map[row, col])
                if k >= 0:
                    hit = (0, k)
        if hit is None:
            index = self._store.roi_at(row, col)
            if index >= 0:
                hit = (-1, index)
        if "Ctrl" in mods:
            if hit is not None:
                self.buffer_toggle(*hit)
            return
        if "Shift" in mods:
            if hit is not None:
                self.buffer_add(*hit)
            return
        if hit is None:
            # empty space: keep the selection, so its traces stay on the plot
            return
        self.buffer_clear()
        if hit[0] < 0:
            self.select_roi(hit[1])
        else:
            self.select_signal(hit[1])

    def select_row(self, row: Optional[int]):
        """Route a table row to the kind of ROI it holds."""
        if row is None or not 0 <= row < len(self._rows):
            self.select_roi(-1)
            return
        si, k = self._rows[row]
        if si < 0:
            self.select_roi(k)
        else:
            self.select_signal(k)

    def select_roi(self, index: Optional[int]):
        """Select a drawn ROI; anything out of range clears the selection."""
        self._selected_signal = None
        self._selected = index if index is not None and 0 <= index < self.n_rois else -1
        self._scroll_to_selection = True
        if self._selected < 0:
            self._note = ""
            self._status = f"{self.n_rois} ROIs"
        else:
            record = self._store.rois[self._selected]
            self._note = record.note
            cleared = self._order.reveal(self._selected)
            self._status = f"ROI {self._selected}: {record.area} px" + _cleared_note(
                cleared
            )
            if self._follow:
                self._center_on(*self._feather(self._selected)[:2])
        self._sync_trace_sel()
        self.refresh_overlays()

    def select_signal(self, k: int):
        """Select demixed signal ``k``, clearing any drawn selection."""
        if self._signals is None or not 0 <= k < len(self._signals):
            self.select_roi(-1)
            return
        self._selected = -1
        self._note = ""
        self._selected_signal = int(k)
        self._scroll_to_selection = True
        row = self._row_index.get((0, k))
        cleared = self._order.reveal(row) if row is not None else []
        tail = " · promoted" if k in self._promoted else ""
        self._status = f"signal {k}: {self._signals.area(k)} px{tail}" + _cleared_note(
            cleared
        )
        if self._follow:
            ypix, xpix, _lam = self._signals.footprints[k]
            self._center_on(ypix, xpix)
        self._sync_trace_sel()
        self.refresh_overlays()

    def _selected_pair(self) -> Optional[tuple]:
        if self._selected >= 0:
            return (-1, self._selected)
        if self._selected_signal is not None:
            return (0, self._selected_signal)
        return None

    def _center_on(self, ypix, xpix):
        """Frame the camera on one mask with some context around it."""
        if not len(ypix):
            return
        y0, y1 = float(np.min(ypix)), float(np.max(ypix))
        x0, x1 = float(np.min(xpix)), float(np.max(xpix))
        cy, cx = (y0 + y1) / 2, (x0 + x1) / 2
        half = max(max(y1 - y0, x1 - x0, 1.0) * 2.0, 40.0)
        self._fov_subplot.camera.show_rect(cx - half, cx + half, cy - half, cy + half)

    def _center_selection(self):
        pair = self._selected_pair()
        if pair is None:
            return
        if pair[0] < 0:
            ypix, xpix, _lam = self._feather(pair[1])
        else:
            ypix, xpix, _lam = self._signals.footprints[pair[1]]
        self._center_on(ypix, xpix)

    def step(self, delta: int):
        """Move the table cursor and select what it lands on."""
        if self._order.step(delta):
            self.select_row(self._order.current)

    def next_unlabeled(self):
        if self._order.next_unlabeled():
            self.select_row(self._order.current)

    # ------------------------------------------------------------------
    # group buffer
    # ------------------------------------------------------------------

    def _seed_buffer(self):
        """A first ctrl or shift click keeps the current selection grouped."""
        if self._buffer:
            return
        pair = self._selected_pair()
        if pair is not None:
            self._buffer.append(pair)

    def buffer_add(self, si: int, k: int):
        self._seed_buffer()
        if (si, k) not in self._buffer:
            self._buffer.append((si, k))
        self._after_buffer_change(si, k)

    def buffer_toggle(self, si: int, k: int):
        self._seed_buffer()
        if (si, k) in self._buffer:
            self._buffer.remove((si, k))
        else:
            self._buffer.append((si, k))
        self._after_buffer_change(si, k)

    def buffer_extend_to(self, item: int):
        """Add every table row between the cursor and ``item`` to the group."""
        if not 0 <= item < len(self._rows):
            return
        self._seed_buffer()
        order = list(self._order.order)
        if item not in order:
            self.buffer_add(*self._rows[item])
            return
        current = self._order.current
        start = order.index(current) if current in order else order.index(item)
        stop = order.index(item)
        for row in order[min(start, stop) : max(start, stop) + 1]:
            pair = self._rows[row]
            if pair not in self._buffer:
                self._buffer.append(pair)
        self._after_buffer_change(*self._rows[item])

    def buffer_clear(self):
        if self._buffer:
            self._buffer.clear()
            self.refresh_overlays()

    def _after_buffer_change(self, si: int, k: int):
        if si < 0:
            self.select_roi(k)
        else:
            self.select_signal(k)

    def _row_grouped(self, item: int) -> bool:
        return 0 <= item < len(self._rows) and self._rows[item] in self._buffer

    def set_group_color(self, rgb: Optional[tuple]):
        """Give every grouped ROI one color; None reverts to class or hue."""
        pairs = self._buffer or [pair for pair in (self._selected_pair(),) if pair]
        for si, k in pairs:
            if si < 0:
                self._store.set_color(
                    k, None if rgb is None else tuple(int(round(v * 255)) for v in rgb)
                )
            elif rgb is None:
                self._signals.colors.pop(k, None)
            else:
                self._signals.colors[k] = tuple(rgb)
        self._trace_display.clear()
        self.refresh_overlays()

    # ------------------------------------------------------------------
    # labels
    # ------------------------------------------------------------------

    def assign_class(self, class_index: int):
        """Label the group, or the selection when there is no group."""
        rows = [
            self._row_index[pair] for pair in self._buffer if pair in self._row_index
        ]
        if not rows:
            pair = self._selected_pair()
            if pair is None or pair not in self._row_index:
                return
            rows = [self._row_index[pair]]
        self._classes.assign(rows, class_index)
        self._sync_classes_to_model()
        self._order.rebuild()
        self.refresh_overlays()
        if self._follow and len(rows) == 1:
            self.step(1)

    def unlabel_all(self):
        self._classes.clear()
        self._sync_classes_to_model()
        self._order.rebuild()
        self.refresh_overlays()

    def add_label(self, name: str):
        if self._classes.add(name):
            self._order.rebuild()

    # ------------------------------------------------------------------
    # overlays
    # ------------------------------------------------------------------

    def _feather(self, index: int) -> tuple:
        """
        ``(ypix, xpix, lam)`` of one drawn mask, soft-edged and cached by uid;
        a mask's pixels never change once drawn.
        """
        record = self._store.rois[index]
        got = self._feathers.get(record.uid)
        if got is None:
            mask = self._store.mask(index)
            weights = feather_mask(mask)
            ypix, xpix = np.nonzero(mask)
            got = (ypix.astype(np.int32), xpix.astype(np.int32), weights[ypix, xpix])
            self._feathers[record.uid] = got
        return got

    def refresh_overlays(self):
        self._refresh_drawn_overlay()
        self._refresh_signal_overlay()

    def _refresh_drawn_overlay(self):
        self._overlay.visible = self._show_masks
        if not self._show_masks:
            return
        grouped = {k for si, k in self._buffer if si < 0}
        comps = []
        selected = None
        for i in range(self.n_rois):
            ypix, xpix, lam = self._feather(i)
            rgb = np.asarray(self._store.rgb(i), np.float32) / 255.0
            fill = SELECTED_ALPHA if i in grouped else self._opacity
            comps.append((ypix, xpix, lam, rgb, fill))
            if i == self._selected:
                selected = (ypix, xpix, rgb)
        self._overlay.data = feathered_rgba((self._ny, self._nx), comps, selected)

    def _refresh_signal_overlay(self):
        show = self._signals is not None and self._signals.visible
        self._signal_overlay.visible = show
        if not show:
            return
        grouped = {k for si, k in self._buffer if si >= 0}
        self._signal_overlay.data = self._signals.rgba(
            (self._ny, self._nx), self._signal_opacity, self._selected_signal, grouped
        )

    def toggle_drawn_overlay(self):
        self._show_masks = not self._show_masks
        self._refresh_drawn_overlay()

    def toggle_signal_overlay(self):
        if self._signals is None:
            return
        self._signals.visible = not self._signals.visible
        self._resync()
        self._refresh_signal_overlay()

    # ------------------------------------------------------------------
    # trace extraction
    # ------------------------------------------------------------------

    @property
    def trace_busy(self) -> bool:
        self._trace_threads = [t for t in self._trace_threads if t.is_alive()]
        return bool(self._trace_threads)

    def trace_rois(self, indices: Sequence[int]):
        """
        Pull every movie's trace for each drawn ROI, on one background thread.

        The PMD trace sets the baseline the residual trace is scaled by, so both
        land on the same percent axis.
        """
        work = []
        for index in indices:
            if 0 <= index < self.n_rois:
                mask = self._store.mask(index)
                work.append((self._store.rois[index].uid, mask, feather_mask(mask)))
        if not work:
            return
        thread = threading.Thread(
            target=self._trace_worker,
            args=(work,),
            name="signal-selection-trace",
            daemon=True,
        )
        self._trace_threads.append(thread)
        self._status = (
            f"tracing ROI {indices[0]}"
            if len(work) == 1
            else f"tracing {len(work)} ROIs"
        )
        thread.start()

    def _trace_worker(self, work):
        """Extract traces off the render thread; results arrive through a queue."""
        for uid, mask, weights in work:
            try:
                entries = {}
                f0 = None
                for name, movie in self._movies.items():
                    trace = roi_trace(movie, mask, weights=weights)
                    if name == "pmd":
                        f0 = baseline(trace)
                    entries[name] = make_entry(trace, f0, zeroed=name != "pmd")
            except Exception as error:
                self._trace_results.put((uid, None, f"{type(error).__name__}: {error}"))
                continue
            self._trace_results.put((uid, entries, None))

    def _poll_traces(self):
        """Move finished traces into the trace sets; called once per frame."""
        changed = False
        while True:
            try:
                uid, entries, error = self._trace_results.get_nowait()
            except queue.Empty:
                break
            if error is not None:
                self._status = f"trace failed: {error}"
                continue
            for name, entry in entries.items():
                self._traces[name].data[uid] = entry
            changed = True
            index = self._store.uid_index(uid)
            self._status = f"traced ROI {index}" if index is not None else "traced"
        if changed:
            self._traces_changed()
            self._sync_trace_sel()

    def _traces_changed(self):
        """Trace entries moved: drop stale stats and selections, refit the plot."""
        self._trace_stats.clear()
        self._trace_display.clear()
        self._trace_sel &= set(self._trace_rows())
        self._trace_fit = True

    def _signal_entry(self, movie: str, k: int) -> Optional[dict]:
        """One demixed signal's trace, from the ROI averages the demixer made."""
        cached = self._signal_entries.get((movie, k))
        if cached is not None:
            return cached
        averages = {
            "pmd": getattr(self._results, "pmd_roi_averages", None),
            "residual": getattr(self._results, "residual_roi_averages", None),
        }.get(movie)
        if averages is None or not 0 <= k < averages.shape[0]:
            return None
        trace = averages[k].cpu().numpy()
        f0 = self._signal_baseline(k)
        entry = make_entry(trace, f0, zeroed=movie != "pmd")
        self._signal_entries[(movie, k)] = entry
        return entry

    def _signal_baseline(self, k: int) -> Optional[float]:
        """The PMD baseline of one demixed signal, shared by its other traces."""
        averages = getattr(self._results, "pmd_roi_averages", None)
        if averages is None or not 0 <= k < averages.shape[0]:
            return None
        return baseline(averages[k].cpu().numpy())

    def _trace_entry(self, key) -> Optional[dict]:
        movie, kind, index = key
        if kind == "signal":
            return self._signal_entry(movie, index)
        return self._traces[movie].data.get(index)

    def collect_signal_traces(self, k: int):
        """List one demixed signal's traces, which the demixer already computed."""
        for movie in self._movies:
            self._signal_entry(movie, k)
        self._traces_changed()

    def _trace_rows(self) -> list:
        """Every listed trace key: the ones pulled so far, of either kind."""
        rows = [
            (movie, "roi", uid)
            for movie, trace_set in self._traces.items()
            for uid in trace_set.data
        ]
        rows += [(movie, "signal", k) for movie, k in self._signal_entries]
        return rows

    def _key_to_pair(self, key) -> Optional[tuple]:
        """``(si, k)`` behind one trace key, or None when its ROI is gone."""
        _movie, kind, index = key
        if kind == "signal":
            return (0, index)
        found = self._store.uid_index(index)
        return (-1, found) if found is not None else None

    def _trace_color(self, key) -> Optional[tuple]:
        """
        The mask color of the ROI behind a trace, lightened once per movie past
        the first, so one ROI's traces read as one ROI but stay apart.
        """
        pair = self._key_to_pair(key)
        if pair is None:
            return None
        si, k = pair
        rgb = (
            tuple(v / 255.0 for v in self._store.rgb(k))
            if si < 0
            else self._signals.color(k)
        )
        fade = _MOVIE_FADE * list(self._movies).index(key[0])
        return tuple(v + (1.0 - v) * fade for v in rgb)

    def _trace_name(self, key) -> tuple:
        """``(sort value, display text)`` for a trace's roi column."""
        _movie, kind, index = key
        if kind == "signal":
            promoted = self._promoted.get(index)
            if promoted is not None:
                return float(promoted), f"{promoted}"
            return float((1 << 30) + index), f"signal {index}"
        found = self._store.uid_index(index)
        if found is not None:
            return float(found), f"{found}"
        return float((1 << 30) + index), f"uid {index}"

    def _display(self, key) -> Optional[np.ndarray]:
        """Cached display array for one trace key."""
        got = self._trace_display.get(key)
        if got is None:
            entry = self._trace_entry(key)
            if entry is None:
                return None
            got = np.ascontiguousarray(display_trace(entry, self._dff), np.float32)
            self._trace_display[key] = got
        return got

    def _stats(self, key) -> tuple:
        got = self._trace_stats.get(key)
        if got is None:
            trace = self._display(key)
            got = trace_stats(trace if trace is not None else np.zeros(0, np.float32))
            self._trace_stats[key] = got
        return got

    def _selection_trace_keys(self) -> list:
        """Trace keys of the selection, across every movie."""
        pair = self._selected_pair()
        if pair is None:
            return []
        si, k = pair
        if si >= 0:
            keys = [(movie, "signal", k) for movie in self._movies]
        else:
            uid = self._store.rois[k].uid
            keys = [
                (movie, "roi", uid)
                for movie, trace_set in self._traces.items()
                if uid in trace_set.data
            ]
        return [key for key in keys if self._trace_entry(key) is not None]

    def _sync_trace_sel(self):
        """
        Point the plot at the selection, so what it draws is the ROI the image
        is showing. A multi-row selection that already covers it is left alone.
        """
        keys = set(self._selection_trace_keys())
        if keys and self._trace_sel & keys:
            return
        if self._trace_sel != keys:
            self._trace_sel = keys
            self._trace_fit = True

    def select_trace(self, key):
        self._trace_sel = {key}
        self._trace_fit = True
        pair = self._key_to_pair(key)
        if pair is not None:
            self.buffer_clear()
            self.select_row(self._row_index.get(pair))

    def toggle_trace(self, key):
        self._trace_sel ^= {key}
        self._trace_fit = True

    def _plot_lines(self) -> Optional[tuple]:
        """``(header, [(label, key), ...])`` of what the plot should draw."""
        selection = set(self._selection_trace_keys())
        if self._trace_sel and self._trace_sel != selection:
            lines = []
            for key in sorted(self._trace_sel):
                if self._trace_entry(key) is not None:
                    lines.append((f"{self._trace_name(key)[1]} · {key[0]}", key))
            if lines:
                return f"{len(lines)} selected", lines
        if not selection:
            return None
        keys = self._selection_trace_keys()
        return self._trace_name(keys[0])[1], [(key[0], key) for key in keys]

    # ------------------------------------------------------------------
    # imgui: top controls
    # ------------------------------------------------------------------

    def _draw_controls(self):
        self._poll_traces()
        self._poll_file_dialog()
        self._handle_keys()
        # a left/right dock, so the sections stack as rows: view and draw take
        # only the height their content needs, labels take everything left, so
        # its buttons get the most columns the panel can hold
        self._draw_view_card(0.0, -1.0)
        self._draw_draw_card(0.0, -1.0)
        room = imgui.get_content_region_avail().y - imgui.get_frame_height_with_spacing()
        self._draw_labels_card(max(room, em(8)), -1.0)
        self._draw_status()
        self._keybinds_open = draw_keybinds_popup(_KEYBINDS, self._keybinds_open)
        self._draw_export_popup()

    def _draw_view_card(self, height: float, width: float):
        with card("##view", "VIEW", height, width):
            imgui.set_next_item_width(em(10))
            changed, index = imgui.combo("source", self._source_idx, self._source_names)
            if changed:
                self._set_source(index)
            set_tooltip(
                "what the FOV panel shows (left / right); "
                "the movies stream fastest on cuda"
            )
            dirty, changed = False, False
            changed, self._show_masks = imgui.checkbox("drawn", self._show_masks)
            dirty |= changed
            imgui.same_line(0, em(0.3))
            imgui.text_disabled("(b)")
            imgui.same_line(0, em(0.5))
            imgui.set_next_item_width(em(7))
            changed, self._opacity = imgui.slider_float(
                "##opacity", self._opacity, 0.05, 1.0, "opacity %.2f"
            )
            dirty |= changed
            if dirty:
                self._refresh_drawn_overlay()
            if self._signals is None:
                return
            dirty = False
            changed, self._signals.visible = imgui.checkbox(
                "demixed", self._signals.visible
            )
            if changed:
                self._resync()
                dirty = True
            imgui.same_line(0, em(0.3))
            imgui.text_disabled("(d)")
            imgui.same_line(0, em(0.5))
            imgui.set_next_item_width(em(7))
            changed, self._signal_opacity = imgui.slider_float(
                "##signal-opacity", self._signal_opacity, 0.05, 1.0, "opacity %.2f"
            )
            if dirty or changed:
                self._refresh_signal_overlay()

    def _draw_draw_card(self, height: float, width: float):
        with card("##draw", "DRAW", height, width):
            with toggle_button(self.drawing):
                if imgui.button("draw roi"):
                    self.set_drawing(not self.drawing)
            imgui.same_line(0, em(0.3))
            imgui.text_disabled("(a)")
            imgui.same_line(0, em(0.6))
            if imgui.button("undo"):
                self.delete_roi(self.n_rois - 1)
            imgui.same_line(0, em(0.3))
            imgui.text_disabled("(ctrl+z)")
            imgui.same_line(0, em(0.6))
            nothing = self._selected < 0
            if nothing:
                imgui.begin_disabled()
            if imgui.button("delete"):
                self.delete_roi(self._selected)
            if nothing:
                imgui.end_disabled()
            imgui.same_line(0, em(0.3))
            imgui.text_disabled("(del)")

            with danger_button():
                if imgui.button("clear rois"):
                    self.clear_rois()
            imgui.same_line(0, em(0.6))
            if imgui.button("export rois"):
                self._export.start()
            set_tooltip("write the drawn masks to an .npz for a seeded re-demix")
            imgui.same_line(0, em(0.6))
            if imgui.button("keys"):
                self._keybinds_open = True

            changed, self._follow = imgui.checkbox("center & advance", self._follow)
            if changed and self._follow:
                self._center_selection()
            imgui.same_line(0, em(0.3))
            imgui.text_disabled("(f)")

    def _draw_labels_card(self, height: float, width: float):
        with card("##labels", "LABELS", height, width):
            if draw_progress(self._classes):
                self.next_unlabeled()
            self._new_label, changed = draw_label_editor(self._classes, self._new_label)
            if changed:
                self._sync_classes_to_model()
                self._order.rebuild()
                self.refresh_overlays()
            picked = draw_label_buttons(self._classes)
            if picked == UNLABEL_ALL:
                self.unlabel_all()
            elif picked is not None:
                self.assign_class(picked)

    def _draw_status(self):
        if self._device == "cpu":
            imgui.text_colored(
                to_vec4(THEME.err), f"{fa.ICON_FA_TRIANGLE_EXCLAMATION} {CPU_BANNER}"
            )
            set_tooltip(CPU_WARNING)
            imgui.same_line(0, em(1))
        color = THEME.warn if self.trace_busy else THEME.text_dim
        imgui.text_colored(to_vec4(color), self._status)
        if self.drawing:
            imgui.same_line(0, em(1))
            imgui.text_disabled("click to add points; release to close the stroke")

    def step_source(self, delta: int):
        """Cycle the FOV image, like left / right in the classification GUI."""
        if self._source_names:
            self._set_source((self._source_idx + delta) % len(self._source_names))

    def _set_source(self, index: int):
        """Show one source and pause the others, so only it fetches frames."""
        if index == self._source_idx:
            return
        old = self._nd_images[self._source_names[self._source_idx]]
        new = self._nd_images[self._source_names[index]]
        old.compute_histogram = False
        old.graphic.visible = False
        old.pause = True
        new.pause = False
        new.graphic.visible = True
        _enable_histogram(new)
        self._source_idx = index
        self._fov_subplot.title = self._source_names[index]
        if "time" in new.dims:
            self._ndw_fov.indices = {"time": self.reference_index["time"]}

    # ------------------------------------------------------------------
    # imgui: roi panel
    # ------------------------------------------------------------------

    def _draw_roi_panel(self):
        if imgui.begin_child("##roi_body", imgui.ImVec2(-HANDLE_THICKNESS, 0)):
            self._draw_roi_body()
        imgui.end_child()
        draw_edge_handle(self._roi_window)

    def _draw_roi_body(self):
        changed = draw_label_filter(self._order, self._classes, "_roi")
        set_tooltip("filter by class label")
        if self._signals is not None:
            changed |= draw_category_filter(
                self._order, ["drawn", SIGNAL_SET_NAME], "_roi"
            )
            set_tooltip("filter by source: drawn by hand, or found by the demixer")
        changed |= draw_range_filter(self._order, "_roi")
        imgui.text_disabled(f"{len(self._order.order)}/{self._order.n_items} in view")
        if changed:
            self._order.rebuild()

        footer = self._footer_height()
        if imgui.begin_child("##roi_table", imgui.ImVec2(0, -footer)):
            if self._rows:
                pos = self._order.pos
                self._scroll_to_selection = draw_roi_table(
                    self._order,
                    self._classes,
                    _ROI_COLUMNS,
                    self._formatters,
                    self._scroll_to_selection,
                    table_id="signal_rois",
                    on_select=self._table_select,
                    actions=self._row_actions,
                    is_grouped=self._row_grouped,
                    on_ctrl_select=self._table_ctrl,
                    on_shift_select=self.buffer_extend_to,
                )
                if self._order.pos != pos and self._order.current is not None:
                    self.select_row(self._order.current)
            else:
                imgui.text_disabled(
                    "no ROIs yet: press draw roi, then drag around a cell"
                )
        imgui.end_child()

        pending, self._pending_delete = self._pending_delete, None
        if pending is not None and pending[0] < 0:
            self.delete_roi(pending[1])

        imgui.separator()
        self._draw_selection_footer()
        self._draw_trace_all()

    def _footer_height(self) -> float:
        """
        Exactly the rows the footer draws, so no dead space opens under the
        table. The trace-all button shares the last row, making it frame tall.
        """
        text = imgui.get_text_line_height_with_spacing()
        frame = imgui.get_frame_height_with_spacing()
        pad = imgui.get_style().item_spacing.y + 2
        if len(self._buffer) > 1 or self._selected >= 0:
            return 2 * frame + pad
        if self._selected_signal is not None:
            return text + frame + pad
        return frame + pad

    def listed_rows(self) -> list:
        """The (kind, index) pairs the table is currently showing."""
        return [self._rows[item] for item in self._order.order]

    def trace_listed(self):
        """Pull traces for every ROI the table lists, drawn and demixed alike."""
        listed = self.listed_rows()
        self.trace_rois([k for si, k in listed if si < 0])
        for si, k in listed:
            if si >= 0:
                self.collect_signal_traces(k)

    def _draw_trace_all(self):
        """Right-aligned on the footer's last row, level with its buttons."""
        label = f"{_TRACE_ICON} {_EXTRACT_LABEL} for all"
        width = imgui.calc_text_size(label).x + imgui.get_style().frame_padding.x * 2
        imgui.same_line(0, em(0.5))
        avail = imgui.get_content_region_avail().x
        if width < avail:
            imgui.set_cursor_pos_x(imgui.get_cursor_pos_x() + avail - width)
        listed = self.listed_rows()
        if not listed:
            imgui.begin_disabled()
        if imgui.button(label):
            self.trace_listed()
        if not listed:
            imgui.end_disabled()
        set_tooltip(f"{_EXTRACT_TOOLTIP}\nRuns for all {len(listed)} ROIs the table lists.")

    def _draw_selection_footer(self):
        if len(self._buffer) > 1:
            self._draw_group_footer()
            return
        if self._selected >= 0:
            imgui.set_next_item_width(-1)
            changed, self._note = imgui.input_text_with_hint(
                "##note", "note", self._note
            )
            if changed:
                self._store.set_note(self._selected, self._note)
            if imgui.button(f"{_TRACE_ICON} {_EXTRACT_LABEL}"):
                self.trace_rois([self._selected])
            set_tooltip(_EXTRACT_TOOLTIP)
            imgui.same_line(0, em(0.5))
            if imgui.button("delete"):
                self.delete_roi(self._selected)
            return
        if self._selected_signal is not None:
            k = self._selected_signal
            promoted = k in self._promoted
            imgui.text_disabled(f"signal {k}" + (" · promoted" if promoted else ""))
            if promoted:
                imgui.begin_disabled()
            if imgui.button("add to drawn"):
                self.promote_signal(k)
            if promoted:
                imgui.end_disabled()
            set_tooltip(
                "copy this footprint into the drawn set so it exports with them"
            )
            return
        imgui.text_disabled("click an ROI in the image or the table")

    def _draw_group_footer(self):
        imgui.text_disabled(f"{len(self._buffer)} ROIs grouped")
        imgui.same_line(0, em(0.5))
        changed, color = imgui.color_edit3(
            "##group_color", list(self._group_color), imgui.ColorEditFlags_.no_inputs
        )
        if changed:
            self._group_color = tuple(color)
        imgui.same_line(0, em(0.3))
        if imgui.small_button("color group"):
            self.set_group_color(self._group_color)
        imgui.same_line(0, em(0.3))
        if imgui.small_button("reset"):
            self.set_group_color(None)
        imgui.same_line(0, em(0.3))
        if imgui.small_button("ungroup"):
            self.buffer_clear()
        set_tooltip("empty the group (esc)")
        imgui.text_disabled("labels and keys 1-9 apply to the whole group")

    def _table_select(self, item: int):
        self.buffer_clear()
        self.select_row(item)

    def _table_ctrl(self, item: int):
        if 0 <= item < len(self._rows):
            self.buffer_toggle(*self._rows[item])

    def _act_trace(self, item: int):
        si, k = self._rows[item]
        if si < 0:
            self.trace_rois([k])
        else:
            self.collect_signal_traces(k)

    def _act_remove(self, item: int):
        # mutating mid-draw rebuilds the rows the clipper is still walking, so
        # the delete waits until draw_roi_table has returned
        self._pending_delete = self._rows[item]

    def _remove_row_disabled(self, item: int) -> Optional[str]:
        return "demixed signals cannot be deleted" if self._rows[item][0] >= 0 else None

    @property
    def _row_actions(self) -> tuple:
        return (
            RowAction(_TRACE_ICON, _EXTRACT_TOOLTIP, self._act_trace),
            RowAction(
                _REMOVE_ICON,
                "delete this drawn ROI",
                self._act_remove,
                self._remove_row_disabled,
            ),
        )

    # ------------------------------------------------------------------
    # imgui: trace panel
    # ------------------------------------------------------------------

    def _draw_trace_panel(self):
        self._draw_trace_plot()
        imgui.dummy(imgui.ImVec2(1, HANDLE_THICKNESS))
        self._fit_trace_window()
        window = self._trace_window
        before = None if window is None else window.size
        draw_edge_handle(window)
        if window is not None and window.size != before:
            self._trace_manual = True

    def _fit_trace_window(self):
        """
        Size the dock to whatever the panel drew this frame, until the user
        drags the handle; after that the size they chose is the size it keeps.
        """
        window = self._trace_window
        if window is None or self._trace_manual or window._collapsed:
            return
        pad = imgui.get_style().window_padding.y * 2
        wanted = int(imgui.get_cursor_pos_y() + pad)
        # a 1px deadband, since every set relays out the whole figure
        if abs(wanted - (window.size or 0)) > 1:
            window.size = wanted

    def _draw_trace_plot(self):
        target = self._plot_lines()
        if target is None:
            imgui.text_disabled(f"no traces yet: {_EXTRACT_LABEL} on an ROI, or {_TRACE_ICON} on a table row")
            return
        header, lines = target
        self._draw_trace_options(header)
        if implot.get_current_context() is None:
            implot.create_context()
        key = tuple(label for label, _ in lines)
        if key != self._plot_key:
            self._plot_key = key
            self._trace_fit = True
        fit = (self._trace_fit and self._autofit) or self._force_fit
        self._trace_fit = False
        self._force_fit = False
        if fit:
            implot.set_next_axes_to_fit()
        flags = implot.Flags_.no_title
        if len(lines) <= 1:
            flags |= implot.Flags_.no_legend
        height = (
            max(imgui.get_content_region_avail().y - HANDLE_THICKNESS, em(6))
            if self._trace_manual
            else self._trace_plot_height
        )
        if not implot.begin_plot("##trace_plot", imgui.ImVec2(-1, height), flags):
            return
        try:
            self._plot_lines_into(lines)
        finally:
            implot.end_plot()

    def _draw_trace_options(self, header: str):
        changed, self._dff = imgui.checkbox("dF/F", self._dff)
        set_tooltip("percent change over the ROI's resting fluorescence")
        if changed:
            self._trace_display.clear()
            self._trace_stats.clear()
            self._trace_fit = True
        imgui.same_line(0, em(0.8))
        changed, self._autofit = imgui.checkbox("autofit", self._autofit)
        set_tooltip("refit the axes whenever the plotted traces change")
        if changed and self._autofit:
            self._force_fit = True
        imgui.same_line(0, em(0.4))
        if imgui.button("fit"):
            self._force_fit = True
        imgui.same_line(0, em(0.6))
        units = self._x_units()
        imgui.set_next_item_width(em(6))
        changed, index = imgui.combo("##x_unit", units.index(self._x_unit), list(units))
        if changed:
            self._x_unit = units[index]
            self._force_fit = True
        imgui.text_disabled(f"{header}, frame {self.current_frame()}")
        set_tooltip(
            "drag pans, scroll zooms, shift+scroll zooms time only, double-click fits"
        )

    def _x_units(self) -> tuple:
        """Frames always; the reference axis only when it is not frame indices."""
        if np.array_equal(self._x_time, self._x_frames):
            return ("frames",)
        return ("frames", "time")

    def _x_values(self) -> np.ndarray:
        return self._x_frames if self._x_unit == "frames" else self._x_time

    def _plot_lines_into(self, lines):
        """Draw the trace lines and the playhead inside an open implot plot."""
        # holding shift locks y, so the wheel zooms the time axis alone
        y_flags = (
            implot.AxisFlags_.lock
            if imgui.get_io().key_shift
            else implot.AxisFlags_.none
        )
        implot.setup_axes(
            "frame" if self._x_unit == "frames" else "time",
            "dF/F (%)" if self._dff else "intensity",
            implot.AxisFlags_.none,
            y_flags,
        )
        # a legend inside the frame covers the traces once a group is plotted
        implot.setup_legend(implot.Location_.north_west, implot.LegendFlags_.outside)
        xs = self._x_values()
        ctrl = imgui.get_io().key_ctrl
        for label, key in lines:
            trace = self._display(key)
            if trace is None:
                continue
            rgb = self._trace_color(key)
            if rgb is not None:
                implot.push_colormap(_line_colormap(rgb))
            implot.plot_line(label, xs, trace)
            if rgb is not None:
                implot.pop_colormap()
            pair = self._key_to_pair(key)
            if pair is None:
                continue
            if (
                ctrl
                and implot.is_legend_entry_hovered(label)
                and imgui.is_mouse_clicked(0)
            ):
                self.buffer_toggle(*pair)
            if implot.begin_legend_popup(label):
                grouped = pair in self._buffer
                if imgui.menu_item_simple(
                    "remove from group" if grouped else "add to group"
                ):
                    self.buffer_toggle(*pair)
                if imgui.menu_item_simple("select this ROI"):
                    self.buffer_clear()
                    self.select_row(self._row_index.get(pair))
                implot.end_legend_popup()
        frame = self.current_frame()
        moved, at = implot.drag_line_x(0, float(xs[frame]), _CURSOR_COLOR, 1.5)[:2]
        if moved:
            self.set_frame(int(np.searchsorted(xs, at)))

    def _sorted_trace_rows(self) -> list:
        """Trace keys in the order the table shows them."""
        rows = self._trace_rows()
        column, ascending = self._trace_sort
        rows.sort(
            key=lambda key: self._trace_sort_key(key)[column], reverse=not ascending
        )
        return rows

    def _trace_sort_key(self, key) -> tuple:
        return (self._trace_name(key)[0], key[0], *self._stats(key))

    def _draw_trace_table(self):
        rows = self._sorted_trace_rows()
        if not rows:
            imgui.text_disabled(f"no traces yet: {_EXTRACT_LABEL} on an ROI, or {_TRACE_ICON} on a table row")
            return
        imgui.text_disabled(f"{len(rows)} traces · {len(self._trace_sel)} plotted")
        imgui.same_line(0, em(0.8))
        if imgui.small_button("clear"):
            self._trace_sel.clear()
            self._trace_fit = True
        flags = (
            imgui.TableFlags_.sortable
            | imgui.TableFlags_.row_bg
            | imgui.TableFlags_.borders_inner_h
            | imgui.TableFlags_.scroll_y
            | imgui.TableFlags_.resizable
            | imgui.TableFlags_.hideable
            | imgui.TableFlags_.sizing_stretch_prop
        )
        avail = imgui.get_content_region_avail()
        if not imgui.begin_table(
            "##trace_table", len(_TRACE_COLUMNS), flags, imgui.ImVec2(0, avail.y)
        ):
            return
        imgui.table_setup_scroll_freeze(0, 1)
        for i, (name, weight, hidden) in enumerate(_TRACE_COLUMNS):
            column_flags = imgui.TableColumnFlags_.width_stretch
            if i == 0:
                column_flags |= imgui.TableColumnFlags_.default_sort
            if hidden:
                column_flags |= imgui.TableColumnFlags_.default_hide
            imgui.table_setup_column(name, column_flags, weight)
        imgui.table_headers_row()
        specs = imgui.table_get_sort_specs()
        if specs is not None and specs.specs_dirty:
            if specs.specs_count > 0:
                self._trace_sort = (
                    int(specs.specs.column_index),
                    specs.specs.sort_direction == imgui.SortDirection.ascending,
                )
            specs.specs_dirty = False
        ctrl = imgui.get_io().key_ctrl
        for key in rows:
            self._draw_trace_row(key, ctrl)
        imgui.end_table()

    def _draw_trace_row(self, key, ctrl: bool):
        movie, kind, index = key
        imgui.table_next_row()
        imgui.table_next_column()
        rgb = self._trace_color(key)
        if rgb is not None:
            imgui.push_style_color(imgui.Col_.text, imgui.ImVec4(*rgb, 1.0))
        clicked, _ = imgui.selectable(
            f"{self._trace_name(key)[1]}##tr_{movie}_{kind}_{index}",
            key in self._trace_sel,
            imgui.SelectableFlags_.span_all_columns,
        )
        if rgb is not None:
            imgui.pop_style_color()
        if clicked and ctrl:
            self.toggle_trace(key)
        elif clicked:
            self.select_trace(key)
        frames, mean, peak, snr = self._stats(key)
        for text in (movie, f"{frames}", f"{mean:.1f}", f"{peak:.1f}", f"{snr:.1f}"):
            if imgui.table_next_column():
                imgui.text(text)

    # ------------------------------------------------------------------
    # keys and dialogs
    # ------------------------------------------------------------------

    def _handle_keys(self):
        io = imgui.get_io()
        if io.want_text_input:
            return
        if imgui.is_key_pressed(imgui.Key.escape, False):
            if self.drawing:
                self.set_drawing(False)
            else:
                self.buffer_clear()
        if imgui.is_key_pressed(imgui.Key.delete, False) and self._selected >= 0:
            self.delete_roi(self._selected)
        stride = 10 if io.key_shift else 1
        if imgui.is_key_pressed(imgui.Key.down_arrow, True):
            self.step(stride)
        if imgui.is_key_pressed(imgui.Key.up_arrow, True):
            self.step(-stride)
        if imgui.is_key_pressed(imgui.Key.left_arrow, True):
            self.step_source(-1)
        if imgui.is_key_pressed(imgui.Key.right_arrow, True):
            self.step_source(1)
        if io.key_ctrl:
            if imgui.is_key_pressed(imgui.Key.z, False):
                self.delete_roi(self.n_rois - 1)
            return
        if imgui.is_key_pressed(imgui.Key.a, False):
            self.set_drawing(not self.drawing)
        if imgui.is_key_pressed(imgui.Key.u, False):
            self.next_unlabeled()
        if imgui.is_key_pressed(imgui.Key.f, False):
            self._follow = not self._follow
            if self._follow:
                self._center_selection()
        if imgui.is_key_pressed(imgui.Key.t, False) and self._selected >= 0:
            self.trace_rois([self._selected])
        if imgui.is_key_pressed(imgui.Key.b, False):
            self.toggle_drawn_overlay()
        if imgui.is_key_pressed(imgui.Key.d, False):
            self.toggle_signal_overlay()
        if imgui.is_key_pressed(imgui.Key.k, False):
            self._keybinds_open = not self._keybinds_open
        picked = self._classes.hotkey_pressed()
        if picked is not None:
            self.assign_class(picked)

    def _save_rois(self, path: str) -> bool:
        """Write the drawn ROIs, reporting either way in the status row."""
        try:
            written = self.export_rois(path)
        except (OSError, ValueError) as error:
            self._status = self._export.status = f"export failed: {error}"
            return False
        self._export.path = written
        self._status = self._export.status = f"exported {self.n_rois} roi(s) to {written}"
        return True

    def _draw_export_popup(self):
        path, browse = draw_path_prompt(self._export)
        if path is not None and self._save_rois(path):
            self._export.open = False
        if browse:
            self._browse_export()

    def _browse_export(self):
        if self._file_dialog is None:
            self._file_dialog = pfd.save_file("Export ROIs", self._export.path, _NPZ_FILTERS)

    def _poll_file_dialog(self):
        if self._file_dialog is None or not self._file_dialog.ready(0):
            return
        result = self._file_dialog.result()
        self._file_dialog = None
        if result:
            self._export.path = result


def main(argv=None):
    import argparse
    import h5py

    parser = argparse.ArgumentParser(
        description="Draw, label and trace signals over a PMD or residual movie"
    )
    parser.add_argument("path", help="masknmf demixing_results or PMD .hdf5 file")
    parser.add_argument(
        "--device",
        default=None,
        help="'cpu' or 'cuda'; the default takes a gpu when there is one, since on "
        "cpu each movie frame is a full sparse reconstruction of the field of view",
    )
    parser.add_argument(
        "--labels",
        default=None,
        help="comma-separated class names, e.g. cell,dendrite,junk",
    )
    args = parser.parse_args(argv)
    label_names = args.labels.split(",") if args.labels else DEFAULT_LABEL_NAMES
    device = resolve_device(args.device)

    with h5py.File(args.path, "r") as f:
        is_demixing = "DemixingResults" in f
    if is_demixing:
        results = DemixingResults.from_hdf5(args.path, device=device)
    else:
        results = PMDArray.from_hdf5(args.path, device=device)
    vis = SignalSelectionVis(results, label_names=label_names, device=device)
    vis.export_path = os.path.join(os.path.dirname(os.path.abspath(args.path)), "rois.npz")
    vis.show()
    fpl.loop.run()


if __name__ == "__main__":
    main()
