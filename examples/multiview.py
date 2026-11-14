"""
Every session of a tracking run on one screen, as static images.

One panel per session shows the session's aligned FOV with its ROIs on top. A cell tracked across sessions has
the same color in every panel. Pan and zoom are shared by all panels. Double-click a cell to mark it in every
session; the controls on the right filter cells by the number of sessions they were found in, step through
them, zoom every panel onto the selected one and save the grid as a png.

It reads only the tracking folder (ROICaT's aligned footprints, labels and FOV images), so it needs neither the
sessions' results files nor a GPU.

    python examples/multiview.py tracking_folder
    python examples/multiview.py tracking_folder --save grid.png
"""

from typing import List, Optional, Tuple

import argparse
import math
import os
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path

import cv2
import fastplotlib as fpl
import imageio.v3 as iio
import numpy as np
import scipy.sparse
from imgui_bundle import imgui

from masknmf import cli
from masknmf.multisession import RoicatTrackingResults

COLOR_UNCLUSTERED = (0.45, 0.45, 0.45)
COLOR_BOX = (1.0, 1.0, 1.0)
DIM_UNSELECTED = 0.35
MARGIN_BOX = 4
SPAN_ZOOM_MINIMUM = 80.0


@dataclass
class Multiview:
    """The tracking run, the per-session images derived from it once, and the display settings."""

    tracking: RoicatTrackingResults
    names_session: List[str]
    images_fov: List[np.ndarray]  ## each (height, width), 0-1
    rois_atPixel: List[np.ndarray]  ## each (height, width), session-local ROI index, -1 where no ROI
    weights_atPixel: List[np.ndarray]  ## each (height, width), 0-1
    colors_cluster: np.ndarray  ## (num_clusters, 3)
    sessions_min: int = 1
    opacity: float = 0.6
    show_rois: bool = True
    show_unclustered: bool = False
    cluster_selected: Optional[int] = None
    graphics: list = field(default_factory=list)
    figure: Optional[fpl.Figure] = None
    filepath_saved: str = ""


def load_tracking(path: str) -> RoicatTrackingResults:
    """
    Load a tracking run from a manifest, a tracking folder or an experiment folder holding one.

    Parameters
    ----------
    path : str
        Anything ``masknmf view`` accepts as a tracking run.

    Returns
    -------
    RoicatTrackingResults
    """
    source = cli.find_tracking(entry=path)
    if source is None:
        raise ValueError(f"{path} holds no tracking run")
    if source.is_file():
        return RoicatTrackingResults.from_manifest(source)
    return RoicatTrackingResults.from_roicat_dir(source)


def name_sessions(filepaths_session: Tuple[str, ...]) -> List[str]:
    """
    Name each session by the first folder at which its results path leaves the others'. Names that repeat are numbered.

    Parameters
    ----------
    filepaths_session : tuple of str
        One results file per session.

    Returns
    -------
    list of str
    """
    if len(filepaths_session) == 1:
        return [Path(filepaths_session[0]).parent.name]
    dir_common = os.path.commonpath([str(filepath) for filepath in filepaths_session])
    names = [Path(os.path.relpath(str(filepath), dir_common)).parts[0] for filepath in filepaths_session]
    if len(set(names)) < len(names):
        names = [f"{session + 1}: {name}" for session, name in enumerate(names)]
    return names


def scale_to_unit(image: np.ndarray) -> np.ndarray:
    """
    Scale an image to 0-1 between its 0.5 and 99.5 percentiles.

    Parameters
    ----------
    image : np.ndarray
        Shape (height, width). NaNs count as 0.

    Returns
    -------
    np.ndarray
        float32, shape (height, width).
    """
    image = np.nan_to_num(np.asarray(image, dtype=np.float32))
    low, high = np.percentile(image, (0.5, 99.5))
    if high <= low:
        return np.zeros_like(image)
    return np.clip((image - low) / (high - low), 0, 1)


def project_rois(rois: scipy.sparse.spmatrix, frame_shape: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
    """
    The strongest ROI at each pixel, each ROI scaled to a peak of 1.

    Parameters
    ----------
    rois : scipy.sparse.spmatrix
        Shape (num_pixels, num_rois), pixels flattened in C order.
    frame_shape : tuple of int
        (height, width).

    Returns
    -------
    rois_atPixel : np.ndarray
        int64, shape (height, width). -1 where no ROI has weight.
    weights_atPixel : np.ndarray
        float32, shape (height, width).
    """
    rois = scipy.sparse.csc_matrix(rois, dtype=np.float32)
    peaks = np.asarray(rois.max(axis=0).todense()).reshape(-1)
    rois = scipy.sparse.csr_matrix(rois @ scipy.sparse.diags(1.0 / np.maximum(peaks, 1e-12)))
    weights = np.asarray(rois.max(axis=1).todense()).reshape(-1)
    indices = np.asarray(rois.argmax(axis=1)).reshape(-1).astype(np.int64)
    indices[weights <= 0] = -1
    return indices.reshape(frame_shape), weights.reshape(frame_shape).astype(np.float32)


def make_cluster_colors(num_clusters: int, seed: int) -> np.ndarray:
    """
    One saturated random color per cluster.

    Parameters
    ----------
    num_clusters : int
    seed : int

    Returns
    -------
    np.ndarray
        float32, shape (num_clusters, 3), each row's brightest channel at 1.
    """
    colors = np.random.default_rng(seed).uniform(low=0.12, high=1.0, size=(num_clusters, 3))
    return (colors / colors.max(axis=1, keepdims=True)).astype(np.float32)


def build_multiview(tracking: RoicatTrackingResults, seed: int = 0) -> Multiview:
    """
    Derive the per-session images of a tracking run.

    Parameters
    ----------
    tracking : RoicatTrackingResults
    seed : int
        Seeds the cluster colors.

    Returns
    -------
    Multiview
    """
    images_fov = tracking.aligned_fov_images
    if images_fov is None:
        images_fov = [np.zeros(tracking.frame_shape, dtype=np.float32)] * tracking.num_sessions
    projections = [project_rois(rois=rois, frame_shape=tracking.frame_shape) for rois in tracking.aligned_rois]
    return Multiview(
        tracking=tracking,
        names_session=name_sessions(filepaths_session=tracking.session_files),
        images_fov=[scale_to_unit(image=image) for image in images_fov],
        rois_atPixel=[indices for indices, _ in projections],
        weights_atPixel=[weights for _, weights in projections],
        colors_cluster=make_cluster_colors(num_clusters=tracking.num_clusters, seed=seed),
    )


def clusters_shown(view: Multiview) -> np.ndarray:
    """Ids of the clusters found in at least ``view.sessions_min`` sessions."""
    return np.flatnonzero(view.tracking.num_sessions_per_cluster >= view.sessions_min)


def find_cluster_box(view: Multiview, session: int, cluster: int) -> Optional[Tuple[int, int, int, int]]:
    """
    Bounding box of a cluster's ROIs in one session.

    Returns
    -------
    tuple of int or None
        (row_min, row_max, col_min, col_max), or None when the session has no ROI of the cluster.
    """
    labels_atPixel = label_pixels(view=view, session=session)
    rows, cols = np.nonzero(labels_atPixel == cluster)
    if rows.size == 0:
        return None
    return int(rows.min()), int(rows.max()), int(cols.min()), int(cols.max())


def label_pixels(view: Multiview, session: int) -> np.ndarray:
    """
    Cluster id at each pixel of a session.

    Returns
    -------
    np.ndarray
        int64, shape (height, width). -1 under an unclustered ROI, -2 where there is no ROI.
    """
    rois_atPixel = view.rois_atPixel[session]
    labels = view.tracking.labels_by_session[session]
    return np.where(rois_atPixel >= 0, labels[np.maximum(rois_atPixel, 0)], -2)


def render_session(view: Multiview, session: int) -> np.ndarray:
    """
    A session's aligned FOV with its ROIs blended on top under the current settings; the selected cluster is opaque and boxed, the others dimmed.

    Returns
    -------
    np.ndarray
        float32 rgb, shape (height, width, 3), 0-1.
    """
    image_fov = view.images_fov[session][:, :, None]  ## (height, width, 1)
    if not view.show_rois:
        return np.repeat(image_fov, 3, axis=2)

    labels_atPixel = label_pixels(view=view, session=session)
    is_shown_cluster = np.zeros(view.tracking.num_clusters + 2, dtype=bool)  ## index label + 2
    is_shown_cluster[clusters_shown(view=view) + 2] = True
    is_shown_cluster[1] = view.show_unclustered
    colors = np.concatenate([np.zeros((1, 3)), [COLOR_UNCLUSTERED], view.colors_cluster]).astype(np.float32)
    if view.cluster_selected is not None:
        colors = colors * DIM_UNSELECTED
        colors[view.cluster_selected + 2] = view.colors_cluster[view.cluster_selected]

    weights = view.weights_atPixel[session] * is_shown_cluster[labels_atPixel + 2]  ## (height, width)
    opacity = np.where(labels_atPixel == view.cluster_selected, 1.0, view.opacity)
    alpha = (opacity * weights)[:, :, None]
    image = image_fov * (1 - alpha) + colors[labels_atPixel + 2] * alpha

    if view.cluster_selected is not None:
        box = find_cluster_box(view=view, session=session, cluster=view.cluster_selected)
        if box is not None:
            row_min, row_max, col_min, col_max = box
            image = np.ascontiguousarray(image)
            cv2.rectangle(
                image,
                (col_min - MARGIN_BOX, row_min - MARGIN_BOX),
                (col_max + MARGIN_BOX, row_max + MARGIN_BOX),
                COLOR_BOX,
                1,
            )
    return image.astype(np.float32)


def grid_shape(num_sessions: int) -> Tuple[int, int]:
    """(rows, cols) of the squarest grid, wider than tall, that holds every session."""
    cols = math.ceil(math.sqrt(num_sessions))
    return math.ceil(num_sessions / cols), cols


def render_grid(view: Multiview, gap: int = 6) -> np.ndarray:
    """
    Every session's rendered image tiled into one picture, session names in the corners.

    Returns
    -------
    np.ndarray
        uint8 rgb, shape (rows * (height + gap) - gap, cols * (width + gap) - gap, 3).
    """
    height, width = view.tracking.frame_shape
    rows, cols = grid_shape(num_sessions=view.tracking.num_sessions)
    grid = np.zeros((rows * (height + gap) - gap, cols * (width + gap) - gap, 3), dtype=np.uint8)
    for session, name in enumerate(view.names_session):
        row, col = divmod(session, cols)
        image = np.ascontiguousarray((render_session(view=view, session=session) * 255).astype(np.uint8))
        cv2.putText(image, name, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        top, left = row * (height + gap), col * (width + gap)
        grid[top : top + height, left : left + width] = image
    return grid


def refresh(view: Multiview) -> None:
    """Redraw every panel under the current settings."""
    for session, graphic in enumerate(view.graphics):
        graphic.data = render_session(view=view, session=session)


def select_cluster(view: Multiview, cluster: Optional[int]) -> None:
    """Mark a cluster in every session, or clear the selection with None."""
    view.cluster_selected = cluster
    refresh(view=view)


def select_at_pixel(view: Multiview, session: int, event) -> None:
    """Double-click handler of a session's image: select the cluster of the ROI under the pointer."""
    col, row = event.pick_info["index"]
    height, width = view.tracking.frame_shape
    if not (0 <= row < height and 0 <= col < width):
        return
    label = int(label_pixels(view=view, session=session)[int(row), int(col)])
    select_cluster(view=view, cluster=label if label >= 0 else None)


def step_cluster(view: Multiview, delta: int) -> None:
    """Select the cluster ``delta`` places from the selected one among those shown, wrapping."""
    clusters = clusters_shown(view=view)
    if clusters.size == 0:
        return
    if view.cluster_selected is None:
        position = 0 if delta > 0 else clusters.size - 1
    else:
        position = (int(np.searchsorted(clusters, view.cluster_selected)) + delta) % clusters.size
    select_cluster(view=view, cluster=int(clusters[position]))
    zoom_to_selected(view=view)


def zoom_to_selected(view: Multiview) -> None:
    """Center every panel on the selected cluster; zoom in on it only from the whole-FOV view."""
    boxes = [
        find_cluster_box(view=view, session=session, cluster=view.cluster_selected)
        for session in range(view.tracking.num_sessions)
    ]
    boxes = np.array([box for box in boxes if box is not None])  ## (sessions with the cluster, 4)
    if boxes.size == 0:
        return
    row_min, col_min = boxes[:, 0].min(), boxes[:, 2].min()
    row_max, col_max = boxes[:, 1].max(), boxes[:, 3].max()
    center_row, center_col = (row_min + row_max) / 2, (col_min + col_max) / 2
    height, width = view.tracking.frame_shape
    camera = view.figure[0, 0].camera
    if camera.width >= width or camera.height >= height:
        span_x = span_y = max(4.0 * max(row_max - row_min, col_max - col_min), SPAN_ZOOM_MINIMUM)
    else:
        span_x, span_y = camera.width, camera.height
    for session in range(view.tracking.num_sessions):
        subplot = view.figure[divmod(session, view.figure.shape[1])]
        subplot.camera.show_rect(
            center_col - span_x / 2, center_col + span_x / 2, center_row - span_y / 2, center_row + span_y / 2
        )


def reset_zoom(view: Multiview) -> None:
    """Show the whole FOV in every panel."""
    for session in range(view.tracking.num_sessions):
        view.figure[divmod(session, view.figure.shape[1])].auto_scale(zoom=0.95)


def save_grid(view: Multiview, filepath: str) -> None:
    """Write the tiled grid of every session, as currently displayed, to a png."""
    iio.imwrite(filepath, render_grid(view=view))
    view.filepath_saved = filepath


def draw_controls(view: Multiview) -> None:
    """The imgui panel docked right of the session grid."""
    tracking = view.tracking
    changed_min, view.sessions_min = imgui.slider_int("min sessions", view.sessions_min, 1, tracking.num_sessions)
    changed_rois, view.show_rois = imgui.checkbox("ROIs", view.show_rois)
    changed_opacity, view.opacity = imgui.slider_float("opacity", view.opacity, 0.0, 1.0)
    changed_unclustered, view.show_unclustered = imgui.checkbox("unclustered ROIs", view.show_unclustered)
    if changed_min and view.cluster_selected is not None:
        if view.cluster_selected not in clusters_shown(view=view):
            view.cluster_selected = None
    if changed_min or changed_rois or changed_opacity or changed_unclustered:
        refresh(view=view)

    imgui.text(f"{clusters_shown(view=view).size} of {tracking.num_clusters} cells shown")
    imgui.separator()

    if imgui.button("previous"):
        step_cluster(view=view, delta=-1)
    imgui.same_line()
    if imgui.button("next"):
        step_cluster(view=view, delta=1)
    imgui.same_line()
    if imgui.button("clear"):
        select_cluster(view=view, cluster=None)
    if imgui.button("zoom to cell") and view.cluster_selected is not None:
        zoom_to_selected(view=view)
    imgui.same_line()
    if imgui.button("whole FOV"):
        reset_zoom(view=view)

    if view.cluster_selected is None:
        imgui.text_disabled("double-click a cell to select it")
    else:
        members = tracking.find_members(view.cluster_selected)
        imgui.text(f"cell {view.cluster_selected}")
        for session, name in enumerate(view.names_session):
            rois = ", ".join(str(int(roi)) for roi in members[session])
            imgui.text(f"  {name}: {'ROI ' + rois if rois else 'not found'}")
    imgui.separator()

    if imgui.button("save png"):
        save_grid(view=view, filepath=str(Path.cwd() / "multiview.png"))
    if view.filepath_saved:
        imgui.text_wrapped(f"saved {view.filepath_saved}")


def open_multiview(view: Multiview, size: Tuple[int, int] = (1700, 1000)) -> fpl.Figure:
    """
    Build the window: a grid of session panels sharing one camera controller, and the controls.

    Parameters
    ----------
    view : Multiview
    size : tuple of int
        Window (width, height) in logical pixels.

    Returns
    -------
    fpl.Figure
        Not yet shown.
    """
    figure = fpl.Figure(
        shape=grid_shape(num_sessions=view.tracking.num_sessions),
        controller_ids="sync",
        size=size,
        canvas_kwargs={"title": "masknmf multiview"},
    )
    view.figure = figure
    for subplot in figure:
        subplot.title = ""
        subplot.toolbar = False
        subplot.tooltip.enabled = False
        subplot.axes.visible = False
    for session, name in enumerate(view.names_session):
        subplot = figure[divmod(session, figure.shape[1])]
        subplot.title = name
        graphic = subplot.add_image(render_session(view=view, session=session))
        graphic.add_event_handler(partial(select_at_pixel, view, session), "double_click")
        view.graphics.append(graphic)
    figure.add_imgui_window(partial(draw_controls, view), location="right", size=330, title="Cells")
    return figure


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("tracking", help="a tracking manifest, a tracking folder, or the folder holding one")
    parser.add_argument("--min-sessions", type=int, default=1, help="show only cells found in at least this many")
    parser.add_argument("--cell", type=int, default=None, help="cluster id to select on opening")
    parser.add_argument("--save", default=None, help="write the grid to this png and exit without opening a window")
    parser.add_argument("--seed", type=int, default=0, help="seeds the cell colors")
    args = parser.parse_args()

    view = build_multiview(tracking=load_tracking(path=args.tracking), seed=args.seed)
    view.sessions_min = args.min_sessions
    view.cluster_selected = args.cell
    if args.save is not None:
        save_grid(view=view, filepath=args.save)
        print(f"saved {args.save}")
        return
    open_multiview(view=view).show()
    reset_zoom(view=view)
    if args.cell is not None:
        zoom_to_selected(view=view)
    fpl.loop.run()


if __name__ == "__main__":
    main()
