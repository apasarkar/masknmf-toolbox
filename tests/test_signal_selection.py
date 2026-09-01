"""
Offscreen tests for the signal selection GUI.

``SignalSelectionVis`` paints masks into a uint16 label image over a real
fastplotlib figure, so these need the offscreen rendercanvas backend that
``tests/conftest.py`` selects for the whole suite. They pin the mask
bookkeeping, the trace extraction and its storage, the table order, the
pointer wiring through the real renderer, and that every panel draws.
"""

import numpy as np
import pytest
import torch

from masknmf.compression import PMDArray
from masknmf.demixing.demixing_results import DemixingResults
from masknmf.visualization.imgui.labels import UNLABELED, LabelSet
from masknmf.visualization.imgui.table import RoiOrder
from masknmf.visualization.rois import (
    FootprintSet,
    RoiLabelStore,
    build_pick_map,
    feather_mask,
    feathered_rgba,
)
from masknmf.visualization.signal_selection_vis import SignalSelectionVis
from masknmf.visualization.traces import (
    TraceSet,
    baseline,
    display_trace,
    make_entry,
    roi_trace,
    trace_stats,
)

T, NY, NX = 40, 32, 32
RANK = 5
N_SIGNALS = 3
FIGURE_SIZE = (1200, 900)


def offscreen_selected() -> bool:
    from rendercanvas.auto import RenderCanvas

    return "offscreen" in RenderCanvas.__module__


pytestmark = pytest.mark.skipif(
    not offscreen_selected(),
    reason="offscreen rendercanvas backend not selected (another backend was "
    "imported before RENDERCANVAS_FORCE_OFFSCREEN took effect)",
)


def sparse_columns(n_pixels: int, n_columns: int, seed: int) -> torch.Tensor:
    """A sparse (pixels, columns) matrix with a contiguous block per column."""
    generator = torch.Generator().manual_seed(seed)
    dense = torch.rand(n_pixels, n_columns, generator=generator)
    dense[dense < 0.85] = 0
    dense[0, :] = 1.0  # every column keeps at least one pixel
    return dense.to_sparse_coo().coalesce()


@pytest.fixture
def pmd_array() -> PMDArray:
    u = sparse_columns(NY * NX, RANK, seed=0)
    v = torch.rand(RANK, T, generator=torch.Generator().manual_seed(1))
    return PMDArray.from_tensors(
        (T, NY, NX), u, v, torch.rand(NY, NX) + 1.0, torch.rand(NY, NX) + 1.0
    )


@pytest.fixture
def demixing_results() -> DemixingResults:
    generator = torch.Generator().manual_seed(2)
    return DemixingResults(
        (T, NY, NX),
        sparse_columns(NY * NX, RANK, seed=0),
        torch.rand(RANK, T, generator=generator),
        sparse_columns(NY * NX, N_SIGNALS, seed=3),
        torch.rand(T, N_SIGNALS, generator=generator),
        mean_img=torch.rand(NY, NX) + 1.0,
        var_img=torch.rand(NY, NX) + 1.0,
        # DemixingResults builds a (pixels, 1) baseline when b is omitted, which
        # its own ROI averages then cannot multiply, so give it a flat one
        b=torch.rand(NY * NX),
    )


@pytest.fixture
def vis(pmd_array):
    widget = SignalSelectionVis(pmd_array, size=FIGURE_SIZE)
    widget.show()
    yield widget
    widget.close()


@pytest.fixture
def demix_vis(demixing_results):
    widget = SignalSelectionVis(demixing_results, size=FIGURE_SIZE)
    widget.show()
    yield widget
    widget.close()


def square(x0, y0, size):
    """Stroke tracing a square with its top left corner at (x0, y0)."""
    return [
        (float(x0), float(y0)),
        (float(x0 + size), float(y0)),
        (float(x0 + size), float(y0 + size)),
        (float(x0), float(y0 + size)),
    ]


def draw_frames(widget, n=2):
    """Render n frames, which runs every imgui panel."""
    canvas = widget.fov_widget.figure.canvas
    canvas.request_draw(canvas._draw_frame)
    for _ in range(n):
        canvas.draw()


def wait_for_traces(widget, timeout=30.0):
    import time

    deadline = time.perf_counter() + timeout
    while widget.trace_busy and time.perf_counter() < deadline:
        time.sleep(0.02)
    assert not widget.trace_busy, "trace threads did not finish"
    widget._poll_traces()


# ----------------------------------------------------------------------
# store
# ----------------------------------------------------------------------


def test_store_add_and_delete_renumbers():
    store = RoiLabelStore(NY, NX)
    mask = np.zeros((NY, NX), bool)
    mask[2:6, 2:6] = True
    assert store.add_roi(mask) == 0
    other = np.zeros((NY, NX), bool)
    other[10:14, 10:14] = True
    assert store.add_roi(other) == 1
    assert store.labels.max() == 2
    assert store.roi_at(3, 3) == 0
    assert store.roi_at(11, 11) == 1
    assert store.roi_at(0, 0) == -1

    uid = store.rois[1].uid
    store.delete_roi(0)
    assert len(store.rois) == 1
    assert store.labels.max() == 1
    assert store.roi_at(11, 11) == 0
    assert store.rois[0].uid == uid, "uid must survive a renumber"


def test_store_rejects_overlap_and_small_masks():
    store = RoiLabelStore(NY, NX, min_pixels=4)
    mask = np.zeros((NY, NX), bool)
    mask[2:6, 2:6] = True
    assert store.add_roi(mask) == 0
    assert store.add_roi(mask) is None, "every pixel is already claimed"
    overlapping = np.zeros((NY, NX), bool)
    overlapping[4:10, 4:10] = True
    index = store.add_roi(overlapping)
    assert index == 1
    assert store.rois[1].area == 36 - 4, "the shared pixels stay with the first ROI"
    tiny = np.zeros((NY, NX), bool)
    tiny[20, 20] = True
    assert store.add_roi(tiny) is None


def test_store_colors_follow_class_then_explicit():
    store = RoiLabelStore(NY, NX)
    mask = np.zeros((NY, NX), bool)
    mask[2:6, 2:6] = True
    store.add_roi(mask)
    hue = store.rgb(0)
    store.set_class(0, 1)
    labeled = store.rgb(0)
    assert labeled != hue
    store.set_color(0, (10, 20, 30))
    assert store.rgb(0) == (10, 20, 30)
    store.set_color(0, None)
    assert store.rgb(0) == labeled


# ----------------------------------------------------------------------
# overlays
# ----------------------------------------------------------------------


def test_feather_mask_falls_off_at_the_edge():
    mask = np.zeros((NY, NX), bool)
    mask[8:24, 8:24] = True
    weights = feather_mask(mask, edge_width=3)
    assert weights[mask].max() == pytest.approx(1.0)
    assert weights[8, 8] < weights[16, 16]
    assert weights[~mask].max() == 0.0


def test_feathered_rgba_selection_wins_and_gets_a_rim():
    ypix, xpix = np.mgrid[4:8, 4:8]
    ypix, xpix = ypix.ravel(), xpix.ravel()
    lam = np.ones(ypix.size, np.float32)
    rgba = feathered_rgba(
        (NY, NX), [(ypix, xpix, lam, (1.0, 0.0, 0.0), 0.5)], (ypix, xpix, (1.0, 0.0, 0.0))
    )
    assert rgba.shape == (NY, NX, 4)
    assert rgba[6, 6, 3] > 200, "the selection fills at the selected alpha"
    assert tuple(rgba[4, 4]) == (255, 255, 255, rgba[6, 6, 3]), "boundary pixels are the rim"
    assert rgba[0, 0, 3] == 0


def test_build_pick_map_gives_contested_pixels_to_the_stronger():
    weak = (np.array([1, 1]), np.array([1, 2]), np.array([0.1, 0.1], np.float32))
    strong = (np.array([1]), np.array([1]), np.array([1.0], np.float32))
    pick = build_pick_map([weak, strong], (4, 4))
    assert pick[1, 1] == 1
    assert pick[1, 2] == 0
    assert pick[0, 0] == -1


def test_footprint_set_from_sparse_matches_the_matrix(demixing_results):
    signals = FootprintSet.from_sparse("demixed", demixing_results.ac_array.a, (NY, NX))
    assert len(signals) == N_SIGNALS
    dense = demixing_results.ac_array.a.to_dense().numpy().reshape(NY, NX, -1)
    for k, (ypix, xpix, lam) in enumerate(signals.footprints):
        assert np.allclose(dense[ypix, xpix, k], lam)
        assert (dense[:, :, k] > 0).sum() == len(ypix)


# ----------------------------------------------------------------------
# traces
# ----------------------------------------------------------------------


def test_roi_trace_matches_a_direct_mean(pmd_array):
    mask = np.zeros((NY, NX), bool)
    mask[4:9, 6:11] = True
    movie = np.asarray(pmd_array[:])
    expected = movie[:, mask].mean(axis=1)
    assert np.allclose(roi_trace(pmd_array, mask), expected, atol=1e-4)


def test_roi_trace_weights_the_mean(pmd_array):
    mask = np.zeros((NY, NX), bool)
    mask[4:9, 6:11] = True
    weights = np.zeros((NY, NX), np.float32)
    weights[6, 8] = 1.0
    movie = np.asarray(pmd_array[:])
    assert np.allclose(roi_trace(pmd_array, mask, weights=weights), movie[:, 6, 8], atol=1e-4)


def test_roi_trace_batches_a_plain_array():
    movie = np.arange(T * NY * NX, dtype=np.float32).reshape(T, NY, NX)
    mask = np.zeros((NY, NX), bool)
    mask[1:3, 1:3] = True
    assert np.allclose(roi_trace(movie, mask, batch=7), movie[:, mask].mean(axis=1))


def test_roi_trace_rejects_an_empty_mask(pmd_array):
    with pytest.raises(ValueError):
        roi_trace(pmd_array, np.zeros((NY, NX), bool))


def test_display_trace_scales_a_zeroed_trace_by_the_same_baseline():
    trace = np.linspace(10.0, 20.0, 100).astype(np.float32)
    f0 = baseline(trace)
    assert f0 == pytest.approx(float(np.percentile(trace, 20)))
    entry = make_entry(trace, f0)
    assert display_trace(entry)[0] == pytest.approx((trace[0] - f0) / f0 * 100)
    residual = make_entry(trace - f0, f0, zeroed=True)
    assert np.allclose(display_trace(residual), display_trace(entry))
    assert np.allclose(display_trace(entry, dff=False), trace)


def test_display_trace_stays_raw_without_a_positive_baseline():
    trace = np.linspace(-5.0, 5.0, 50).astype(np.float32)
    assert baseline(trace) is None
    entry = make_entry(trace, baseline(trace))
    assert np.allclose(display_trace(entry), trace)


def test_trace_stats_reports_frames_mean_and_snr():
    trace = np.zeros(100, np.float32)
    trace[50] = 10.0
    frames, mean, peak, snr = trace_stats(trace)
    assert (frames, peak) == (100, 10.0)
    assert mean == pytest.approx(0.1)
    assert snr == 0.0, "a flat baseline has no spread to measure against"
    noisy = np.random.default_rng(0).normal(0, 1, 500).astype(np.float32)
    noisy[100] = 20.0
    assert trace_stats(noisy)[3] > 5


def test_trace_set_prunes_dropped_keys():
    trace_set = TraceSet("pmd", {1: {}, 2: {}, 3: {}})
    trace_set.prune([1, 3])
    assert sorted(trace_set.data) == [1, 3]


# ----------------------------------------------------------------------
# table order
# ----------------------------------------------------------------------


def test_roi_order_filters_compose_and_reveal_undoes_them():
    labels = np.array([0, 1, 0, UNLABELED], np.int64)
    columns = {
        "source": np.array([0, 0, 1, 1], np.int64),
        "area": np.array([10, 20, 30, 40], np.int64),
    }
    order = RoiOrder(columns, labels, 4)
    order.category_column = "source"
    order.set_range_column("area")
    order.rebuild()
    assert list(order.order) == [0, 1, 2, 3]

    order.filter_label = 0
    order.category = 0
    order.rebuild()
    assert list(order.order) == [0]
    assert sorted(order.hidden_by(2)) == ["source"]
    assert sorted(order.hidden_by(3)) == ["label", "source"]

    assert sorted(order.reveal(3)) == ["label", "source"]
    assert order.current == 3

    order.sort_column = 2  # the area column, after labels and source
    order.ascending = False
    order.rebuild()
    assert list(order.order) == [3, 2, 1, 0]


def test_roi_order_next_unlabeled_wraps():
    labels = np.array([0, UNLABELED, 0, UNLABELED], np.int64)
    order = RoiOrder({"area": np.arange(4)}, labels, 4)
    order.rebuild()
    assert order.next_unlabeled() and order.current == 1
    assert order.next_unlabeled() and order.current == 3
    assert order.next_unlabeled() and order.current == 1


def test_label_set_remove_shifts_higher_labels():
    labels = LabelSet(3, ("a", "b", "c"), [0, 1, 2])
    assert labels.remove(1)
    assert list(labels.labels) == [0, UNLABELED, 1]
    assert labels.names == ("a", "c")


# ----------------------------------------------------------------------
# widget
# ----------------------------------------------------------------------


def test_stroke_becomes_an_roi_and_selects_it(vis):
    assert vis.add_roi(square(4, 4, 8)) == 0
    assert vis.n_rois == 1
    assert vis.store.rois[0].area == 81
    assert vis._selected == 0
    assert vis.roi_masks.shape == (NY, NX, 1)


def test_short_stroke_is_rejected(vis):
    assert vis.add_roi([(1.0, 1.0), (2.0, 2.0)]) is None
    assert vis.n_rois == 0


def test_pointer_drag_draws_through_the_renderer(vis):
    draw_frames(vis)
    vis.set_drawing(True)
    subplot = vis.fov_widget.figure["fov"]
    corners = [(4, 4), (16, 4), (16, 16), (4, 16), (4, 4)]
    screen = [subplot.map_world_to_screen((x, y, 0)) for x, y in corners]
    drawer = vis._drawer
    drawer._down(type("Event", (), {"button": 1, "x": screen[0][0], "y": screen[0][1]})())
    for x, y in screen[1:]:
        drawer._move(type("Event", (), {"button": 1, "x": x, "y": y})())
    drawer._up(type("Event", (), {"button": 1, "x": screen[-1][0], "y": screen[-1][1]})())
    assert vis.n_rois == 1
    assert vis.store.rois[0].area > 100


def test_click_selects_the_roi_under_the_cursor(vis):
    vis.add_roi(square(4, 4, 8))
    vis.add_roi(square(16, 16, 8))
    vis._pick(18, 18)
    assert vis._selected == 1
    vis._pick(6, 6)
    assert vis._selected == 0
    vis._pick(0, 0)
    assert vis._selected == -1


def test_ctrl_click_builds_a_group_that_labels_and_colors_together(vis):
    vis.add_roi(square(4, 4, 8))
    vis.add_roi(square(16, 16, 8))
    vis.select_roi(0)
    vis._pick(18, 18, frozenset({"Ctrl"}))
    assert sorted(vis._buffer) == [(-1, 0), (-1, 1)]
    vis.add_label("soma")
    vis.assign_class(0)
    assert [r.class_index for r in vis.store.rois] == [0, 0]
    vis.set_group_color((1.0, 0.0, 0.0))
    assert vis.store.rgb(0) == vis.store.rgb(1) == (255, 0, 0)
    vis.buffer_clear()
    assert vis._buffer == []


def test_deleting_an_roi_prunes_only_its_traces(vis):
    vis.add_roi(square(4, 4, 8))
    vis.add_roi(square(16, 16, 8))
    kept_uid = vis.store.rois[1].uid
    vis.trace_rois([0, 1])
    wait_for_traces(vis)
    assert len(vis._traces["pmd"].data) == 2
    vis.delete_roi(0)
    assert list(vis._traces["pmd"].data) == [kept_uid]


def test_traces_land_on_every_movie_with_a_shared_baseline(demix_vis):
    demix_vis.add_roi(square(4, 4, 8))
    demix_vis.trace_rois([0])
    wait_for_traces(demix_vis)
    uid = demix_vis.store.rois[0].uid
    pmd = demix_vis._traces["pmd"].data[uid]
    residual = demix_vis._traces["residual"].data[uid]
    assert pmd["trace"].shape == (T,)
    assert residual["f0"] == pmd["f0"]
    assert residual["zeroed"] and not pmd["zeroed"]


def test_table_rows_cover_drawn_and_demixed(demix_vis):
    demix_vis.add_roi(square(4, 4, 8))
    assert demix_vis._rows[0] == (-1, 0)
    assert len(demix_vis._rows) == 1 + N_SIGNALS
    assert demix_vis._formatters["source"](0) == "drawn"
    assert demix_vis._formatters["source"](1) == "demixed"


def test_selecting_a_signal_lists_its_traces(demix_vis):
    demix_vis.select_signal(1)
    assert demix_vis._selected_signal == 1
    keys = demix_vis._selection_trace_keys()
    assert sorted(keys) == [("pmd", "signal", 1), ("residual", "signal", 1)]
    assert set(demix_vis._trace_rows()) == set(keys)
    assert demix_vis._trace_color(keys[0]) == demix_vis.signals.color(1)


def test_promoting_a_signal_copies_it_into_the_drawn_set(demix_vis):
    index = demix_vis.promote_signal(2)
    assert index == 0
    assert demix_vis.store.rois[0].source == "demixed:2"
    assert demix_vis.store.rois[0].area == demix_vis.signals.area(2)
    assert demix_vis._formatters["source"](demix_vis._row_index[(0, 2)]) == "demixed · promoted"
    assert demix_vis.promote_signal(2) is None, "a promoted signal cannot be promoted twice"


def test_combined_footprints_appends_the_drawn_masks(demix_vis):
    demix_vis.add_roi(square(4, 4, 8))
    combined = demix_vis.combined_footprints()
    assert combined.shape == (NY, NX, N_SIGNALS + 1)
    assert np.array_equal(combined[:, :, -1], demix_vis.roi_masks[:, :, 0])


def test_export_writes_masks_labels_and_names(demix_vis, tmp_path):
    demix_vis.add_roi(square(4, 4, 8))
    demix_vis.add_label("soma")
    soma = demix_vis.label_names.index("soma")
    demix_vis.select_roi(0)
    demix_vis.assign_class(soma)
    path = demix_vis.export_rois(str(tmp_path / "rois"))
    assert path.endswith(".npz")
    data = np.load(path)
    assert data["spatial_footprints"].shape == (NY, NX, 1)
    assert list(data["class_labels"]) == [soma]
    assert list(data["label_names"]) == list(demix_vis.label_names)
    assert data["spatial_footprints_combined"].shape == (NY, NX, N_SIGNALS + 1)


def test_export_without_rois_raises(vis, tmp_path):
    with pytest.raises(ValueError):
        vis.export_rois(str(tmp_path / "rois.npz"))


def test_source_switch_pauses_the_other_movies(demix_vis):
    residual = demix_vis._source_names.index("residual movie")
    demix_vis._set_source(residual)
    for name, nd in demix_vis._nd_images.items():
        assert nd.pause == (name != "residual movie")
        assert nd.graphic.visible == (name == "residual movie")


def test_frame_round_trips_through_the_reference_index(vis):
    vis.set_frame(11)
    assert vis.current_frame() == 11
    vis.set_frame(T + 100)
    assert vis.current_frame() == T - 1


def test_every_panel_draws(demix_vis):
    demix_vis.add_roi(square(4, 4, 8))
    demix_vis.add_label("soma")
    demix_vis.select_roi(0)
    demix_vis.assign_class(0)
    demix_vis.trace_rois([0])
    wait_for_traces(demix_vis)
    draw_frames(demix_vis, n=3)
    demix_vis.select_signal(0)
    demix_vis.trace_listed()
    wait_for_traces(demix_vis)
    demix_vis._dff = False
    draw_frames(demix_vis, n=3)
    demix_vis.toggle_drawn_overlay()
    demix_vis.toggle_signal_overlay()
    demix_vis.set_drawing(True)
    draw_frames(demix_vis, n=2)
