# Shared imgui/fastplotlib widgets

Target: `masknmf.visualization.imgui`. Consumers: masknmf `ClassificationVis`, mbo_utilities `ManualRoiWidget` (`mbo <path> --widget manualroi`). Dependency direction is `mbo_utilities -> masknmf`.

Status: `have` = already in the package (curation-viz) · `extract` = pull out of an existing GUI · `merge` = one widget from two divergent copies.

| Widget | Module | Status | Inputs | Outputs | fastplotlib / array surface | Intended use |
|---|---|---|---|---|---|---|
| `CheckboxWindow` | `widgets` | have | `label: str`, `value: bool` | `.value` | `fpl.ui.ImguiWindow`; `figure.add_imgui_window(win, location=, size=)` | one-off toggle panel |
| `component_at_pixel` | `picking` | have | sparse `a` (px, comp), `centers` (comp, 2), `fov_shape`, `pick_index` (col, row), `mask`, `radius` | comp index \| `None` | `graphic.pick_info["index"]`; torch sparse `.indices()/.values()` | click -> ROI id, nearest center wins |
| `contours_to_bbox` | `picking` | have | `fov_shape`, `contour` (N, 2), `extra_space` | `(lower, upper)` xyz | numpy only | ROI -> zoom box |
| `zoom_to_bbox` | `picking` | have | `subplot`, `graphic`, `lower`, `upper` | none (mutates) | `graphic.map_model_to_world`; `subplot.x_range/.y_range` | frame one ROI |
| `resolve_time_reference` | `layout` | have | `num_frames`, `frame_timings`, `ref_range`, `axis` | `(ref_range, frame_timings)` | NDWidget `ref_range` dict | uniform time axis across viewers |
| `is_notebook_canvas` | `layout` | have | `figure` | `bool` | `figure.canvas.__class__.__name__` | `show()` dispatch (HBox vs window) |
| `MoviePlayer` | `movie_player` | have | lazy `(T, H, W)`, `fps` | `frame(region)` 2D; `draw() -> changed` | reads `ACArray`/`LazyArray`; feeds `ImageGraphic.data` | transport bar over a lazy movie |
| `LabelSet` | `labels` | extract | `names: tuple[str]`, `class_labels` (n,) int64 | `add/remove/assign`, `-1` = unlabeled | none (pure data) | class registry, tab10 colors, `1`-`9`/`0` hotkeys |
| `RoiOrder` | `table` | extract | per-ROI columns dict, filter label, area range, sort col/dir | `order` (k,) int idx | none | filter + stable sort model behind the table |
| `RoiTable` | `table` | extract | `RoiOrder`, `LabelSet`, column formatters, `current` | clicked row -> new current | `imgui.ListClipper`, `table_get_sort_specs` | scrollable sortable ROI list, scroll-to-current |
| `OverlayCompositor` | `overlay` | merge | bg 2D, label image `uint16` or mask `(H,W)`, per-label colors, `alpha`, `selected` | `rgba (H, W, 4)` | `ImageGraphic.data/.vmin/.vmax/.alpha/.visible`, `alpha_mode="blend"`, `offset` | bg + mask blend, selected ROI rim |
| `LabelImageMasks` | `masks` | extract | closed stroke `[(x, y)]`, `labels uint16` | updated `labels`, per-ROI px `counts` | `cv2.fillPoly`, `cv2.morphologyEx(MORPH_GRADIENT)` | stroke -> non-overlapping mask, add/delete/renumber |
| `StrokeDrawer` | `draw` | extract | `subplot`, armed flag | closed stroke, click-pick `(row, col)` | `subplot.renderer.add_event_handler("pointer_down/move/up")`, `map_screen_to_world`, `LineGraphic.data/.visible`, `subplot.controller.controls["mouse1"]` pop/restore, `world_object.material.pick_write = False` | freehand outline; suppress pan while armed |
| `CropView` | `crop` | extract | `fov` 2D, `centroid` (y, x), `(h, w)` | crop `(h, w)`, origin `(top, left)` | none | zero-padded context crop, roicat-centred convention |
| `BackgroundSelector` | `panels` | extract | `{name: [img per session]}`, `{name: movie}` | active 2D image or movie | `ImageGraphic.data/.vmin/.vmax` | swap FOV background / demixed movie |
| `ProgressRow` | `panels` | extract | `class_labels`, `session_sizes` | draws; "next unlabeled" click | none | `labeled n/total`, per-session split |
| `KeybindsPopup` | `panels` | extract | `tuple[(key, action)]` | draws | `imgui.begin` + 2-col table | help overlay |
| `LabelStore` | `store` | extract | npz path and/or hdf5 file list, `session_sizes` | writes `class_labels`, `label_names`, `roi_masks`, `labels_complete` | none | autosave on every label change |
| `AsyncLoad` | `loader` | extract | callable, poll per frame | result \| error, status text | none | non-blocking build (ROICaT) behind a live GUI |
| `SummaryImageViewer` | `summary` | merge | `{name: 2D}`, `{name: (T,H,W)}`, `figure`; optional metadata scan, ROI contours | popup; `set_highlight((y0,x0,h,w))`; PNG export | `figure.imgui_renderer.backend.register_texture/unregister_texture`, `wgpu` texture + `queue.write_texture`, `imgui.draw_list.add_image` | full-FOV browse: pan/zoom/cmap/contrast/histogram/pixel values |

## `SummaryImageViewer` merge

| Aspect | mbo_utilities `gui/widgets/summary_image.py` (809 L) | masknmf `visualization/summary_widget.py` (~500 L) | Unified |
|---|---|---|---|
| Base | `Widget` ABC (`is_supported`, `draw`, `cleanup`) | plain class, takes `figure` | plain core + optional `Widget` adapter |
| Image source | auto-scan array metadata (ops.npy) | explicit `{name: 2D}` dict | dict core; metadata scan = optional provider |
| Movies | none | `MoviePlayer`, per-movie fixed range | core |
| Highlight | none | `set_highlight` rect | core |
| ROI overlay | suite2p `stat.npy` contours | none | optional provider |
| Export | PNG via `portable_file_dialogs` | none | optional |
| cmap | syncs with fpl graphic | fixed 5-cmap list | core list + optional sync hook |
| Shared | `_GpuImage`, `_to_rgba`, `_data_range`, `_auto_range`, `_format_value`, pan/zoom, contrast modes, histogram, pixel-value overlay | identical | core |

## Consumer wiring

| | `ClassificationVis` (masknmf) | `--widget manualroi` (mbo_utilities) |
|---|---|---|
| Figure | owns `fpl.Figure(size=(1200,900))` | `iw.figure` (`MboNDViewer`) |
| Host | two `add_imgui_window` panels: `top`, `right` | `ROI` tab inside `TimeSeriesViewer` tab bar |
| ROI source | `(n, Y, X)` footprint stack (ROICaT) | `uint16` label image drawn by hand |
| Adds ROIs | no (fixed set) | yes, `StrokeDrawer` + `LabelImageMasks` |
| Table columns | `id, label, area, peak, snr, skew` | `id, label, area` |
| Persistence | hdf5 `DemixingResults/*` + npz | `manual_masks.npy` + `LabelStore` |

## manualroi host change

Today `run_gui.py:660` *replaces* `PreviewDataWidget` with `ManualRoiWidget`, so manualroi loses windowing (contrast, z/t sliders, projections). Required end state: manualroi keeps every `PreviewDataWidget` tab and gains one more.

| Item | Now | Target |
|---|---|---|
| `run_gui.py` `elif widget == "manualroi"` | `ManualRoiWidget(iw, path)` only | build `PreviewDataWidget` as for `preview`, then attach ROI state to it |
| Tab bar (`viewers/time_series.py:39`) | `Preview \| Signal Quality \| Run \| BioHPC` | `+ ROI`, shown only when ROI state is attached |
| `ManualRoiWidget` panels | `add_imgui_window("top")` + `("right")` | `draw_tab()` drawing tools + ROI list inline; edge windows dropped |
| Canvas handlers / graphics | `pointer_down/move/up`, overlay + stroke graphics | unchanged, independent of panel host |
