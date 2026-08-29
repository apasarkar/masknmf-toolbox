# Shared imgui/fastplotlib widgets

Target: `masknmf.visualization.imgui`. Consumers: masknmf `ClassificationVis`, mbo_utilities `ManualRoiWidget` (`mbo <path> --widget manualroi`). Dependency direction is `mbo_utilities -> masknmf`.

Status: `have` = pre-existing · `done` = extracted and in use by both consumers · `one host` = extracted but only one consumer wires it today.

| Widget | Module | Status | Inputs | Outputs | fastplotlib / array surface | Intended use |
|---|---|---|---|---|---|---|
| `CheckboxWindow` | `widgets` | have | `label: str`, `value: bool` | `.value` | `fpl.ui.ImguiWindow`; `figure.add_imgui_window(win, location=, size=)` | one-off toggle panel |
| `component_at_pixel` | `picking` | have | sparse `a` (px, comp), `centers` (comp, 2), `fov_shape`, `pick_index` (col, row), `mask`, `radius` | comp index \| `None` | `graphic.pick_info["index"]`; torch sparse `.indices()/.values()` | click -> ROI id, nearest center wins |
| `contours_to_bbox` | `picking` | have | `fov_shape`, `contour` (N, 2), `extra_space` | `(lower, upper)` xyz | numpy only | ROI -> zoom box |
| `zoom_to_bbox` | `picking` | have | `subplot`, `graphic`, `lower`, `upper` | none (mutates) | `graphic.map_model_to_world`; `subplot.x_range/.y_range` | frame one ROI |
| `resolve_time_reference` | `layout` | have | `num_frames`, `frame_timings`, `ref_range`, `axis` | `(ref_range, frame_timings)` | NDWidget `ref_range` dict | uniform time axis across viewers |
| `is_notebook_canvas` | `layout` | have | `figure` | `bool` | `figure.canvas.__class__.__name__` | `show()` dispatch (HBox vs window) |
| `MoviePlayer` | `movie_player` | done | lazy `(T, H, W)`, `fps` | `frame(region)` 2D; `draw() -> changed` | reads `ACArray`/`LazyArray`/any `[t]`-indexable; feeds `ImageGraphic.data` | transport bar over a lazy movie |
| `LabelSet` | `labels` | done | `names: tuple[str]`, `class_labels` (n,) int64 | `add/remove/assign/clear/resize`, `-1` = unlabeled | none (pure data) | class registry, tab10 colors, `1`-`9`/`0` hotkeys |
| `RoiOrder` | `table` | done | per-ROI columns dict, filter label, range column, sort col/dir | `order` (k,) int idx; `step`/`goto`/`next_unlabeled`/`step_group` | none | filter + stable sort model behind the table |
| `draw_roi_table` / `draw_filter_row` | `table` | done | `RoiOrder`, `LabelSet`, column formatters, `current` | clicked row -> new current | `imgui.ListClipper`, `table_get_sort_specs` | scrollable sortable ROI list, scroll-to-current |
| `label_image_rgba` / `footprint_rgba` / `OverlayPair` | `overlay` | done | bg 2D, label image `uint16` or mask `(H,W)`, per-label colors, `alpha`, `selected`, `outline_width` | `rgba (H, W, 4)` | `ImageGraphic.data/.vmin/.vmax/.alpha/.visible`, `alpha_mode="blend"`, `offset` | bg + mask blend, selected ROI rim |
| `OUTLINE_WIDTH` / `rim_kernel` | `masks` | done | half-width in px (default `1`) | `(2w+1, 2w+1)` uint8 kernel | `cv2.morphologyEx(MORPH_GRADIENT)`, `cv2.erode` | one knob for mask boundary and selected-rim thickness |
| `LabelImage` | `masks` | done | closed stroke `[(x, y)]`, `labels uint16` | updated `labels`, per-ROI px `counts`, `areas()`, `edges(width)`, `footprints()` | `cv2.fillPoly`, `cv2.morphologyEx(MORPH_GRADIENT)` | stroke -> non-overlapping mask, add/delete/renumber |
| `StrokeDrawer` | `draw` | done | `subplot`, armed flag | closed stroke, click-pick `(row, col)` | `subplot.renderer.add_event_handler("pointer_down/move/up")`, `map_screen_to_world`, `LineGraphic.data/.visible`, `subplot.controller.controls["mouse1"]` pop/restore, `world_object.material.pick_write = False` | freehand outline; suppress pan while armed |
| `crop_origin` / `context_crop` | `crop` | one host | `fov` 2D, `centroid` (y, x), `(h, w)` | crop `(h, w)`, origin `(top, left)` | none | zero-padded context crop, roicat-centred convention |
| `draw_label_buttons` / `draw_label_editor` | `panels` | one host | `LabelSet` | clicked label index, `UNLABELED` (-1), `UNLABEL_ALL` (-2); add/remove | none | assign and manage classes |
| `draw_progress` | `panels` | done | `class_labels`, `session_sizes` | draws; "next unlabeled" click | none | `labeled n/total`, per-session split |
| `draw_keybinds_popup` | `panels` | done | `tuple[(key, action)]` | draws | `imgui.begin` + 2-col table | help overlay |
| `LabelStore` | `store` | done | npz path and/or hdf5 file list, `session_sizes` | writes `class_labels`, `label_names`, `roi_masks`, `labels_complete` | none | autosave on every label change |
| `AsyncLoad` | `loader` | done | callable, poll per frame | result \| error, status text | none | non-blocking build behind a live GUI (ROICaT; projection reduces) |
| `SummaryImageViewer` | `summary` | done | `{name: 2D}`, `{name: (T,H,W)}`, `figure`; hooks `roi_provider`, `on_export`, `extra_toolbar` | popup; `set_highlight((y0,x0,h,w))` | `figure.imgui_renderer.backend.register_texture/unregister_texture`, `wgpu` texture + `queue.write_texture`, `imgui.draw_list.add_image` | full-FOV browse: pan/zoom/cmap/contrast/histogram/pixel values/ROI contours/export |

## Outline thickness

`OUTLINE_WIDTH` in `masks.py` is the half-width, in pixels, of both the mask boundary and the selected-ROI rim; `rim_kernel(w)` turns it into the `(2w+1, 2w+1)` structuring element that `MORPH_GRADIENT` and `erode` take. It is `1` — a one-pixel line each side of a boundary, which is what a dense field of small ROIs needs.

Three places take it, widest to narrowest scope:

| Change | Effect |
|---|---|
| `masks.OUTLINE_WIDTH` | the default everywhere, both GUIs |
| `label_image_rgba(..., outline_width=w)` / `LabelImage.edges(w)` | one call |
| `ManualRoiWidget.outline_width` | one mbo_utilities widget; passed to both of the above on every `refresh_overlay()` |

It was `2` (a hard-coded 5x5 `_RIM`), which drew a 4px band across a boundary between touching ROIs.

## Consumer wiring

| | `ClassificationVis` (masknmf) | `--widget manualroi` (mbo_utilities) |
|---|---|---|
| Figure | owns `fpl.Figure(size=(1200,900))` | `iw.figure` (`MboNDViewer`), shared with `PreviewDataWidget` |
| Controls | `add_imgui_window(location="top", size=150)` | `add_imgui_window(location="top", size=115)` |
| Table | `add_imgui_window(location="right", size=360)` | `add_imgui_window(location="left", size=300)` |
| Why not right | — | `PreviewDataWidget` owns `right`, the NDWidget sliders own `bottom` |
| ROI source | `(n, Y, X)` footprint stack (ROICaT) | `uint16` label image drawn by hand |
| Adds ROIs | no (fixed set) | yes, `StrokeDrawer` + `LabelImage` |
| Table columns | `id, label, area, peak, snr, skew` | `id, label, area` |
| Background graphic | its own; FOV context crop, mask MIP, or demixed movie | the viewer's own `ImageGraphic` |
| Background sources | per-session FOV images + demixed movie | live movie + `mean` / `max` / `std` over the plane on screen |
| Full FOV images | per-session FOV images, demixed movie | current frame + the three projections, and the `(T, Y, X)` plane as a movie |
| Full FOV highlight | the ROI's crop box | the selected ROI's bounding box |
| Full FOV hooks | none | `roi_provider` (ROI contours), `on_export` (tiff beside the data) |
| Persistence | hdf5 `DemixingResults/*` + npz | `manual_masks.npy` + `LabelStore` npz |

## Top-panel parity

Both hosts draw one control row set in the same order. `-` means the control does not apply to that host.

| Control | `ClassificationVis` | `manualroi` |
|---|---|---|
| Add ROI / Undo / Clear / Save | - (fixed ROI set) | yes |
| prev / next / position slider | yes | yes |
| Open full FOV | yes | yes |
| save note tooltip | hdf5 + npz snippet | npy + npz snippet |
| keybinds (right-aligned) | yes | yes |
| loading / status text | `_loading` | `AsyncLoad.status`, else the ROI status |
| bg checkbox + source combo + `(b)` | yes | yes |
| bg opacity | yes | yes |
| mask checkbox + `(m)` + opacity | yes | yes |
| outlines checkbox + `(o)` | - (footprints have no label image) | yes |
| inline `MoviePlayer` transport | when bg is the demixed movie | - (the NDWidget's own `t` slider is the transport) |
| `draw_progress` + next unlabeled `(u)` | yes | yes |
| new label / add / del | inline copy | `draw_label_editor` |
| per-class buttons, unlabel `(0)`, unlabel all | inline copy | `draw_label_buttons` |

Keybinds are the same set in both, plus `a` / `esc` / `ctrl+z` for drawing in `manualroi`.

## Remaining extraction candidates

- **`ClassificationVis` -> `LabelSet` / `draw_label_editor` / `draw_label_buttons`.** The biggest duplication left: `ClassificationVis` keeps `_label_names` + `_class_labels` as bare fields and inlines its own copies of all three panel widgets (~60 lines that `panels.py` already covers). Adopting `LabelSet` would delete them and put both GUIs on one label implementation. `UNLABEL_ALL` was added to `draw_label_buttons` so the shared version is already feature-complete against the inline one.
- **`ClassificationVis` -> `RoiOrder`.** `_rebuild_order` / `_order` / `_pos` / `step` / `goto` / `goto_next_unlabeled` / `_step_group` / `_draw_table` are a hand-rolled copy of `RoiOrder` + `draw_roi_table` + `draw_filter_row`, differing only in the column set.
- **Background source combo.** Both hosts now have one, but over genuinely different source sets (per-session FOV images and a demixed movie vs. a live viewer graphic and projections of it). A `BackgroundSources` model holding `{name: image}` + an optional movie, with a `draw()` that returns the selection, would cover both.
- **Projection reduce.** `mbo_utilities.gui.manual_roi.compute_projections` (evenly spaced sample -> mean/max/std) is generic and belongs beside `AsyncLoad`.
- **`PlaneMovie`.** `mbo_utilities.gui.manual_roi.PlaneMovie` pins the non-`(T, Y, X)` dims of an n-d array to give `MoviePlayer` and the reducer a lazy 3D view. Only mbo needs it today because only mbo hosts an n-d viewer, but it is pure index bookkeeping with no mbo dependencies.
- **`roi.py`** — `PMDWidget`'s rectangle-ROI machinery in `interactive_guis.py` (`add_rectangle` / `resize_rect` / `end_resize`, the per-graphic selector `OrderedDict`, `rect_selector_kwargs`): a self-contained "draw one rect, mirror it across synced subplots, fire a callback on release" widget.
- **`selection.py`** — the `ImageHighlightSelector` wiring repeated in curation / multisession / demixing (`lut="tab10"`, `lut_wrap="repeat"`, contour pixel options, white options color): a `make_contour_selector(contours, **overrides)` factory, plus the `SelectionVector` + global/local index-map pattern from multisession.
- **`layout.grid_extents`** — curation and demixing build fractional extents dicts by hand; a rows/cols -> extents helper would remove both.
- **`layout` camera linking** — multisession shares one camera across the video and MIP figures; motion_vis links trace subplots on x only (`add_camera(..., include_state={"x", "width"})`). Two patterns: `share_camera(figures)` and `link_x(subplots)`.
- **traces** — normalize/offset trace stacks fed to `fpl.utils.heatmap_to_positions` (`CurationVis._refresh_traces`, the trace panels in `SingleSessionDemixingVis._click_update`). Data-side siblings (`extract_per_trace_roi_averages` in demixing_vis, `get_roi_avg` in plots.py) belong in `masknmf.demixing` rather than here.

## `SummaryImageViewer` vs mbo's own summary widget

mbo_utilities still has a second, unrelated summary widget at `gui/widgets/summary_image.py` (809 L) in the Preview tab, which duplicates most of this one. `ManualRoiWidget` uses the shared `SummaryImageViewer`, so the merge below is now only about retiring that Preview-tab widget.

| Aspect | mbo `gui/widgets/summary_image.py` | `imgui/summary.py` | Unified |
|---|---|---|---|
| Base | `Widget` ABC (`is_supported`, `draw`, `cleanup`) | plain class, takes `figure` | plain core + optional `Widget` adapter |
| Image source | auto-scan array metadata (ops.npy) | explicit `{name: 2D}` dict | dict core; metadata scan = optional provider |
| Movies | none | `MoviePlayer`, per-movie fixed range | core |
| Highlight | none | `set_highlight` rect | core |
| ROI overlay | suite2p `stat.npy` contours | `roi_provider` hook | core hook; suite2p contours become one provider |
| Export | PNG via `portable_file_dialogs` | `on_export` hook | core hook |
| cmap | syncs with fpl graphic | fixed 5-cmap list | core list + optional sync hook |
| Shared | `_GpuImage`, `_to_rgba`, `_data_range`, `_auto_range`, `_format_value`, pan/zoom, contrast modes, histogram, pixel-value overlay | identical | core |
