# Shared imgui/fastplotlib widgets

Everything here is importable from `masknmf.visualization.imgui`. Consumers: masknmf `ClassificationVis`, mbo_utilities `ManualRoiWidget` (`mbo <path> --widget manualroi`). Dependency direction is `mbo_utilities -> masknmf`.

`draw_*` helpers are called inside an imgui frame and return what the user did. Classes hold state; construct once, call their draw/refresh methods per frame.

| Widget | Use case | How to use |
|---|---|---|
| `LabelSet` | Class names plus a per-item label vector, `-1` unlabeled. The label model behind every panel here. | `s = LabelSet(n, ("cell", "junk"))`, `s.assign([i], 0)`, `s.add/remove/clear/resize`, `s.color(i)`, `s.count(i)`, `s.name_of(i)`; `i = s.hotkey_pressed()` maps `1`-`9` to an index and `0` to `UNLABELED`. |
| `RoiOrder` | Filter + stable sort over per-ROI columns; owns the cursor into the visible order. | `o = RoiOrder({"area": areas}, s.labels, n)`, `o.set_range_column("area")`, then `o.rebuild()` after any change. Move with `o.step(±1)`, `o.goto(i)`, `o.next_unlabeled()`, `o.step_group(±1)`; read `o.current` / `o.order` / `o.pos`. `sort_column` is an imgui column index: 0 the id, 1 the label, the rest `columns` in insertion order — that order must match `draw_roi_table`'s `column_names[2:]`. |
| `draw_roi_table` | Scrollable sortable ROI list, clipped to the visible rows, scrolls to the current item. | `scroll = draw_roi_table(o, s, ("id", "label", "area"), {"area": fmt}, scroll, table_id="rois", on_select=cb, actions=acts)` — pass the returned flag back next frame. `column_names[0]` is the id, `"label"` draws in its class colour, the rest come from `formatters`. |
| `draw_filter_row` | The class filter combo + range slider + "n/n in view" that sits above the table. | `draw_filter_row(o, s, "_roi")` — rebuilds `o` itself, returns True if the view changed. |
| `RowAction` | Per-row icon buttons in a trailing, unsortable column — run something on one ROI without leaving the table. | `RowAction(icon, tooltip, on_click, disabled)`, passed to `draw_roi_table(..., actions=(a, b))`. `on_click(item)` fires on press; `disabled(item)` returns a reason that greys the button and replaces the tooltip, or None. |
| `draw_label_buttons` | One coloured button per class, plus unlabel and unlabel all. | `picked = draw_label_buttons(s, "_roi")`; branch on `UNLABEL_ALL` (-2) before treating it as an index, `UNLABELED` (-1) clears the current item, `None` means no click. |
| `draw_label_editor` | Add and remove classes (text field, `add`, `del` popup). | `text, changed = draw_label_editor(s, text, "_roi")` — keep `text` in your own state; `changed` means rebuild the order and redraw the overlay. |
| `draw_progress` | `labeled n/total` with an optional per-session split and a "next unlabeled" button. | `if draw_progress(s.labels, session_sizes): o.next_unlabeled()`. |
| `draw_keybinds_popup` | Key reference overlay. | `open = draw_keybinds_popup((("a", "arm drawing"), ...), open, "ROI keys")` — call unconditionally, it no-ops while closed. |
| `LabelImage` | ROIs as one `uint16` label image, so they can never overlap: pixels already claimed are dropped from a new stroke. | `m = LabelImage((ny, nx))`, `i = m.add(stroke)` (-1 and `m.last_error` on reject), `m.delete(i)` renumbers, `m.at(row, col)` picks, `m.areas()` / `m.counts` feed the table, `m.edges(w, placement)` feeds the overlay, `m.footprints()` gives `(n, ny, nx)`. |
| `label_image_rgba` | A label image as a blended RGBA overlay: tinted fills, coloured boundaries, a white rim on the selected ROI. | `graphic.data = label_image_rgba(m.labels, colors=per_roi_rgb, alpha=0.45, selected=i, edges=m.edges(w, p), outline_width=w, outline_alpha=1.0, outline_placement=p)`. `alpha` is the fill, `outline_alpha` the boundary. `show_outlines=False` clears every line, rim included. Put it on an `add_image(..., alpha_mode="blend", offset=(0, 0, 1))` and pin `graphic.vmin, graphic.vmax = 0, 255` — the data is literal bytes, and auto-ranging an all-zero array saturates to white. |
| `footprint_rgba` | One weighted footprint as a flat colour whose alpha is the normalised weight. | `graphic.data = footprint_rgba(footprints[i], color, peak=peaks[i])`. |
| `OUTLINE_WIDTH` / `OUTLINE_PLACEMENT` | How thick mask outlines are and which side of the boundary they sit on. 1 px is the floor, so on a 1-3 px ROI the placement decides whether the line costs a mask pixel. | Globals in `masks.py`; per call `label_image_rgba(..., outline_width=w, outline_placement=p)` and `LabelImage.edges(w, p)`. `outer` (default) takes background pixels, `inner` the mask's own, `center` straddles. `rim_kernel(w)` is the `(2w+1, 2w+1)` element behind all three. |
| `outline_labels` / `selected_rim` | The outline of a label image, and the rim around one selected mask; both honour `placement`. | `outline_labels(labels, w, "outer")` -> uint16 labels, 0 off the line, so each segment keeps its ROI's colour. `selected_rim(labels == i, w, p)` -> bool. |
| `OverlayPair` | A background image plus an RGBA overlay on one subplot, with visibility and alpha for each. | `p = OverlayPair(subplot, (ny, nx))`, `p.set_background(img)`, `p.set_overlay(rgba)`, set `p.show_bg/.show_fg/.bg_alpha/.fg_alpha` then `p.apply()`; `p.exclude_from_picking()` keeps tooltips reading the image beneath, `p.remove()` tears down. |
| `StrokeDrawer` | Freehand outline on a subplot; lifts the left-drag pan binding while armed so the stroke isn't a pan. | `d = StrokeDrawer(subplot, on_stroke, on_click)`, `d.arm(True)` to start. `on_stroke(points)` fires on release with a closed `[(x, y)]` stroke; `on_click(row, col)` fires on a click while disarmed. `d.line` is the live stroke graphic. |
| `MoviePlayer` | Transport bar (play/pause, frame slider, fps) over any lazy `(T, H, W)` array. | `p = MoviePlayer(movie)`, then per frame `if p.draw(): img = p.frame()`. `p.frame((top, left, h, w))` crops lazily and zero-pads off the edge; `p.set_movie` / `p.jump_to` drive it. |
| `SummaryImageViewer` | Full-FOV popup over a set of images and lazy movies: pan/zoom, cmap, contrast modes, histogram, pixel values, ROI contours, export. | `v = SummaryImageViewer(figure, roi_provider=contours_fn, on_export=save_fn)`, then `v.set_movies({...})`, `v.set_images({name: img2d}, selected=name)`, `v.set_highlight((y0, x0, h, w))`, `v.open()`. Call `v.draw()` every imgui frame and `v.cleanup()` on teardown to free its GPU textures. |
| `LabelStore` | Autosave labels to an npz and/or into per-session hdf5, so they travel with the data. | `store = LabelStore(npz_path=..., hdf5_files=[...], session_sizes=(...))`, `store.save(s.names, s.labels, masks)` on every label change; `store.load(n)` restores, `store.error` reports a failed write. hdf5 writes land in `DemixingResults/{class_labels,label_names,labels_complete,roi_masks}`. |
| `AsyncLoad` | Run something slow on a thread while the GUI stays live — a ROICaT build, a projection reduce. | `load.start(fn, "computing...")`, then per frame `result = load.poll()` (non-None once). `load.busy` / `load.status` drive the loading text, `load.error` the failure. |
| `crop_origin` / `context_crop` | A fixed-size crop centred on a centroid, roicat's convention, zero-padded off the edge. | `top, left = crop_origin(centroid, (h, w))`; `crop = context_crop(fov, centroid, (h, w))`. |
| `component_at_pixel` | Turn a fastplotlib pick into a component index; nearest centre wins ties. | `i = component_at_pixel(a, centers, fov_shape, graphic.pick_info["index"], mask=..., radius=...)` — `a` is a coalesced sparse `(pixels, components)` torch tensor, `pick_index` is `(col, row)`. |
| `contours_to_bbox` / `zoom_to_bbox` | Frame one ROI in a subplot. | `lower, upper = contours_to_bbox(fov_shape, contour, extra_space=10)`, then `zoom_to_bbox(subplot, graphic, lower, upper)`. |
| `resolve_time_reference` | One time axis across the NDWidget viewers, from frame timings or plain frame indices. | `ref_range, frame_timings = resolve_time_reference(n, frame_timings, ref_range, axis="time")`. A `ref_range` without `frame_timings` raises. |
| `is_notebook_canvas` | Branch `show()` between an ipywidgets canvas and a native window. | `if is_notebook_canvas(figure): return HBox([...])`. |
| `CheckboxWindow` | A one-off toggle in its own imgui panel. | `w = CheckboxWindow("show contours")`, `figure.add_imgui_window(w, location="top", size=40)`, read `w.value`. |

## Hosting the panels

Both consumers put every ROI control in one edge window and the table in another:

```python
figure.add_imgui_window(draw_controls, location="top", size=150, title="Classification")
figure.add_imgui_window(draw_table, location="right", size=360, title="ROIs")
```

`ClassificationVis` owns its figure and takes `top` + `right`. `ManualRoiWidget` shares the viewer's figure, where `PreviewDataWidget` already owns `right` and the NDWidget sliders own `bottom`, so it takes only `top` — controls plus a collapsible trace viewer, growing the window from 140 to 360 when that opens — and hands the edge back with `figure.remove_imgui_window(location)` when toggled off. Its table is a tab inside `PreviewDataWidget` rather than a second edge window.

An edge window's `size` is writable, so a section that folds out can grow its host:

```python
window = figure.imgui_windows["top"]
window.size = PANEL_HEIGHT + (TRACE_HEIGHT if expanded else 0)
```

## Not yet shared

`ClassificationVis` still keeps `_label_names` / `_class_labels` as bare fields and hand-rolls its own copies of `LabelSet`, `RoiOrder`, `draw_roi_table`, `draw_filter_row`, `draw_label_editor` and `draw_label_buttons`. Moving it onto the shared versions is the one change that would put both GUIs on one implementation. `mbo_utilities/gui/widgets/summary_image.py` likewise duplicates `SummaryImageViewer` for the Preview tab.

Candidates from the older viewers (`interactive_guis`, `multisession_vis`, `demixing_vis`, `motion_vis`): `PMDWidget`'s rectangle-ROI machinery; the repeated `ImageHighlightSelector` contour-selector wiring; a rows/cols -> fractional extents helper; `share_camera(figures)` / `link_x(subplots)`; and the normalize/offset trace stacks fed to `heatmap_to_positions`. From mbo: `compute_projections` (sampled mean/max/std reduce) and `PlaneMovie` (pins the non-`(T, Y, X)` dims of an n-d array to give `MoviePlayer` a lazy 3D view) are both generic.
