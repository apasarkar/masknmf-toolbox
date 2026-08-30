# masknmf.visualization.imgui

Reusable GUI-layer building blocks for the interactive viewers (`classification_vis`,
`multisession_vis`, `demixing_vis`, `motion_vis`, `interactive_guis`) and for
mbo_utilities' `ManualRoiWidget`. `WIDGETS.md` has the per-widget table.

## Modules

- `theme.py` — colour tokens and push/pop context managers; `THEME.md` lists them.
- `widgets.py` — `CheckboxWindow`, an `fpl.ui.ImguiWindow` with one checkbox.
- `picking.py` — data-space picking: `component_at_pixel`, `contours_to_bbox`, `zoom_to_bbox`.
- `layout.py` — figure-level helpers: `resolve_time_reference`, `is_notebook_canvas`.
- `labels.py` — `LabelSet`: class names, per-item labels, tab10 colours, `1`-`9`/`0` hotkeys.
- `table.py` — `RoiOrder` filter/sort model plus `draw_roi_table` / `draw_filter_row`.
- `panels.py` — `draw_progress`, `draw_keybinds_popup`, `draw_label_buttons`, `draw_label_editor`.
- `overlay.py` — `footprint_rgba`, `label_image_rgba`, `OverlayPair` (bg + RGBA overlay graphics).
- `masks.py` — `LabelImage`: non-overlapping ROIs in one uint16 label image; `outline_labels` / `selected_rim` with `OUTLINE_WIDTH` / `OUTLINE_PLACEMENT`.
- `draw.py` — `StrokeDrawer`: freehand stroke capture with pan suppression.
- `crop.py` — `crop_origin`, `crop_slices`, `context_crop`: roicat-centred zero-padded crops.
- `movie_player.py` — `MoviePlayer`: transport bar over a lazy `(T, H, W)` array.
- `store.py` — `LabelStore`: labels to npz and/or per-session hdf5.
- `loader.py` — `AsyncLoad`: background build polled from the draw loop.
- `summary.py` — `SummaryImageViewer`: full-FOV popup with pan/zoom/cmap/contrast/pixel values.

`widgets`/`panels`/`table`/`summary` are imgui proper; the rest are data or
fastplotlib-level helpers that live here because they only serve the interactive GUIs.

## Candidates not yet extracted

- **roi.py** — `PMDWidget`'s rectangle-ROI machinery in `interactive_guis.py`
  (`add_rectangle` / `resize_rect` / `end_resize`, the per-graphic selector `OrderedDict`,
  `rect_selector_kwargs`): a self-contained "draw one rect, mirror it across synced
  subplots, fire a callback on release" widget. Biggest remaining extraction.
- **selection.py** — the `ImageHighlightSelector` wiring repeated in multisession / demixing
  (`lut="tab10"`, `lut_wrap="repeat"`, contour pixel options, white options color): a
  `make_contour_selector(contours, **overrides)` factory, plus the `SelectionVector` +
  global/local index-map pattern from multisession.
- **layout.grid_extents** — demixing builds fractional extents dicts by hand;
  a rows/cols -> extents helper would remove it.
- **layout camera linking** — multisession shares one camera across the video and MIP
  figures; motion_vis links trace subplots on x only (`add_camera(..., include_state={"x", "width"})`).
  Two patterns: `share_camera(figures)` and `link_x(subplots)`.
- **traces** — normalize/offset trace stacks fed to `fpl.utils.heatmap_to_positions`
  (the trace panels in `SingleSessionDemixingVis._click_update`). Data-side siblings
  (`extract_per_trace_roi_averages` in demixing_vis, `get_roi_avg` in plots.py) belong in
  `masknmf.demixing` utils rather than here.
