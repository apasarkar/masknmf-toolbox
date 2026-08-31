# masknmf.visualization.imgui

Reusable GUI-layer building blocks shared by the interactive viewers
(`curation_vis`, `multisession_vis`, `demixing_vis`, `motion_vis`, `interactive_guis`).

## Modules

- `widgets.py` — imgui `EdgeWindow` panels. `CheckboxWindow` (was `ROIManager` in `interactive_guis.py`).
- `picking.py` — data-space picking. `component_at_pixel` (was duplicated as `CurationVis._neuron_at` and `MultiSessionDemixingVis.neuron_selection`), `contours_to_bbox` / `zoom_to_bbox` (from `multisession_vis.py`).
- `layout.py` — figure-level helpers. `resolve_time_reference` (frame_timings/ref_range block that was triplicated across curation/demixing/motion) and `is_notebook_canvas` (canvas-class check in `show()` dispatch).

`widgets` is imgui proper; `picking`/`layout` are fastplotlib-level helpers that
live here because they only serve the interactive GUIs. If this package grows, they could
move to a sibling `visualization/common/`.

## Candidates not yet extracted

- **roi.py** — `PMDWidget`'s rectangle-ROI machinery in `interactive_guis.py`
  (`add_rectangle` / `resize_rect` / `end_resize`, the per-graphic selector `OrderedDict`,
  `rect_selector_kwargs`): a self-contained "draw one rect, mirror it across synced
  subplots, fire a callback on release" widget. Biggest remaining extraction.
- **panels.py** — imgui draw-callback side panels registered via
  `figure.add_imgui_window(...)`. `CurationVis._draw_panel` (counts + action buttons) is
  the only instance so far; generalize once a second GUI needs one.
- **selection.py** — the `ImageHighlightSelector` wiring repeated in curation /
  multisession / demixing (`lut="tab10"`, `lut_wrap="repeat"`, contour pixel options,
  white options color): a `make_contour_selector(contours, **overrides)` factory, plus the
  `SelectionVector` + global/local index-map pattern from multisession.
- **layout.grid_extents** — curation and demixing build fractional extents dicts by hand;
  a rows/cols -> extents helper would remove both.
- **layout camera linking** — multisession shares one camera across the video and MIP
  figures; motion_vis links trace subplots on x only (`add_camera(..., include_state={"x", "width"})`).
  Two patterns: `share_camera(figures)` and `link_x(subplots)`.
- **traces** — normalize/offset trace stacks fed to `fpl.utils.heatmap_to_positions`
  (`CurationVis._refresh_traces`, the trace panels in `SingleSessionDemixingVis._click_update`).
  Data-side siblings (`extract_per_trace_roi_averages` in demixing_vis, `get_roi_avg` in
  plots.py) belong in `masknmf.demixing` utils rather than here.
