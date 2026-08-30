# `masknmf.visualization.imgui.theme`

## Tokens

| Token | Value (r, g, b, a) | Meaning | Used by |
|---|---|---|---|
| `OK` | 0.35, 0.90, 0.35, 1.0 | progress complete | `panels.draw_progress` |
| `WARN` | 1.00, 0.75, 0.25, 1.0 | progress incomplete, loading status | `panels.draw_progress`, `ClassificationVis._draw_panel` |
| `ERROR` | 1.00, 0.30, 0.30, 1.0 | failure text | `summary.SummaryImageViewer.draw` |
| `HINT` | 1.00, 0.85, 0.40, 1.0 | key names in the keybinds table | `panels.draw_keybinds_popup` |
| `CODE` | 0.55, 0.75, 1.00, 1.0 | code snippets in tooltips | `ClassificationVis._draw_save_note` |
| `HIGHLIGHT` | 1.00, 0.90, 0.20, 0.9 | current-ROI box on the full FOV | `summary.SummaryImageViewer.draw` |
| `CONTOUR` | 0.20, 1.00, 0.40, 0.9 | ROI outlines on the full FOV | `summary.SummaryImageViewer._draw_rois` |
| `TEXT_ON_DARK` | 1.0, 1.0, 1.0, 1.0 | pixel-value text over dark pixels | `theme.text_on` |
| `TEXT_ON_LIGHT` | 0.0, 0.0, 0.0, 1.0 | pixel-value text over light pixels | `theme.text_on` |
| `LUMA_LIGHT` | 140 | luma threshold between the two | `theme.text_on` |
| `DANGER` | 0.75, 0.15, 0.15, 0.8 | destructive button | `theme.danger_button` |
| `DANGER_HOVERED` | 0.90, 0.20, 0.20, 1.0 | destructive button, hovered | `theme.danger_button` |
| `LABEL_BUTTON_ALPHA` | 0.5 | class-colour button fill alpha | `theme.label_button` |

## Helpers

| Helper | Signature | Returns | Used by |
|---|---|---|---|
| `label_color` | `(rgb, alpha=1.0)` | `ImVec4` from a `LabelSet.color` tuple | `panels.draw_label_editor`, `table.draw_roi_table` |
| `u32` | `(color)` | packed int for `draw_list.add_*` | `summary` |
| `text_on` | `(luma)` | packed text colour readable over that luma | `summary._draw_pixel_values` |
| `button_colors` | `(button, hovered=None, active=None)` | context manager; pops on exit even on error | `label_button`, `danger_button` |
| `label_button` | `(rgb)` | context manager, class-coloured button | `panels.draw_label_buttons` |
| `danger_button` | `()` | context manager, red button | `panels.draw_label_buttons` ("unlabel all") |

## Consolidated literals

| Before | Where | After |
|---|---|---|
| `DONE_COLOR (0.35, 0.9, 0.35)` | `panels.py` | `OK` |
| `TODO_COLOR (1.0, 0.75, 0.25)` | `panels.py` | `WARN` |
| `(1.0, 0.8, 0.2)` loading text | `classification_vis.py` | `WARN` |
| `HINT_COLOR (1.0, 0.85, 0.4)` | `panels.py` | `HINT` |
| `(1.0, 0.9, 0.2, 0.9)` highlight box | `summary.py` | `HIGHLIGHT` |
| `(0.2, 1.0, 0.4, 0.9)` contours | `summary.py` | `CONTOUR` |
| `(1.0, 0.3, 0.3)` GPU unavailable | `summary.py` | `ERROR` |
| white / black pixel-value text, `> 140` | `summary.py` | `text_on(luma)` |
| `(0.55, 0.75, 1.0)` tooltip code ×2 | `classification_vis.py` | `CODE` |
| `(0.75, 0.15, 0.15, 0.8)` / `(0.90, 0.20, 0.20)` | `panels.py`, `classification_vis.py` | `danger_button()` |
| `ImVec4(*color, 0.5)` button ×2 | `panels.py`, `classification_vis.py` | `label_button(rgb)` |
| `ImVec4(*color, 1.0)` text ×3 | `panels.py`, `table.py`, `classification_vis.py` | `label_color(rgb)` |
