"""
The Curation tab's guide (h): the steps, the panels and what fills them, the motion shifts, what a click, a
group and the selection tools do, and what Demix does, as diagrams and tables, with the viewer's keybinds
behind a button. From the repo root,
``python -m masknmf.visualization.imgui.curation_help`` opens it in a window of its own.
"""

import math
from functools import partial
from pathlib import Path

import imgui_bundle
import wgpu
from imgui_bundle import icons_fontawesome_6 as fa
from imgui_bundle import imgui
from wgpu.utils.imgui import ImguiRenderer

from masknmf.visualization.imgui.keybinds import DEMIXING
from masknmf.visualization.imgui.panels import draw_keybinds_button, draw_keybinds_popup
from masknmf.visualization.imgui.theme import popup

ACCENT = imgui.ImVec4(0.40, 0.68, 1.00, 1.0)
KEY = imgui.ImVec4(1.00, 0.80, 0.20, 1.0)
DIM = imgui.ImVec4(0.62, 0.62, 0.65, 1.0)
TEXT = imgui.ImVec4(0.92, 0.92, 0.94, 1.0)
DROP = imgui.ImVec4(1.00, 0.40, 0.40, 1.0)
CARD = imgui.ImVec4(0.17, 0.18, 0.21, 1.0)
EDGE = imgui.ImVec4(0.35, 0.35, 0.37, 1.0)
PLOT = imgui.ImVec4(0.06, 0.06, 0.08, 1.0)
# the trace plot's compressed / signal / background / residual line colors, then its first group colors
COMPRESSED = imgui.ImVec4(0.85, 0.85, 0.85, 1.0)
SIGNAL = imgui.ImVec4(0.30, 0.85, 0.40, 1.0)
BACKGROUND = imgui.ImVec4(0.95, 0.55, 0.15, 1.0)
RESIDUAL = imgui.ImVec4(0.35, 0.65, 0.95, 1.0)
GROUP = (
    imgui.ImVec4(1.00, 0.55, 0.10, 1.0),
    imgui.ImVec4(0.25, 0.85, 0.35, 1.0),
    imgui.ImVec4(0.95, 0.35, 0.90, 1.0),
)
# the trace plot's frame while a selection shows, and while a double-click's sources do
SELECTED = imgui.ImVec4(0.05, 0.14, 0.08, 1.0)
SOURCED = imgui.ImVec4(0.06, 0.09, 0.16, 1.0)
# a made-up field of view: four cells at (column, row) fractions of the panel, each with a radius in em
CELLS = ((0.18, 0.34, 0.85), (0.42, 0.68, 1.0), (0.66, 0.30, 0.75), (0.86, 0.66, 0.65))
WIDTH_EM = 44
FONT_SIZE = 14

TITLE = "Curation"
TOOLTIP = (
    "the steps, the panels and what fills them, the motion shifts, what a click, a group and the selection "
    "tools do, what Demix does, the keybinds"
)
STEPS = (
    (fa.ICON_FA_ARROW_POINTER, "Click", "a signal: its parts"),
    (fa.ICON_FA_DRAW_POLYGON, "Select", "draw (a), filter, ctrl / shift"),
    (fa.ICON_FA_PLUS, "Add ROI", "r: keep the drawn region"),
    (fa.ICON_FA_TRASH, "Delete", "mark wrong signals"),
    (fa.ICON_FA_WAND_MAGIC_SPARKLES, "Demix", "refit to a new file"),
)
# the movies a panel can show, in the viewer's order, and what their thumbnails draw: the smooth background,
# the cells pulsing with their traces, and how strong the speckle, 0 to 1
THUMBS = (
    ("raw", True, True, 1.0),
    ("registered", True, True, 1.0),
    ("compressed+denoised", True, True, 0.2),
    ("residual", False, False, 0.6),
    ("background", True, False, 0.0),
    ("signals", False, True, 0.0),
)
# the movies the three panels open on, one per column of the Panels window
DEFAULTS = ("raw", "compressed+denoised", "signals")
# the plot panels' lines as (trace kind, cell, color): the shift panel's height / width, one clicked signal's
# parts, and a group of three
SHIFT_LINES = (("slow", 3, imgui.ImVec4(1.0, 0.6, 0.2, 1.0)), ("slow", 4, imgui.ImVec4(0.4, 0.7, 1.0, 1.0)))
TRACE_LINES = (("sum", 0, COMPRESSED), ("bumps", 0, SIGNAL), ("slow", 0, BACKGROUND), ("noise", 0, RESIDUAL))
GROUP_LINES = (("bumps", 0, GROUP[0]), ("bumps", 1, GROUP[1]), ("bumps", 2, GROUP[2]))
# what the Static images window offers, behind its (?)
STILLS = (
    "the registration template",
    "the summary image given",
    "the residual correlation image, once demixed",
    "the mean and noise variance images of the compression",
    "lag-1 autocorrelation images, with masknmf view --compression",
    "pan, zoom, colormap and contrast; pixel values when zoomed in",
)
PANELS = (
    ("movie", "shows"),
    ("raw", "the movie as recorded, when given or found beside the results"),
    ("registered", "the raw movie with the file's registration replayed, when both are there"),
    (
        "compressed+denoised",
        "the raw movie compressed and denoised by PMD: the noise left out, what the demixer fits",
    ),
    ("signals", "every footprint x trace summed: the demixed movie"),
    ("background", "the fitted fluctuating background"),
    ("residual", "compressed minus signals minus background"),
    ("Static images", "a window over the stills the results hold, from the button above the tabs", STILLS),
)
# the compressed movie's parts in the trace plot's colors, each followed by its operator
PARTS = (
    ("compressed", COMPRESSED, "="),
    ("signal", SIGNAL, "+"),
    ("background", BACKGROUND, "+"),
    ("residual", RESIDUAL, ""),
)
SHIFTS = (
    ("registration", "the height and width lines"),
    ("rigid", "one shift per frame: shift height, shift width"),
    (
        "pw-rigid",
        "a shift per block per frame, shown as the largest absolute one over the blocks: "
        "max block shift height, max block shift width",
    ),
    (
        "registration only",
        "a file holding only a registration opens raw beside registered, with the shifts; no signals, rois "
        "or Demix",
    ),
)
MEMBERS = (
    ("row", "line"),
    ("signal k", "its demixed trace, scaled by its footprint's mean weight"),
    ("pixel avg (r, c)", "the compressed movie's 5x5 average at a clicked empty pixel (pixel traces on)"),
    ("roi n", "the compressed movie averaged inside a drawn ROI"),
)
# what a double-click plots, behind its (?)
SOURCES = (
    "takes the 3x3 square around the pixel",
    "averages compressed, background and residual over it",
    "one line per signal whose footprint touches the square, in its mask color",
    "scaled by its mean footprint weight there",
    "the traces frame turns navy while sources show, titled Traces (mode: source)",
    "needs demixed signals and traces shown; both presses on one spot",
)
MOUSE = (
    ("mouse", "does"),
    ("click", "select a signal or roi; an empty pixel adds its 5x5 average (pixel traces on)"),
    ("ctrl + click", "toggle it in the group"),
    ("shift + click", "add it; in the table, every row up to it"),
    ("double-click", "show demixed sources for selected pixels", SOURCES),
    ("drag", "pan; the selection stays"),
    ("right-click", "pick the panel's movie, over the usual menu"),
)
# the SELECTION row's buttons as (icon, name, color): draw lit, the rest as they sit
TOOL_BUTTONS = (
    (fa.ICON_FA_DRAW_POLYGON, "draw (a)", KEY),
    (fa.ICON_FA_SIGNATURE, "freehand", DIM),
    (fa.ICON_FA_PLUS, "add roi (r)", TEXT),
    (fa.ICON_FA_TRASH, "delete", DROP),
    (fa.ICON_FA_OBJECT_GROUP, "merge", DIM),
    (fa.ICON_FA_LOCATION_CROSSHAIRS, "center (f)", ACCENT),
)
TOOLS = (
    ("tool", "does"),
    (
        "draw (a)",
        "one region at a time, a polygon on any panel: every signal in view whose center is on the switch's "
        "side of it is selected, live as it is drawn or dragged; a or esc drops it, the selection stays, and so "
        "does a click or table pick that edits the selection by hand",
    ),
    ("freehand", "the region drawn in one stroke, not yet implemented"),
    (
        "add roi (r)",
        "keep the drawn region as a roi, where a cell is missed: its average joins the plot and Demix seeds the "
        "nmf pass with it; r with nothing drawn starts drawing; Export writes the drawn rois to a .npz",
    ),
    (
        "delete",
        "mark the selected signals for the next demix, again unmarks; a selected roi or pixel average is "
        "dropped right away",
    ),
    ("merge", "the grouped signals into one, not yet implemented"),
    ("center (f)", "every panel on the selected signal, following it as the selection moves"),
    (
        "inside / outside",
        "the switches: which side of the region it takes (inside by default) and which side of the range the "
        "filter takes (outside by default), each lit while its tool drives the selection",
    ),
    (
        "filter, range",
        "on the Signals tab, a stats column and a range on it: the table shows what is inside; apply makes "
        "its switch's side the selection, live as the lines move, as ctrl+a does for the table",
    ),
    (
        "applied",
        "a Delete made from the filter is listed under it and named in the curated file's description; "
        "ctrl+z undoes it",
    ),
    (
        "overlay",
        "masks (m), the selected masks, contours (c), the selected contours, each with an opacity; color by "
        "ranks a column; traces: center (t) keeps the current frame mid-plot, quick pixel trace (p), show selected traces",
    ),
)


def u32(color: imgui.ImVec4, alpha: float = 1.0) -> int:
    return imgui.color_convert_float4_to_u32(imgui.ImVec4(color.x, color.y, color.z, alpha))


def box(dl, x: float, y: float, w: float, h: float, label: str, color: imgui.ImVec4, fill_alpha: float = 0.0):
    """A rounded box with its label centered; fill_alpha tints it with the label color over the card."""
    a, b = imgui.ImVec2(x, y), imgui.ImVec2(x + w, y + h)
    dl.add_rect_filled(a, b, u32(CARD), 4.0)
    if fill_alpha:
        dl.add_rect_filled(a, b, u32(color, fill_alpha), 4.0)
    dl.add_rect(a, b, u32(color, 0.6), 4.0)
    size = imgui.calc_text_size(label)
    dl.add_text(imgui.ImVec2(x + (w - size.x) / 2, y + (h - size.y) / 2), u32(color), label)


def arrow(dl, x0: float, x1: float, y: float) -> None:
    col = u32(DIM)
    dl.add_line(imgui.ImVec2(x0, y), imgui.ImVec2(x1 - 5, y), col, 1.5)
    dl.add_triangle_filled(imgui.ImVec2(x1, y), imgui.ImVec2(x1 - 7, y - 4), imgui.ImVec2(x1 - 7, y + 4), col)


def hash01(a: float, b: float) -> float:
    """A deterministic value in [0, 1) for the pair: the same speckle every run."""
    return math.sin(12.9898 * a + 78.233 * b) * 43758.5453 % 1.0


SPECKLE = tuple((hash01(k, -1), hash01(k, -2)) for k in range(300))


def trace(kind: str, cell: int, t: float) -> float:
    """A made-up trace in [0, 1] at frame t: the cell's calcium bumps, a slow wander, noise, or their sum."""
    if kind == "bumps":
        return 0.03 + 0.03 * math.sin(2.7 * t + cell) + 0.9 * max(0.0, math.sin(0.4 * t + 1.9 * cell)) ** 4
    if kind == "slow":
        return 0.45 + 0.35 * math.sin(0.11 * t + 1.3 * cell) + 0.03 * math.sin(3.1 * t + cell)
    if kind == "noise":
        return 0.3 + 0.4 * hash01(t, cell)
    return 0.55 * trace("bumps", cell, t) + 0.3 * trace("slow", cell, t) + 0.15 * trace("noise", cell, t)


def lines(dl, x: float, y: float, w: float, h: float, specs: tuple, frame: float) -> None:
    """One trace per (kind, cell, color) spec, stacked over 60 frames, and the cursor at the frame."""
    em = imgui.get_font_size()
    x0, x1 = x + 0.5 * em, x + w - 0.5 * em
    slot = (h - 0.6 * em) / len(specs)
    for i, (kind, cell, color) in enumerate(specs):
        base = y + 0.3 * em + slot * (i + 1)
        points = [
            imgui.ImVec2(x0 + t * (x1 - x0) / 60, base - 0.9 * slot * trace(kind, cell, t)) for t in range(61)
        ]
        dl.add_polyline(points, u32(color), 1.5, 0)
    cx = x0 + frame * (x1 - x0) / 60
    dl.add_line(imgui.ImVec2(cx, y + 0.3 * em), imgui.ImVec2(cx, y + h - 0.3 * em), u32(TEXT, 0.7), 1.5)


def card(dl, x: float, y: float, w: float, h: float, title: str, tint: imgui.ImVec4 = PLOT) -> float:
    """A panel's card: its title in a strip on top, the plot area below in tint. Returns the area's top."""
    em = imgui.get_font_size()
    a, b = imgui.ImVec2(x, y), imgui.ImVec2(x + w, y + h)
    dl.add_rect_filled(a, b, u32(CARD), 4.0)
    area = imgui.ImVec2(x, y + 1.2 * em)
    dl.add_rect_filled(area, b, u32(tint), 4.0, imgui.ImDrawFlags_.round_corners_bottom)
    dl.add_rect(a, b, u32(EDGE), 4.0)
    size = imgui.calc_text_size(title)
    dl.add_text(imgui.ImVec2(x + (w - size.x) / 2, y + (1.2 * em - size.y) / 2), u32(TEXT), title)
    return y + 1.2 * em


def plot(dl, w: float, title: str, specs: tuple, h: float, frame: float, tint: imgui.ImVec4 = PLOT) -> None:
    """A full-width plot card at the cursor, its lines under the title; the cursor moves past it."""
    em = imgui.get_font_size()
    p = imgui.get_cursor_screen_pos()
    lines(dl, p.x, card(dl, p.x, p.y, w, 1.2 * em + h, title, tint), w, h, specs, frame)
    imgui.dummy(imgui.ImVec2(w, 1.2 * em + h))


def heading(icon: str, text: str) -> None:
    em = imgui.get_font_size()
    imgui.dummy(imgui.ImVec2(0, 0.7 * em))
    imgui.text_colored(ACCENT, f"{icon}  {text}")
    imgui.dummy(imgui.ImVec2(0, 0.1 * em))


def table(name: str, rows: tuple) -> None:
    """Two columns: the first row dim as the header, the rest a key name and its wrapped meaning, with a (?)
    whose tooltip lists a row's third element as bullets."""
    em = imgui.get_font_size()
    flags = imgui.TableFlags_.row_bg | imgui.TableFlags_.borders_inner_h
    if not imgui.begin_table(f"##{name}", 2, flags, imgui.ImVec2(WIDTH_EM * em, 0)):
        return
    imgui.table_setup_column("name", imgui.TableColumnFlags_.width_fixed, 11 * em)
    imgui.table_setup_column("meaning")
    for i, row in enumerate(rows):
        imgui.table_next_row()
        imgui.table_next_column()
        imgui.text_colored(DIM if i == 0 else KEY, row[0])
        imgui.table_next_column()
        if i == 0:
            imgui.text_colored(DIM, row[1])
        else:
            imgui.text_wrapped(row[1])
        if len(row) == 3:
            imgui.same_line()
            imgui.text_disabled("(?)")
            if imgui.is_item_hovered():
                imgui.begin_tooltip()
                for line in row[2]:
                    imgui.bullet_text(line)
                imgui.end_tooltip()
    imgui.end_table()


def draw_curation_help(is_open: bool, keys_open: bool) -> tuple[bool, bool]:
    """The page as a centered window with its keybinds button, whose popup the caller draws; returns (page open, popup open)."""
    if not is_open:
        return False, keys_open
    em = imgui.get_font_size()
    w = WIDTH_EM * em
    viewport = imgui.get_main_viewport()
    imgui.set_next_window_size_constraints(
        imgui.ImVec2(0, 0), imgui.ImVec2(viewport.size.x, 0.94 * viewport.size.y)
    )
    opened, is_open = popup(f"{TITLE} help", is_open)
    if not opened:
        imgui.end()
        return is_open, keys_open
    dl = imgui.get_window_draw_list()
    # the frame slider, sweeping the sketches' 60 frames every 7.5 s
    frame = imgui.get_time() * 8.0 % 60
    imgui.push_text_wrap_pos(w)
    imgui.text_colored(ACCENT, f"{fa.ICON_FA_CIRCLE_QUESTION}  {TITLE}")
    keys_open = draw_keybinds_button(keys_open, right=w)
    imgui.separator()
    imgui.dummy(imgui.ImVec2(0, 0.5 * em))

    gap = 1.4 * em
    card_w = (w - 4 * gap) / 5
    card_h = 6.8 * em
    imgui.push_style_color(imgui.Col_.child_bg, CARD)
    imgui.push_style_color(imgui.Col_.border, EDGE)
    imgui.push_style_var(imgui.StyleVar_.child_rounding, 5.0)
    imgui.push_style_var(imgui.StyleVar_.window_padding, imgui.ImVec2(0.5 * em, 0.6 * em))
    for i, (icon, name, hint) in enumerate(STEPS):
        if i:
            right = imgui.get_item_rect_max()
            mid = imgui.get_item_rect_min().y + card_h / 2
            arrow(dl, right.x + 0.25 * em, right.x + gap - 0.25 * em, mid)
            imgui.same_line(0, gap)
        imgui.begin_child(
            f"##step{i}",
            imgui.ImVec2(card_w, card_h),
            child_flags=imgui.ChildFlags_.borders,
            window_flags=imgui.WindowFlags_.no_scrollbar,
        )
        imgui.push_font(None, 1.8 * em)
        imgui.set_cursor_pos_x((card_w - imgui.calc_text_size(icon).x) / 2)
        imgui.text_colored(ACCENT, icon)
        imgui.pop_font()
        imgui.dummy(imgui.ImVec2(0, 0.2 * em))
        label = f"{i + 1}  {name}"
        imgui.set_cursor_pos_x((card_w - imgui.calc_text_size(label).x) / 2)
        imgui.text_colored(DIM, f"{i + 1}")
        imgui.same_line(0, 0.5 * em)
        imgui.text_colored(KEY, name)
        imgui.push_text_wrap_pos(card_w - 0.5 * em)
        imgui.text_colored(DIM, hint)
        imgui.pop_text_wrap_pos()
        imgui.end_child()
    imgui.pop_style_var(2)
    imgui.pop_style_color(2)

    # the Panels heading with the compressed movie's parts centered beside it, over the six panels
    imgui.dummy(imgui.ImVec2(0, 0.7 * em))
    p = imgui.get_cursor_screen_pos()
    bh = 1.9 * em
    title = f"{fa.ICON_FA_TABLE_CELLS_LARGE}  Panels"
    dl.add_text(imgui.ImVec2(p.x, p.y + (bh - em) / 2), u32(ACCENT), title)
    parts_w = sum(imgui.calc_text_size(t).x + 1.6 * em + (1.8 * em if op else 0) for t, _, op in PARTS)
    x = p.x + (w - parts_w) / 2
    for label, color, op in PARTS:
        bw = imgui.calc_text_size(label).x + 1.6 * em
        box(dl, x, p.y, bw, bh, label, color, 0.18)
        x += bw
        if op:
            size = imgui.calc_text_size(op)
            dl.add_text(imgui.ImVec2(x + (1.8 * em - size.x) / 2, p.y + (bh - size.y) / 2), u32(DIM), op)
            x += 1.8 * em
    imgui.dummy(imgui.ImVec2(w, bh))
    imgui.dummy(imgui.ImVec2(0, 0.1 * em))
    p = imgui.get_cursor_screen_pos()
    bgap, bw, ih = 0.4 * em, (w - 0.8 * em) / 3, 4.4 * em
    # the speckle's gray, its alpha filled into the top byte per dot
    grain = u32(TEXT) & 0xFFFFFF
    y = p.y
    for i, (name, shaded, cells, speckle) in enumerate(THUMBS):
        if i and i % 3 == 0:
            y += 1.2 * em + ih + bgap
        x = p.x + (i % 3) * (bw + bgap)
        top = card(dl, x, y, bw, 1.2 * em + ih, name)
        if shaded:
            glow = 0.5 + 0.7 * trace("slow", 0, frame)
            dl.add_rect_filled_multi_color(
                imgui.ImVec2(x, top),
                imgui.ImVec2(x + bw, top + ih),
                u32(TEXT, 0.05 * glow),
                u32(TEXT, 0.30 * glow),
                u32(TEXT, 0.18 * glow),
                u32(TEXT, 0.03 * glow),
            )
        for k, (u, v, r) in enumerate(CELLS if cells else ()):
            lit = 0.15 + 0.85 * trace("bumps", k, frame)
            center = imgui.ImVec2(x + u * bw, top + v * ih)
            for ring in (1.0, 0.85, 0.7, 0.55, 0.4, 0.25):
                dl.add_circle_filled(center, ring * r * em, u32(TEXT, 0.2 * lit))
        # the speckle re-rolls with the frame; fainter speckle is sparser too
        for k, (u, v) in enumerate(SPECKLE[: int(len(SPECKLE) * speckle)]):
            at = imgui.ImVec2(x + u * (bw - 0.2 * em), top + v * (ih - 0.2 * em))
            alpha = int(140 * speckle * hash01(k, int(frame)))
            dl.add_rect_filled(at, imgui.ImVec2(at.x + 0.2 * em, at.y + 0.2 * em), grain | alpha << 24)
    imgui.dummy(imgui.ImVec2(w, y + 1.2 * em + ih - p.y))
    # the Panels window: one radio per movie and panel, the defaults picked
    p = imgui.get_cursor_screen_pos()
    row_h = 1.3 * em
    mw, mh = 15 * em, row_h * (len(THUMBS) + 1) + 0.4 * em
    dl.add_rect_filled(p, imgui.ImVec2(p.x + mw, p.y + mh), u32(CARD), 4.0)
    dl.add_rect(p, imgui.ImVec2(p.x + mw, p.y + mh), u32(EDGE), 4.0)
    for j in range(3):
        dl.add_text(imgui.ImVec2(p.x + (9.9 + 1.8 * j) * em, p.y + 0.2 * em), u32(DIM), str(j + 1))
    for i, (name, _shaded, _cells, _speckle) in enumerate(THUMBS):
        ry = p.y + 0.2 * em + (i + 1) * row_h
        dl.add_text(imgui.ImVec2(p.x + 0.5 * em, ry), u32(TEXT), name)
        for j in range(3):
            center = imgui.ImVec2(p.x + (10.1 + 1.8 * j) * em, ry + 0.55 * em)
            dl.add_circle(center, 0.4 * em, u32(DIM), 0, 1.0)
            if DEFAULTS[j] == name:
                dl.add_circle_filled(center, 0.25 * em, u32(ACCENT))
    imgui.dummy(imgui.ImVec2(mw, mh))
    imgui.same_line(0, em)
    imgui.text_colored(
        DIM,
        "up to three panels, any movie in each: the Panels button above the tabs, or a right-click on a "
        "panel, picks what it shows; the zoom and the drawn rois stay when it switches. They open on raw, "
        "compressed+denoised and signals, and the frame slider runs all three.",
    )
    table("panels", PANELS)

    heading(fa.ICON_FA_ARROWS_UP_DOWN_LEFT_RIGHT, "Motion correction shifts")
    plot(dl, w, "shift (px)", SHIFT_LINES, 2.4 * em, frame)
    imgui.text_colored(
        DIM,
        "if motion shifts are detected, these will be displayed and should be used to see if activity "
        "correlates with brain motion",
    )
    table("shifts", SHIFTS)

    heading(fa.ICON_FA_ARROW_POINTER, "One signal: click")
    plot(dl, w, "Traces (mode: selection)", TRACE_LINES, 4.4 * em, frame, SELECTED)
    imgui.text_colored(
        DIM,
        "compressed, signal, background and residual averaged over its footprint, the cursor on the frame "
        "slider; the frame goes green while a selection shows, titled Traces (mode: selection)",
    )

    heading(fa.ICON_FA_OBJECT_GROUP, "A group: ctrl / shift click")
    p = imgui.get_cursor_screen_pos()
    pw, ph = 24 * em, 5.2 * em
    dl.add_rect_filled(p, imgui.ImVec2(p.x + pw, p.y + ph), u32(SELECTED), 4.0)
    dl.add_rect(p, imgui.ImVec2(p.x + pw, p.y + ph), u32(EDGE), 4.0)
    lines(dl, p.x, p.y, pw, ph, GROUP_LINES, frame)
    for i, (label, color) in enumerate(zip(("signal 12", "roi 0", "pixel avg (40, 61)"), GROUP)):
        lx, ly = p.x + pw + em, p.y + 0.4 * em + 1.5 * i * em
        swatch = imgui.ImVec2(lx, ly + 0.25 * em), imgui.ImVec2(lx + 0.8 * em, ly + 1.05 * em)
        dl.add_rect_filled(*swatch, u32(color), 2.0)
        dl.add_text(imgui.ImVec2(lx + 1.2 * em, ly), u32(TEXT), label)
    imgui.dummy(imgui.ImVec2(w, ph))
    imgui.text_colored(
        DIM,
        "one line per member, in its mask and row color; alike traces: one cell split in two, or two that "
        "fire together",
    )
    table("members", MEMBERS)

    heading(fa.ICON_FA_SLIDERS, "Selection tools")
    p = imgui.get_cursor_screen_pos()
    bw, bh, bgap = 4.6 * em, 2.2 * em, 0.6 * em
    for i, (icon, name, color) in enumerate(TOOL_BUTTONS):
        x = p.x + i * (bw + bgap)
        box(dl, x, p.y, bw, bh, icon, color, 0.18 if color in (KEY, ACCENT) else 0.0)
        size = imgui.calc_text_size(name)
        dl.add_text(imgui.ImVec2(x + (bw - size.x) / 2, p.y + bh + 0.3 * em), u32(DIM), name)
    # the side switch, outside picked and lit as while the region or the filter drives the selection
    x = p.x + len(TOOL_BUTTONS) * (bw + bgap) + em
    dl.add_text(imgui.ImVec2(x, p.y + 0.55 * em), u32(DIM), "inside")
    x += imgui.calc_text_size("inside").x + 0.5 * em
    pill = imgui.ImVec2(x, p.y + 0.45 * em), imgui.ImVec2(x + 2.4 * em, p.y + 1.65 * em)
    dl.add_rect_filled(*pill, u32(ACCENT, 0.35), 0.6 * em)
    dl.add_rect(*pill, u32(ACCENT), 0.6 * em)
    dl.add_circle_filled(imgui.ImVec2(x + 1.8 * em, p.y + 1.05 * em), 0.45 * em, u32(TEXT))
    dl.add_text(imgui.ImVec2(x + 2.9 * em, p.y + 0.55 * em), u32(TEXT), "outside")
    imgui.dummy(imgui.ImVec2(w, bh + 1.6 * em))
    imgui.text_colored(
        DIM,
        "the Curation tab's SELECTION row and its side switch; a mode that is on is lit, a tool with nothing "
        "to act on is greyed; the filter's column and range sit under the switch",
    )
    table("tools", TOOLS)

    heading(fa.ICON_FA_COMPUTER_MOUSE, "Mouse")
    table("mouse", MOUSE)

    heading(fa.ICON_FA_WAND_MAGIC_SPARKLES, "Demix")
    p = imgui.get_cursor_screen_pos()
    bh = 1.9 * em
    y = p.y + 1.4 * em
    x = p.x
    box(dl, x, y, 9 * em, bh, "results.hdf5", TEXT)
    x += 9 * em
    arrow(dl, x + 0.4 * em, x + 9.6 * em, y + bh / 2)
    notes = (("+ drawn ROIs", SIGNAL, -1.3 * em), ("- marked signals", DROP, 0.3 * em))
    for text, color, dy in notes:
        size = imgui.calc_text_size(text)
        dl.add_text(imgui.ImVec2(x + (10 * em - size.x) / 2, y + bh / 2 + dy), u32(color), text)
    x += 10 * em
    box(dl, x, y, 6.5 * em, bh, "NMF pass", ACCENT)
    x += 6.5 * em
    arrow(dl, x + 0.4 * em, x + 2.4 * em, y + bh / 2)
    x += 2.8 * em
    box(dl, x, y, 13 * em, bh, "<time>.curated.hdf5", KEY)
    imgui.dummy(imgui.ImVec2(w, 1.4 * em + bh + 1.4 * em))
    imgui.text_colored(
        DIM,
        "every signal refit, the original file untouched; a filter behind a Delete is named in the new "
        "file's description; the viewer moves on to the new file",
    )

    imgui.pop_text_wrap_pos()
    imgui.dummy(imgui.ImVec2(0, 0.4 * em))
    if imgui.button("Close", imgui.ImVec2(6 * em, 0)):
        is_open = False
    imgui.end()
    return is_open, keys_open


def draw_window(state: dict) -> None:
    """One frame of the standalone window: the ? as it will sit in the tab, then the page."""
    viewport = imgui.get_main_viewport()
    imgui.set_next_window_pos(viewport.pos)
    imgui.set_next_window_size(viewport.size)
    imgui.begin(
        "##curation-help-host",
        flags=imgui.WindowFlags_.no_decoration
        | imgui.WindowFlags_.no_saved_settings
        | imgui.WindowFlags_.no_bring_to_front_on_focus,
    )
    imgui.text_disabled(TITLE)
    imgui.same_line(0, 12)
    if imgui.small_button("?##curation_help"):
        state["open"] = True
    if imgui.is_item_hovered():
        imgui.set_tooltip(TOOLTIP)
    imgui.end()
    state["open"], state["keys"] = draw_curation_help(state["open"], state["keys"])
    state["keys"] = draw_keybinds_popup(DEMIXING, state["keys"], f"{TITLE} keybinds")


def load_fonts() -> None:
    """Roboto with Font Awesome 6 merged in, as fastplotlib's imgui figures load it."""
    fonts = Path(imgui_bundle.__file__).parent / "assets" / "fonts"
    io = imgui.get_io()
    io.fonts.add_font_from_file_ttf(str(fonts / "Roboto/Roboto-Regular.ttf"), FONT_SIZE, imgui.ImFontConfig())
    merge = imgui.ImFontConfig()
    merge.merge_mode = True
    io.fonts.add_font_from_file_ttf(str(fonts / "Font_Awesome_6_Free-Solid-900.otf"), FONT_SIZE, merge)


def main() -> None:
    from rendercanvas.auto import RenderCanvas, loop

    canvas = RenderCanvas(title="curation help", size=(760, 1000), update_mode="continuous", max_fps=60)
    renderer = ImguiRenderer(wgpu.utils.get_default_device(), canvas)
    load_fonts()
    renderer.set_gui(partial(draw_window, {"open": True, "keys": False}))
    canvas.request_draw(renderer.render)
    loop.run()


if __name__ == "__main__":
    main()
