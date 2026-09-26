"""
The Classification window's (?) page: the steps, its windows, the backgrounds, what a label does, train and
classify, the keys and the files, as diagrams and tables. Only imgui_bundle at import:
``python classification_help.py`` opens it in a window of its own.
"""

import math
from functools import partial
from pathlib import Path

import imgui_bundle
import wgpu
from imgui_bundle import icons_fontawesome_6 as fa
from imgui_bundle import imgui
from wgpu.utils.imgui import ImguiRenderer

ACCENT = imgui.ImVec4(0.40, 0.68, 1.00, 1.0)
KEY = imgui.ImVec4(1.00, 0.80, 0.20, 1.0)
DIM = imgui.ImVec4(0.62, 0.62, 0.65, 1.0)
TEXT = imgui.ImVec4(0.92, 0.92, 0.94, 1.0)
WARN = imgui.ImVec4(0.95, 0.68, 0.25, 1.0)
CARD = imgui.ImVec4(0.17, 0.18, 0.21, 1.0)
EDGE = imgui.ImVec4(0.35, 0.35, 0.37, 1.0)
PLOT = imgui.ImVec4(0.06, 0.06, 0.08, 1.0)
# the viewer's first class colors, given to classes in the order they are added, and its unlabeled gray
CLASSES = (
    ("soma", imgui.ImVec4(0.12, 0.47, 0.71, 1.0)),
    ("dendrite", imgui.ImVec4(1.00, 0.50, 0.05, 1.0)),
    ("junk", imgui.ImVec4(0.17, 0.63, 0.17, 1.0)),
)
UNLABELED = imgui.ImVec4(0.78, 0.78, 0.78, 1.0)
# the pred column where the prediction disagrees with the label
DISAGREE = imgui.ImVec4(1.00, 0.40, 0.40, 1.0)
WIDTH_EM = 44
FONT_SIZE = 14

TITLE = "Classification"
TOOLTIP = (
    "the steps, the windows, the backgrounds, what a label does, train and classify, the keys, the files"
)
STEPS = (
    (fa.ICON_FA_FOLDER_OPEN, "Load", "sessions: results files or a folder"),
    (fa.ICON_FA_TAG, "Label", "1-9 each ROI; 0 clears"),
    (fa.ICON_FA_IMAGE, "View", "backgrounds, the full FOV"),
    (fa.ICON_FA_GRADUATION_CAP, "Train", "every ROI labeled, 2 per class"),
    (fa.ICON_FA_WAND_MAGIC_SPARKLES, "Classify", "new sessions; fix, train again"),
)
# the ROI view's layers, each with its key, followed by the operator drawn after it
PARTS = (("background (b)", TEXT, "+"), ("mask (m)", ACCENT, "=" ), ("ROI view", KEY, ""))
# the top strip's cards and three lines of what each holds
CARDS = (
    ("NAVIGATE", ("prev   next", "13 / 505", "Open full FOV")),
    ("VIEW", ("mean img (enhanced)", "bg    opacity 0.50", "mask  opacity 1.00")),
    ("LABELS", ("labeled 393/505", "soma  (210)   (1)", "dendrite  (180)   (2)")),
    ("CLASSIFIER", ("save to  classifier...", "Select classifier", "train   classify")),
)
# a made-up run as (roi id, its label, the classifier's call, its confidence), stepped through as the
# current ROI
ROIS = (
    (12, "soma", "soma", 0.97),
    (13, "dendrite", "dendrite", 0.88),
    (14, None, "soma", 0.71),
    (15, "junk", "soma", 0.62),
    (16, "soma", "soma", 0.94),
    (17, None, "junk", 0.83),
)
# the cells in the current ROI's crop as (column, row) fractions and a radius in em; the first is the ROI
CELLS = ((0.50, 0.50, 1.3), (0.15, 0.25, 0.7), (0.82, 0.30, 0.6), (0.30, 0.85, 0.55), (0.88, 0.80, 0.5))
# the LABELS card's list as it might stand: (class, color, count)
LIST = (
    ("soma", CLASSES[0][1], 210),
    ("dendrite", CLASSES[1][1], 180),
    ("junk", CLASSES[2][1], 1),
    ("unlabeled", UNLABELED, 114),
)
WINDOWS = (
    ("window", "holds"),
    (
        "ROI view",
        "the current ROI's mask in its class color, white unlabeled, over a background cropped around it; "
        "the title names it",
    ),
    ("NAVIGATE", "prev / next, the position slider, Open full FOV"),
    (
        "VIEW",
        "the background source, bg and mask on / off with their opacity, class masks on the full FOV; the "
        "movie player when the demixed movie is the background",
    ),
    (
        "LABELS",
        "labeled n / N and next unlabeled; one row per class: eye, swatch, name (count), its key, x; add a "
        "class, unlabel, unlabel all",
    ),
    ("CLASSIFIER", "where train saves, Select classifier, train, classify, classify on load"),
    (
        "ROIs",
        "the table, filtered by class and area: id, sess, label, pred, area, peak, f, skew; click a row "
        "to go there, a header to sort",
    ),
    (
        "full FOV",
        "the session's background with the ROI's crop outlined; pan, zoom, colormap, contrast",
        (
            "class masks on: every ROI of the session in its class color",
            "the eye next to a class in the list hides it there",
            "the demixed movie plays there too",
            "it follows the ROI you move to",
        ),
    ),
)
BACKGROUNDS = (
    ("source", "shows"),
    ("mean img (enhanced)", "the processed mean image"),
    ("corr img (roi)", "the movie's correlation with the current ROI's trace"),
    ("resid corr img (roi)", "the residual movie's correlation with the current ROI's trace"),
    ("corr img (global)", "each pixel's correlation with its neighbors"),
    ("resid corr img (global)", "the residual movie's neighbor correlations"),
    ("mask MIP", "the max projection of every ROI mask"),
    ("demixed movie", "the demixed movie around the ROI, from its peak frame; the player plays it"),
)
LABELS = (
    ("action", "does"),
    ("click a row, or its key", "labels the current ROI and moves to the next"),
    ("0, unlabel", "clears its label"),
    ("add", "a new class, colored in order; its key is its position"),
    ("x", "deletes the class: its ROIs become unlabeled"),
    ("eye", "hides the class on the full FOV"),
    ("warning", "a class with fewer than 2 ROIs blocks train"),
    ("next unlabeled (u)", "the next ROI in view without a label"),
)
CLASSIFY = (
    ("button", "does"),
    (
        "train",
        "fits a ROICaT classifier on the labels, every ROI labeled and at least 2 per class, and saves it to "
        "the path",
    ),
    ("Select classifier", "a saved .roicat_classifier: its classes join the list, classify on load turns on"),
    ("classify", "predicts every ROI: unlabeled ROIs take the prediction, your labels are kept"),
    (
        "pred",
        "the prediction and its confidence in the table, red where it disagrees with your label: fix it, "
        "then train again",
    ),
    ("classify on load", "runs the selected classifier on each session as it arrives"),
)
KEYS = (
    ("key", "does"),
    ("up / down", "previous / next ROI; shift: ten at a time"),
    ("left / right", "previous / next background; shift: the previous / next class's first ROI"),
    ("1-9", "label the current ROI; 0 clears"),
    ("u", "next unlabeled ROI"),
    ("m / b", "the mask / the background on or off"),
    ("h / k", "the help / the keybinds"),
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


SPECKLE = tuple((hash01(k, -1), hash01(k, -2)) for k in range(200))


def card(dl, x: float, y: float, w: float, h: float, title: str) -> float:
    """A panel's card: its title in a strip on top, the dark area below. Returns the area's top."""
    em = imgui.get_font_size()
    a, b = imgui.ImVec2(x, y), imgui.ImVec2(x + w, y + h)
    dl.add_rect_filled(a, b, u32(CARD), 4.0)
    area = imgui.ImVec2(x, y + 1.2 * em)
    dl.add_rect_filled(area, b, u32(PLOT), 4.0, imgui.ImDrawFlags_.round_corners_bottom)
    dl.add_rect(a, b, u32(EDGE), 4.0)
    size = imgui.calc_text_size(title)
    dl.add_text(imgui.ImVec2(x + (w - size.x) / 2, y + (1.2 * em - size.y) / 2), u32(TEXT), title)
    return y + 1.2 * em


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


def draw_classification_help(is_open: bool) -> bool:
    """The page as a centered window. Returns whether it is still open."""
    if not is_open:
        return False
    em = imgui.get_font_size()
    w = WIDTH_EM * em
    viewport = imgui.get_main_viewport()
    imgui.set_next_window_pos(viewport.get_center(), imgui.Cond_.appearing, pivot=imgui.ImVec2(0.5, 0.5))
    imgui.set_next_window_size_constraints(
        imgui.ImVec2(0, 0), imgui.ImVec2(viewport.size.x, 0.94 * viewport.size.y)
    )
    imgui.push_style_var(imgui.StyleVar_.window_rounding, 6.0)
    imgui.push_style_var(imgui.StyleVar_.window_padding, imgui.ImVec2(em, 0.8 * em))
    opened, is_open = imgui.begin(
        f"{TITLE} help###classification-help",
        is_open,
        flags=imgui.WindowFlags_.no_saved_settings | imgui.WindowFlags_.always_auto_resize,
    )
    imgui.pop_style_var(2)
    if not opened:
        imgui.end()
        return is_open
    dl = imgui.get_window_draw_list()
    # the current ROI of the made-up run, the next one every 1.5 s
    cur = int(imgui.get_time() / 1.5) % len(ROIS)
    imgui.push_text_wrap_pos(w)
    imgui.text_colored(ACCENT, f"{fa.ICON_FA_CIRCLE_QUESTION}  {TITLE}")
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

    # the Windows heading with the ROI view's layers centered beside it, over the window sketch
    imgui.dummy(imgui.ImVec2(0, 0.7 * em))
    p = imgui.get_cursor_screen_pos()
    bh = 1.9 * em
    title = f"{fa.ICON_FA_WINDOW_RESTORE}  Windows"
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
    gap = 0.4 * em
    cw, ch = (w - 3 * gap) / 4, 5.2 * em
    for i, (name, rows) in enumerate(CARDS):
        x = p.x + i * (cw + gap)
        top = card(dl, x, p.y, cw, ch, name)
        for j, text in enumerate(rows):
            dl.add_text(imgui.ImVec2(x + 0.5 * em, top + 0.35 * em + j * 1.2 * em), u32(DIM), text)
    y = p.y + ch + gap
    strip = "load demixing result     load folder     clear all     help (h)     keybinds (k)"
    box(dl, p.x, y, w, 1.4 * em, strip, DIM)
    y += 1.4 * em + gap
    # the ROI view, the figure itself: the crop's cells under the current ROI's mask in its class color
    rid, label, pred, conf = ROIS[cur]
    color = dict(CLASSES).get(label, TEXT)
    tw = 13.5 * em
    vw, vh = w - tw - gap, 10.5 * em
    top = card(dl, p.x, y, vw, vh, f"ROI {rid}  [{label or 'unlabeled'}]  ({cur + 13}/505)")
    ih = vh - 1.2 * em
    dl.add_rect_filled_multi_color(
        imgui.ImVec2(p.x, top),
        imgui.ImVec2(p.x + vw, top + ih),
        u32(TEXT, 0.05),
        u32(TEXT, 0.22),
        u32(TEXT, 0.14),
        u32(TEXT, 0.03),
    )
    for k, (u, v, r) in enumerate(CELLS):
        center = imgui.ImVec2(p.x + u * vw, top + v * ih)
        for ring in (1.0, 0.85, 0.7, 0.55, 0.4, 0.25):
            dl.add_circle_filled(center, ring * r * em, u32(TEXT, 0.08))
        if k == 0:
            for ring in (1.0, 0.85, 0.7, 0.55, 0.4, 0.25):
                dl.add_circle_filled(center, ring * r * em, u32(color, 0.2))
    # the speckle's gray, its alpha filled into the top byte per dot
    grain = u32(TEXT) & 0xFFFFFF
    for k, (u, v) in enumerate(SPECKLE):
        at = imgui.ImVec2(p.x + u * (vw - 0.2 * em), top + v * (ih - 0.2 * em))
        alpha = int(90 * hash01(k, 0))
        dl.add_rect_filled(at, imgui.ImVec2(at.x + 0.2 * em, at.y + 0.2 * em), grain | alpha << 24)
    # the ROIs table beside it, the current row lit
    tx = p.x + vw + gap
    top = card(dl, tx, y, tw, vh, "ROIs")
    row_h = 1.25 * em
    for text, dx in (("id", 0.4), ("label", 3.0), ("pred", 7.6)):
        dl.add_text(imgui.ImVec2(tx + dx * em, top + 0.3 * em), u32(DIM), text)
    for i, (rid, label, pred, conf) in enumerate(ROIS):
        ry = top + 0.3 * em + (i + 1) * row_h
        if i == cur:
            a, b = imgui.ImVec2(tx + 1, ry - 0.1 * em), imgui.ImVec2(tx + tw - 1, ry + row_h - 0.15 * em)
            dl.add_rect_filled(a, b, u32(TEXT, 0.12))
        dl.add_text(imgui.ImVec2(tx + 0.4 * em, ry), u32(TEXT), str(rid))
        tint = dict(CLASSES)[label] if label else TEXT
        dl.add_text(imgui.ImVec2(tx + 3.0 * em, ry), u32(tint), label or "-")
        wrong = label is not None and pred != label
        dl.add_text(imgui.ImVec2(tx + 7.6 * em, ry), u32(DISAGREE if wrong else DIM), f"{pred} {conf:.2f}")
    imgui.dummy(imgui.ImVec2(w, y + vh - p.y))
    imgui.text_colored(
        DIM,
        "the ROI view is the figure; the cards and the table are its windows, the full FOV opens over them",
    )
    table("windows", WINDOWS)

    heading(fa.ICON_FA_LAYER_GROUP, "Backgrounds")
    imgui.text_colored(
        DIM, "the VIEW card's source, stepped with left / right; each is cropped around the current ROI"
    )
    table("backgrounds", BACKGROUNDS)

    heading(fa.ICON_FA_TAG, "Labels")
    p = imgui.get_cursor_screen_pos()
    row_h = 1.4 * em
    lw, lh = 22 * em, row_h * len(LIST) + 0.4 * em
    dl.add_rect_filled(p, imgui.ImVec2(p.x + lw, p.y + lh), u32(CARD), 4.0)
    dl.add_rect(p, imgui.ImVec2(p.x + lw, p.y + lh), u32(EDGE), 4.0)
    for i, (name, color, count) in enumerate(LIST):
        ry = p.y + 0.3 * em + i * row_h
        dl.add_text(imgui.ImVec2(p.x + 0.5 * em, ry), u32(DIM), fa.ICON_FA_EYE)
        swatch = imgui.ImVec2(p.x + 2.3 * em, ry), imgui.ImVec2(p.x + 3.4 * em, ry + 1.1 * em)
        dl.add_rect_filled(*swatch, u32(color), 2.0)
        dl.add_text(imgui.ImVec2(p.x + 4.0 * em, ry), u32(TEXT), f"{name}  ({count})")
        if count < 2:
            dl.add_text(imgui.ImVec2(p.x + 14.5 * em, ry), u32(WARN), fa.ICON_FA_TRIANGLE_EXCLAMATION)
        dl.add_text(imgui.ImVec2(p.x + 16.5 * em, ry), u32(DIM), f"({(i + 1) % len(LIST)})")
        if name != "unlabeled":
            dl.add_text(imgui.ImVec2(p.x + 19.8 * em, ry), u32(DIM), "x")
    imgui.dummy(imgui.ImVec2(w, lh))
    imgui.text_colored(
        DIM,
        "the LABELS card's list: the eye, the swatch, the class and its count, the too-few warning, its key, "
        "x",
    )
    table("labels", LABELS)

    heading(fa.ICON_FA_GRADUATION_CAP, "Train and classify")
    p = imgui.get_cursor_screen_pos()
    bh = 1.9 * em
    y = p.y + 1.4 * em
    x = p.x
    box(dl, x, y, 6.5 * em, bh, "labels", KEY)
    x += 6.5 * em
    for text, color, dy in (("train", ACCENT, -1.3 * em), ("2 per class", DIM, 0.3 * em)):
        size = imgui.calc_text_size(text)
        dl.add_text(imgui.ImVec2(x + (8 * em - size.x) / 2, y + bh / 2 + dy), u32(color), text)
    arrow(dl, x + 0.4 * em, x + 7.6 * em, y + bh / 2)
    x += 8 * em
    box(dl, x, y, 14.5 * em, bh, "classifier.roicat_classifier", KEY)
    x += 14.5 * em
    for text, color, dy in (("classify", ACCENT, -1.3 * em), ("new sessions", DIM, 0.3 * em)):
        size = imgui.calc_text_size(text)
        dl.add_text(imgui.ImVec2(x + (8 * em - size.x) / 2, y + bh / 2 + dy), u32(color), text)
    arrow(dl, x + 0.4 * em, x + 7.6 * em, y + bh / 2)
    x += 8 * em
    box(dl, x, y, 7 * em, bh, "pred column", TEXT)
    imgui.dummy(imgui.ImVec2(w, 1.4 * em + bh + 1.4 * em))
    imgui.text_colored(
        DIM,
        "unlabeled ROIs take the prediction, yours stay; red where the two disagree: fix it and train again",
    )
    table("classify", CLASSIFY)

    heading(fa.ICON_FA_KEYBOARD, "Keys")
    table("keys", KEYS)

    heading(fa.ICON_FA_FLOPPY_DISK, "Files")
    p = imgui.get_cursor_screen_pos()
    y = p.y + 1.4 * em
    x = p.x
    box(dl, x, y, 9 * em, bh, "results.hdf5", TEXT)
    x += 9 * em
    for text, color, dy in (("label, classify", ACCENT, -1.3 * em), ("saved as you go", DIM, 0.3 * em)):
        size = imgui.calc_text_size(text)
        dl.add_text(imgui.ImVec2(x + (10 * em - size.x) / 2, y + bh / 2 + dy), u32(color), text)
    arrow(dl, x + 0.4 * em, x + 9.6 * em, y + bh / 2)
    x += 10 * em
    box(dl, x, y, 13 * em, bh, "results.labels.hdf5", KEY)
    imgui.dummy(imgui.ImVec2(w, 1.4 * em + bh + 1.4 * em))
    imgui.text_colored(
        DIM,
        "the results file is never written; the sidecar beside it holds the labels, their names, the masks, "
        "the predictions and the classifier's path; train saves the classifier to the CLASSIFIER card's "
        "path, with a .training.json of what it learned from",
    )

    imgui.pop_text_wrap_pos()
    imgui.dummy(imgui.ImVec2(0, 0.4 * em))
    if imgui.button("Close", imgui.ImVec2(6 * em, 0)):
        is_open = False
    imgui.end()
    return is_open


def draw_window(state: dict) -> None:
    """One frame of the standalone window: the ? as it will sit in the window, then the page."""
    viewport = imgui.get_main_viewport()
    imgui.set_next_window_pos(viewport.pos)
    imgui.set_next_window_size(viewport.size)
    imgui.begin(
        "##classification-help-host",
        flags=imgui.WindowFlags_.no_decoration
        | imgui.WindowFlags_.no_saved_settings
        | imgui.WindowFlags_.no_bring_to_front_on_focus,
    )
    imgui.text_disabled(TITLE)
    imgui.same_line(0, 12)
    if imgui.small_button("?##classification_help"):
        state["open"] = True
    if imgui.is_item_hovered():
        imgui.set_tooltip(TOOLTIP)
    imgui.end()
    state["open"] = draw_classification_help(state["open"])


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

    canvas = RenderCanvas(title="classification help", size=(760, 1000), update_mode="continuous", max_fps=60)
    renderer = ImguiRenderer(wgpu.utils.get_default_device(), canvas)
    load_fonts()
    renderer.set_gui(partial(draw_window, {"open": True}))
    canvas.request_draw(renderer.render)
    loop.run()


if __name__ == "__main__":
    main()
