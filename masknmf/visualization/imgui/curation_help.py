"""
The Curation tab's (?) page: the steps, the panels, what a click and a group plot, and what Demix does, as
diagrams and tables. Only imgui_bundle at import: ``python curation_help.py`` opens it in a window of its own.
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
DROP = imgui.ImVec4(1.00, 0.40, 0.40, 1.0)
CARD = imgui.ImVec4(0.17, 0.18, 0.21, 1.0)
EDGE = imgui.ImVec4(0.35, 0.35, 0.37, 1.0)
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
WIDTH_EM = 44
FONT_SIZE = 14

TITLE = "Curation"
TOOLTIP = "the steps, the panels, what a click and a group plot, and what Demix does"
STEPS = (
    (fa.ICON_FA_ARROW_POINTER, "Click", "a signal: its parts"),
    (fa.ICON_FA_OBJECT_GROUP, "Group", "ctrl / shift: compare"),
    (fa.ICON_FA_DRAW_POLYGON, "Draw ROI", "where a cell is missed"),
    (fa.ICON_FA_TRASH, "Delete", "mark wrong signals"),
    (fa.ICON_FA_WAND_MAGIC_SPARKLES, "Demix", "refit to a new file"),
)
PANEL_GRID = (
    ("raw", "compressed+denoised", "signals"),
    ("background", "residual", "colorful signals"),
    ("summary img",),
)
PANELS = (
    ("panel", "shows"),
    ("raw", "the raw movie, when given or found beside the results"),
    ("compressed+denoised", "the compressed movie: what the demixer sees"),
    ("signals", "every footprint x trace summed: the demixed movie"),
    ("background", "the fitted fluctuating background"),
    ("residual", "compressed minus signals minus background"),
    ("colorful signals", "each signal's footprint in its own color"),
    ("summary img", "the residual correlation image, or the image given"),
    ("shift (px)", "the registration shift per frame, when found"),
    ("traces", "the selection's traces, linked to the frame slider"),
)
PARTS = (
    ("line", "averaged over the footprint's pixels"),
    ("compressed", "the compressed movie there"),
    ("signal", "the demixed movie there, overlapping neighbours included"),
    ("background", "the fitted background there; absent when none was fit"),
    ("residual", "what nothing explains"),
)
MEMBERS = (
    ("row", "line"),
    ("signal k", "its demixed trace, scaled by its footprint's mean weight"),
    ("pixel avg (r, c)", "the compressed movie's 5x5 average at a clicked empty pixel (pixel traces on)"),
    ("roi n", "the compressed movie averaged inside a drawn ROI"),
)
MOUSE = (
    ("mouse", "does"),
    ("click", "select a signal or roi; an empty pixel adds its 5x5 average (pixel traces on)"),
    ("ctrl + click", "toggle it in the group"),
    ("shift + click", "add it; in the table, every row up to it"),
    ("double-click", "split the 3x3 square there into its sources"),
    ("drag", "pan; the selection stays"),
    ("k", "every key"),
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


def heading(icon: str, text: str) -> None:
    em = imgui.get_font_size()
    imgui.dummy(imgui.ImVec2(0, 0.7 * em))
    imgui.text_colored(ACCENT, f"{icon}  {text}")
    imgui.dummy(imgui.ImVec2(0, 0.1 * em))


def table(name: str, rows: tuple) -> None:
    """Two columns: the first row dim as the header, the rest a key name and its wrapped meaning."""
    em = imgui.get_font_size()
    flags = imgui.TableFlags_.row_bg | imgui.TableFlags_.borders_inner_h
    if not imgui.begin_table(f"##{name}", 2, flags, imgui.ImVec2(WIDTH_EM * em, 0)):
        return
    imgui.table_setup_column("name", imgui.TableColumnFlags_.width_fixed, 11 * em)
    imgui.table_setup_column("meaning")
    for i, (key, meaning) in enumerate(rows):
        imgui.table_next_row()
        imgui.table_next_column()
        imgui.text_colored(DIM if i == 0 else KEY, key)
        imgui.table_next_column()
        if i == 0:
            imgui.text_colored(DIM, meaning)
        else:
            imgui.text_wrapped(meaning)
    imgui.end_table()


def draw_curation_help(is_open: bool) -> bool:
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
        f"{TITLE} help###curation-help",
        is_open,
        flags=imgui.WindowFlags_.no_saved_settings | imgui.WindowFlags_.always_auto_resize,
    )
    imgui.pop_style_var(2)
    if not opened:
        imgui.end()
        return is_open
    dl = imgui.get_window_draw_list()
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

    heading(fa.ICON_FA_TABLE_CELLS_LARGE, "Panels")
    p = imgui.get_cursor_screen_pos()
    bgap, bw, bh = 0.4 * em, (w - 0.8 * em) / 3, 2.0 * em
    y = p.y
    for row in PANEL_GRID:
        for j, name in enumerate(row):
            box(dl, p.x + j * (bw + bgap), y, bw, bh, name, TEXT)
        y += bh + bgap
    box(dl, p.x, y, w, 1.3 * em, "shift (px)", TEXT)
    y += 1.3 * em + bgap
    box(dl, p.x, y, w, bh, "traces", TEXT)
    imgui.dummy(imgui.ImVec2(w, y + bh - p.y))
    imgui.text_colored(
        DIM, "three to a row, panned and zoomed together; a panel the results cannot fill is left out"
    )
    table("panels", PANELS)

    heading(fa.ICON_FA_ARROW_POINTER, "One signal: click")
    p = imgui.get_cursor_screen_pos()
    bh = 1.9 * em
    x = p.x
    for label, color, op in (
        ("compressed", COMPRESSED, "="),
        ("signal", SIGNAL, "+"),
        ("background", BACKGROUND, "+"),
        ("residual", RESIDUAL, ""),
    ):
        bw = imgui.calc_text_size(label).x + 1.6 * em
        box(dl, x, p.y, bw, bh, label, color, 0.18)
        x += bw
        if op:
            size = imgui.calc_text_size(op)
            at = imgui.ImVec2(x + (1.8 * em - size.x) / 2, p.y + (bh - size.y) / 2)
            dl.add_text(at, u32(DIM), op)
            x += 1.8 * em
    imgui.dummy(imgui.ImVec2(w, bh))
    table("parts", PARTS)

    heading(fa.ICON_FA_OBJECT_GROUP, "A group: ctrl / shift click")
    p = imgui.get_cursor_screen_pos()
    pw, ph = 24 * em, 5.2 * em
    dl.add_rect_filled(p, imgui.ImVec2(p.x + pw, p.y + ph), u32(CARD), 4.0)
    dl.add_rect(p, imgui.ImVec2(p.x + pw, p.y + ph), u32(EDGE), 4.0)
    for i, (label, color) in enumerate(zip(("signal 12", "roi 0", "pixel avg (40, 61)"), GROUP)):
        base = p.y + ph * (0.3 + 0.28 * i)
        points = []
        for t in range(61):
            bump = 1.1 * em * max(0.0, math.sin(0.4 * t + 1.9 * i)) ** 4
            noise = 0.08 * em * math.sin(2.7 * t + i)
            points.append(imgui.ImVec2(p.x + 0.6 * em + t * (pw - 1.2 * em) / 60, base - bump - noise))
        dl.add_polyline(points, u32(color), 1.5, 0)
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
        DIM, "every signal refit, the original file untouched; the viewer moves on to the new file"
    )

    imgui.pop_text_wrap_pos()
    imgui.dummy(imgui.ImVec2(0, 0.4 * em))
    if imgui.button("Close", imgui.ImVec2(6 * em, 0)):
        is_open = False
    imgui.end()
    return is_open


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
    state["open"] = draw_curation_help(state["open"])


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
    renderer.set_gui(partial(draw_window, {"open": True}))
    canvas.request_draw(renderer.render)
    loop.run()


if __name__ == "__main__":
    main()
