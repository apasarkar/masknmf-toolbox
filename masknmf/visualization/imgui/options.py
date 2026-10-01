"""File > Options: the settings the viewers share, kept for the session."""

from dataclasses import dataclass

from imgui_bundle import icons_fontawesome_6 as fa
from imgui_bundle import imgui

from masknmf.visualization.imgui.layout import is_notebook_canvas
from masknmf.visualization.imgui.theme import THEME, close_button, em, help_mark, popup, to_vec4


@dataclass
class Options:
    """What File > Options sets, for every viewer in the process."""

    separate_image_window: bool = True


OPTIONS = Options()

OPTIONS_LABEL = f"{fa.ICON_FA_GEARS}  options"


def draw_options_menu() -> bool:
    """A menu bar holding File > options, for a panel with no File menu of its own. True when options is picked."""
    picked = False
    # a child carries the menu bar, so the docked window itself needs no flag
    imgui.begin_child(
        "##menu",
        imgui.ImVec2(0, 0),
        imgui.ChildFlags_.auto_resize_y | imgui.ChildFlags_.always_auto_resize,
        imgui.WindowFlags_.menu_bar,
    )
    if imgui.begin_menu_bar():
        if imgui.begin_menu("File"):
            picked = imgui.menu_item_simple(OPTIONS_LABEL)
            imgui.end_menu()
        imgui.end_menu_bar()
    imgui.end_child()
    return picked


def draw_options_popup(figure, is_open: bool) -> bool:
    """The Options window of the viewer on ``figure``. Returns the new open state."""
    if not is_open:
        return False
    opened, is_open = popup("Options", is_open)
    if opened:
        imgui.text_colored(to_vec4(THEME.accent), f"{fa.ICON_FA_GEARS}  Options")
        imgui.separator()
        imgui.dummy(imgui.ImVec2(em(24), em(0.3)))
        imgui.text_disabled("Static images / Full FOV")
        # a notebook has no windows to open: the images stay a popup inside the viewer
        notebook = is_notebook_canvas(figure)
        imgui.begin_disabled(notebook)
        changed, separate = imgui.checkbox("Open as separate window", OPTIONS.separate_image_window and not notebook)
        if changed:
            OPTIONS.separate_image_window = separate
        imgui.end_disabled()
        help_mark(
            "Separate window: the images open in a window of their own, movable off this one, e.g. to another screen.\n"
            "Off, or in a notebook, they open as a popup inside this window."
        )
        imgui.dummy(imgui.ImVec2(0, em(0.3)))
        imgui.separator()
        if close_button():
            is_open = False
    imgui.end()
    return is_open
