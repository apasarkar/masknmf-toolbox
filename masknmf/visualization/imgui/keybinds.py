"""
Every viewer's keybinds in one place. A :class:`Bind` is one key with its modifiers and the words the keybinds popup
and the help pages show for it; a viewer's table maps action names to binds, its key handler asks :func:`pressed`
about each, and the popup lists the table's rows. Rows without a key (clicks, scrolls) are listed only. Both viewers
share ``UP``, ``DOWN``, ``MASKS``, ``HELP`` and ``KEYBINDS`` so those read the same everywhere; a custom mapping
replaces a table here and every handler and popup follows.
"""

from typing import Mapping, NamedTuple

from imgui_bundle import imgui


class Bind(NamedTuple):
    label: str
    action: str
    key: imgui.Key | None = None
    ctrl: bool = False
    shift: bool | None = None
    repeat: bool = False


def pressed(bind: Bind) -> bool:
    """
    Whether ``bind``'s key went down this frame with its modifiers: ctrl exactly as given, shift as given or, when
    None, either way (the handler then reads it as a stride). Never while a text field has the keyboard.
    """
    io = imgui.get_io()
    if bind.key is None or io.want_text_input or io.key_ctrl != bind.ctrl:
        return False
    if bind.shift is not None and io.key_shift != bind.shift:
        return False
    return imgui.is_key_pressed(bind.key, bind.repeat)


UP = Bind("up", "previous in the table (shift: by 10)", imgui.Key.up_arrow, repeat=True)
DOWN = Bind("down", "next in the table (shift: by 10)", imgui.Key.down_arrow, repeat=True)
MASKS = Bind("m", "toggle the masks overlay", imgui.Key.m)
HELP = Bind("h", "the help page", imgui.Key.h)
KEYBINDS = Bind("k", "these keybinds", imgui.Key.k)
LABEL_KEYS = (
    imgui.Key._1, imgui.Key._2, imgui.Key._3, imgui.Key._4, imgui.Key._5,
    imgui.Key._6, imgui.Key._7, imgui.Key._8, imgui.Key._9,
)

DEMIXING: Mapping[str, Bind] = {
    "up": UP,
    "down": DOWN,
    "left": Bind("left", "previous frame (shift: by 10)", imgui.Key.left_arrow, repeat=True),
    "right": Bind("right", "next frame (shift: by 10)", imgui.Key.right_arrow, repeat=True),
    "click": Bind(
        "click",
        "on an empty pixel: add its 5x5 pixel average to the plot as if grouped; on a drawn roi: plot its average alone",
    ),
    "ctrl_click": Bind("ctrl + click", "toggle a signal, drawn roi or pixel average in the group, in the image or the table"),
    "shift_click": Bind("shift + click", "add a signal or drawn roi to the group; in the table, every row up to it"),
    "scroll": Bind("shift / alt + scroll", "in the trace plot, zoom x only / y only"),
    "masks": MASKS,
    "contours": Bind("c", "toggle every other footprint's contour", imgui.Key.c),
    "follow": Bind("f", "center the view on the selection and keep following it", imgui.Key.f),
    "trace_follow": Bind(
        "t", "toggle center in the traces: the current frame stays in the middle as the movie plays", imgui.Key.t
    ),
    "pixel_trace": Bind(
        "p", "toggle quick pixel trace: a click on an empty pixel adds its 5x5 average to the plot", imgui.Key.p
    ),
    "roi": Bind("r", "add a roi: the next click on a panel starts its polygon", imgui.Key.r),
    "poly": Bind(
        "a", "poly-select: the next click on a panel starts its polygon; again or esc leaves it, the selection stays", imgui.Key.a
    ),
    "delete": Bind(
        "delete",
        "remove the selected roi, drop the active pixel average, or mark the selected signals for deletion (unmark "
        "when all are)",
        imgui.Key.delete,
    ),
    "escape": Bind(
        "esc",
        "close the help and keybinds windows, cancel a new roi, stop a poly-select (the selection stays), else "
        "deselect everything and drop the pixel averages",
        imgui.Key.escape,
    ),
    "select_all": Bind("ctrl + a", "group every signal the table shows", imgui.Key.a, ctrl=True),
    "undo": Bind("ctrl + z", "undo the last mark, drawn roi, pixel average or deselect", imgui.Key.z, ctrl=True),
    "help": HELP,
    "keybinds": KEYBINDS,
}

CLASSIFICATION: Mapping[str, Bind] = {
    "up": UP,
    "down": DOWN,
    "left": Bind("left", "previous background image", imgui.Key.left_arrow, shift=False, repeat=True),
    "right": Bind("right", "next background image", imgui.Key.right_arrow, shift=False, repeat=True),
    "group_prev": Bind("shift + left", "previous label group", imgui.Key.left_arrow, shift=True, repeat=True),
    "group_next": Bind("shift + right", "next label group", imgui.Key.right_arrow, shift=True, repeat=True),
    "label": Bind("1-9", "assign that label and move to the next ROI in view"),
    "clear": Bind("0", "clear the label", imgui.Key._0),
    "clear_delete": Bind("delete", "clear the label", imgui.Key.delete),
    "unlabeled": Bind("u", "jump to the next unlabeled ROI", imgui.Key.u),
    "masks": MASKS,
    "background": Bind("b", "toggle the background image", imgui.Key.b),
    "class_masks": Bind("c", "toggle every ROI's class mask on the full FOV", imgui.Key.c),
    "fov": Bind("f", "open the full FOV window on the current ROI", imgui.Key.f),
    "undo": Bind("ctrl + z", "undo the last label change", imgui.Key.z, ctrl=True),
    "escape": Bind("esc", "close the help and keybinds windows", imgui.Key.escape),
    "help": HELP,
    "keybinds": KEYBINDS,
}
