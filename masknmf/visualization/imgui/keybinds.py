"""
Every viewer's keybinds in one place: a table of (key, action) rows per viewer, what the keybinds popup draws
and what a help page shows for its viewer. The viewers still read their keys inline; a custom mapping would
replace a table here, and the handlers would look their keys up in it.
"""

DEMIXING = (
    ("up / down", "previous / next signal in the table (shift: by 10)"),
    (
        "click",
        "on an empty pixel: add its 5x5 pixel average to the plot as if grouped; on a drawn roi: plot its "
        "average alone",
    ),
    (
        "ctrl + click",
        "toggle a signal, drawn roi or pixel average in the group, in the image or the table",
    ),
    (
        "shift + click",
        "add a signal or drawn roi to the group; in the table, every row up to it",
    ),
    (
        "esc",
        "cancel a new roi, stop a poly-select (the selection stays), else deselect everything and drop the "
        "pixel averages",
    ),
    ("ctrl + a", "group every signal the table shows"),
    ("ctrl + z", "undo the last mark, drawn roi, pixel average or deselect"),
    ("f", "center the view on the selection and keep following it"),
    (
        "p",
        "toggle quick pixel trace: a click on an empty pixel adds its 5x5 average to the plot",
    ),
    (
        "delete",
        "remove the selected roi, drop the active pixel average, or mark the selected signals for deletion "
        "(unmark when all are)",
    ),
    ("shift / alt + scroll", "in the trace plot, zoom x only / y only"),
    ("k", "show these keybinds"),
)
CLASSIFICATION = (
    ("up / down", "previous / next ROI"),
    ("shift + up / down", "jump 10 ROIs"),
    ("left / right", "previous / next background image"),
    ("shift + left / right", "previous / next label group"),
    ("1-9", "assign label"),
    ("0", "clear label"),
    ("u", "jump to next unlabeled ROI"),
    ("m", "toggle mask overlay"),
    ("b", "toggle background"),
    ("h", "toggle help"),
    ("k", "toggle keybinds"),
)
