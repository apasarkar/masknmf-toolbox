"""File > run parameters: the record of the run that made the results (:func:`masknmf.io.read_run_record`), read-only."""

import json

from imgui_bundle import icons_fontawesome_6 as fa
from imgui_bundle import imgui

from masknmf.visualization.imgui.theme import THEME, close_button, em, popup, to_vec4

RECORD_LABEL = f"{fa.ICON_FA_LIST_CHECK}  run parameters"

RECORD_TIP = "the values the run that made these results used, from the config.json beside them"

SECTIONS = ("run", "inputs", "configs", "timings")


def draw_json_tree(name: str, value) -> None:
    """``value`` under a tree node named ``name``; a dict or list opens into its items, a config shows its kind, anything else prints beside its name."""
    if isinstance(value, (dict, list)):
        items = value.items() if isinstance(value, dict) else enumerate(value)
        kind = value.get("kind") if isinstance(value, dict) else None
        label = name if kind is None else f"{name}  ({kind})"
        if imgui.tree_node(f"{label}##{name}"):
            for key, item in items:
                if key != "kind":
                    draw_json_tree(str(key), item)
            imgui.tree_pop()
        return
    imgui.text_disabled(f"{name}:")
    imgui.same_line()
    imgui.text(json.dumps(value))


def draw_run_record_popup(record: dict | None, is_open: bool) -> bool:
    """The Run parameters window over ``record``, a run record. Returns the new open state."""
    if not is_open or record is None:
        return False
    opened, is_open = popup("Run parameters", is_open)
    if opened:
        imgui.text_colored(to_vec4(THEME.accent), RECORD_LABEL)
        imgui.separator()
        imgui.text_disabled(f"{record.get('pipeline', 'unknown pipeline')}, masknmf {record.get('masknmf_version', '?')}")
        imgui.begin_child("##record", imgui.ImVec2(em(44), em(28)), imgui.ChildFlags_.borders)
        for name in SECTIONS:
            if name in record:
                imgui.set_next_item_open(True, imgui.Cond_.once)
                draw_json_tree(name, record[name])
        imgui.end_child()
        command = record.get("run", {}).get("command")
        if command is not None and imgui.button("Copy command"):
            imgui.set_clipboard_text(command)
        if close_button():
            is_open = False
    imgui.end()
    return is_open
