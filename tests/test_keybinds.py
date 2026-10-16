"""The keybinds registry: in one viewer no key does two things, and the shared binds are the same objects."""

import pytest

from masknmf.visualization.imgui.keybinds import CLASSIFICATION, DEMIXING, DOWN, HELP, KEYBINDS, MASKS, UP


@pytest.mark.parametrize("table", [DEMIXING, CLASSIFICATION], ids=["demixing", "classification"])
def test_no_key_with_its_modifiers_binds_twice_in_a_table(table):
    seen = {}
    for name, bind in table.items():
        if bind.key is None:
            continue
        for key in bind.key if isinstance(bind.key, tuple) else (bind.key,):
            for shift in (False, True) if bind.shift is None else (bind.shift,):
                assert (key, bind.ctrl, shift) not in seen, f"{name} and {seen[(key, bind.ctrl, shift)]}"
                seen[(key, bind.ctrl, shift)] = name


def test_both_viewers_share_the_common_binds():
    for bind in (UP, DOWN, MASKS, HELP, KEYBINDS):
        assert bind in DEMIXING.values() and bind in CLASSIFICATION.values()
