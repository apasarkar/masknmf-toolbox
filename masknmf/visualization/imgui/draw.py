"""Freehand stroke capture on a fastplotlib subplot."""

from typing import Callable, Optional

import numpy as np

# a press travelling further than this many screen px is a drag, not a click
CLICK_SLOP = 4


class StrokeDrawer:
    """
    Freehand stroke capture on one subplot.

    Left-drag pans by default, so the pan binding is lifted while drawing is
    armed; wheel zoom and right-drag zoom stay live. Emits a closed stroke on
    release, or a click position plus the held keyboard modifiers (a frozenset
    of pygfx names such as "Ctrl" and "Shift") when disarmed.

    Args:
        subplot: the subplot to draw on
        on_stroke (Callable[[list], None]): called with the stroke's world-space
            (x, y) points on release
        on_click (Optional[Callable[[int, int, frozenset], None]]): called with
            (row, col, modifiers) for a click while disarmed
        color (str): stroke color while drawing
    """

    def __init__(
        self,
        subplot,
        on_stroke: Callable[[list], None],
        on_click: Optional[Callable[[int, int, frozenset], None]] = None,
        color: str = "magenta",
    ):
        self.subplot = subplot
        self.on_stroke = on_stroke
        self.on_click = on_click
        self.armed = False
        self.stroke: list = []
        self._pan_control = None
        self._press = None

        self.line = subplot.add_line(
            np.zeros((2, 3), np.float32), colors=color, thickness=2.0,
            name="stroke", offset=(0, 0, 2), visible=False,
        )
        self.line.world_object.material.pick_write = False

        subplot.renderer.add_event_handler(self._down, "pointer_down")
        subplot.renderer.add_event_handler(self._move, "pointer_move")
        subplot.renderer.add_event_handler(self._up, "pointer_up")

    def arm(self, on: bool):
        """Take over the left mouse button for drawing, or give it back to pan."""
        if on == self.armed:
            return
        self.armed = on
        controls = self.subplot.controller.controls
        if on:
            self._pan_control = controls.pop("mouse1", None)
        elif self._pan_control is not None:
            controls["mouse1"] = self._pan_control
            self._pan_control = None
        self.stroke = []
        self.line.visible = False

    def close(self):
        """Disarm and take the pointer handlers back off the renderer."""
        self.arm(False)
        renderer = self.subplot.renderer
        for handler, event in (
            (self._down, "pointer_down"),
            (self._move, "pointer_move"),
            (self._up, "pointer_up"),
        ):
            try:
                renderer.remove_event_handler(handler, event)
            except (KeyError, ValueError):
                pass

    def _down(self, ev):
        if ev.button != 1:
            return
        pos = self.subplot.map_screen_to_world((ev.x, ev.y))
        if pos is None:
            return
        if self.armed:
            self.stroke = [(float(pos[0]), float(pos[1]))]
        else:
            self._press = (ev.x, ev.y)

    def _move(self, ev):
        if not self.stroke:
            return
        pos = self.subplot.map_screen_to_world((ev.x, ev.y), allow_outside=True)
        if pos is None:
            return
        self.stroke.append((float(pos[0]), float(pos[1])))
        vertices = np.zeros((len(self.stroke), 3), np.float32)
        vertices[:, :2] = self.stroke
        self.line.data = vertices
        self.line.visible = True

    def _up(self, ev):
        if self.stroke:
            stroke, self.stroke = self.stroke, []
            self.line.visible = False
            self.on_stroke(stroke)
            return
        if self._press is None or self.on_click is None:
            return
        press, self._press = self._press, None
        if abs(ev.x - press[0]) > CLICK_SLOP or abs(ev.y - press[1]) > CLICK_SLOP:
            return
        pos = self.subplot.map_screen_to_world((ev.x, ev.y))
        if pos is not None:
            mods = frozenset(getattr(ev, "modifiers", ()) or ())
            self.on_click(int(pos[1]), int(pos[0]), mods)
