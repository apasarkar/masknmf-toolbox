from typing import Callable, Optional

import numpy as np

# a press travelling further than this many screen px is a drag, not a pick
CLICK_SLOP = 4


class StrokeDrawer:
    """
    Freehand stroke capture on one subplot.

    Left-drag pans by default, so the pan binding is lifted while drawing is
    armed; wheel zoom and right-drag zoom stay live. Emits a closed stroke on
    release, or a click position when disarmed.
    """

    def __init__(
        self,
        subplot,
        on_stroke: Callable[[list], None],
        on_click: Optional[Callable[[int, int], None]] = None,
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
            self.on_click(int(pos[1]), int(pos[0]))
