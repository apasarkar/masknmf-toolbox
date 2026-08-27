from typing import *
import numpy as np
import fastplotlib as fpl
from imgui_bundle import imgui
import h5py
import torch
from functools import partial
import masknmf
from masknmf.visualization.imgui import (
    component_at_pixel,
    resolve_time_reference,
)


class CurationVis:
    """
    Curation widget to accept/reject signals.
    """

    def __init__(
        self,
        demixing_results: masknmf.DemixingResults,
        frame_timings: Optional[np.ndarray] = None,
        ref_range: Optional[dict] = None,
        iscell: Optional[np.ndarray] = None,
        save_path: Optional[str] = None,
        click_radius: int = 10,
        device: str = "cuda",
    ):
        """save_path: the results hdf5 file; iscell is written into it on every flip"""
        self._demixing_results = demixing_results
        self._demixing_results.to(device)
        self._device = device
        self._save_path = save_path
        self._save_error: Optional[str] = None

        num_neurons = self._demixing_results.a.shape[1]
        if iscell is None:
            iscell = np.ones((num_neurons,), dtype=bool)
        else:
            iscell = np.asarray(iscell).astype(bool)
            if iscell.shape[0] != num_neurons:
                raise ValueError(
                    f"iscell has {iscell.shape[0]} entries, expected {num_neurons}"
                )
        self._iscell = iscell

        ref_range, frame_timings = resolve_time_reference(
            self._demixing_results.shape[0], frame_timings, ref_range
        )
        self._frame_timings = frame_timings

        fov_shape = self._demixing_results.fov_shape
        flyweight = self._demixing_results.flyweight
        self._accepted_array = masknmf.ColorfulACArray.from_flyweight(
            fov_shape, flyweight
        )
        self._rejected_array = masknmf.ColorfulACArray.from_flyweight(
            fov_shape, flyweight
        )
        self._rejected_array.colors = self._accepted_array.colors
        self._ac_array = self._demixing_results.ac_array
        self._a = self._demixing_results.a.coalesce()
        self._apply_masks()

        self._selected: list[int] = []
        self._click_radius = click_radius
        self._pointer_down_xy = (0.0, 0.0)

        self._panels = ("accepted", "rejected", "traces")
        extents = {
            self._panels[0]: (0, 0.5, 0.0, 0.65),
            self._panels[1]: (0.5, 1, 0.0, 0.65),
            self._panels[2]: (0, 1, 0.65, 1.0),
        }
        self._ndw = fpl.NDWidget(
            ref_range,
            extents=extents,
            names=[*self._panels],
            controller_ids=[tuple(self._panels[:2])],
            size=(1200, 1200),
        )
        # the standard right-click menu can reset the camera (autoscale/center)
        self._ndw.figure.remove_imgui_right_click()

        movie_dims = ("time", "m", "n", "c")
        spatial_dims = ("m", "n", "c")
        slider_transforms = {"time": frame_timings}
        self._accepted_graphic = self._ndw["accepted"].add_nd_image(
            self._accepted_array,
            movie_dims,
            spatial_dims,
            rgb_dim="c",
            slider_dim_transforms=slider_transforms.copy(),
            name="accepted",
        )
        self._rejected_graphic = self._ndw["rejected"].add_nd_image(
            self._rejected_array,
            movie_dims,
            spatial_dims,
            rgb_dim="c",
            slider_dim_transforms=slider_transforms.copy(),
            name="rejected",
        )
        self._trace_graphic = self._ndw["traces"].add_nd_timeseries(
            None,
            ("l", "time", "d"),
            ("l", "time", "d"),
            slider_dim_transforms=slider_transforms.copy(),
            max_display_datapoints=5000,
            x_range_mode="auto",
            display_window=None,
            name="traces",
        )

        self._selector = fpl.ImageHighlightSelector(
            lut="tab10",
            lut_wrap="repeat",
            selection_options={"pixels": self._ac_array.contours},
            options_color="w",
            options_alpha=0.0,
            alpha=0.9,
        )
        self._image_graphics = {
            self._accepted_graphic.graphic: True,
            self._rejected_graphic.graphic: False,
        }
        for graphic, accepted in self._image_graphics.items():
            self._selector.add_graphic(graphic)
            graphic.add_event_handler(partial(self._select_click, accepted), "click")
            graphic.add_event_handler(partial(self._flip_click, accepted), "double_click")
        self._selector.selection = None
        self._ndw.figure.renderer.add_event_handler(
            self._record_pointer_down, "pointer_down"
        )

        for name in self._panels[:2]:
            self._ndw.figure[name].tooltip.enabled = False

        self._ndw.figure["traces"].camera.maintain_aspect = False
        self._refresh_traces()

        self._ndw.figure.add_imgui_window(
            self._draw_panel, location="top", size=70, title="Curation"
        )

    def _apply_masks(self):
        keep = torch.from_numpy(self._iscell)
        self._accepted_array.mask = keep
        self._rejected_array.mask = ~keep

    def _neuron_at(self, pick_index, accepted: bool) -> Optional[int]:
        side = self._iscell if accepted else ~self._iscell
        return component_at_pixel(
            self._a,
            self._ac_array.centers,
            self._demixing_results.fov_shape,
            pick_index,
            mask=torch.from_numpy(side),
            radius=self._click_radius,
        )

    def _record_pointer_down(self, ev):
        self._pointer_down_xy = (ev.x, ev.y)

    def _is_drag(self, ev) -> bool:
        # pygfx fires "click" even after a pan; ignore clicks that moved
        x, y = self._pointer_down_xy
        return abs(ev.x - x) > 4 or abs(ev.y - y) > 4

    def _select_click(self, accepted: bool, ev):
        if ev.button not in (1, 2) or self._is_drag(ev):
            return
        neuron = self._neuron_at(ev.pick_info["index"], accepted)
        if neuron is None:
            return
        if ev.button == 2:
            ids = list(self._selected) if neuron in self._selected else [neuron]
            self._set_selection(ids)
            self.flip(ids)
            return
        same_side = self._selected and bool(self._iscell[self._selected[0]]) == accepted
        if "Shift" in ev.modifiers and same_side:
            if neuron in self._selected:
                if len(self._selected) > 1:
                    self._selected.remove(neuron)
            else:
                self._selected.append(neuron)
            self._set_selection(self._selected)
        else:
            self._set_selection([neuron])

    def _flip_click(self, accepted: bool, ev):
        if ev.button != 1 or self._is_drag(ev):
            return
        neuron = self._neuron_at(ev.pick_info["index"], accepted)
        if neuron is None:
            return
        self._set_selection([neuron])
        self.flip([neuron])

    def flip(self, neuron_ids: Sequence[int]):
        """Move the given neurons to the other side (accepted <-> rejected)"""
        self._iscell[list(neuron_ids)] = ~self._iscell[list(neuron_ids)]
        self._apply_masks()
        current_index = dict(self._ndw.indices)["time"]
        self._ndw.indices.set_dim_index("time", current_index)
        if self._save_path is not None:
            try:
                self.save()
                self._save_error = None
            except OSError as e:
                self._save_error = str(e)

    def _set_selection(self, neuron_ids: list[int]):
        self._selected = list(neuron_ids)
        self._selector.selection = self._selected if self._selected else None
        self._refresh_traces()

    def _refresh_traces(self):
        num_frames = self._demixing_results.shape[0]
        subplot = self._ndw.figure["traces"]
        if not self._selected:
            self._trace_graphic.data = fpl.utils.heatmap_to_positions(
                np.zeros((1, num_frames), dtype=np.float32), self._frame_timings
            )
            y_range = (0.0, 1.0)
            subplot.title = "0 selected"
        else:
            traces = self._demixing_results.c[:, self._selected].T.cpu().numpy()
            traces = traces - np.amin(traces, axis=1, keepdims=True)
            scale = np.amax(traces, axis=1, keepdims=True)
            scale[scale == 0] = 1
            traces = traces / scale + np.arange(len(self._selected))[:, None]
            self._trace_graphic.data = fpl.utils.heatmap_to_positions(
                traces, self._frame_timings
            )
            self._trace_graphic.graphic.colors = (
                self._accepted_array.colors[self._selected].cpu().numpy()
            )
            y_range = (-0.2, len(self._selected) + 0.2)
            subplot.title = f"{len(self._selected)} selected"

        # setting NDTimeseries data rescales the camera; fit to the data span after
        subplot.camera.maintain_aspect = False
        subplot.camera.zoom = 1.0
        subplot.x_range = (float(self._frame_timings[0]), float(self._frame_timings[-1]))
        subplot.y_range = y_range

    def _draw_panel(self):
        imgui.text(f"accepted: {int(self._iscell.sum())}")
        imgui.same_line(0, 30)
        imgui.text(f"rejected: {int((~self._iscell).sum())}")
        imgui.same_line(0, 30)
        if imgui.button("flip selected") and self._selected:
            self.flip(self._selected)
        imgui.same_line(0, 30)
        if imgui.button("accept all") and not self._iscell.all():
            self.flip(np.where(~self._iscell)[0])
        imgui.same_line(0, 30)
        imgui.text(f"autosave: {self._save_path if self._save_path else 'off'}")
        if self._save_error is not None:
            imgui.same_line(0, 30)
            imgui.text(f"save failed: {self._save_error}")

    def save(self, path: Optional[str] = None):
        """Write iscell into the results hdf5 as DemixingResults/iscell"""
        path = path if path is not None else self._save_path
        with h5py.File(path, "a") as f:
            group = f.require_group("DemixingResults")
            if "iscell" in group:
                del group["iscell"]
            group.create_dataset("iscell", data=self._iscell)

    @property
    def iscell(self) -> np.ndarray:
        return self._iscell

    @property
    def demixing_results(self) -> masknmf.DemixingResults:
        return self._demixing_results

    @property
    def widget(self) -> fpl.NDWidget:
        return self._ndw

    def show(self):
        canvas = self._ndw.show()
        # show() autoscales all subplots; re-fit the traces to the exact data span
        self._refresh_traces()
        return canvas


def main(argv=None):
    import argparse

    parser = argparse.ArgumentParser(
        description="Accept/reject curation GUI for demixing results"
    )
    parser.add_argument(
        "path", help="hdf5 file containing DemixingResults; iscell is saved into it"
    )
    parser.add_argument(
        "--iscell", default=None, help="import labels from an external iscell.npy"
    )
    parser.add_argument(
        "--device", default=None, help="torch device (default: cuda if available)"
    )
    args = parser.parse_args(argv)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    if args.iscell is not None:
        iscell = np.load(args.iscell)
        if iscell.ndim == 2:  # suite2p-style (num_neurons, 2)
            iscell = iscell[:, 0]
    else:
        with h5py.File(args.path, "r") as f:
            iscell = (
                f["DemixingResults/iscell"][()]
                if "DemixingResults/iscell" in f
                else None
            )

    results = masknmf.DemixingResults.from_hdf5(args.path)
    vis = CurationVis(results, iscell=iscell, save_path=args.path, device=device)
    vis.show()
    fpl.loop.run()


if __name__ == "__main__":
    main()
