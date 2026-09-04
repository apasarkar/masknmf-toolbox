from masknmf.arrays.array_interfaces import ArrayLike
from typing import *
import numpy as np
from masknmf.compression import PMDArray, PMDResidualArray
from masknmf.utils import display
from masknmf.visualization.imgui import resolve_time_reference, is_notebook_canvas
from masknmf.diagnostics import pmd_autocovariance_diagnostics
import fastplotlib as fpl
from collections import OrderedDict
from masknmf.visualization.imgui import CheckboxWindow
import pygfx
from functools import partial

def mean_subtract_func(mean, frame):
    return frame - mean

class CompressionVis:
    def __init__(self,
                 moco_stack: ArrayLike,
                 pmd_stack: PMDArray,
                 frame_batch_size: int = 200,
                 mean_subtract: bool = False,
                 device="cpu",
                 frame_timings: Optional[np.ndarray | List[np.ndarray]] = None,
                 ref_range: Optional[dict] = None):

        self._mean_subtract = mean_subtract
        self._pmd_stack = PMDArray.from_flyweight(pmd_stack.shape,
                                                          pmd_stack.flyweight,
                                                          device=device,
                                                          rescale=True,
                                                          include_trend=True)
        self._moco_stack = moco_stack

        # Tricky: comparison stack is not mean subtracted, which is what we need to pass in to the residual array and to the autocov diagnostic
        self._residual_stack = PMDResidualArray(self.moco_stack, self.pmd_stack)
        display('Computing Residual Statistics')
        raw_lag1, pmd_lag1, resid_lag1 = pmd_autocovariance_diagnostics(self.moco_stack,
                                                                                            self.pmd_stack,
                                                                                            batch_size=frame_batch_size,
                                                                                            device=device)
        display('Residual Statistics: Complete')
        self._mcorr_name = "motion corrected mean 0" if mean_subtract else "motion corrected"
        self._pmd_name = "compressed+denoised mean 0" if mean_subtract else "compressed + denoised"
        self._residual_name = "residual"

        ref_range, frame_timings = resolve_time_reference(
            self.moco_stack.shape[0], frame_timings, ref_range
        )
        movie_index_mapping = {"time": frame_timings}

        self._video_names = [self._mcorr_name,
                             self._pmd_name,
                             self._residual_name]

        self._diagnostic_names = ["mcorr lag1 acf",
                                  "compressed lag1 acf",
                                  "resid lag1 acf"]

        self._ndw_videos = fpl.NDWidget(
            ref_range,
            shape=(1, 3),
            names=[*self._video_names],
            controller_ids=[
                tuple(self._video_names),
            ],
        )
        self._reference_index = self._ndw_videos.indices

        spatial_dims = ["m", "n"]
        dims = ["time", "m", "n"]
        self._moco_graphic = self._ndw_videos[self._mcorr_name].add_nd_image(self.moco_stack,
                                                                             dims,
                                                                             spatial_dims,
                                                                             slider_dim_transforms=movie_index_mapping.copy(),
                                                                             spatial_func=None if not mean_subtract else partial(
                                                                                 mean_subtract_func,
                                                                                 self.pmd_stack.mean_img.cpu().numpy()),
                                                                             name=self._mcorr_name)
        self._moco_graphic.graphic.cmap = "gray"

        self._pmd_graphic = self._ndw_videos[self._pmd_name].add_nd_image(self.pmd_stack,
                                                                          dims,
                                                                          spatial_dims,
                                                                          slider_dim_transforms=movie_index_mapping.copy(),
                                                                          spatial_func=None if not mean_subtract else partial(
                                                                              mean_subtract_func,
                                                                              self.pmd_stack.mean_img.cpu().numpy()),
                                                                          name=self._pmd_name)
        self._pmd_graphic.graphic.cmap = "gray"

        self._residual_graphic = self._ndw_videos[self._residual_name].add_nd_image(self.residual_stack,
                                                                                    dims,
                                                                                    spatial_dims,
                                                                                    slider_dim_transforms=movie_index_mapping.copy(),
                                                                                    name=self._residual_name)
        self._residual_graphic.graphic.cmap = "gray"

        self._ndw_diagnostics = fpl.NDWidget(ref_ranges=self.reference_index.ref_ranges,
                                             ref_index=self.reference_index,
                                             shape=(1, 3),
                                             names=[*self._diagnostic_names],
                                             controller_ids=[tuple(self._diagnostic_names)],
                                             )

        self._moco_lag1_graphic = self.ndw_diagnostics[self._diagnostic_names[0]].add_nd_image(raw_lag1,
                                                                                               spatial_dims,
                                                                                               spatial_dims,
                                                                                               slider_dim_transforms=None,
                                                                                               name=
                                                                                               self._diagnostic_names[
                                                                                                   0])

        self._pmd_lag1_graphic = self._ndw_diagnostics[self._diagnostic_names[1]].add_nd_image(pmd_lag1,
                                                                                               spatial_dims,
                                                                                               spatial_dims,
                                                                                               slider_dim_transforms=None,
                                                                                               name=
                                                                                               self._diagnostic_names[
                                                                                                   1])

        self._residual_lag1_graphic = self._ndw_diagnostics[self._diagnostic_names[2]].add_nd_image(resid_lag1,
                                                                                                    spatial_dims,
                                                                                                    spatial_dims,
                                                                                                    slider_dim_transforms=None,
                                                                                                    name=
                                                                                                    self._diagnostic_names[
                                                                                                        2])

        ## Use one camera for all of these spatial panels
        self._synchronize_spatial_panels()

        self._trace_panel_names = ['motion corrected', 'compressed + denoised', 'residual']

        self._ndw_temporal = fpl.NDWidget(ref_ranges=self.reference_index.ref_ranges,
                                          ref_index=self.reference_index,
                                          shape=(3, 1),
                                          names=[*self._trace_panel_names],
                                          controller_ids=[
                                              tuple(self._trace_panel_names),
                                          ],
                                          )
        self._synchronize_cameras_temporal()

        self._moco_trace_graphic = self._ndw_temporal[self._trace_panel_names[0]].add_nd_timeseries(
            None,
            ("l", "time", "d"),
            ("l", "time", "d"),
            slider_dim_transforms=movie_index_mapping.copy(),
            max_display_datapoints=5000,
            x_range_mode="auto",
            display_window=None,
            name=self._trace_panel_names[1],
        )

        self._pmd_trace_graphic = self._ndw_temporal[self._trace_panel_names[1]].add_nd_timeseries(
            None,
            ("l", "time", "d"),
            ("l", "time", "d"),
            slider_dim_transforms=movie_index_mapping.copy(),
            max_display_datapoints=5000,
            x_range_mode="auto",
            display_window=None,
            name=self._trace_panel_names[1],
        )

        self._residual_trace_graphic = self._ndw_temporal[self._trace_panel_names[2]].add_nd_timeseries(
            None,
            ("l", "time", "d"),
            ("l", "time", "d"),
            slider_dim_transforms=movie_index_mapping.copy(),
            max_display_datapoints=5000,
            x_range_mode="auto",
            display_window=None,
            name=self._trace_panel_names[2],
        )

        for subplot in self.ndw_temporal.figure:
            subplot.toolbar = False

        for subplot in self._ndw_videos.figure:
            subplot.tooltip.enabled = False

        for subplot in self._ndw_diagnostics.figure:
            subplot.tooltip.enabled = False

        self.rect_selector_kwargs = dict(
            edge_thickness=1,
            edge_color="w",
            vertex_size=3.0,
            vertex_color="cyan"
        )

        self.image_graphics = [self._moco_graphic.graphic,
                               self._pmd_graphic.graphic,
                               self._residual_graphic.graphic,
                               self._moco_lag1_graphic.graphic,
                               self._pmd_lag1_graphic.graphic,
                               self._residual_lag1_graphic.graphic]

        self.selectors = OrderedDict()

        for img in self.image_graphics:
            self.selectors[img] = list()

        self.roi_manager = CheckboxWindow("Add ROI")
        self.ndw_videos.figure.add_imgui_window(self.roi_manager, location="right", size=100, title="ROI Selector")

        self.RESIZING_NEW_RECT = False

        for graphic in self.image_graphics:
            graphic.add_event_handler(self.add_rectangle, "pointer_down")

        self.ndw_videos.figure.renderer.add_event_handler(self.resize_rect_vids, "pointer_move")
        self.ndw_videos.figure.renderer.add_event_handler(self.end_resize, "pointer_up")
        self.ndw_diagnostics.figure.renderer.add_event_handler(self.resize_rect_diagnostics, "pointer_move")
        self.ndw_diagnostics.figure.renderer.add_event_handler(self.end_resize, "pointer_up")

    @property
    def reference_index(self):
        return self._reference_index

    @property
    def ndw_videos(self):
        return self._ndw_videos

    @property
    def ndw_diagnostics(self):
        return self._ndw_diagnostics

    @property
    def ndw_temporal(self):
        return self._ndw_temporal

    @property
    def moco_stack(self):
        return self._moco_stack

    @property
    def pmd_stack(self):
        return self._pmd_stack

    @property
    def residual_stack(self):
        return self._residual_stack

    @property
    def trace_xdata(self):
        ## This is used every time to update traces
        return np.arange(self.moco_stack.shape[0])

    def _synchronize_cameras_temporal(self):
        for i, subplot in enumerate(self.ndw_temporal.figure):
            curr_camera = subplot.camera
            curr_controller = subplot.controller
            for j, other_subplot in enumerate(self.ndw_temporal.figure):
                if i >= j:
                    continue
                else:
                    other_camera = other_subplot.camera
                    other_controller = other_subplot.controller
                    curr_controller.add_camera(other_camera, include_state={"x", "width"})
                    other_controller.add_camera(curr_camera, include_state={"x", "width"})

    def _synchronize_spatial_panels(self):
        common_camera = self.ndw_videos.figure[0].camera
        for subplot in self.ndw_videos.figure:
            subplot.camera = common_camera
        for subplot in self.ndw_diagnostics.figure:
            subplot.camera = common_camera

    def rect_selector_moved(self, selectors_pair: Tuple[fpl.RectangleSelector], ev: fpl.GraphicFeatureEvent):
        for selector in selectors_pair:
            selector.selection = ev.info["value"]

        row_ixs, col_ixs = ev.get_selected_indices()
        self._row_slice = slice(row_ixs[0], row_ixs[-1] + 1)
        self._col_slice = slice(col_ixs[0], col_ixs[-1] + 1)

    def add_rectangle(self, ev: pygfx.PointerEvent):

        if not self.roi_manager.value:
            return

        if ev.button != 1:
            return

        for subplot in self.ndw_videos.figure:
            subplot.controller.enabled = False
        for subplot in self.ndw_diagnostics.figure:
            subplot.controller.enabled = False

        # in world space
        x, y = ev.pick_info["index"]

        new_selectors = list()

        for subplot in self.ndw_videos.figure:
            if len(subplot.graphics) < 1:
                continue  # empty subplot

            for img in subplot.graphics:

                new_selector = img.add_rectangle_selector(
                    selection=[x, x + 1, y, y + 1],
                    **self.rect_selector_kwargs
                )

                if len(self.selectors[img]) > 0:
                    old_selector = self.selectors[img].pop()
                    subplot.remove_graphic(old_selector)

                self.selectors[img].append(new_selector)
                new_selectors.append(new_selector)

        for subplot in self.ndw_diagnostics.figure:
            if len(subplot.graphics) < 1:
                continue  # empty subplot

            for img in subplot.graphics:

                new_selector = img.add_rectangle_selector(
                    selection=[x, x + 1, y, y + 1],
                    **self.rect_selector_kwargs
                )

                if len(self.selectors[img]) > 0:
                    old_selector = self.selectors[img].pop()
                    subplot.remove_graphic(old_selector)

                self.selectors[img].append(new_selector)
                new_selectors.append(new_selector)

        for sel in new_selectors:
            sel.add_event_handler(partial(self.rect_selector_moved, new_selectors), "selection")

        self.RESIZING_NEW_RECT = True

    def resize_rect_vids(self, ev: pygfx.PointerEvent):
        if not self.RESIZING_NEW_RECT:
            return

        img = self.image_graphics[0]

        for subplot in self.ndw_videos.figure:
            # world (x, y)
            pos = subplot.map_screen_to_world(ev)
            if pos is None:
                continue
            else:
                break

        if pos is None:
            # if pointer was moved outside the subplot
            self.RESIZING_NEW_RECT = False
            return

        x2, y2, _ = pos

        # most recently added selector
        x1, _, y1, _ = self.selectors[img][-1].selection

        self.selectors[img][-1].selection = [x1, x2, y1, y2]

    def resize_rect_diagnostics(self, ev: pygfx.PointerEvent):
        if not self.RESIZING_NEW_RECT:
            return

        img = self.image_graphics[3]  ## First graphic belonging to the diagnostics plot

        for subplot in self.ndw_diagnostics.figure:
            # world (x, y)
            pos = subplot.map_screen_to_world(ev)
            if pos is None:
                continue
            else:
                break

        if pos is None:
            # if pointer was moved outside the subplot
            self.RESIZING_NEW_RECT = False
            return

        x2, y2, _ = pos

        # most recently added selector
        x1, _, y1, _ = self.selectors[img][-1].selection

        self.selectors[img][-1].selection = [x1, x2, y1, y2]

    def end_resize(self, ev: pygfx.PointerEvent):
        if ev.button != 1:
            return
        if not self.RESIZING_NEW_RECT:
            return

        self._crop_and_display()

    def _crop_and_display(self):
        xdata = self.trace_xdata
        mcorr_temporal = self.moco_stack[:, self._row_slice, self._col_slice].mean(axis=(1, 2))
        pmd_temporal = self.pmd_stack[:, self._row_slice, self._col_slice].mean(axis=(1, 2))
        residual_temporal = mcorr_temporal - pmd_temporal

        self._moco_trace_graphic.data = fpl.utils.heatmap_to_positions(mcorr_temporal[None, :], xdata)
        self._pmd_trace_graphic.data = fpl.utils.heatmap_to_positions(pmd_temporal[None, :], xdata)
        self._residual_trace_graphic.data = fpl.utils.heatmap_to_positions(residual_temporal[None, :], xdata)

        for subplot in self.ndw_temporal.figure:
            subplot.auto_scale()

        for subplot in self.ndw_videos.figure:
            subplot.controller.enabled = True
        for subplot in self.ndw_diagnostics.figure:
            subplot.controller.enabled = True

        self.RESIZING_NEW_RECT = False
        self._row_slice = None
        self._col_slice = None

    def show(self):
        # parse based on canvas type
        if is_notebook_canvas(self.ndw_videos.figure):
            from ipywidgets import VBox
            return VBox([VBox([self.ndw_videos.show(), self.ndw_temporal.show(maintain_aspect=False)]),
                         self.ndw_diagnostics.show()])
        else:
            return self.ndw_videos.show(), self.ndw_diagnostics.show(), self.ndw_temporal.show(maintain_aspect=False)
