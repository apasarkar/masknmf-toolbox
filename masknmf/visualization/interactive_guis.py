from typing import *
import numpy as np
from fastplotlib.widgets import ImageWidget
from ipywidgets import HBox, VBox
import fastplotlib as fpl
from imgui_bundle import imgui
from fastplotlib import ui
import pygfx
from functools import partial
from ipywidgets import VBox, HBox
from collections import OrderedDict
import masknmf.arrays
from masknmf.utils import display
from masknmf.demixing import DemixingResults
from masknmf.compression import PMDArray


class ROIManager(ui.EdgeWindow):
    def __init__(self, figure, size):
        super().__init__(
            figure=figure,
            size=size,
            location="right",
            title="ROI Selector"
        )
        self.add_rois_mode = False

    def update(self):
        _, self.add_rois_mode = imgui.checkbox("Add ROI", self.add_rois_mode)

class PMDWidget:
    def __init__(self,
                 comparison_stack: Union[np.ndarray, masknmf.LazyFrameLoader],
                 pmd_stack: masknmf.PMDArray,
                 frame_batch_size: int=200,
                 mean_subtract: bool = False,
                 device="cpu"):

        self._mean_subtract = mean_subtract
        pmd_stack.to(device)
        self._pmd_stack = pmd_stack
        self._input_raw_stack = comparison_stack
        if self._mean_subtract:
            ref_mean_img = self.pmd_stack.mean_img.cpu().numpy()[None, :, :]
            self._comparison_stack = masknmf.FilteredArray(self._input_raw_stack,
                                       lambda x: x - ref_mean_img,
                                      device='cpu')
            self._pmd_stack.rescale = False
        else:
            self._comparison_stack = comparison_stack
            self._pmd_stack.rescale = True

        # Tricky: comparison stack is not mean subtracted, which is what we need to pass in to the residual array and to the autocov diagnostic
        self._residual_stack = masknmf.PMDResidualArray(comparison_stack, self.pmd_stack)
        display('Computing Residual Statistics')
        raw_lag1, pmd_lag1, resid_lag1 = masknmf.diagnostics.pmd_autocovariance_diagnostics(self._input_raw_stack,
                                                                                            self.pmd_stack,
                                                                                            batch_size=frame_batch_size,
                                                                                            device=device)
        display('Residual Statistics: Complete')
        mcorr_name = "mcorr mean 0" if mean_subtract else "mcorr"
        pmd_name = "pmd mean 0" if mean_subtract else "pmd"
        self._iw = fpl.ImageWidget([self.comparison_stack,
                                    self.pmd_stack,
                                    self._residual_stack,
                                    raw_lag1,
                                    pmd_lag1,
                                    resid_lag1],
                                   names=[mcorr_name,
                                          pmd_name,
                                          "residual",
                                          'mcorr lag1 acf',
                                          'pmd lag1 acf',
                                          'resid lag1 acf'],
                                   figure_shape=(2, 3)
                                   )

        self.image_graphics = [k for k in self.iw.managed_graphics]

        self._fig_temporal = fpl.Figure(shape=(3, 1), names=["mcorr", "pmd", "residual"])
        self._mcorr_line = self.fig_temporal["mcorr"].add_line(np.zeros(self.pmd_stack.shape[0]))
        self._pmd_line = self.fig_temporal["pmd"].add_line(np.zeros(self.pmd_stack.shape[0]))
        self._resid_line = self.fig_temporal["residual"].add_line(np.zeros(self.pmd_stack.shape[0]))

        self._mcorr_selectors = list()
        self._pmd_selectors = list()
        self._residual_selectors = list()

        self._moco_ls = self._mcorr_line.add_linear_selector()
        self._pmd_ls = self._pmd_line.add_linear_selector()
        self._resid_ls= self._resid_line.add_linear_selector()

        self._mcorr_selectors.append(self._moco_ls)
        self._pmd_selectors.append(self._pmd_ls)
        self._residual_selectors.append(self._resid_ls)

        self._iw.add_event_handler(self._sync_time, "current_index")
        self._moco_ls.add_event_handler(self._sync_time, "selection")
        self._pmd_ls.add_event_handler(self._sync_time, "selection")
        self._resid_ls.add_event_handler(self._sync_time, "selection")


        for subplot in self.fig_temporal:
            subplot.toolbar = False

        self.rect_selector_kwargs = dict(
            edge_thickness=1,
            edge_color="w",
            vertex_size=3.0,
            vertex_color="cyan"
        )

        self.selectors = OrderedDict()

        for img in self.image_graphics:
            self.selectors[img] = list()

        self.roi_manager = ROIManager(self.iw.figure, size=100)
        self.iw.figure.add_gui(self.roi_manager)

        self.RESIZING_NEW_RECT = False

        for img in self.image_graphics:
            img.add_event_handler(self.add_rectangle, "pointer_down")

        self.iw.figure.renderer.add_event_handler(self.resize_rect, "pointer_move")
        self.iw.figure.renderer.add_event_handler(self.end_resize, "pointer_up")


    def _sync_time(self, ev: dict | fpl.GraphicFeatureEvent):
        if isinstance(ev, dict):
            index = ev["t"]  # event from imagewidget
        else:
            # event from linear selector
            index = int(ev.info["value"])
        self.iw.current_index = {"t": index}
        self._moco_ls.selection = index
        self._pmd_ls.selection = index
        self._resid_ls.selection = index

    @property
    def comparison_stack(self):
        return self._comparison_stack

    @property
    def pmd_stack(self):
        return self._pmd_stack

    @property
    def iw(self):
        return self._iw

    @property
    def fig_temporal(self):
        return self._fig_temporal

    def rect_selector_moved(self, selectors_pair: Tuple[fpl.RectangleSelector], ev: fpl.GraphicFeatureEvent):
        for selector in selectors_pair:
            selector.selection = ev.info["value"]

        row_ixs, col_ixs = ev.get_selected_indices()
        self._row_slice = slice(row_ixs[0], row_ixs[-1] + 1)
        self._col_slice = slice(col_ixs[0], col_ixs[-1] + 1)
        #
        # mcorr_temporal = self.comparison_stack[:, row_slice, col_slice].mean(axis=(1, 2))
        #
        # pmd_temporal = self.pmd_stack[:, row_slice, col_slice].mean(axis=(1, 2))
        # residual_temporal = mcorr_temporal - pmd_temporal
        # self.fig_temporal["mcorr"].graphics[0].data[:, 1] = mcorr_temporal
        # self.fig_temporal["pmd"].graphics[0].data[:, 1] = pmd_temporal
        # self.fig_temporal["residual"].graphics[0].data[:, 1] = residual_temporal
        #
        # for subplot in self.fig_temporal:
        #     subplot.auto_scale()

    def add_rectangle(self, ev: pygfx.PointerEvent):

        if not self.roi_manager.add_rois_mode:
            return

        if ev.button != 1:
            return

        for subplot in self.iw.figure:
            subplot.controller.enabled = False

        # in world space
        x, y = ev.pick_info["index"]

        new_selectors = list()

        for subplot in self.iw.figure:
            if len(subplot.graphics) < 1:
                continue  # empty subplot

            img = subplot["image_widget_managed"]
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

    def resize_rect(self, ev: pygfx.PointerEvent):
        if not self.RESIZING_NEW_RECT:
            return

        img = self.image_graphics[0]

        for subplot in self.iw.figure:
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

        mcorr_temporal = self._input_raw_stack[:, self._row_slice, self._col_slice].mean(axis=(1, 2))
        pmd_temporal = self.pmd_stack[:, self._row_slice, self._col_slice].mean(axis=(1, 2))
        if self._mean_subtract:
            mcorr_temporal -= np.mean(mcorr_temporal.flatten())
            pmd_temporal -= np.mean(pmd_temporal.flatten())

        residual_temporal = mcorr_temporal - pmd_temporal
        self.fig_temporal["mcorr"].graphics[0].data[:, 1] = mcorr_temporal
        self.fig_temporal["pmd"].graphics[0].data[:, 1] = pmd_temporal
        self.fig_temporal["residual"].graphics[0].data[:, 1] = residual_temporal

        for subplot in self.fig_temporal:
            subplot.auto_scale()

        for subplot in self.iw.figure:
            subplot.controller.enabled = True

        self.RESIZING_NEW_RECT = False
        self._row_slice = None
        self._col_slice = None

    def show(self):
        return VBox([self.iw.show(), self.fig_temporal.show(maintain_aspect=False)])


def signal_space_demixing(demixing_results: masknmf.DemixingResults,
                          v_range: tuple,
                          device: str = 'cpu'):
    demixing_results.to(device)
    pmd_arr = demixing_results.pmd_array
    pmd_arr.rescale = False
    ac_arr = demixing_results.ac_array
    num_frames, fov_dim1, fov_dim2 = pmd_arr.shape

    data_order = demixing_results.ac_array.order
    a_dense = demixing_results.ac_array.export_a()
    c_numpy = demixing_results.ac_array.export_c()
    print(c_numpy.shape)
    colors = demixing_results.colorful_ac_array.colors.cpu().numpy()

    color_projection_img = np.tensordot(a_dense, colors, axes=(2, 0))

    iw = fpl.ImageWidget(
        data=[pmd_arr, ac_arr, color_projection_img],
        names=["pmd", "ac_movie", "color projection"],
        rgb=[False, False, True],
        figure_shape=(1, 3),
        histogram_widget=True,
        graphic_kwargs={"vmin": v_range[0], "vmax": v_range[1]},
    )

    ig = iw.figure[0, 2]["image_widget_managed"]
    iw.vmin = 0
    ig.vmax = 255

    line_fig = fpl.Figure((2, 1))

    placeholder = np.column_stack([np.arange(num_frames), np.zeros((num_frames))])
    lgraphic_1 = line_fig[0, 0].add_line(data=placeholder)
    lgraphic_2 = line_fig[1, 0].add_line(data=placeholder)

    def clickEvent(ev):
        dim2_coord, dim1_coord = ev.pick_info["index"]
        print(type(dim2_coord))
        print(dim2_coord)
        print(isinstance(dim2_coord, np.integer))

        a_identified = a_dense[dim1_coord, dim2_coord, :] != 0
        num_neurons = np.sum(a_identified.astype("int"))
        if num_neurons == 0:
            line_fig[0, 0].clear()
            line_fig[0, 0].add_line(data=placeholder)
            line_fig[0, 0].title = f"No Signals at {dim2_coord, dim2_coord}"
            line_fig[1, 0].clear()
            trace_to_show = pmd_arr[:, slice(dim1_coord, dim1_coord + 1), slice(dim2_coord, dim2_coord + 1)]
            mean_pmd_trace = np.column_stack([np.arange(num_frames), trace_to_show])
            line_fig[1, 0].add_line(mean_pmd_trace)
            line_fig[1, 0].title = f"PMD Signal"
        else:
            line_fig[0, 0].clear()
            line_fig[1, 0].clear()
            c_traces = c_numpy[:, a_identified]
            colors_used = colors[a_identified, :]

            if c_traces.ndim == 1:
                c_traces = c_traces[:, None]
            if colors_used.ndim == 1:
                colors_used = colors_used[None, :]

            rgba_colors = np.zeros((colors_used.shape[0], 4))
            rgba_colors[:, :3] = colors_used
            rgba_colors[:, 3] = 1.0

            list_elts = []
            for k in range(num_neurons):
                curr = np.column_stack(
                    [np.arange(num_frames), c_traces[:, k] / np.amax(c_traces[:, k])]
                )
                list_elts.append(curr)

            list_elts = np.array(list_elts)
            if list_elts.ndim == 2:
                list_elts = list_elts[None, :, :]
            line_fig[0, 0].add_line_stack(
                list_elts, colors=rgba_colors.squeeze(), separation=2
            )
            line_fig[0, 0].title = f"Signals at {dim2_coord, dim1_coord}."
            trace_to_show = pmd_arr[:, dim1_coord, dim2_coord]
            mean_pmd_trace = np.column_stack([np.arange(num_frames), trace_to_show])
            line_fig[1, 0].add_line(mean_pmd_trace)
            line_fig[1, 0].title = f"PMD Signal"

        line_fig[1, 0].auto_scale(maintain_aspect=False)
        line_fig[0, 0].auto_scale(maintain_aspect=False)

    iw.figure[0, 0].graphics[0].add_event_handler(clickEvent, "click")
    iw.figure[0, 1].graphics[0].add_event_handler(clickEvent, "click")
    iw.figure[0, 2].graphics[0].add_event_handler(clickEvent, "click")

    return VBox([iw.show(), line_fig.show()])

def stack_comparison_interface(
    stack_1: Union[np.ndarray, PMDArray],
    stack_2: Union[np.ndarray, PMDArray],
    summary_img: np.ndarray,
    names: Optional[List] = ["Stack 1", "Stack 2", "Summary Img"],
):
    num_frames = stack_1.shape[0]

    def clickEvent(ev):
        dim2_coord, dim1_coord = ev.pick_info["index"]

        data_list = [stack_2, stack_1]
        print(plot_trace_graphic.data[:].shape)
        for k in range(2):
            curr = data_list[k][:, dim1_coord, dim2_coord]
            plot_trace_graphic[k].data[:, 1] = curr
        line_fig[0, 0].set_title(f"Plots at {dim2_coord, dim1_coord}.")
        line_fig[0, 0].auto_scale(maintain_aspect=False)

    iw = fpl.ImageWidget(
        data=[stack_1, stack_2, summary_img], names=names, figure_shape=(1, 3)
    )

    iw.cmap = "gray"

    iw.figure[0, 0].graphics[0].add_event_handler(clickEvent, "click")
    iw.figure[0, 1].graphics[0].add_event_handler(clickEvent, "click")
    iw.figure[0, 2].graphics[0].add_event_handler(clickEvent, "click")

    line_fig = fpl.Figure((1, 1))
    plot_trace_graphic = fpl.LineStack(
        data=[
            np.column_stack([np.arange(num_frames), np.zeros((num_frames))]),
            np.column_stack([np.arange(num_frames), np.zeros((num_frames))]),
        ],
        colors=["red", "w"],
    )
    line_fig[0, 0].add_graphic(plot_trace_graphic)
    line_fig[0, 0].auto_scale(maintain_aspect=False)

    return VBox([iw.show(), line_fig.show()])


def get_correlation_widget(image_stack: np.ndarray) -> HBox:
    num_frames = image_stack.shape[0]
    mean_img = np.mean(image_stack, axis=0)
    std_img = np.std(image_stack, axis=0)
    mean_zero_norms = std_img * (num_frames**0.5)

    std_img_fig = fpl.Figure((1, 1))
    std_img_graphic = std_img_fig[0, 0].add_image(data=std_img, name="Std Img")
    correlation_image_widget = fpl.ImageWidget(
        data=[np.zeros_like(std_img)], names=["Select pixel on std img"]
    )

    def click_pixel(ev):
        x, y = ev.pick_info["index"]
        curr_pixel = image_stack[:, y, x].copy()
        curr_pixel = (curr_pixel - mean_img[y, x]) / mean_zero_norms[y, x]

        local_corr_img = (
            np.tensordot(curr_pixel[None, :], image_stack, axes=(1, 0))
            - mean_img[None, :, :] * np.sum(curr_pixel)
        ).squeeze()
        local_corr_img /= mean_zero_norms

        correlation_image_widget.set_data(new_data=np.nan_to_num(local_corr_img, nan=0))
        correlation_image_widget.figure[0, 0].auto_scale(maintain_aspect=True)
        correlation_image_widget.figure[0, 0].set_title(f"Corr_Img at ({x}, {y})")

    std_img_graphic.add_event_handler(click_pixel, "click")

    return HBox([std_img_fig.show(), correlation_image_widget.show()])


def make_demixing_video(
    results: DemixingResults,
    device: str,
    v_range: Tuple[float, float],
    show_histogram: bool = False,
) -> ImageWidget:
    results.to(device)

    ac_arr = results.ac_array
    fluctuating_arr = results.fluctuating_background_array
    pmd_arr = results.pmd_array
    residual_arr = results.residual_array
    colorful_arr = results.colorful_ac_array
    global_residual_img = results.global_residual_correlation_image.cpu().numpy()

    iw = ImageWidget(
        data=[pmd_arr, ac_arr, fluctuating_arr, residual_arr, colorful_arr, global_residual_img],
        names=[
            "pmd",
            "signals",
            "fluctuating bkgd",
            "residual",
            "colorful signals",
            "global resid corr img",
        ],
        rgb=[False, False, False, False, True, False],
        histogram_widget=show_histogram,
        graphic_kwargs={"vmin": v_range[0], "vmax": v_range[1]}
        if v_range is not None
        else None,
    )

    for i, subplot in enumerate(iw.figure):
        if i == 4:
            ig = subplot["image_widget_managed"]
            ig.vmin = 0
            ig.vmax = 255
        if i == 5:
            ig = subplot["image_widget_managed"]
            ig.vmin = 0.0
            ig.vmax = 1.0


    return iw


def visualize_superpixels_peaks(superpixel_results: dict):
    superpixel_map = superpixel_results['superpixel_map']
    pure_superpixel_map = superpixel_results['pure_superpixel_map']
    correlation_image = superpixel_results['correlation_image']

    superpixel_img = np.stack([correlation_image.copy()] * 3, axis=-1)
    superpixel_img[superpixel_map > 0] = [4, 0, 0]

    pure_superpixel_img = np.stack([correlation_image.copy()] * 3, axis=-1)
    pure_superpixel_img[pure_superpixel_map > 0] = [4, 0, 0]
    iw = fpl.ImageWidget(data=[np.stack([correlation_image] * 3, axis=-1),
                               superpixel_img,
                               pure_superpixel_img],
                         rgb=[True, True, True],
                         figure_shape=(1, 3),
                         names=['corr', 'superpix', 'pure superpix'])
    return iw
