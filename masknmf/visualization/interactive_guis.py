from typing import *
import numpy as np
from fastplotlib.widgets.image_widget import ImageWidget
from ipywidgets import HBox, VBox
import fastplotlib as fpl
import pygfx
import torch
from functools import partial
from ipywidgets import VBox, HBox
from collections import OrderedDict
import masknmf.arrays
from masknmf.utils import display
from masknmf.demixing import DemixingResults
from masknmf.compression import PMDArray
from masknmf.demixing import InitializationResults
from masknmf.demixing.demixing_arrays import ACArray, ColorfulACArray
from masknmf.demixing.demixing_utils import brightness_order

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

    # Demixing is run on the U/V representation, without rescaling, so we set rescale = False here to make sure scales match
    pmd_arr.rescale = False
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


def brightness_demix_init(curr_dr, splits=4, device='cpu'):
    """
    The goal of this is to segregate the signals based on "max brightness" and see whether the dimmer vs. brighter signals are significantly different in any way
    """
    curr_dr.to(device)
    a, c = curr_dr.a, curr_dr.c
    matched = torch.arange(a.shape[1], device=a.device).long()
    subset_ind = matched
    a_subset = torch.index_select(a, 1, subset_ind)
    c_subset = torch.index_select(c, 1, subset_ind)
    brightness_ordering, _ = masknmf.demixing.demixing_utils.brightness_order(a_subset, c_subset)
    matched_ordered = matched[brightness_ordering]

    points = [int(i) for i in np.linspace(0, brightness_ordering.shape[0], splits + 1)]

    subset_indices = [matched_ordered[points[i]:points[i + 1]] for i in range(splits)]

    curr_dr.to(device)
    pmd_arr = curr_dr.pmd_array
    pmd_arr.rescale = False

    pseudo_residuals = []
    subset_ac = []

    fluctuating_bg = curr_dr.fluctuating_background_array
    static_bg = curr_dr.b.reshape(curr_dr.fov_shape)
    for k in range(splits):
        curr_resid_ac_arr = curr_dr.ac_array
        curr_ac_arr = curr_dr.ac_array

        curr_resid_mask = torch.ones_like(curr_resid_ac_arr.mask)
        curr_subset_indices = subset_indices[k]
        curr_resid_mask[curr_subset_indices] = 0.0
        curr_resid_ac_arr.mask = curr_resid_mask

        curr_mask = torch.zeros_like(curr_ac_arr.mask)
        curr_mask[curr_subset_indices] = 1.0
        curr_ac_arr.mask = curr_mask

        resid_arr = masknmf.ResidualArray(pmd_arr,
                                          curr_resid_ac_arr,
                                          fluctuating_bg,
                                          static_bg)

        pseudo_residuals.append(resid_arr)
        subset_ac.append(curr_ac_arr)

    pmd_list = [pmd_arr for i in range(0, splits)]

    pmd_names = [f'pmd {i}' for i in range(0, splits)]
    ac_names = ['matched 75-100', 'matched 50-75', 'matched 25-50', 'matched 0-25']
    resid_names = ['pseudo resid 75-100', 'pseudo resid 50-75', 'pseudo resid 25-50', 'pseudo resid 0-25']
    iw = fpl.ImageWidget(names=[*pmd_names,
                                *ac_names,
                                *resid_names],
                         data=[*pmd_list,
                               *subset_ac,
                               *pseudo_residuals],
                         figure_shape=(3, splits))
    return iw


def quantile_segregated_signal_gui(ac_arr: masknmf.ACArray,
                                   partitions = 4) -> fpl.Figure:
    brightness_ordering, _ = brightness_order(ac_arr.a, ac_arr.c)
    points = [int(i) for i in np.linspace(0, brightness_ordering.shape[0], partitions + 1)]
    ac_arr_list = []
    colorful_arr_list = []
    for k in range(len(points) - 1):
        start = points[k]
        end = points[k+1]
        current_subset = brightness_ordering[start:end]
        curr_ac = ACArray(ac_arr.shape[1:], ac_arr.a, ac_arr.c)
        curr_colorful_ac = ColorfulACArray(ac_arr.shape[1:], ac_arr.a, ac_arr.c)
        curr_mask = torch.zeros_like(curr_ac.mask)
        curr_mask[current_subset] = 1.0
        curr_ac.mask = curr_mask
        curr_colorful_ac.mask = curr_mask
        ac_arr_list.append(curr_ac)
        colorful_arr_list.append(curr_colorful_ac)


    rgb = [*[False for i in range(len(ac_arr_list))], *[True for i in range(len(ac_arr_list))]]
    iw = fpl.ImageWidget(data = [*ac_arr_list, *colorful_arr_list],
                         figure_shape = (2, len(ac_arr_list)),
                         rgb = rgb)
    iw.cmap = "gray"
    return iw
