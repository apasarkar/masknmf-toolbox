from typing import *
import numpy as np
import torch
import fastplotlib as fpl
from fastplotlib.widgets import ImageWidget
from ipywidgets import HBox, VBox
import os
import re
from masknmf import DemixingResults, PMDArray
import torch


def signal_space_demixing(demixing_results: DemixingResults,
                          pmd_array: PMDArray,
                          v_range: tuple):
    mean_img = pmd_array.mean_img
    dense_ac_movie = demixing_results.ac_array[:]

    num_frames, fov_dim1, fov_dim2 = dense_ac_movie.shape

    data_order = demixing_results.ac_array.order
    a_dense = demixing_results.ac_array.a.cpu().to_dense().numpy().reshape((fov_dim1, fov_dim2, -1), order=data_order)
    c_numpy = demixing_results.ac_array.c.cpu().numpy()
    colors = demixing_results.colorful_ac_array.colors.cpu().numpy()

    color_projection_img = np.tensordot(a_dense, colors, axes=(2, 0))

    normalized_ac_movie = dense_ac_movie / np.amax(dense_ac_movie)
    normalized_mean_img = mean_img / np.amax(mean_img)

    superimposed_movie = normalized_ac_movie + 5 * normalized_mean_img[None, :, :]
    iw = fpl.ImageWidget(data=[color_projection_img, dense_ac_movie, superimposed_movie],
                         names=['Signal Img', 'Signal Movie', 'Superimposed'],
                         rgb=[True, False, False],
                         figure_shape=(1, 3),
                         histogram_widget=True,
                        graphic_kwargs = {'vmin':v_range[0], 'vmax': v_range[1]})

    
    ig = iw.figure[0, 0]["image_widget_managed"]
    iw.vmin = 0
    ig.vmax = 255

    

    line_fig = fpl.Figure((2, 1))

    placeholder = np.column_stack([np.arange(num_frames), np.zeros((num_frames))])
    lgraphic_1 = line_fig[0, 0].add_line(data=placeholder)
    lgraphic_2 = line_fig[1, 0].add_line(data=placeholder)

    def clickEvent(ev):
        dim2_coord, dim1_coord = ev.pick_info['index']

        a_identified = a_dense[dim1_coord, dim2_coord, :] != 0
        num_neurons = np.sum(a_identified.astype("int"))
        if num_neurons == 0:
            line_fig[0, 0].clear()
            line_fig[0, 0].add_line(data=placeholder)
            line_fig[0, 0].set_title(f"No Signals at {dim2_coord, dim2_coord}")
            line_fig[1, 0].clear()
            trace_to_show = (pmd_array[:, dim1_coord, dim2_coord] - pmd_array.mean_img[
                dim1_coord, dim2_coord]) / pmd_array.var_img[dim1_coord, dim2_coord]
            mean_pmd_trace = np.column_stack([np.arange(num_frames), trace_to_show])
            line_fig[1, 0].add_line(mean_pmd_trace)
            line_fig[1, 0].set_title(f"PMD Signal")
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
                curr = np.column_stack([np.arange(num_frames), c_traces[:, k] / np.amax(c_traces[:, k])])
                list_elts.append(curr)

            list_elts = np.array(list_elts)
            if list_elts.ndim == 2:
                list_elts = list_elts[None, :, :]
            line_fig[0, 0].add_line_stack(
                list_elts,
                colors=rgba_colors.squeeze(),
                separation=2
            )
            line_fig[0, 0].set_title(f"Signals at {dim2_coord, dim1_coord}.")
            trace_to_show = (pmd_array[:, dim1_coord, dim2_coord] - pmd_array.mean_img[
                dim1_coord, dim2_coord]) / pmd_array.var_img[dim1_coord, dim2_coord]
            mean_pmd_trace = np.column_stack([np.arange(num_frames), trace_to_show])
            line_fig[1, 0].add_line(mean_pmd_trace)
            line_fig[1, 0].set_title(f"PMD Signal")

        line_fig[1, 0].auto_scale(maintain_aspect=False)
        line_fig[0, 0].auto_scale(maintain_aspect=False)

    iw.figure[0, 0].graphics[0].add_event_handler(clickEvent, "click")
    iw.figure[0, 1].graphics[0].add_event_handler(clickEvent, "click")
    iw.figure[0, 2].graphics[0].add_event_handler(clickEvent, "click")

    return VBox([iw.show(), line_fig.show()])


def stack_comparison_interface(stack_1: Union[np.ndarray, PMDArray],
                               stack_2: Union[np.ndarray, PMDArray],
                               summary_img: np.ndarray,
                               names: Optional[List] = ["Stack 1", "Stack 2", "Summary Img"]):

    num_frames = stack_1.shape[0]
    def clickEvent(ev):
        dim2_coord, dim1_coord = ev.pick_info['index']

        data_list = [stack_2, stack_1]
        print(plot_trace_graphic.data[:].shape)
        for k in range(2):
            curr = data_list[k][:, dim1_coord, dim2_coord]
            plot_trace_graphic[k].data[:, 1] = curr
        line_fig[0, 0].set_title(f"Plots at {dim2_coord, dim1_coord}.")
        line_fig[0, 0].auto_scale(maintain_aspect=False)

    iw = fpl.ImageWidget(data=[stack_1, stack_2, summary_img],
                         names=names,
                         figure_shape=(1, 3))

    iw.cmap = "gray"

    iw.figure[0, 0].graphics[0].add_event_handler(clickEvent, "click")
    iw.figure[0, 1].graphics[0].add_event_handler(clickEvent, "click")
    iw.figure[0, 2].graphics[0].add_event_handler(clickEvent, "click")

    line_fig = fpl.Figure((1, 1))
    plot_trace_graphic = fpl.LineStack(
        data=[np.column_stack([np.arange(num_frames), np.zeros((num_frames))]),
              np.column_stack([np.arange(num_frames), np.zeros((num_frames))])],
        colors=['red', 'w'])
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
    correlation_image_widget = fpl.ImageWidget(data=[np.zeros_like(std_img)],
                                               names=['Select pixel on std img'])

    def click_pixel(ev):
        x, y = ev.pick_info['index']
        curr_pixel = image_stack[:, y, x].copy()
        curr_pixel = (curr_pixel - mean_img[y, x]) / mean_zero_norms[y, x]

        local_corr_img = (np.tensordot(curr_pixel[None, :], image_stack, axes = (1, 0)) -
                          mean_img[None, :, :] * np.sum(curr_pixel)).squeeze()
        local_corr_img /= mean_zero_norms

        correlation_image_widget.set_data(new_data=np.nan_to_num(local_corr_img, nan=0))
        correlation_image_widget.figure[0, 0].auto_scale(maintain_aspect=True)
        correlation_image_widget.figure[0, 0].set_title(f"Corr_Img at ({x}, {y})")

    std_img_graphic.add_event_handler(click_pixel, "click")

    return HBox([std_img_fig.show(), correlation_image_widget.show()])



def make_demixing_video(results: DemixingResults,
                        device: str,
                        v_range: tuple[float, float],
                        show_histogram: bool=False) -> ImageWidget:


    results.to(device)

    ac_arr = results.ac_array
    fluctuating_arr = results.fluctuating_background_array
    pmd_arr = results.pmd_array
    residual_arr = results.residual_array
    colorful_arr = results.colorful_ac_array
    static_bg = results.baseline.cpu().numpy()

    iw = ImageWidget(data=[pmd_arr,
                           ac_arr,
                           fluctuating_arr,
                           residual_arr,
                           colorful_arr,
                           static_bg],
                     names=['pmd',
                            'signals',
                            'fluctuating bkgd',
                            'residual',
                            'colorful signals',
                            'static Bkgd'],
                     rgb=[False, False, False, False, True, False],
                     histogram_widget=show_histogram,
                     graphic_kwargs = {'vmin':v_range[0], 'vmax':v_range[1]} if v_range is not None else None)
   
    for i, subplot in enumerate(iw.figure):
       if i == 4:
           ig = subplot["image_widget_managed"]
           iw.vmin = 0
           ig.vmax = 255
    
    return iw



