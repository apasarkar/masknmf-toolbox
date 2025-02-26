import localnmf
import localmd
from typing import *
import numpy as np
import torch
import os
import sys
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

import fastplotlib as fpl
from fastplotlib.widgets import ImageWidget
from ipywidgets import HBox, VBox
import os
import re
from .utils import display
import plotly.graph_objects as go
import plotly.subplots as sp




def signal_space_demixing(demixing_results: localnmf.DemixingResults,
                          pmd_array: localmd.PMDArray,
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


def stack_comparison_interface(stack_1: Union[np.ndarray, localmd.PMDArray],
                               stack_2: Union[np.ndarray, localmd.PMDArray],
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
    mean_img = np.mean(image_stack, axis = 0)
    std_img = np.std(image_stack, axis = 0)
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

# For every signal, need to look at the temporal trace and the PMD average, superimposed
def get_roi_avg(array, p1, p2, normalize=True):
    """
    Given nonzero dim1 and dim2 indices p1 and p2, get the ROI average
    """
    if np.amin(p1) == np.amax(p1):
        term1 = slice(np.amin(p1), np.amin(p1) + 1)
        expand_first = True
    else:
        term1 = slice(np.amin(p1), np.amax(p1) + 1)
        expand_first = False

    if np.amin(p2) == np.amax(p2):
        term2 = slice(np.amin(p2), np.amin(p2) + 1)
        expand_second = True
    else:
        term2 = slice(np.amin(p2), np.amax(p2) + 1)
        expand_second = False

    selected_pixels = array[:, term1, term2]
    if expand_first:
        selected_pixels = np.expand_dims(selected_pixels, 1)
    if expand_second:
        selected_pixels = np.expand_dims(selected_pixels, 2)


    if selected_pixels.ndim < 3:
        print(f"error in roi avg, {p1} and {p2}")
        print(f"term 1 is {term1} and term2 is {term2}")
    data_2d = selected_pixels[:, p1 - np.amin(p1), p2 - np.amin(p2)]
    avg_trace = np.mean(data_2d, axis=1)
    if normalize:
        return avg_trace / np.amax(avg_trace)
    else:
        return avg_trace



# Custom sorting function to sort based on the numerical part after 'neuron_'



def construct_index(folder: str, file_prefix = "neuron", index_name = "index.html"):

    def numerical_sort(file):
        match = re.search(rf'{file_prefix}[_\s]*(\d+)', file)
        return int(match.group(1)) if match else float('inf')  # Default to large number if no match
    
    index_file = os.path.join(folder, index_name)
    
    # List all HTML files in the directory
    html_files = [f for f in os.listdir(folder) if f.endswith('.html')]
    html_files.sort(key=numerical_sort)  # Sort files by numerical order

    # Create the index.html file
    with open(index_file, 'w') as f:
        f.write('<!DOCTYPE html>\n')
        f.write('<html lang="en">\n')
        f.write('<head>\n')
        f.write('    <meta charset="UTF-8">\n')
        f.write('    <meta name="viewport" content="width=device-width, initial-scale=1.0">\n')
        f.write('    <title>Navigation Index</title>\n')
        f.write('    <style>\n')
        f.write('        body { font-family: Arial, sans-serif; margin: 20px; text-align: center; }\n')
        f.write('        .content { margin-bottom: 20px; }\n')
        f.write('        .nav-buttons { margin-top: 20px; }\n')
        f.write('        button { padding: 10px 20px; margin: 5px; font-size: 16px; }\n')
        f.write('    </style>\n')
        f.write('</head>\n')
        f.write('<body>\n')
        f.write('    <h1>Navigate Through Files</h1>\n')
        f.write('    <div class="content" id="content">\n')
        f.write('        <iframe src="" style="width:100%; height:600px; border:none;"></iframe>\n')
        f.write('    </div>\n')
        f.write('    <div class="nav-buttons">\n')
        f.write('        <button id="prev-btn" onclick="navigate(-1)">Previous</button>\n')
        f.write('        <button id="next-btn" onclick="navigate(1)">Next</button>\n')
        f.write('    </div>\n')
        f.write('\n')
        f.write('    <script>\n')
        f.write('        const files = [\n')
        for file in html_files:
            f.write(f'            \'{file}\',\n')
        f.write('        ];\n')
        f.write('        let currentIndex = 0;\n')
        f.write('        const contentDiv = document.getElementById(\'content\');\n')
        f.write('        const prevBtn = document.getElementById(\'prev-btn\');\n')
        f.write('        const nextBtn = document.getElementById(\'next-btn\');\n')
        f.write('\n')
        f.write('        function loadContent() {\n')
        f.write('            contentDiv.innerHTML = `<iframe src=\"${files[currentIndex]}\" style="width:100%; height:600px; border:none;"></iframe>`;\n')
        f.write('            prevBtn.disabled = currentIndex === 0;\n')
        f.write('            nextBtn.disabled = currentIndex === files.length - 1;\n')
        f.write('        }\n')
        f.write('\n')
        f.write('        function navigate(direction) {\n')
        f.write('            currentIndex += direction;\n')
        f.write('            if (currentIndex >= 0 && currentIndex < files.length) {\n')
        f.write('                loadContent();\n')
        f.write('            }\n')
        f.write('        }\n')
        f.write('\n')
        f.write('        // Initial load\n')
        f.write('        loadContent();\n')
        f.write('    </script>\n')
        f.write('</body>\n')
        f.write('</html>\n')
    
    print(f'Index file "{index_file}" created successfully.')



    

def plot_ith_roi(i: int, results, folder=".", name="neuron.html", radius:int = 5,
                 residual_mode: Optional[localnmf.ResidCorrMode] = None):
    """
    Generates a diagnostic plot of the i-th ROI using Plotly
    Args:
        i (int): The neuron data
        results: the results from the demixing procedure
        folder (str): folder where the data is saved. This folder must exist already.
        name (str): The name of the output .html file
        radius (int): For each ROI we show, we provide a residual correlation image to show the broader context of the data. 
            This param specifies how big that radius is
        residual_mode (localnmf.ResidCorrMode): The residual correlation mode of the localnmf resid corr object.
    """
    if not os.path.exists(folder):
        raise ValueError(f"folder {folder} does not exist; please make it then run this code")

    order = results.order
    current_a = torch.index_select(results.a, 1,
                                   torch.arange(i, i + 1).to(results.device)).to_dense().cpu().numpy()
    a = current_a.reshape((results.shape[1], results.shape[2]), order=order)

    p1, p2 = a.nonzero()
    T, d1, d2 = results.pmd_array.shape
    pmd_roi_avg = get_roi_avg(results.pmd_array, p1, p2, normalize=False)
    static_bg_roi_avg = np.ones_like(pmd_roi_avg) * np.mean(results.baseline[p1, p2].cpu().numpy().flatten())
    fluctuating_bg_roi_avg = get_roi_avg(results.fluctuating_background_array, p1, p2, normalize=False)
    signal_roi_avg = np.mean(a[a > 0]) * results.c[:, i].cpu().numpy()
    residual_roi_avg = get_roi_avg(results.residual_array, p1, p2, normalize=False)

    lb_dim1 = max(int(np.amin(p1)) - radius, 0)
    ub_dim1 = min(int(np.amax(p1)) + radius, d1)
    lb_dim2 = max(int(np.amin(p2)) - radius, 0)
    ub_dim2 = min(int(np.amax(p2)) + radius, d2)
    
    # Spatial Footprint
    residual_data = results.residual_array[:, lb_dim1:ub_dim1, lb_dim2:ub_dim2]
    residual_img = np.std(residual_data, axis=0)
    
    mean_pmd_img = np.std(results.pmd_array[:, lb_dim1:ub_dim1, lb_dim2:ub_dim2], axis=0)

    if residual_mode is None:
        results.residual_correlation_image.mode = localnmf.ResidCorrMode.DEFAULT
    else:
        results.residual_correlation_image.mode = residual_mode
    resid_corr_img = results.residual_correlation_image[i, lb_dim1:ub_dim1, lb_dim2:ub_dim2]

    std_corr_img = results.standard_correlation_image[i, lb_dim1:ub_dim1, lb_dim2:ub_dim2]

    # Create x and y axes that match the p1 and p2 coordinates
    x_ticks = np.arange(lb_dim2, ub_dim2)
    y_ticks = np.arange(lb_dim1, ub_dim1)

    # Create a Plotly subplot
    fig = sp.make_subplots(
        rows=5, cols=5,
        subplot_titles=[
            "Spatial Footprint", "Residual Std Dev Image", "PMD Std Dev Image", "Corr(PMD, c_i)", "Corr(Resid, c_i)", 
            "Temporal Trace", "Background Trace", "Residual", "PMD, Signal, Background"
        ],
        specs=[[{'type': 'heatmap'}, {'type': 'heatmap'}, {'type': 'heatmap'}, {'type': 'heatmap'}, {'type': 'heatmap'}],
               [{'colspan': 5}, None, None, None, None],
               [{'colspan': 5}, None, None, None, None],
               [{'colspan': 5}, None, None, None, None],
               [{'colspan': 5}, None, None, None, None]]
    )

    # Adding heatmaps with synchronized zooming and custom axes (using p1 and p2)
    fig.add_trace(go.Heatmap(z=a[lb_dim1:ub_dim1, lb_dim2:ub_dim2], x=x_ticks, y=y_ticks, 
                             showscale=False, colorscale='Viridis'), row=1, col=1)
    fig.add_trace(go.Heatmap(z=residual_img, x=x_ticks, y=y_ticks, 
                             showscale=False, colorscale='Viridis'), row=1, col=2)
    fig.add_trace(go.Heatmap(z=mean_pmd_img, x=x_ticks, y=y_ticks, 
                             showscale=False, colorscale='Viridis'), row=1, col=3)

    #Fix the values of the zmin and zmax for these correlation images so it's easier to visually compare them
    fig.add_trace(go.Heatmap(z=std_corr_img, x=x_ticks, y=y_ticks, 
                             showscale=False, colorscale='Viridis', zmin = 0, zmax = 1), row=1, col=4)
    fig.add_trace(go.Heatmap(z=resid_corr_img, x=x_ticks, y=y_ticks, 
                             showscale=False, colorscale='Viridis', zmin = 0, zmax = 1), row=1, col=5)

    # Temporal Trace
    fig.add_trace(go.Scatter(y=signal_roi_avg, mode='lines', name='Signal'), row=2, col=1)

    #Background ROI average
    fig.add_trace(go.Scatter(y=fluctuating_bg_roi_avg, mode='lines', name='Background'), row=3, col=1)

    # Residual Trace
    fig.add_trace(go.Scatter(y=residual_roi_avg, mode='lines', name='Residual'), row=4, col=1)

    # ROI Avg + Signal + Background
    normalizer = np.amax(pmd_roi_avg)
    fig.add_trace(go.Scatter(y=4 + pmd_roi_avg / normalizer, mode='lines', name='PMD'), row=5, col=1)
    fig.add_trace(go.Scatter(y=3 + signal_roi_avg / normalizer, mode='lines', name='Source'), row=5, col=1)
    fig.add_trace(go.Scatter(y=2 + fluctuating_bg_roi_avg / normalizer, mode='lines', name='Net Bkgd'), row=5, col=1)

    # Update the layout to adjust titles and color axes
    fig.update_layout(
        title=f"ROI {i} Diagnostic Plot",
        height=800,
        xaxis=dict(matches='x1', scaleanchor="y1", scaleratio=1),
        yaxis=dict(matches='y1', scaleanchor="x1", scaleratio=1),
        xaxis1=dict(matches='x1', scaleanchor="y1", scaleratio=1),
        yaxis1=dict(matches='y1', scaleanchor="x1", scaleratio=1),
        xaxis2=dict(matches='x1', scaleanchor="y1", scaleratio=1),
        yaxis2=dict(matches='y1', scaleanchor="x1", scaleratio=1),
        xaxis3=dict(matches='x1', scaleanchor="y1", scaleratio=1),
        yaxis3=dict(matches='y1', scaleanchor="x1", scaleratio=1),
        xaxis4=dict(matches='x1', scaleanchor="y1", scaleratio=1),
        yaxis4=dict(matches='y1', scaleanchor="x1", scaleratio=1),
        xaxis5=dict(matches='x1', scaleanchor="y1", scaleratio=1),
        yaxis5=dict(matches='y1', scaleanchor="x1", scaleratio=1),
    )

    

    # Save to an HTML file
    fig.write_html(os.path.join(folder, name))
    
    # Return the figure for further inspection (optional)
    return fig



def make_demixing_video(results: localnmf.DemixingResults,
                        device: str,
                        v_range: tuple[float, float],
                        show_histogram: bool=False) -> ImageWidget:


    results.to(device) #Hardcoded for now until fastplotlib updates the vmin/vmax computations

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



