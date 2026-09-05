from typing import *
import numpy as np
import fastplotlib as fpl
from imgui_bundle import imgui
from fastplotlib import ui
import pygfx
import torch
from collections import OrderedDict
import masknmf.arrays
from masknmf.utils import display
from functools import partial
from fastplotlib.widgets.nd_widget._index import ReferenceIndex
from masknmf.visualization.imgui import TracePlot, resolve_time_reference

class SingleSessionDemixingVis:
    """
    This is a general viewer for analyzing demixing results a single imaging field of view (say, frames x height x width)
    """
    def __init__(
        self,
        demixing_results: masknmf.DemixingResults | List[masknmf.DemixingResults],
        frame_timings: Optional[np.ndarray | List[np.ndarray]] = None,
        ref_range: Optional[dict] = None,
        roi_radius: int = 1,
        summary_img: np.ndarray | masknmf.ArrayLike | None = None,
        summary_img_name: str | None = None,
        show_contours: bool = True,
        device='cpu'
    ):
        self._roi_radius = roi_radius
        if device=='cpu':
            display("Using CPU; it will be much slower. Use CUDA for much faster rendering")
        self._demixing_results = demixing_results
        self._device = device

        self._demixing_results.to(self.device)

        ref_range, frame_timings = resolve_time_reference(
            self.demixing_results.shape[0], frame_timings, ref_range
        )

        self._video_panels = ("compressed+denoised",
                        "signals",
                        "background",
                        "residual",
                        "colorful_signals",
                        "summary img")



        self._pmd_array = self.demixing_results.pmd_array
        self._fluctuating_background_array = self.demixing_results.fluctuating_background_array
        self._residual_array = self.demixing_results.residual_array
        self._colorful_ac_array = self.demixing_results.colorful_ac_array
        self._ac_array = self.demixing_results.ac_array

        self._video_extents =  {
                self._video_panels[0]: (0, 0.333, 0.0, 0.5),
                self._video_panels[1]: (0.33, 0.666, 0.0, 0.5),
                self._video_panels[2]: (0.666, 1, 0.0, 0.5),
                self._video_panels[3]: (0.0, 0.333, 0.5, 1.0),
                self._video_panels[4]: (0.333, 0.666, 0.5, 1.0),
                self._video_panels[5]: (0.666, 1, 0.5, 1.0)}


        self._ndw_fov = fpl.NDWidget(
            ref_range,
            extents=self._video_extents,
            names=[*self._video_panels],
            controller_ids=[
                tuple(self._video_panels),
            ],
            size=(1200, 1200),
        )

        self._reference_index = self._ndw_fov.indices

        movie_dims = ["time", "m", "n"]
        movie_spatial_dims = ["m", "n"]
        movie_index_mapping = {"time": frame_timings}
        self._pmd_graphic = self._ndw_fov[self._video_panels[0]].add_nd_image(
            self._pmd_array,
            movie_dims,
            movie_spatial_dims,
            slider_dim_transforms=movie_index_mapping.copy(),
            name=self._video_panels[0],
        )

        self._ac_graphic = self._ndw_fov[self._video_panels[1]].add_nd_image(
            self._ac_array,
            movie_dims,
            movie_spatial_dims,
            slider_dim_transforms=movie_index_mapping.copy(),
            name=self._video_panels[1],
        )

        self._background_graphic = self._ndw_fov[self._video_panels[2]].add_nd_image(
            self._fluctuating_background_array,
            movie_dims,
            movie_spatial_dims,
            slider_dim_transforms=movie_index_mapping.copy(),
            name=self._video_panels[2],
        )

        self._residual_graphic = self._ndw_fov[self._video_panels[3]].add_nd_image(
            self._residual_array,
            movie_dims,
            movie_spatial_dims,
            slider_dim_transforms=movie_index_mapping.copy(),
            name=self._video_panels[3],
        )

        movie_dims_rgb = ["time", "m", "n", "c"]
        movie_spatial_dims_rgb= ["m", "n", "c"]
        movie_index_mapping = {"time": frame_timings}
        self._colorful_signal_graphic = self._ndw_fov[self._video_panels[4]].add_nd_image(
            self._colorful_ac_array,
            movie_dims_rgb,
            movie_spatial_dims_rgb,
            slider_dim_transforms=movie_index_mapping.copy(),
            rgb_dim="c",
            name=self._video_panels[4],
        )

        if summary_img is not None:
            dimension_data = ["m", "n"] if summary_img.ndim == 2 else ["time", "m", "n"]
            self._summary_image = self._ndw_fov[self._video_panels[5]].add_nd_image(
                summary_img,
                dimension_data,
                ["m", "n"],
                name=self._video_panels[5],
            )
            ##Code to show contours

            self._image_selector = fpl.ImageHighlightSelector(lut="tab10",
                                                       lut_wrap="repeat",
                                                       selection_options={"pixels": self.demixing_results.ac_array.contours},
                                                       options_color="w",
                                                       options_alpha=0.1,
                                                       alpha=0.7
                                                       )

            if show_contours:
                self._image_selector.add_graphic(self._summary_image.graphic)

            self._ndw_fov.figure[self._video_panels[5]].title = summary_img_name if summary_img_name is not None else "Summary Image"

        else:
            self._summary_image = self._ndw_fov[self._video_panels[5]].add_nd_image(
                self.demixing_results.global_residual_correlation_image.cpu().numpy(),
                ["m", "n"],
                ["m", "n"],
                name=self._video_panels[5],
            )

            self._ndw_fov.figure[self._video_panels[5]].title = summary_img_name if summary_img_name is not None else "Residual Correlation Image"


        self._traces = TracePlot(("traces",), self.demixing_results.shape[0], frame_timings)
        self._traces.dock(self._ndw_fov.figure, size=360, title="traces")
        self._traces.link(self.reference_index)
        self._traces.on_pick = self._select_signal
        self._base_lines = ("compressed", "background", "residual")
        # footprint of the signal picked in the trace dock, drawn over the signals movie
        self._pick_selector = fpl.ImageHighlightSelector(
            color="w",
            selection_options={"pixels": self.demixing_results.ac_array.contours},
            options_alpha=0.0,
            alpha=0.6,
        )
        self._pick_selector.add_graphic(self._ac_graphic.graphic)

        self._selected_signals = None

        for name in self._video_panels:
            self._ndw_fov[name][name].graphic.add_event_handler(partial(self._click_update), "double_click")

        for subplot in self._ndw_fov.figure:
            subplot.tooltip.enabled = False

    def _select_signal(self, panel: str, index: int):
        """Highlight the footprint of the demixed signal whose line was double-clicked."""
        index -= len(self._base_lines)
        if self._selected_signals is None or not 0 <= index < len(self._selected_signals):
            self._pick_selector.selection = []
            return
        self._pick_selector.selection = [int(self._selected_signals[index])]

    ## Let's make a dummy click event for now
    def _click_update(self, ev: pygfx.PointerEvent):
        num_frames, height, width = self.demixing_results.shape
        col, row = ev.pick_info["index"]

        col_start, col_stop = max(0, col - self._roi_radius), min(width, col + self._roi_radius + 1)
        row_start, row_stop = max(0, row - self._roi_radius), min(height, row + self._roi_radius + 1)
        ## For each array, add the appropriate data

        pmd_trace = np.mean(self._pmd_array[:, row_start:row_stop, col_start:col_stop], axis = (1,2))
        residual_trace = np.mean(self._residual_array[:, row_start:row_stop, col_start:col_stop], axis = (1, 2))
        background_trace = np.mean(self._fluctuating_background_array[:, row_start:row_stop, col_start:col_stop], axis = (1, 2))

        #Pull out colorful signals
        separated_ac_signals, separated_colors, unique_signals = extract_per_trace_roi_averages(self._colorful_ac_array,
                                                              slice(row_start, row_stop),
                                                              slice(col_start, col_stop))

        self._selected_signals = unique_signals
        self._pick_selector.selection = []
        greys = ((0.9, 0.9, 0.9), (0.6, 0.6, 0.6), (0.4, 0.4, 0.4))
        lines = list(zip(self._base_lines, (pmd_trace, background_trace, residual_trace), greys))
        if separated_ac_signals is not None:
            # the movie's colors sum to 1 per signal; scaled up so the lines read on a dark plot
            lines += [
                (f"signal {k}", trace, tuple(color / color.max()))
                for k, trace, color in zip(unique_signals, separated_ac_signals, separated_colors)
            ]
        self._traces.set("traces", lines)

    @property
    def roi_radius(self) -> int:
        return self._roi_radius

    @roi_radius.setter
    def roi_radius(self, new_radius):
        self._roi_radius = new_radius

    @property
    def device(self) -> str:
        return self._device

    @property
    def demixing_results(self) -> masknmf.DemixingResults:
        return self._demixing_results

    @property
    def fov_widget(self) -> fpl.NDWidget:
        return self._ndw_fov

    @property
    def traces(self) -> TracePlot:
        return self._traces

    @property
    def reference_index(self) -> ReferenceIndex:
        return self._reference_index

    def show(self):
        return self.fov_widget.show()


def extract_per_trace_roi_averages(colorful_ac_array: masknmf.ACArray,
                                   rowslice: slice,
                                   colslice: slice):
    """

    Args:
        ac_array (masknmf.ACArray): The signal array that contains the factorized signals
        coloring (torch.tensor): Shape (num_neurons, 3) #Each row is RGB coloring
    """
    device = colorful_ac_array.device
    num_frames, height, width, _ = colorful_ac_array.shape
    a = colorful_ac_array.a.coalesce() #Shape (num_pixels, num_signals)
    c = colorful_ac_array.c #Shape (num_frames, num_signals)

    pixel_space = torch.arange(height * width, device = device).reshape(height, width).long()
    good_row_values = pixel_space[rowslice, colslice].flatten()
    num_pixels = good_row_values.shape[0]

    row, col = a.indices()
    values = a.values()

    valid_indices = torch.isin(row, good_row_values)
    if torch.count_nonzero(valid_indices) == 0:
        return None, None, None
    else:
        valid_columns = col[valid_indices]
        unique_signals = torch.unique(valid_columns)

        a_subset = torch.index_select(a, 1, unique_signals).coalesce()
        filtered_rows, filtered_col = a_subset.indices()
        filtered_values = a_subset.values()

        valid_indices = valid_indices = torch.isin(filtered_rows, good_row_values)
        filtered_rows = filtered_rows[valid_indices]
        filtered_col = filtered_col[valid_indices]
        filtered_values = filtered_values[valid_indices]

        reduce_tensor = torch.zeros(a_subset.shape[1], device=device)
        reduce_tensor.scatter_reduce_(0, filtered_col, filtered_values, reduce="sum")
        reduce_tensor = reduce_tensor / num_pixels

        # unique_signals = torch.unique(filtered_col)
        # unique_scales = reduce_tensor[unique_signals]

        weighted_signals = reduce_tensor[None, :] * c[:, unique_signals] #Shape (num_frames, neural_signals)
        colors = colorful_ac_array.colors[unique_signals, :] #(neural_signals, 3)

        return weighted_signals.T.cpu().numpy(), colors.cpu().numpy(), unique_signals.cpu().numpy()


def visualize_superpixels_peaks(init_results: masknmf.InitializationResults):
    superpixel_map = init_results.nmf_seed_map
    pure_superpixel_map = init_results.pure_nmf_seed_map
    correlation_image = init_results.correlation_img

    superpixel_img = np.stack([correlation_image.copy()] * 3, axis=-1)
    superpixel_img[superpixel_map > 0] = [4, 0, 0]

    pure_superpixel_img = np.stack([correlation_image.copy()] * 3, axis=-1)
    pure_superpixel_img[pure_superpixel_map > 0] = [4, 0, 0]

    corr_rgb = np.stack([correlation_image] * 3, axis=-1)

    image_panels = ("corr image",
                    "nmf seed map",
                    "pure nmf seed map")

    extents = {
        image_panels[0]: (0, 0.333, 0.0, 1),
        image_panels[1]: (0.33, 0.666, 0.0, 1),
        image_panels[2]: (0.666, 1, 0.0, 1)}

    ndw_corr = fpl.NDWidget(
        extents=extents,
        names=[*image_panels],
        controller_ids=[
            tuple(image_panels),
        ],
        size=(1200, 1200),
    )

    corr_img_graphic = ndw_corr[image_panels[0]].add_nd_image(corr_rgb, ["m", "n", "c"], ["m", "n", "c"], rgb_dim="c",
                                                              name=image_panels[0])
    nmf_seed_graphic = ndw_corr[image_panels[1]].add_nd_image(superpixel_img,
                                                              ["m", "n", "c"],
                                                              ["m", "n", "c"],
                                                              rgb_dim="c",
                                                              name=image_panels[1])
    pure_seed_graphic = ndw_corr[image_panels[2]].add_nd_image(pure_superpixel_img,
                                                               ["m", "n", "c"],
                                                               ["m", "n", "c"],
                                                               rgb_dim="c",
                                                               name=image_panels[2])

    return ndw_corr.show()





