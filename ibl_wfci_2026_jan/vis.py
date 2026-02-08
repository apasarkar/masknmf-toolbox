import fastplotlib as fpl
import os
import masknmf
import sys
import numpy as np

from ipywidgets import HBox, VBox
import math

def raster_view(img_stack_1, img_stack_2, radius):
    ## This gives a basic GUI that allows us to look at two sets of videos. Double click on the top panels and in the bottom panel a rastermap of a local region of pixels will pop up
    
    def find_points_by_radius(y, x, radius):
    
        lower_point_y = max(0, y - radius)
        lower_point_x = max(0, x - radius)
        upper_point_y = lower_point_y + 2*radius
        upper_point_x = lower_point_x + 2*radius
        return lower_point_y, upper_point_y, lower_point_x, upper_point_x
        
    num_frames = img_stack_1.shape[0]
    data_shape = img_stack_1.shape[1], img_stack_1.shape[2]
    
    start_coordinates = (data_shape[0] // 2, data_shape[1] // 2)
    lower_point_y, upper_point_y, lower_point_x, upper_point_x = find_points_by_radius(start_coordinates[0], start_coordinates[1], radius)

    raster_1 = img_stack_1[:, lower_point_y:upper_point_y, lower_point_x:upper_point_x].reshape(num_frames, -1).T
    raster_2 = img_stack_2[:, lower_point_y:upper_point_y, lower_point_x:upper_point_x].reshape(num_frames, -1).T
    movie_widget = fpl.ImageWidget(data = [img_stack_1, 
                                    img_stack_2],
                           names = ['stack1',
                                    'stack2'])
    movie_widget.cmap = "gray"

    raster_widget = fpl.ImageWidget(data = [raster_1,
                                            raster_2],
                                    names = ['raster1',
                                             'raster2'])
    raster_widget.cmap = "gray"

    rect_selector_kwargs = dict(
                edge_thickness=1,
                edge_color="w",
                vertex_size=3.0,
                vertex_color="cyan"
            )
    
    sel_1 = movie_widget.managed_graphics[0].add_rectangle_selector(
                    selection=[lower_x, upper_x, lower_y, upper_y],
                    **rect_selector_kwargs
                )
    
    sel_2 = movie_widget.managed_graphics[1].add_rectangle_selector(
                    selection=[lower_x, upper_x, lower_y, upper_y],
                    **rect_selector_kwargs
                )

    time_sel_1 = raster_widget.managed_graphics[0].add_linear_selector(0, axis="x")
    time_sel_2 = raster_widget.managed_graphics[1].add_linear_selector(0, axis="x")
    
    def update_values(ev):
        x, y = ev.pick_info['index']
        lower_point_y, upper_point_y, lower_point_x, upper_point_x = find_points_by_radius(y, x, radius)
    
        raster_1 = img_stack_1[:, lower_point_y:upper_point_y, lower_point_x:upper_point_x].reshape(num_frames, -1).T
        raster_2 = img_stack_2[:, lower_point_y:upper_point_y, lower_point_x:upper_point_x].reshape(num_frames, -1).T
    
        sel_1.selection = [lower_point_x, upper_point_x, lower_point_y, upper_point_y]
        sel_2.selection = [lower_point_x, upper_point_x, lower_point_y, upper_point_y]
        raster_widget.managed_graphics[0].data = raster_1
        raster_widget.managed_graphics[1].data = raster_2

    def temporal_sync(ev):
        ## First implementation will be selector --> widget (change the selector, update widget)

        ##In this case, the event came from imagewidget
        if isinstance(ev, dict):
            print(ev['t'])
            curr_index = int(ev['t'])
            time_sel_1.selection = curr_index
            time_sel_2.selection = curr_index
            
        else:
            curr_index = math.floor(ev.info['value'])

            time_sel_1.selection = curr_index
            time_sel_2.selection = curr_index
            movie_widget.current_index = {'t': curr_index}
    
    
    movie_widget.managed_graphics[0].add_event_handler(update_values, "double_click")
    movie_widget.managed_graphics[1].add_event_handler(update_values, "double_click")

    time_sel_1.add_event_handler(temporal_sync, "selection")
    time_sel_2.add_event_handler(temporal_sync, "selection")
    movie_widget.add_event_handler(temporal_sync, "current_index")

    time_sel_1.visible = False
    time_sel_2.visible = False
    
    
    return VBox([movie_widget.show(), raster_widget.show()])