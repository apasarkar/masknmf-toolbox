import numpy as np

_NOTEBOOK_CANVASES = ("JupyterRenderCanvas", "AnywidgetRenderCanvas")


def is_notebook_canvas(figure) -> bool:
    """True when the figure renders through an ipywidgets-based canvas (wrap shows in HBox/VBox)."""
    return figure.canvas.__class__.__name__ in _NOTEBOOK_CANVASES


def resolve_time_reference(num_frames, frame_timings=None, ref_range=None, axis="time"):
    """
    Standardize the (ref_range, frame_timings) pair used by the NDWidget-based viewers.

    With frame_timings and no ref_range, builds one spanning the timings at their finest step.
    With neither, defaults to integer frame indices. A ref_range without frame_timings is an error.
    """
    if frame_timings is not None:
        if ref_range is None:
            ref_range = {
                axis: (
                    0,
                    np.amax(frame_timings),
                    np.amin(frame_timings[1:] - frame_timings[:-1]),
                )
            }
    else:
        if ref_range is not None:
            raise ValueError(
                "If you provide a reference range, you need to provide frame_timings"
            )
        ref_range = {axis: (0, num_frames, 1)}
        frame_timings = np.arange(num_frames)
    return ref_range, frame_timings


def close_figure(figure):
    """
    Close a figure whether or not it was ever shown.

    ``Figure.close`` goes through the output widget ``show`` creates, so closing
    a figure that was only built raises; the canvas is what needs closing then.
    """
    try:
        figure.close()
    except AttributeError:
        figure.canvas.close()


HANDLE_THICKNESS = 14.0
_HANDLE_TOOLTIP = "Drag to resize, double click to expand/collapse"
_HANDLE_CURSORS = {"top": "ns_resize", "left": "ew_resize"}


def _handle_rect(location, origin, width, height, thickness):
    """The strip on the window's inboard edge, in screen coordinates."""
    from imgui_bundle import imgui

    if location == "top":
        return (
            imgui.ImVec2(origin.x, origin.y + height - thickness),
            imgui.ImVec2(origin.x + width, origin.y + height),
        )
    return (
        imgui.ImVec2(origin.x + width - thickness, origin.y),
        imgui.ImVec2(origin.x + width, origin.y + height),
    )


def draw_edge_handle(window) -> None:
    """
    Draw a resize / collapse handle for a "top" or "left" edge window.

    Call it last in the window's draw callback. It hit-tests in screen space
    rather than claiming layout space, so reserve ``HANDLE_THICKNESS`` on the
    inboard edge yourself or the handle will sit over the content.
    """
    from imgui_bundle import imgui

    location = getattr(window, "location", None)
    if location not in _HANDLE_CURSORS:
        return
    thickness = window._separator_thickness
    rect_min, rect_max = _handle_rect(
        location,
        imgui.get_window_pos(),
        imgui.get_window_width(),
        imgui.get_window_height(),
        thickness,
    )
    mouse = imgui.get_mouse_pos()
    hovered = (
        rect_min.x <= mouse.x <= rect_max.x and rect_min.y <= mouse.y <= rect_max.y
    )

    if hovered and imgui.is_mouse_clicked(0):
        window._right_gui_resizing = True
    if not imgui.is_mouse_down(0):
        window._right_gui_resizing = False
    active = window._right_gui_resizing

    if hovered and imgui.is_mouse_double_clicked(0):
        if not window._collapsed:
            window._old_size = window.size
            window.size = int(thickness)
            window._collapsed = True
        else:
            window.size = int(window._old_size or window.size)
            window._collapsed = False

    if hovered or active:
        if not window._resize_cursor_set:
            window._figure.canvas.set_cursor(_HANDLE_CURSORS[location])
            window._resize_cursor_set = True
        imgui.set_tooltip(_HANDLE_TOOLTIP)
    elif window._resize_cursor_set:
        window._figure.canvas.set_cursor("default")
        window._resize_cursor_set = False

    if active and imgui.is_mouse_dragging(0):
        drag = imgui.get_mouse_drag_delta(0)
        # these handles sit on the inboard edge, so a positive drag grows them
        delta = drag.y if location == "top" else drag.x
        imgui.reset_mouse_drag_delta(0)
        if delta:
            window.size = max(30, round(window.size + delta))
            window._collapsed = False

    draw_list = imgui.get_window_draw_list()
    line = imgui.get_color_u32(
        imgui.ImVec4(0.9, 0.9, 0.9, 1.0)
        if (hovered or active)
        else imgui.ImVec4(0.5, 0.5, 0.5, 0.8)
    )
    background = imgui.get_color_u32(
        imgui.ImVec4(0.2, 0.2, 0.2, 0.8)
        if (hovered or active)
        else imgui.ImVec4(0.15, 0.15, 0.15, 0.6)
    )
    draw_list.add_rect_filled(rect_min, rect_max, background)
    center = imgui.ImVec2(
        (rect_min.x + rect_max.x) * 0.5, (rect_min.y + rect_max.y) * 0.5
    )
    for i in (-1, 0, 1):
        offset = i * 7.0
        dot = (
            imgui.ImVec2(center.x + offset, center.y)
            if location == "top"
            else imgui.ImVec2(center.x, center.y + offset)
        )
        draw_list.add_circle_filled(dot, 2, line)
