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
