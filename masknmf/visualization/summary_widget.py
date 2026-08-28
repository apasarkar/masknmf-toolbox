"""Backwards-compatible alias; the implementation is shared with mbo_utilities."""

from masknmf.visualization.imgui.summary import (
    CMAPS,
    CONTRAST_AUTO,
    CONTRAST_FULL,
    CONTRAST_MANUAL,
    CONTRAST_MODES,
    GpuImage,
    SummaryImageViewer,
    auto_range,
    data_range,
    format_value,
    to_rgba,
)

__all__ = [
    "CMAPS",
    "CONTRAST_AUTO",
    "CONTRAST_FULL",
    "CONTRAST_MANUAL",
    "CONTRAST_MODES",
    "GpuImage",
    "SummaryImageViewer",
    "auto_range",
    "data_range",
    "format_value",
    "to_rgba",
]
