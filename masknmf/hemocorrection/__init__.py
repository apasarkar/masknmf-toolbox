"""Removing a reference (e.g. hemodynamic) channel from a functional channel, both held as CompressionArrays."""
from .regression import (
    HemocorrectionResults,
    hemocorrect,
    pixelwise_coefficients,
    reference_term_in_functional_basis,
    align_temporal,
    interleaved_frame_positions,
)

__all__ = [
    "HemocorrectionResults",
    "hemocorrect",
    "pixelwise_coefficients",
    "reference_term_in_functional_basis",
    "align_temporal",
    "interleaved_frame_positions",
]
