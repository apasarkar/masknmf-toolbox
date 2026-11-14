from dataclasses import dataclass
from typing import *


@dataclass
class PixelwiseHemocorrectionConfig:
    """
    Parameters of masknmf.hemocorrection.hemocorrect: one reference coefficient per pixel, fitted and removed on the
    compressed functional and reference channels.

    The defaults reproduce the January 2026 IBL widefield pipeline (no filtering, local projector) with one change:
    center=True fits with an intercept.
    """
    # subtract each pixel's temporal mean before fitting (regression with an intercept)
    center: bool = True
    # how the removed reference term is mapped into the functional spatial basis: "local" uses the functional
    # CompressionArray's local projector; "least_squares" solves the normal equations of the spatial basis exactly
    projection: Literal["local", "least_squares"] = "local"
    # highpass both channels before fitting only (Hz); slow drift then does not drive the coefficients. None: no filter
    fit_highpass_hz: float | None = None
    # lowpass the reference before fitting and subtracting (Hz). None: no filter. Cutoffs at or above Nyquist are skipped
    reference_lowpass_hz: float | None = None
    filter_order: int = 2
    # added to each coefficient's denominator
    eps: float = 1e-6
    # pixels handled at once while fitting
    pixel_batch_size: int = 2000


# every config a pipeline's hemocorrection step can take; see MotionCorrectionConfigs
HemocorrectionConfigs = PixelwiseHemocorrectionConfig
