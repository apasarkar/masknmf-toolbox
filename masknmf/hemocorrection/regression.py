"""
Pixelwise hemodynamic correction carried out entirely on compressed representations.

Setting: a functional channel F (e.g. 470 nm, calcium dependent) and a reference channel R (e.g. 405 nm, isosbestic,
hemodynamics only), each compressed separately into a CompressionArray:

    F[p, t] = mean_F[p] + sigma_F[p] * (U_F V_F)[p, t]  (+ trend)
    R[p, t] = mean_R[p] + sigma_R[p] * (U_R V_R)[p, t]  (+ trend)

The correction works on the normalized fluctuations f = U_F V_F and r = U_R V_R~ (V_R~ = reference temporal factors
aligned to the functional frame times). For every pixel p it fits one coefficient

    c[p] = <f_p, r_p> / <r_p, r_p>          (optionally after centering and temporal filtering)

and removes c[p] * r_p from f_p. A per-pixel coefficient is unaffected by per-pixel scaling, so fitting in each
channel's normalized units is equivalent to fitting on the raw fluctuations.

Nothing is reconstructed in time: <f_p, r_p> = U_F[p] (V_F V_R~^T) U_R[p]^T and <r_p, r_p> = U_R[p] (V_R~ V_R~^T) U_R[p]^T,
so the fit costs two small Gram matrices plus one pass over the pixels.

The removed term diag(c) U_R V_R~ is mapped into the functional spatial basis with a projector P,

    V_corrected = V_F - (P^T diag(c) U_R) V_R~,

so the result is again a CompressionArray in the functional channel's basis, with its mean, normalizer and trend. The
part of the blood term outside the functional basis is dropped; ``projection_relative_residual`` in the results says
how much that is.

This is the compressed-domain method of ibl_wfci_2026_jan/pipeline.py (January 2026), with centering, timestamp
alignment, optional filtering, an exact least-squares projector option and diagnostics added.
"""
from __future__ import annotations

from typing import Literal

import numpy as np
import torch
from scipy.interpolate import interp1d
from scipy.signal import butter, sosfiltfilt

from masknmf.compression import CompressionArray
from masknmf.utils import Serializer, torch_select_device, display


class HemocorrectionResults(Serializer):
    """
    What a hemocorrection fit leaves besides the corrected movie: the per-pixel coefficients and how well the removed
    reference term was represented in the functional spatial basis.
    """

    _serialized = {"coefficients", "projection_relative_residual", "projection_retained_fraction", "num_frames_fit"}

    def __init__(self,
                 coefficients: np.ndarray | torch.Tensor,
                 projection_relative_residual: float | np.ndarray | None = None,
                 projection_retained_fraction: float | np.ndarray | None = None,
                 num_frames_fit: int | np.ndarray | None = None):
        """
        Args:
            coefficients: Shape (height, width). The reference coefficient fitted at each pixel, in normalized units
                (functional fluctuation per unit of reference fluctuation, each divided by its channel's normalizer).
                0 where a pixel has no support in either basis.
            projection_relative_residual: ||blood term - its projection|| / ||blood term|| over the whole movie.
            projection_retained_fraction: ||projection||^2 / ||blood term||^2.
            num_frames_fit: Frames the coefficients were fitted on.
        """
        self.coefficients = np.asarray(coefficients.cpu() if isinstance(coefficients, torch.Tensor) else coefficients,
                                       dtype=np.float32)
        self.projection_relative_residual = None if projection_relative_residual is None else float(projection_relative_residual)
        self.projection_retained_fraction = None if projection_retained_fraction is None else float(projection_retained_fraction)
        self.num_frames_fit = None if num_frames_fit is None else int(num_frames_fit)


def interleaved_frame_positions(num_frames: int, channel: int, num_channels: int = 2) -> np.ndarray:
    """Acquisition-order position of each frame of one channel of an interleaved recording: k * num_channels + channel."""
    return np.arange(num_frames, dtype=np.float64) * num_channels + channel


def align_temporal(temporal: torch.Tensor,
                   source_times: np.ndarray,
                   target_times: np.ndarray) -> torch.Tensor:
    """
    Linearly interpolate temporal factors sampled at source_times onto target_times (extrapolating at the ends), as
    the reference channel of an interleaved recording is sampled between the functional frames.

    Args:
        temporal (torch.Tensor): Shape (rank, len(source_times)).
        source_times (np.ndarray): Increasing sample times (seconds, or acquisition-order frame positions).
        target_times (np.ndarray): Times to interpolate onto, in the same units.
    Returns:
        torch.Tensor: Shape (rank, len(target_times)), same dtype and device as temporal.
    """
    source_times = np.asarray(source_times, dtype=np.float64)
    target_times = np.asarray(target_times, dtype=np.float64)
    if temporal.shape[1] != source_times.size:
        raise ValueError(f"temporal has {temporal.shape[1]} frames but source_times has {source_times.size}")
    if np.any(np.diff(source_times) <= 0):
        raise ValueError("source_times must be strictly increasing")
    values = temporal.detach().cpu().numpy()
    aligned = interp1d(source_times, values, axis=1, kind="linear", fill_value="extrapolate",
                       assume_sorted=True)(target_times)
    return torch.from_numpy(np.ascontiguousarray(aligned)).to(dtype=temporal.dtype, device=temporal.device)


def _filter_temporal(temporal: torch.Tensor,
                     frame_rate: float,
                     highpass_hz: float | None = None,
                     lowpass_hz: float | None = None,
                     order: int = 2) -> torch.Tensor:
    """
    Zero-phase Butterworth filtering of each row of temporal. Filtering is linear and acts on time only, so filtering
    the temporal factors filters every pixel of the reconstructed movie the same way. A cutoff at or above Nyquist is
    skipped, as wfield does.
    """
    nyquist = frame_rate / 2
    out = temporal.detach().cpu().numpy().astype(np.float64)
    if highpass_hz is not None and 0 < highpass_hz < nyquist:
        out = sosfiltfilt(butter(order, highpass_hz / nyquist, btype="highpass", output="sos"), out, axis=1)
    elif highpass_hz is not None:
        display(f"highpass at {highpass_hz} Hz skipped: Nyquist is {nyquist:.3g} Hz")
    if lowpass_hz is not None and 0 < lowpass_hz < nyquist:
        out = sosfiltfilt(butter(order, lowpass_hz / nyquist, btype="lowpass", output="sos"), out, axis=1)
    elif lowpass_hz is not None:
        display(f"lowpass at {lowpass_hz} Hz skipped: Nyquist is {nyquist:.3g} Hz")
    return torch.from_numpy(np.ascontiguousarray(out)).to(dtype=temporal.dtype, device=temporal.device)


class _RowSlicer:
    """Cheap row blocks of a sparse COO matrix: coalesce once, then each block is a slice of its index arrays."""

    def __init__(self, matrix: torch.Tensor):
        matrix = matrix.coalesce()
        self.indices = matrix.indices()
        self.values = matrix.values()
        self.num_rows, self.num_cols = matrix.shape
        rows = self.indices[0].contiguous()
        self.row_ptr = torch.searchsorted(rows, torch.arange(self.num_rows + 1, device=rows.device))

    def block(self, start: int, stop: int) -> torch.Tensor:
        lo, hi = int(self.row_ptr[start]), int(self.row_ptr[stop])
        idx = self.indices[:, lo:hi].clone()
        idx[0] -= start
        return torch.sparse_coo_tensor(idx, self.values[lo:hi], (stop - start, self.num_cols)).coalesce()


def pixelwise_coefficients(spatial_functional: torch.Tensor,
                           temporal_functional: torch.Tensor,
                           spatial_reference: torch.Tensor,
                           temporal_reference: torch.Tensor,
                           center: bool = True,
                           eps: float = 1e-6,
                           pixel_batch_size: int = 2000) -> torch.Tensor:
    """
    Least-squares coefficient of the reference fluctuation in the functional fluctuation at every pixel, computed
    from the factors without reconstructing any frame.

    Args:
        spatial_functional (torch.Tensor): Sparse COO, shape (pixels, rank_F).
        temporal_functional (torch.Tensor): Shape (rank_F, frames).
        spatial_reference (torch.Tensor): Sparse COO, shape (pixels, rank_R).
        temporal_reference (torch.Tensor): Shape (rank_R, frames), already aligned to the functional frames.
        center (bool): Subtract each pixel's temporal mean before fitting (a regression with an intercept).
        eps (float): Added to the denominator.
        pixel_batch_size (int): Pixels handled at once.
    Returns:
        torch.Tensor: Shape (pixels,).
    """
    if temporal_functional.shape[1] != temporal_reference.shape[1]:
        raise ValueError(f"the channels have {temporal_functional.shape[1]} and {temporal_reference.shape[1]} frames; "
                         "align the reference to the functional frames first")
    num_frames = temporal_functional.shape[1]
    vf = temporal_functional.double()
    vr = temporal_reference.double()
    cross = vf @ vr.T
    gram = vr @ vr.T
    if center:
        mf, mr = vf.mean(dim=1), vr.mean(dim=1)
        cross -= num_frames * torch.outer(mf, mr)
        gram -= num_frames * torch.outer(mr, mr)
    cross, gram = cross.float(), gram.float()

    functional_rows = _RowSlicer(spatial_functional)
    reference_rows = _RowSlicer(spatial_reference)
    num_pixels = functional_rows.num_rows
    coefficients = torch.zeros(num_pixels, dtype=torch.float32, device=cross.device)
    for start in range(0, num_pixels, pixel_batch_size):
        stop = min(start + pixel_batch_size, num_pixels)
        uf = functional_rows.block(start, stop)
        ur = reference_rows.block(start, stop).to_dense()
        numerator = (torch.sparse.mm(uf, cross) * ur).sum(dim=1)
        denominator = ((ur @ gram) * ur).sum(dim=1)
        coefficients[start:stop] = torch.where(denominator > 0, numerator / (denominator + eps),
                                               torch.zeros_like(numerator))
    return coefficients


def _scale_rows(matrix: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """diag(scale) @ matrix for a sparse COO matrix, keeping it sparse."""
    matrix = matrix.coalesce()
    idx = matrix.indices()
    return torch.sparse_coo_tensor(idx, matrix.values() * scale[idx[0]], matrix.shape).coalesce()


def reference_term_in_functional_basis(spatial_functional: torch.Tensor,
                                       spatial_reference: torch.Tensor,
                                       coefficients: torch.Tensor,
                                       local_projector: torch.Tensor | None = None,
                                       projection: Literal["local", "least_squares"] = "local") -> torch.Tensor:
    """
    The matrix M (rank_F, rank_R) with U_F M ~= diag(coefficients) U_R, so that M @ V_R~ is the removed reference
    term written in the functional basis.

    projection "local" uses the functional CompressionArray's local projector (as the January 2026 pipeline did);
    "least_squares" solves (U_F^T U_F) M = U_F^T diag(c) U_R exactly.
    """
    scaled_reference = _scale_rows(spatial_reference, coefficients)
    if projection == "local":
        if local_projector is None:
            raise ValueError('projection "local" needs the functional array\'s spatial_compressed_local_projector; '
                             'use projection="least_squares"')
        return torch.sparse.mm(local_projector.coalesce().T, scaled_reference).to_dense()
    if projection == "least_squares":
        uf = spatial_functional.coalesce()
        gram = torch.sparse.mm(uf.T, uf).to_dense().double()
        rhs = torch.sparse.mm(uf.T, scaled_reference).to_dense().double()
        ridge = 1e-8 * torch.diagonal(gram).mean().clamp_min(1e-12)
        gram += ridge * torch.eye(gram.shape[0], dtype=gram.dtype, device=gram.device)
        return torch.linalg.solve(gram, rhs).float()
    raise ValueError(f'projection must be "local" or "least_squares", got {projection!r}')


def _projection_diagnostics(spatial_functional: torch.Tensor,
                            scaled_reference: torch.Tensor,
                            mixing: torch.Tensor,
                            temporal_reference: torch.Tensor) -> tuple[float, float]:
    """Relative residual and retained energy of projecting diag(c) U_R V_R~ onto the functional basis, from Grams."""
    vr = temporal_reference.double()
    gram_t = vr @ vr.T
    uf = spatial_functional.coalesce()
    ref_gram = torch.sparse.mm(scaled_reference.T, scaled_reference).to_dense().double()
    func_gram = torch.sparse.mm(uf.T, uf).to_dense().double()
    cross = torch.sparse.mm(uf.T, scaled_reference).to_dense().double()
    m = mixing.double()
    energy_blood = torch.sum(ref_gram * gram_t)
    energy_projection = torch.sum((m.T @ func_gram @ m) * gram_t)
    energy_cross = torch.sum((m.T @ cross) * gram_t)
    if energy_blood <= 0:
        return 0.0, 1.0
    residual = (energy_blood - 2 * energy_cross + energy_projection).clamp_min(0)
    return float(torch.sqrt(residual / energy_blood)), float(energy_projection / energy_blood)


def hemocorrect(functional: CompressionArray,
                reference: CompressionArray,
                functional_frame_times: np.ndarray | None = None,
                reference_frame_times: np.ndarray | None = None,
                frame_rate: float | None = None,
                center: bool = True,
                projection: Literal["local", "least_squares"] = "local",
                fit_highpass_hz: float | None = None,
                reference_lowpass_hz: float | None = None,
                filter_order: int = 2,
                eps: float = 1e-6,
                pixel_batch_size: int = 2000,
                device: Literal["auto", "cuda", "cpu"] | str = "auto") -> tuple[CompressionArray, HemocorrectionResults]:
    """
    Remove the reference (hemodynamic) channel from the functional channel, pixel by pixel, on their compressed
    representations. See the module docstring for the model.

    Args:
        functional (CompressionArray): The functional channel.
        reference (CompressionArray): The reference channel, same field of view.
        functional_frame_times, reference_frame_times (np.ndarray | None): Sample time of each frame of each channel
            (seconds, or acquisition-order positions; see interleaved_frame_positions). Both None means the channels
            are already frame-aligned and must have the same number of frames.
        frame_rate (float | None): Functional channel frame rate in Hz; needed only for the filters.
        center (bool): Fit with an intercept (subtract each pixel's mean first).
        projection ("local" | "least_squares"): How the removed term is mapped into the functional basis.
        fit_highpass_hz (float | None): Highpass both channels before fitting the coefficients only (the slow drift
            does not drive the fit; the output keeps its slow components). None fits on the unfiltered factors.
        reference_lowpass_hz (float | None): Lowpass the reference before both fitting and subtracting it, so less
            reference noise is subtracted into the functional channel. None leaves it unfiltered.
        filter_order (int): Butterworth order of both filters.
        eps (float): Added to each coefficient's denominator.
        pixel_batch_size (int): Pixels handled at once while fitting.
        device (str): Where the computation runs.
    Returns:
        CompressionArray: The corrected functional channel, in the functional spatial basis, with its mean image,
            normalizer, local projector and trend.
        HemocorrectionResults: Coefficient image and projection diagnostics.
    """
    if tuple(functional.shape[1:]) != tuple(reference.shape[1:]):
        raise ValueError(f"field of view differs: {functional.shape[1:]} vs {reference.shape[1:]}")
    device = torch_select_device(device)
    num_frames = functional.shape[0]

    vf = functional.temporal_compressed.to(device).float()
    vr = reference.temporal_compressed.to(device).float()
    if functional_frame_times is None and reference_frame_times is None:
        if reference.shape[0] != num_frames:
            raise ValueError(f"the channels have {num_frames} and {reference.shape[0]} frames; give frame times")
    elif functional_frame_times is None or reference_frame_times is None:
        raise ValueError("give both functional_frame_times and reference_frame_times, or neither")
    else:
        vr = align_temporal(vr, reference_frame_times, functional_frame_times)

    if (fit_highpass_hz is not None or reference_lowpass_hz is not None) and frame_rate is None:
        raise ValueError("frame_rate is needed to filter")
    if reference_lowpass_hz is not None:
        vr = _filter_temporal(vr, frame_rate, lowpass_hz=reference_lowpass_hz, order=filter_order)
    if fit_highpass_hz is not None:
        vf_fit = _filter_temporal(vf, frame_rate, highpass_hz=fit_highpass_hz, order=filter_order)
        vr_fit = _filter_temporal(vr, frame_rate, highpass_hz=fit_highpass_hz, order=filter_order)
    else:
        vf_fit, vr_fit = vf, vr

    uf = functional.spatial_compressed.to(device).float()
    ur = reference.spatial_compressed.to(device).float()
    projector = functional.spatial_compressed_local_projector
    projector = None if projector is None else projector.to(device).float()

    coefficients = pixelwise_coefficients(uf, vf_fit, ur, vr_fit, center=center, eps=eps,
                                          pixel_batch_size=pixel_batch_size)
    mixing = reference_term_in_functional_basis(uf, ur, coefficients, local_projector=projector,
                                                projection=projection)
    corrected_temporal = vf - mixing @ vr
    residual, retained = _projection_diagnostics(uf, _scale_rows(ur, coefficients), mixing, vr)
    display(f"hemocorrection: projection keeps {retained:.3f} of the removed term's energy, "
            f"relative residual {residual:.3f}")

    corrected = CompressionArray.from_tensors(
        shape=functional.shape,
        spatial_compressed=functional.spatial_compressed.cpu(),
        temporal_compressed=corrected_temporal.cpu(),
        mean_image=functional.mean_image.cpu(),
        noise_variance_image=functional.noise_variance_image.cpu(),
        spatial_compressed_local_projector=None if functional.spatial_compressed_local_projector is None
        else functional.spatial_compressed_local_projector.cpu(),
        spatial_trend_basis=None if functional.spatial_trend_basis is None else functional.spatial_trend_basis.cpu(),
        temporal_trend_basis=None if functional.temporal_trend_basis is None else functional.temporal_trend_basis.cpu(),
        device="cpu",
        rescale=functional.rescale,
        include_trend=functional.include_trend)
    results = HemocorrectionResults(coefficients=coefficients.reshape(functional.shape[1], functional.shape[2]),
                                    projection_relative_residual=residual,
                                    projection_retained_fraction=retained,
                                    num_frames_fit=num_frames)
    return corrected, results
