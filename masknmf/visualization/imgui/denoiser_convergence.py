# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "torch", "imgui-bundle>=1.92.801", "tifffile"]
# ///
"""
Convergence test for the temporal denoiser, as one imgui dialog.

Loads the first frames of a movie into RAM, reduces them to temporal components with a truncated SVD
(a stand-in for PMD), trains the blind-spot denoiser on those components and redraws the denoised trace
of the selected pixels after every epoch. The network, loss, windowing and fusion are copied from
masknmf/compression/denoising.py, so this file needs only numpy, torch and imgui_bundle.

    python denoiser_convergence.py movie.npy
    uv run denoiser_convergence.py            (a local path or a URL; no movie opens a simulated one)
"""

from typing import Any, List, Optional

import dataclasses
import functools
import math
import sys
import threading

import numpy as np
import torch
from imgui_bundle import imgui, immapp, implot

RADIUS_RECEPTIVE = 11  ## samples between a network output and the farthest input it reads
PADDING = 100
SIZE_WINDOW = 900
QUANTILE_NOISE = 0.3  ## CompressDenoiseStrategy's default noise_variance_quantile
CHANNELS = [1, 16, 32, 48, 64, 80, 96]


class BlindSpotTemporal(torch.nn.Module):
    """Dilated 1d convolutions whose output at t never reads the input at t (BlindSpotTemporal in masknmf)."""

    def __init__(self, positive_output: bool):
        super().__init__()
        self.convs_regular = torch.nn.ModuleList(
            [torch.nn.Conv1d(CHANNELS[i], CHANNELS[i + 1], 3, padding="same") for i in range(5)]
        )
        self.convs_blindSpot = torch.nn.ModuleList(
            [torch.nn.Conv1d(CHANNELS[i], CHANNELS[i + 1], 3, dilation=i + 1, padding="same") for i in range(6)]
        )
        self.final = torch.nn.Conv1d(sum(CHANNELS[1:]), 1, 1)
        self.positive_output = positive_output
        self.register_buffer("mask_center", torch.tensor([1.0, 0.0, 1.0]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = []
        for i, conv in enumerate(self.convs_blindSpot):
            out = torch.nn.functional.conv1d(
                x, conv.weight * self.mask_center, conv.bias, dilation=conv.dilation, padding="same"
            )
            features.append(torch.nn.functional.leaky_relu(out, 0.1))
            if i < len(self.convs_regular):
                x = torch.nn.functional.leaky_relu(self.convs_regular[i](x), 0.1)
        out = self.final(torch.cat(features, dim=1))
        return torch.nn.functional.softplus(out) if self.positive_output else out


class DenoiserNetwork(torch.nn.Module):
    """Predicts the mean and the total variance of each sample from its neighbors (TemporalNetwork in masknmf)."""

    def __init__(self):
        super().__init__()
        self.mean_backbone = BlindSpotTemporal(positive_output=False)
        self.var_backbone = BlindSpotTemporal(positive_output=True)

    def forward(self, x: torch.Tensor):
        return self.mean_backbone(x), self.var_backbone(x)


def pad_blind_spot_safe(traces: torch.Tensor) -> torch.Tensor:
    """Mirror-pads the last axis by PADDING per side, with the mirror offset so no output sees its own input."""
    n_frames = traces.shape[-1]
    if n_frames < PADDING + RADIUS_RECEPTIVE + 1:
        raise ValueError(f"need at least {PADDING + RADIUS_RECEPTIVE + 1} frames, got {n_frames}")
    left = traces[..., RADIUS_RECEPTIVE + 1 : RADIUS_RECEPTIVE + 1 + PADDING].flip(-1)
    right = traces[..., n_frames - RADIUS_RECEPTIVE - 1 - PADDING : n_frames - RADIUS_RECEPTIVE - 1].flip(-1)
    return torch.cat([left, traces, right], dim=-1)


def build_windows_training(traces: torch.Tensor) -> torch.Tensor:
    """
    Cuts padded, unit-norm traces into the windows the trainer draws batches from.

    Parameters
    ----------
    traces : torch.Tensor
        Temporal components. Shape (n_traces, n_frames).

    Returns
    -------
    torch.Tensor
        Shape (n_traces * n_windows, 1, size_window).
    """
    padded = pad_blind_spot_safe(traces=traces)
    padded = padded[padded.std(dim=1) > 1e-6]
    padded = padded - padded.mean(dim=1, keepdim=True)
    padded = padded / torch.linalg.norm(padded, dim=1, keepdim=True)
    size = min(SIZE_WINDOW, padded.shape[1])
    starts = list(range(0, padded.shape[1] - size + 1, size - 2 * RADIUS_RECEPTIVE))
    if starts[-1] + size < padded.shape[1]:
        starts.append(padded.shape[1] - size)
    return torch.stack([padded[:, s : s + size] for s in starts], dim=1).reshape(-1, 1, size)


def loss_total_variance(model: DenoiserNetwork, batch: torch.Tensor) -> torch.Tensor:
    """Gaussian negative log-likelihood over the samples whose receptive field lies inside the window."""
    r = RADIUS_RECEPTIVE
    mu_x, total_variance = model(batch)
    x, mu_x, total_variance = batch[..., r:-r], mu_x[..., r:-r], total_variance[..., r:-r]
    total_variance = torch.clamp(total_variance, min=1e-8)
    log_lik = torch.nansum(torch.log(total_variance)) + torch.nansum((x - mu_x) ** 2 / total_variance)
    return log_lik / x.numel()


def denoise_traces(model: DenoiserNetwork, traces: torch.Tensor) -> torch.Tensor:
    """
    Fuses the network's predictions with the observed traces (CompressionTemporalDenoiser in masknmf).

    Parameters
    ----------
    model : DenoiserNetwork
        The network, in eval mode.
    traces : torch.Tensor
        Shape (n_traces, n_frames).

    Returns
    -------
    torch.Tensor
        Denoised traces. Shape (n_traces, n_frames).
    """
    padded = pad_blind_spot_safe(traces=traces)
    means = padded.mean(dim=1, keepdim=True)
    norms = torch.linalg.norm(padded - means, dim=1, keepdim=True)
    norms[norms == 0] = 1
    normalized = (padded - means) / norms
    denoised = []
    with torch.no_grad():
        for chunk in normalized.split(max(1, 100_000 // padded.shape[1])):
            mean_network, variance_total = (out[:, 0] for out in model(chunk[:, None]))
            variance_noise = torch.quantile(variance_total, QUANTILE_NOISE, dim=1, keepdim=True)
            variance_total = torch.clamp(variance_total, min=variance_noise)
            denoised.append((variance_noise * mean_network + (variance_total - variance_noise) * chunk) / variance_total)
    return (torch.cat(denoised) * norms + means)[:, PADDING:-PADDING]


@dataclasses.dataclass
class ConvergenceTest:
    """Settings and live results of one convergence test. ``movie`` is any (n_frames, height, width) array that slices lazily."""

    movie: Any
    n_frames_load: int = 2000
    rank: int = 64
    max_epochs: int = 100
    batch_size: int = 128
    learning_rate: float = 1e-4
    tolerance_percent: float = 10.0
    seed: int = 0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    roi: Optional[List[float]] = None  ## x_min, y_min, x_max, y_max in pixels
    epoch_shown: int = 0
    n_epochs_drawn: int = 0
    status: str = "idle"
    stop_requested: bool = False
    thread: Optional[threading.Thread] = None
    lock: threading.Lock = dataclasses.field(default_factory=threading.Lock)
    image_mean: Optional[np.ndarray] = None  ## (height, width)
    movie_loaded: Optional[np.ndarray] = None  ## (n_frames, height, width), mean image subtracted
    spatial: Optional[np.ndarray] = None  ## (height, width, rank)
    steps_per_epoch: int = 0
    snapshots: List[np.ndarray] = dataclasses.field(default_factory=list)  ## per epoch, (rank, n_frames)
    key_cache: Any = None
    trace_raw: Optional[np.ndarray] = None  ## (n_frames,)
    traces_epochs: Optional[np.ndarray] = None  ## (n_epochs, n_frames)


def run_training(test: ConvergenceTest) -> None:
    """Loads the frames, takes the SVD and trains, appending the denoised components to ``test.snapshots`` each epoch."""
    torch.manual_seed(test.seed)
    test.status = "loading frames"
    ## NOTE: np.array copies, so subtracting the mean below never writes into the caller's movie
    movie = torch.as_tensor(np.array(test.movie[: test.n_frames_load], dtype=np.float32))
    image_mean = movie.mean(dim=0)
    movie -= image_mean
    test.status = "SVD"
    ## (n_frames, n_pixels) = u @ diag(s) @ v.T
    u, s, v = torch.svd_lowrank(movie.reshape(movie.shape[0], -1).to(test.device), q=test.rank, niter=2)
    temporal = (u * s).T  ## (rank, n_frames)
    windows = build_windows_training(traces=temporal)
    model = DenoiserNetwork().to(test.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=test.learning_rate)
    with test.lock:
        test.snapshots, test.epoch_shown, test.n_epochs_drawn = [], 0, 0
        test.image_mean, test.movie_loaded = image_mean.numpy(), movie.numpy()
        test.spatial = v.reshape(*movie.shape[1:], -1).cpu().numpy()
        test.steps_per_epoch = math.ceil(len(windows) / test.batch_size)
    for epoch in range(test.max_epochs):
        test.status = f"training epoch {epoch + 1} of {test.max_epochs}"
        model.train()
        for indices in torch.randperm(len(windows), device=test.device).split(test.batch_size):
            if test.stop_requested:
                test.status = f"stopped after {epoch} epochs"
                return
            optimizer.zero_grad()
            loss_total_variance(model=model, batch=windows[indices]).backward()
            optimizer.step()
        model.eval()
        test.snapshots.append(denoise_traces(model=model, traces=temporal).cpu().numpy())
    test.status = f"finished {test.max_epochs} epochs"


def start_training(test: ConvergenceTest) -> None:
    """Starts ``run_training`` on a background thread."""
    test.stop_requested = False
    test.thread = threading.Thread(target=run_training, kwargs={"test": test}, daemon=True)
    test.thread.start()


def update_traces_roi(test: ConvergenceTest) -> None:
    """Recomputes the ROI's raw trace and its denoised trace at every epoch when the ROI or the epoch count changed."""
    x_min, y_min, x_max, y_max = (int(value) for value in test.roi)
    ## NOTE: the lock keeps a restart on the training thread from swapping the arrays mid-way
    with test.lock:
        key = (x_min, y_min, x_max, y_max, len(test.snapshots), id(test.movie_loaded))
        if key == test.key_cache or len(test.snapshots) == 0:
            return
        test.key_cache = key
        offset = test.image_mean[y_min:y_max, x_min:x_max].mean()
        test.trace_raw = test.movie_loaded[:, y_min:y_max, x_min:x_max].mean(axis=(1, 2)) + offset
        weights = test.spatial[y_min:y_max, x_min:x_max].mean(axis=(0, 1))  ## (rank,)
        test.traces_epochs = np.stack([weights @ snapshot for snapshot in test.snapshots]) + offset


def epochs_to_convergence(test: ConvergenceTest):
    """
    Finds the first epoch after which the ROI trace stays within tolerance of the latest epoch's trace.

    Returns
    -------
    distances : np.ndarray
        RMS difference of each epoch's trace from the latest, in percent of the raw trace's SD. Shape (n_epochs,).
    epoch_converged : int or None
        ``None`` unless training ran at least twice that long, so a trace that is still drifting does not count.
    """
    difference = test.traces_epochs - test.traces_epochs[-1]
    distances = 100 * np.sqrt((difference**2).mean(axis=1)) / test.trace_raw.std()
    epochs_outside = np.flatnonzero(distances > test.tolerance_percent)
    epoch_converged = int(epochs_outside[-1]) + 2 if len(epochs_outside) > 0 else 1
    return distances, epoch_converged if epoch_converged <= 0.5 * len(distances) else None


def line_spec(color: tuple, weight: float = 1.0) -> implot.Spec:
    """An implot line style of this rgba color and weight."""
    spec = implot.Spec()
    spec.line_color, spec.line_weight = imgui.ImVec4(*color), weight
    return spec


def draw_help(text: str) -> None:
    """Draws a (?) after the last widget that shows ``text`` while hovered."""
    imgui.same_line()
    imgui.text_disabled("(?)")
    if imgui.is_item_hovered(imgui.HoveredFlags_.for_tooltip | imgui.HoveredFlags_.allow_when_disabled):
        imgui.set_tooltip(text)


def draw_image_and_settings(test: ConvergenceTest, running: bool) -> None:
    """Draws the mean image with the draggable ROI, and the load and trainer settings beside it."""
    em = imgui.get_font_size()
    height, width = test.image_mean.shape
    if test.roi is None:
        test.roi = [width * 0.4, height * 0.4, width * 0.6, height * 0.6]
    step = math.ceil(max(height, width) / 128)  ## implot draws one quad per heatmap cell
    image = np.ascontiguousarray(test.image_mean[::step, ::step])
    flags_axis = implot.AxisFlags_.no_decorations
    implot.push_colormap(implot.Colormap_.greys)
    if implot.begin_plot("##image", (15 * em * width / max(height, width), 15 * em * height / max(height, width)), implot.Flags_.canvas_only):
        implot.setup_axes("", "", flags_axis | implot.AxisFlags_.lock, flags_axis | implot.AxisFlags_.lock | implot.AxisFlags_.invert)
        implot.setup_axes_limits(0, width, 0, height, implot.Cond_.always)
        low, high = np.percentile(image, [1, 99.5])
        implot.plot_heatmap("##mean", image, high, low, "", implot.Point(0, height), implot.Point(width, 0))
        changed, *corners = implot.drag_rect(0, *test.roi, imgui.ImVec4(1.0, 0.8, 0.2, 1.0))[:5]
        x_min, x_max = sorted(np.clip([corners[0], corners[2]], 0, width - 1))
        y_min, y_max = sorted(np.clip([corners[1], corners[3]], 0, height - 1))
        test.roi = [x_min, y_min, max(x_max, x_min + 1), max(y_max, y_min + 1)]
        implot.end_plot()
    implot.pop_colormap()

    imgui.same_line()
    imgui.begin_group()
    imgui.push_item_width(9 * em)
    imgui.begin_disabled(running)
    gigabytes = test.n_frames_load * height * width * 4 / 1e9
    _, test.n_frames_load = imgui.slider_int(f"frames in RAM ({gigabytes:.2f} GB)", test.n_frames_load, 2 * PADDING, len(test.movie))
    draw_help(text="Frames read from the start of the movie. Each component is cut into 900-frame windows,\nso more frames give more windows and more steps per epoch.")
    _, test.rank = imgui.slider_int("SVD rank", test.rank, 8, 256)
    draw_help(text="Number of temporal components the network trains on, standing in for the rank PMD finds.\nMore components give more windows and more steps per epoch.")
    _, test.max_epochs = imgui.input_int("max epochs", test.max_epochs, 10)
    draw_help(text="Passes over all training windows. Steps = epochs x steps per epoch.\nThis is num_epochs on CompressDenoiseStrategy (default 10).")
    _, test.batch_size = imgui.input_int("batch size", test.batch_size, 16)
    draw_help(text="Windows per optimizer step. Steps per epoch = windows / batch size, rounded up,\nso decreasing it gives more steps per epoch. CompressDenoiseStrategy fixes it at 128.")
    _, test.learning_rate = imgui.input_float("learning rate", test.learning_rate, 0, 0, "%.0e")
    draw_help(text="Adam step size. CompressDenoiseStrategy fixes it at 1e-4.")
    test.max_epochs, test.batch_size = max(test.max_epochs, 1), max(test.batch_size, 1)
    imgui.end_disabled()
    _, test.tolerance_percent = imgui.slider_float("tolerance (% of raw SD)", test.tolerance_percent, 0.5, 20, "%.1f")
    draw_help(text="An epoch counts as converged when its trace and every later one differ from the latest\ntrace by less than this. Increasing it reports an earlier epoch.")
    imgui.pop_item_width()
    if running:
        if imgui.button("Stop"):
            test.stop_requested = True
    elif imgui.button("Start"):
        start_training(test=test)
    imgui.same_line()
    imgui.text_disabled(test.status)
    imgui.text_disabled(f"ROI x {int(test.roi[0])}-{int(test.roi[2])}, y {int(test.roi[1])}-{int(test.roi[3])} px; drag it on the image")
    imgui.text_disabled(f"train_total_variance_denoiser(components, max_epochs={test.max_epochs}, batch_size={test.batch_size}, learning_rate={test.learning_rate:.0e})")
    imgui.end_group()


def draw_results(test: ConvergenceTest) -> None:
    """Draws the epoch count, the ROI trace at the shown epoch over the raw trace, and each epoch's distance to the latest."""
    em = imgui.get_font_size()
    update_traces_roi(test=test)
    n_epochs = len(test.traces_epochs)
    distances, epoch_converged = epochs_to_convergence(test=test)
    if epoch_converged is None:
        imgui.text_colored(imgui.ImVec4(1.0, 0.6, 0.3, 1.0), f"Not converged after {n_epochs} epochs ({n_epochs * test.steps_per_epoch} steps)")
    else:
        imgui.text_colored(imgui.ImVec4(0.4, 0.9, 0.5, 1.0), f"Converged at epoch {epoch_converged} ({epoch_converged * test.steps_per_epoch} steps)")
    imgui.same_line()
    imgui.text_disabled(f"{test.steps_per_epoch} steps per epoch")

    follow_latest = test.epoch_shown >= test.n_epochs_drawn
    test.n_epochs_drawn = n_epochs
    imgui.set_next_item_width(-8 * em)
    _, test.epoch_shown = imgui.slider_int("epoch shown", n_epochs if follow_latest else test.epoch_shown, 1, n_epochs)
    frames = np.arange(len(test.trace_raw), dtype=np.float64)
    if implot.begin_plot("##traces", (-1, 14 * em)):
        implot.setup_axes("frame", "", 0, implot.AxisFlags_.auto_fit | implot.AxisFlags_.range_fit)
        implot.setup_axes_limits(0, len(frames), 0, 1)
        implot.plot_line("raw", frames, test.trace_raw.astype(np.float64), line_spec(color=(0.6, 0.6, 0.6, 0.6)))
        if test.epoch_shown < n_epochs:
            implot.plot_line(f"epoch {n_epochs} (latest)", frames, test.traces_epochs[-1].astype(np.float64), line_spec(color=(0.3, 0.6, 1.0, 1.0), weight=1.5))
        implot.plot_line("epoch shown", frames, test.traces_epochs[test.epoch_shown - 1].astype(np.float64), line_spec(color=(1.0, 0.35, 0.3, 1.0), weight=1.5))
        implot.end_plot()
    if n_epochs > 1 and implot.begin_plot("##distance", (-1, 9 * em), implot.Flags_.no_legend):
        implot.setup_axes("epoch", "% of raw SD", implot.AxisFlags_.auto_fit, implot.AxisFlags_.auto_fit)
        implot.setup_axis_scale(implot.ImAxis_.y1, implot.Scale_.log10)
        ## the latest epoch's distance to itself is 0, which a log axis cannot show
        implot.plot_line("distance", np.arange(1, n_epochs, dtype=np.float64), distances[:-1].astype(np.float64), line_spec(color=(1.0, 0.35, 0.3, 1.0), weight=1.5))
        spec_tolerance = line_spec(color=(0.6, 0.6, 0.6, 0.8))
        spec_tolerance.flags = implot.InfLinesFlags_.horizontal
        implot.plot_inf_lines("tolerance", np.array([test.tolerance_percent]), spec_tolerance)
        if epoch_converged is not None:
            implot.plot_inf_lines("converged", np.array([float(epoch_converged)]), line_spec(color=(0.4, 0.9, 0.5, 0.8)))
        implot.end_plot()


def draw(test: ConvergenceTest) -> None:
    """Draws the dialog's contents into the current imgui window; needs an implot context."""
    if test.image_mean is None:
        test.image_mean = np.asarray(test.movie[:100], dtype=np.float32).mean(axis=0)
        test.n_frames_load = min(test.n_frames_load, len(test.movie))
    draw_image_and_settings(test=test, running=test.thread is not None and test.thread.is_alive())
    if len(test.snapshots) > 0:
        draw_results(test=test)


def run(movie: Any) -> ConvergenceTest:
    """Opens the dialog in its own window on ``movie`` (n_frames, height, width) and returns the test when it closes."""
    test = ConvergenceTest(movie=movie)
    immapp.run(gui_function=functools.partial(draw, test=test), window_title="Denoiser convergence", window_size=(900, 640), with_implot=True, fps_idle=30)
    test.stop_requested = True
    return test


def simulate_movie(n_frames: int = 4000, height: int = 96, width: int = 96, n_cells: int = 30, seed: int = 0) -> np.ndarray:
    """Simulates Gaussian cells with exponentially decaying calcium transients under shot noise. Shape (n_frames, height, width)."""
    rng = np.random.default_rng(seed=seed)
    centers = rng.uniform(low=8, high=min(height, width) - 8, size=(n_cells, 2))
    yy, xx = np.mgrid[:height, :width]
    footprints = np.exp(-((yy[None] - centers[:, 0, None, None]) ** 2 + (xx[None] - centers[:, 1, None, None]) ** 2) / (2 * 3.0**2))
    spikes = rng.random(size=(n_cells, n_frames)) < 0.005
    kernel_decay = np.exp(-np.arange(200) / 30.0)
    calcium = np.stack([np.convolve(row, kernel_decay)[:n_frames] for row in spikes])  ## (n_cells, n_frames)
    return rng.poisson(lam=20 + 30 * np.einsum("ct,chw->thw", calcium, footprints)).astype(np.float32)


if __name__ == "__main__":
    filepath_movie = sys.argv[1] if len(sys.argv) > 1 else None
    if filepath_movie is None:
        run(movie=simulate_movie())
    elif filepath_movie.endswith(".npy"):
        run(movie=np.load(filepath_movie, mmap_mode="r"))
    else:
        import tifffile

        run(movie=tifffile.memmap(filepath_movie))
