import numpy as np
import torch
import torch.nn as nn

import torch
import pytorch_lightning as pl
import torch.nn as nn
import networkx as nx
import numpy as np
from torch.utils.data import DataLoader
import os
import sys
from masknmf.utils import display
from pytorch_lightning.callbacks import TQDMProgressBar


class MaskedConv1d(nn.Conv1d):
    def __init__(self, *args, **kwargs):
        super(MaskedConv1d, self).__init__(*args, **kwargs)
        # Create a mask with the center element zeroed out
        self.mask = nn.Parameter(torch.ones_like(self.weight), requires_grad=False)
        center = self.weight.shape[-1] // 2
        self.mask[:, :, center] = 0

    def forward(self, x):
        # Apply the mask to the weights
        masked_weight = self.weight * self.mask
        return nn.functional.conv1d(
            x,
            masked_weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )


class ConvBlock1d(nn.Module):
    def __init__(
            self, in_channels, out_channels, kernel_size, dilation, use_mask=False
    ):
        super(ConvBlock1d, self).__init__()
        if use_mask:
            self.conv = MaskedConv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                dilation=dilation,
                padding="same",
            )
        else:
            self.conv = nn.Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                dilation=dilation,
                padding="same",
            )
        self.activation = nn.LeakyReLU(0.1)

    def forward(self, x):
        return self.activation(self.conv(x))


class BlindSpotTemporal(nn.Module):
    def __init__(self, out_channels=1, final_activation=None):
        super(BlindSpotTemporal, self).__init__()
        self.out_channels = out_channels
        self.reg_conv1 = ConvBlock1d(
            in_channels=1, out_channels=16, kernel_size=3, dilation=1, use_mask=False
        )
        self.reg_conv2 = ConvBlock1d(
            in_channels=16, out_channels=32, kernel_size=3, dilation=1, use_mask=False
        )
        self.reg_conv3 = ConvBlock1d(
            in_channels=32, out_channels=48, kernel_size=3, dilation=1, use_mask=False
        )
        self.reg_conv4 = ConvBlock1d(
            in_channels=48, out_channels=64, kernel_size=3, dilation=1, use_mask=False
        )
        self.reg_conv5 = ConvBlock1d(
            in_channels=64, out_channels=80, kernel_size=3, dilation=1, use_mask=False
        )

        self.bsconv1 = ConvBlock1d(
            in_channels=1, out_channels=16, kernel_size=3, dilation=1, use_mask=True
        )
        self.bsconv2 = ConvBlock1d(
            in_channels=16, out_channels=32, kernel_size=3, dilation=2, use_mask=True
        )
        self.bsconv3 = ConvBlock1d(
            in_channels=32, out_channels=48, kernel_size=3, dilation=3, use_mask=True
        )
        self.bsconv4 = ConvBlock1d(
            in_channels=48, out_channels=64, kernel_size=3, dilation=4, use_mask=True
        )
        self.bsconv5 = ConvBlock1d(
            in_channels=64, out_channels=80, kernel_size=3, dilation=5, use_mask=True
        )
        self.bsconv6 = ConvBlock1d(
            in_channels=80, out_channels=96, kernel_size=3, dilation=6, use_mask=True
        )

        self.final = nn.Conv1d(
            in_channels=336, out_channels=out_channels, kernel_size=1, dilation=1
        )
        if final_activation is None:
            self.final_activation = nn.Identity()
        else:
            self.final_activation = final_activation

        # Largest distance (in samples) between an output and any input it depends on.
        self.receptive_radius = self._compute_receptive_radius()

    def forward(self, x):

        # run regular convolutions
        enc1 = self.reg_conv1(x)
        enc2 = self.reg_conv2(enc1)
        enc3 = self.reg_conv3(enc2)
        enc4 = self.reg_conv4(enc3)
        enc5 = self.reg_conv5(enc4)

        # run blind spot convolutions
        bs1 = self.bsconv1(x)
        bs2 = self.bsconv2(enc1)
        bs3 = self.bsconv3(enc2)
        bs4 = self.bsconv4(enc3)
        bs5 = self.bsconv5(enc4)
        bs6 = self.bsconv6(enc5)

        out = torch.cat([bs1, bs2, bs3, bs4, bs5, bs6], dim=1)
        out = self.final_activation(self.final(out))
        return out

    def _compute_receptive_radius(self) -> int:
        """Receptive-field radius of forward(). Mirrors its wiring: bsconv1 reads x, and each
        bsconv{i} (i >= 2) reads enc{i-1}. Update this if forward() is rewired."""

        def reach(conv):
            return conv.dilation[0] * (conv.kernel_size[0] // 2)

        regular = [self.reg_conv1, self.reg_conv2, self.reg_conv3, self.reg_conv4, self.reg_conv5]
        blind_spot = [self.bsconv1, self.bsconv2, self.bsconv3, self.bsconv4, self.bsconv5, self.bsconv6]
        feature_radius = [0]  # radius of x, enc1, ..., enc5
        for block in regular:
            feature_radius.append(feature_radius[-1] + reach(block.conv))
        return max(r + reach(b.conv) for r, b in zip(feature_radius, blind_spot)) + reach(self.final)


class TemporalNetwork(nn.Module):
    def __init__(self):
        super(TemporalNetwork, self).__init__()
        self.mean_backbone = BlindSpotTemporal()
        self.var_backbone = BlindSpotTemporal(final_activation=nn.Softplus())

    def forward(self, x):
        return self.mean_backbone(x), self.var_backbone(x)

    @property
    def receptive_radius(self) -> int:
        return max(self.mean_backbone.receptive_radius, self.var_backbone.receptive_radius)


class TotalVarianceTemporalDenoiser(pl.LightningModule):
    """
    PyTorch Lightning module for training a network that predicts
    total variance (signal + noise) instead of just signal variance.
    """

    def __init__(
            self,
            learning_rate=1e-3,
            max_epochs=1,
    ):
        super(TotalVarianceTemporalDenoiser, self).__init__()

        self.temporal_network = TemporalNetwork()

        self.learning_rate = learning_rate
        self.max_epochs = max_epochs

    def training_step(self, batch, batch_idx):
        input_traces = batch
        mu_x, total_variance = self(input_traces)

        num_datapoints = input_traces.numel()

        # make sure all total variances are positive
        total_variance = torch.clamp(total_variance, min=1e-8)

        log_lik = torch.nansum(torch.log(total_variance))
        log_lik = log_lik + torch.nansum(
            (input_traces - mu_x) ** 2 / total_variance
        )
        loss = log_lik / num_datapoints
        self.log("train_loss", loss)

        return loss

    def forward(self, x):
        return self.temporal_network(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    @property
    def receptive_radius(self) -> int:
        return self.temporal_network.receptive_radius


def train_total_variance_denoiser(
        time_series,  # num_timeseries x time_series_length
        learning_rate: float = 1e-2,
        input_size: int = 900,
        overlap: int = 600,
        max_epochs: int = 20,
        batch_size: int = 1,
        devices: int = 1,
        padding: int = 100,
        precision: str = "32-true",
        log_every_n_steps: int = 100,
):
    """Train a total variance prediction network


    Args:
        precision (str): Lightning precision setting. "16-mixed" is the previous behavior. At batch size 1
            the network's tensors are too small to benefit from fp16, so "32-true" may be faster (no cast
            kernels or gradient scaler). "bf16-mixed" drops the gradient scaler on Ampere-or-newer GPUs.
            Worth timing each.
        log_every_n_steps (int): How often metrics are written to the logger and the progress bar is
            redrawn. Each logger write forces a CPU-GPU sync, so the old value of 1 serialized every step.

    Returns:
        model: the trained model.
        dataset: the training dataset, with its data moved back to CPU memory.
    """
    model = TotalVarianceTemporalDenoiser(
        learning_rate=learning_rate,
        max_epochs=max_epochs,
    )

    use_gpu_data = torch.cuda.is_available() and devices == 1
    data_device = "cuda" if use_gpu_data else "cpu"

    padded_timeseries = blind_spot_safe_pad(
        time_series.to(data_device, torch.float32), padding, radius=model.receptive_radius
    )
    dataset = MultivariateTimeSeriesDataset(padded_timeseries, input_size=input_size, overlap=overlap)

    del padded_timeseries

    if use_gpu_data:
        train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    else:
        train_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=True,
        )

    # Trainer(benchmark=True) sets the global torch.backends.cudnn.benchmark flag. Remember the
    # previous value so it can be restored after training (see below).
    previous_cudnn_benchmark = torch.backends.cudnn.benchmark

    trainer = pl.Trainer(
        max_epochs=max_epochs,
        log_every_n_steps=log_every_n_steps,
        devices=devices,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        precision=precision,
        logger=False,
        enable_checkpointing=False,
        callbacks=[TQDMProgressBar(refresh_rate=log_every_n_steps)],
        benchmark=True,
    )

    trainer.fit(model, train_loader)

    # Restore the global cuDNN flag, so inference on many different input shapes doesn't pay a
    # one-time autotuning cost for each new shape.
    torch.backends.cudnn.benchmark = previous_cudnn_benchmark

    dataset.data = dataset.data.cpu()
    return model, dataset

def blind_spot_safe_pad(x: torch.Tensor, pad: int, radius: int) -> torch.Tensor:
    """
    Mirror-pad the last dimension by `pad` samples on each side, without breaking the blind spot.

    Plain reflect padding places a copy of x[t] at position -t, within the receptive field of the
    output at t for small t, so outputs near the trace ends would partly see their own input. Here
    the mirror starts `radius + 1` samples in from each edge, so every copy of x[t] lands more than
    `radius` samples away from t.

    Args:
        x (torch.Tensor): Traces, shape (..., num_frames)
        pad (int): Number of samples to add on each side
        radius (int): The model's receptive radius (model.receptive_radius)

    Returns:
        torch.Tensor: Padded traces, shape (..., num_frames + 2 * pad)
    """
    num_frames = x.shape[-1]
    if num_frames < pad + radius + 1:
        raise ValueError(f"Traces must have at least {pad + radius + 1} samples to pad by {pad} "
                         f"with receptive radius {radius} (got {num_frames})")
    left = x[..., radius + 1:radius + 1 + pad].flip(-1)
    right = x[..., num_frames - radius - 1 - pad:num_frames - radius - 1].flip(-1)
    return torch.cat([left, x, right], dim=-1)

class MultivariateTimeSeriesDataset(torch.utils.data.Dataset):
    def __init__(self, data, input_size=900, overlap=100, provide_indices=False):
        """
        Multivariate time series dataset.

        Args:
            data (torch.Tensor or np.ndarray): An array of shape (num_timeseries, num_frames) containing the time series data.
            input_size (int): Length of the input snippet.
            overlap (int): The number of overlapping samples between consecutive windows.
        """
        if isinstance(data, np.ndarray):
            self.data = torch.from_numpy(data).float()
        else:
            self.data = data.float()

        starting_num_rows = self.data.shape[0]
        # Compute row-wise standard deviation
        std_vals = self.data.std(dim=1)

        # Keep rows with std greater than eps
        non_constant_rows = std_vals > 1e-6
        self.data = self.data[non_constant_rows, :]
        updated_num_rows = self.data.shape[0]

        if starting_num_rows != updated_num_rows:
            display(f"Some of the input time series had no variance, these are excluded from training"
                    f"the input data had {starting_num_rows} time series. "
                    f"After filtering, the training data has {updated_num_rows} time series")

        self.data = self.data - self.data.mean(dim=1, keepdim=True)
        self.data /= torch.linalg.norm(self.data, dim=1, keepdim=True)


        self.num_series = self.data.shape[0]
        # a series shorter than input_size is one window
        self.input_size = min(input_size, self.data.shape[1])
        self.overlap = overlap
        self.stride = max(1, self.input_size - overlap)  # Effective step size for sliding windows
        self.num_windows = (
                                   data.shape[1] - self.input_size
                           ) // self.stride + 1  # Number of windows per time series
        self.provide_indices = provide_indices

        # Check if we need to add a final window at the end
        if (data.shape[1] - self.input_size) % self.stride != 0:
            self.num_windows += 1

    def __len__(self):
        # Total number of snippets: number of windows per time series * number of time series
        return self.num_windows * self.num_series

    def __getitem__(self, dataset_index):
        """
        Given an index, returns the corresponding time series snippet.
        """
        which_series = dataset_index // self.num_windows
        idx = dataset_index % self.num_windows
        start_idx = idx * self.stride
        end_idx = start_idx + self.input_size
        if end_idx >= self.data.shape[1]:
            # If the end index exceeds the data length, adjust it
            end_idx = self.data.shape[1]
            start_idx = end_idx - self.input_size
        data = self.data[which_series:which_series + 1, start_idx:end_idx]
        if self.provide_indices:
            return data, which_series, start_idx, end_idx
        else:
            return data


def denoise_batched(
        model: torch.nn.Module,
        traces: torch.Tensor,
        noise_variance_quantile: float = 0.05,
        input_size: int = 900,
        overlap: int = 200,
):
    """
    Denoise a large dataset by processing in batches. We use a Bayesian update rule to mix observations with network
    predictions during inference time.

    Args:
        model (torch.nn.Module): Trained model
        traces (torch.Tensor): Input traces to denoise [num_traces, num_timesteps]
        noise_variance_quantile (float): quantile for noise variance estimation
        var_partition_timesteps (int): Number of timesteps to use for variance partitioning
        input_size (int): We break each T-length time series into batches of size ``input_size" when running inference.
        overlap (int): The overlap over the input_size chunks.
    Returns:
        denoised_traces: The denoised traces [num_nodes, num_timesteps]
        noise_variance: Estimated noise variance per node. Shape: [num_nodes,]
        signal_weight: weights for signal component. Shape: [num_nodes, num_timesteps]
        observation_weight: weights for observation component. Shape: [num_nodes, num_timesteps]
    """
    if not 0 <= overlap < input_size:
        raise ValueError(f"overlap ({overlap}) must be non-negative and smaller than input_size ({input_size})")
    device = next(model.parameters()).device #Infer device from the model device
    traces = traces.to(device).float().clone()
    traces_means = torch.mean(traces, dim=1, keepdim=True)
    traces -= traces_means
    traces_norms = torch.linalg.norm(traces, dim=1, keepdim=True)
    traces_norms[traces_norms == 0] = 1
    traces /= traces_norms

    # Denoise the entire dataset using the estimated noise variance
    denoised_traces, network_estimates, total_variance_estimates = (
        _denoise_batched_inner(
            model,
            traces,
            noise_variance_quantile,
            input_size=input_size,
            overlap=overlap,
        )
    )

    denoised_traces *= traces_norms
    denoised_traces += traces_means
    return (
        denoised_traces,
        network_estimates,
        total_variance_estimates,
    )

def _denoise_batched_inner(model: torch.nn.Module,
                           traces: torch.Tensor,
                           quantile: float,
                           input_size: int = 900,
                           overlap: int = 200):
    """
    Denoise traces by running the network over overlapping windows and fusing its predictions with the
    observations using a Bayesian update.

    Outputs within RECEPTIVE_RADIUS samples of a window's edges depend on the convolutions' zero padding,
    so they are discarded, except at the trace's true start and end, where no neighboring window exists.
    Every kept output therefore equals what a single full-length pass would produce. Positions covered by
    two windows receive identical values, so they are simply overwritten rather than averaged.

    Args:
        model (torch.nn.Module): Trained model returning (mean, total variance) for inputs of shape (N, 1, L)
        traces (torch.Tensor): Normalized input traces, shape (num_traces, num_frames)
        quantile (float): Quantile in [0, 1] of each trace's predicted total variance, used as that trace's
            observation noise variance. 0 means no denoising: the output equals the input.
        input_size (int): Window length processed per forward pass
        overlap (int): Overlap between consecutive windows. Must satisfy
            2 * RECEPTIVE_RADIUS <= overlap < input_size so the kept regions cover every position.

    Returns:
        denoised_traces (torch.Tensor): (num_traces, num_frames), fusion of network predictions and observations
        network_predictions (torch.Tensor): (num_traces, num_frames), the blind-spot network's mean predictions
        total_variance_estimates (torch.Tensor): (num_traces, num_frames), predicted total variance
            (system + observation), clamped below at the observation variance
        All three are in the normalized units of `traces`.
    """
    radius = model.receptive_radius
    if not 2 * radius <= overlap < input_size:
        raise ValueError(f"overlap ({overlap}) must be at least {2 * radius} "
                         f"and smaller than input_size ({input_size})")

    device = next(model.parameters()).device
    num_timepoints = traces.shape[1]

    # Reuse the dataset's windowing logic to get each window's start and end indices
    placeholder_trace = torch.arange(num_timepoints, device=device)[None, :]
    eval_dataset = MultivariateTimeSeriesDataset(
        placeholder_trace, input_size=input_size, overlap=overlap, provide_indices=True,
    )

    model.eval()
    with torch.no_grad():
        # Every position is written at least once (guaranteed by the overlap check above)
        network_predictions = torch.empty_like(traces)
        total_variance_estimates = torch.empty_like(traces)

        for i in range(eval_dataset.num_windows):
            _, _, start_idx, end_idx = eval_dataset[i]
            window_predictions, window_variances = model(traces[:, None, start_idx:end_idx])

            # Keep only outputs whose receptive field lies entirely inside this window.
            # At the trace's true start/end there is no neighboring window, so keep those outputs too.
            window_length = end_idx - start_idx
            keep_start = 0 if start_idx == 0 else radius
            keep_end = window_length if end_idx == num_timepoints else window_length - radius

            network_predictions[:, start_idx + keep_start:start_idx + keep_end] = \
                window_predictions[:, 0, keep_start:keep_end]
            total_variance_estimates[:, start_idx + keep_start:start_idx + keep_end] = \
                window_variances[:, 0, keep_start:keep_end]

        if quantile == 0:
            # Zero noise variance puts all weight on the observations
            observation_variance = torch.zeros(traces.shape[0], 1, device=device, dtype=traces.dtype)
        else:
            observation_variance = torch.quantile(total_variance_estimates, quantile, dim=1, keepdim=True)

        total_variance_estimates = torch.clamp(total_variance_estimates, min=observation_variance)
        system_variance = torch.clamp(total_variance_estimates - observation_variance, min=0)

        # Apply Bayesian formula for posterior mean
        weight_signal = observation_variance / total_variance_estimates
        weight_observation = system_variance / total_variance_estimates
        denoised_traces = weight_signal * network_predictions + weight_observation * traces

    return denoised_traces, network_predictions, total_variance_estimates

class CompressionTemporalDenoiser(torch.nn.Module):

    def __init__(self,
                 trained_model: torch.nn.Module,
                 noise_variance_quantile:float = 1,
                 input_size: int = 900,
                 padding: int = 100):
        super(CompressionTemporalDenoiser, self).__init__()
        self.noise_variance_quantile = noise_variance_quantile
        self.net = trained_model
        self._padding = padding
        self._input_size = input_size

    def forward(self, traces: torch.Tensor):
        #F.pad(x, (pad_size, pad_size), mode=mode)
        padded = blind_spot_safe_pad(traces, self._padding, radius=self.net.receptive_radius)
        outputs = denoise_batched(self.net,
                               padded,
                               noise_variance_quantile=self.noise_variance_quantile,
                               input_size=self._input_size)[0]
        return outputs[:, self._padding:outputs.shape[1] - self._padding]
