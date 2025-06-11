import numpy as np
import torch
import torch.nn as nn

import torch
import pytorch_lightning as pl
import torch.nn as nn
import networkx as nx
import numpy as np
from typing import *
from pytorch_lightning.loggers import TensorBoardLogger
from copy import deepcopy
from torch.utils.data import DataLoader
import os
import sys
import matplotlib.pyplot as plt


def plot_reconstruction_info(ground_truth,
                             noisy_trace,
                             network_prediction,
                             mixed_prediction,
                             signal_weight,
                             observation_weight,
                             total_var,
                            noise_variance,
                            out_folder = None,
                            out_name = None):
    residual = noisy_trace - mixed_prediction

    signals = [
        ground_truth,
        noisy_trace,
        network_prediction,
        mixed_prediction,
        residual,
        signal_weight,
        observation_weight,
        total_var
    ]

    titles = [
        "Ground Truth",
        "Noisy Trace",
        "Network Prediction",
        "Mixed Prediction",
        "Residual (Noisy - Mixed)",
        "Signal Weight",
        "Observation Weight",
        f"Total Variance. NoiseVar = {float(noise_variance[0]):.2}"
    ]

    n_signals = len(signals)
    fig, axes = plt.subplots(n_signals, 1, figsize=(10, 2 * n_signals), sharex=True)

    if n_signals == 1:
        axes = [axes]

    for ax, signal, title in zip(axes, signals, titles):
        ax.plot(signal, lw=1.5)
        ax.set_title(title)
        ax.grid(True)

    axes[-1].set_xlabel("Time")
    plt.tight_layout()
    if out_folder is not None and out_name is not None:
        final_path = os.path.join(out_folder, out_name)
        print(final_path)
        plt.savefig(final_path, bbox_inches="tight")
    else:
        plt.show()


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


class TemporalNetwork(nn.Module):
    def __init__(self):
        super(TemporalNetwork, self).__init__()
        self.mean_backbone = BlindSpotTemporal()
        self.var_backbone = BlindSpotTemporal(final_activation=nn.Softplus())

    def forward(self, x):
        return self.mean_backbone(x), self.var_backbone(x)


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

        num_datapoints = input_traces.shape[0] * input_traces.shape[1]

        # make sure all total variances are positive
        total_variance = torch.clamp(total_variance, min=1e-8)

        log_lik = torch.nansum(torch.log(total_variance))
        log_lik = log_lik + torch.nansum(
            (input_traces - mu_x) ** 2 / total_variance
        )
        loss = log_lik / num_datapoints
        self.log("train_loss", loss)

        current_lr = self.optimizers().param_groups[0]['lr']
        self.log('learning_rate', current_lr)

        return loss

    def forward(self, x):
        return self.temporal_network(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer


def train_total_variance_denoiser(
        time_series,
        learning_rate: float = 1e-2,
        input_size: int = 900,
        overlap: int = 600,
        max_epochs: int = 20,
        batch_size: int = 1,
        devices: int = 1,
):
    """Train a total variance prediction network"""
    model = TotalVarianceTemporalDenoiser(
        learning_rate=learning_rate,
        max_epochs=max_epochs,
    )

    dataset = MultivariateTimeSeriesDataset(time_series, input_size=input_size, overlap=overlap)
    train_loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=6, pin_memory=True
    )

    logger = TensorBoardLogger("lightning_logs", name="total_variance")
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        log_every_n_steps=1,
        devices=devices,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        precision="16-mixed",
        # strategy="ddp_notebook" if devices > 1 else None,
    )

    trainer.fit(model, train_loader)
    return model, dataset


class MultivariateTimeSeriesDataset(torch.utils.data.Dataset):
    def __init__(self, data, input_size=900, overlap=100, provide_indices=False):
        """
        Multivariate time series dataset.

        Args:
            data (torch.Tensor or np.ndarray): An array of shape (N, T_max) containing the time series data.
            input_size (int): Length of the input snippet.
            overlap (int): The number of overlapping samples between consecutive windows.
        """
        self.data = torch.from_numpy(data).float()
        self.num_series = self.data.shape[0]
        self.input_size = input_size
        self.overlap = overlap
        self.stride = input_size - overlap  # Effective step size for sliding windows
        self.num_windows = (
                                   data.shape[1] - input_size
                           ) // self.stride + 1  # Number of windows per time series
        self.provide_indices = provide_indices

        # Check if we need to add a final window at the end
        if (data.shape[1] - input_size) % self.stride != 0:
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
        data = self.data[[which_series], start_idx:end_idx]
        if self.provide_indices:
            return data, which_series, start_idx, end_idx
        else:
            return data

def denoise_batched(
    model,
    traces,
    noise_variance_percentile=5,
    var_partition_timesteps=5000,
    input_size=900,
    overlap=200,
):
    """
    Denoise a large dataset by processing in batches.
    First, we partition the variance using a subset of the data,
    then we denoise the entire dataset using the estimated noise variance.

    Args:
        model: Trained model
        traces: Input traces to denoise [num_nodes, num_timesteps]
        noise_variance_percentile: Percentile for noise variance estimation
        var_partition_timesteps: Number of timesteps to use for variance partitioning
    Returns:
        denoised_traces: The denoised traces [num_nodes, num_timesteps]
        noise_variance: Estimated noise variance per node. Shape: [num_nodes,]
        signal_weight: weights for signal component. Shape: [num_nodes, num_timesteps]
        observation_weight: weights for observation component. Shape: [num_nodes, num_timesteps]
    """
    noise_variance = partition_variance(
        model,
        traces[:, :var_partition_timesteps],
        percentile=noise_variance_percentile,
    )

    noise_variance = torch.tensor(noise_variance, dtype=torch.float64)

    # Denoise the entire dataset using the estimated noise variance
    denoised_traces, signal_mean, signal_weight, observation_weight, total_var = (
        _denoise_batched_inner(
            model,
            traces,
            noise_variance,
            input_size=input_size,
            overlap=overlap,
        )
    )
    noise_variance = noise_variance.cpu().numpy()
    return (
        denoised_traces,
        signal_mean,
        noise_variance,
        signal_weight,
        observation_weight,
        total_var,
    )


def partition_variance(model, validation_data, percentile=5):
    """
    Partition the total variance into signal and noise components using quantile regression.

    Args:
        model: Trained model that predicts means and total variances
        validation_data: array of validation data used for partitioning. [N x T]
        quantile: Quantile to use (lower quantile represents noise floor)

    Returns:
        noise_variance: Estimated observation noise variance (per node)
    """
    model.eval()
    input_traces = torch.tensor(validation_data, dtype=torch.float32)
    input_traces = input_traces[:, None, :]  # [Batch, channels, num_timesteps]

    # Move tensor to the same device as the model parameters
    device = next(model.parameters()).device
    input_traces = input_traces.to(device)

    # Get variance prediction to use in percentile calculation
    with torch.no_grad():
        _, total_variance = model(input_traces)

    total_variance = total_variance.squeeze(1).cpu().numpy()
    noise_var = np.percentile(total_variance, percentile, axis=1, keepdims = True)
    return noise_var


def _denoise_batched_inner(model, traces, noise_variance, input_size=900, overlap=200):
    """
    Denoise a large dataset by processing in batches.

    Args:
        model: Trained model
        traces: Input traces to denoise [num_nodes, num_timesteps]
        noise_variance: Estimated noise variance per node
        input_size: Input window size
        overlap: Overlap between windows

    Returns:
        denoised_traces: The denoised traces
    """
    # Hacky way to re-use the iteration pattern here over timesteps:
    eval_dataset = MultivariateTimeSeriesDataset(
        traces[[0]], input_size=input_size, overlap=overlap,
        provide_indices=True,
    )

    traces_tensor = torch.from_numpy(traces)

    # Create arrays to hold results
    num_batches = eval_dataset.num_windows

    denoised_traces = np.zeros_like(traces)
    signal_mean = np.zeros_like(traces)
    signal_weights = np.zeros_like(traces)
    observation_weights = np.zeros_like(traces)
    total_var = np.zeros_like(traces)
    counts = np.zeros_like(traces)

    # Process each window
    for i in range(eval_dataset.num_windows):
        _, _, start_idx, end_idx = eval_dataset[i]
        subset = traces_tensor[:, start_idx:end_idx]
        subset = subset.unsqueeze(1)

        denoised_curr, signal_mean_curr, signal_weight_curr, observation_weight_curr, total_var_curr = (
            denoise_with_partitioned_variance(
                model,
                subset,
                noise_variance,
            )
        )
        denoised_traces[:, start_idx:end_idx] += denoised_curr.squeeze(1)
        signal_mean[:, start_idx:end_idx] += signal_mean_curr.squeeze(1)
        signal_weights[:, start_idx:end_idx] += signal_weight_curr.squeeze(1)
        observation_weights[:, start_idx:end_idx] += observation_weight_curr.squeeze(1)
        total_var[:, start_idx:end_idx] += total_var_curr.squeeze(1)

        # Count how many times each window has been added
        counts[:, start_idx:end_idx] += 1

    # Average where windows overlap
    if not np.all(counts > 0):
        raise ValueError("Some windows were not counted!")
    denoised_traces = denoised_traces / counts
    signal_mean = signal_mean / counts
    signal_weights = signal_weights / counts
    observation_weights = observation_weights / counts
    total_var = total_var / counts

    return denoised_traces, signal_mean, signal_weights, observation_weights, total_var


def denoise_with_partitioned_variance(model, traces, noise_variance):
    """
    Denoise a single batch of traces.

    Args:
        model: Trained model
        traces: Input traces [batch_size, num_nodes, num_timesteps]
        noise_variance: Estimated noise variance per node

    Returns:
        denoised_traces: The denoised traces
    """
    model.eval()
    with torch.no_grad():
        traces_tensor = torch.tensor(traces, dtype=torch.float32)

        # Move tensor to the same device as the model parameters
        device = next(model.parameters()).device
        traces_tensor = traces_tensor.to(device)

        # Get predictions
        mu_x, total_variance = model(traces_tensor)

        # Apply noise variance (expand dimensions to match)
        noise_var = noise_variance.to(traces_tensor.device)
        noise_var = noise_var[..., None].expand_as(total_variance)

        # In some regions we may have total_variance <= noise_variance.
        # Since this can't happen, we reset the variance to noise_variance
        # in those regions.
        total_variance = torch.clamp(total_variance, min=noise_var)

        # Ensure signal variance is positive
        signal_var = torch.clamp(total_variance - noise_var, min=0)

        # Apply Bayesian formula for posterior mean
        weight_signal = noise_var / total_variance
        weight_observation = signal_var / total_variance
        denoised_traces = weight_signal * mu_x + weight_observation * traces_tensor

    return (
        denoised_traces.cpu().numpy(),
        mu_x.cpu().numpy(),
        weight_signal.cpu().numpy(),
        weight_observation.cpu().numpy(),
        total_variance.cpu().numpy(),
    )
