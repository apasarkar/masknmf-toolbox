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
            max_epochs=20,
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
        self.data = data.astype(np.float32)
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
        return self.num_windows

    def __getitem__(self, idx):
        """
        Given an index, returns the corresponding time series snippet.
        """
        start_idx = idx * self.stride
        end_idx = start_idx + self.input_size
        if end_idx >= self.data.shape[1]:
            # If the end index exceeds the data length, adjust it
            end_idx = self.data.shape[1]
            start_idx = end_idx - self.input_size
        data = torch.tensor(self.data[:, start_idx:end_idx])
        if self.provide_indices:
            return data, start_idx, end_idx
        else:
            return data
