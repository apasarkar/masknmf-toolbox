"""
Denoiser convergence test
=========================

Opens the convergence dialog on a simulated movie: 30 cells with calcium transients under shot noise.
Drag the yellow box onto a cell, press Start and watch the denoised trace settle epoch by epoch.
"""

# run_example = false

import numpy as np

from masknmf.visualization.imgui import denoiser_convergence

n_frames, height, width, n_cells = 4000, 96, 96, 30
rng = np.random.default_rng(seed=0)

centers = rng.uniform(low=8, high=88, size=(n_cells, 2))
yy, xx = np.mgrid[:height, :width]
footprints = np.exp(-((yy[None] - centers[:, 0, None, None]) ** 2 + (xx[None] - centers[:, 1, None, None]) ** 2) / (2 * 3.0**2))

spikes = rng.random(size=(n_cells, n_frames)) < 0.005
kernel_decay = np.exp(-np.arange(200) / 30.0)
calcium = np.stack([np.convolve(row, kernel_decay)[:n_frames] for row in spikes])  ## (n_cells, n_frames)

movie = 20 + 30 * np.einsum("ct,chw->thw", calcium, footprints)
movie = rng.poisson(lam=movie).astype(np.float32)

denoiser_convergence.run(movie=movie)
