"""
Denoiser convergence test
=========================

Opens the convergence dialog on a simulated movie: 30 cells with calcium transients under shot noise.
Drag the yellow box onto a cell, press Start and watch the denoised trace settle epoch by epoch.
"""

# run_example = false

from masknmf.visualization.imgui import denoiser_convergence

denoiser_convergence.run(movie=denoiser_convergence.simulate_movie())
