"""
PMD Compression
===============

Simple PMD compression, the hello world of masknmf
"""

# test_example = true
# sphinx_gallery_pygfx_docs = 'animate 10s 30fps'

import masknmf
import torch
import fastplotlib as fpl
from urllib.request import urlretrieve
import tifffile


urlretrieve(
    "https://github.com/flatironinstitute/CaImAn/raw/refs/heads/main/example_movies/demoMovie.tif",
    "./demo.tif"
)

# always lazy load raw data by memmaping or other methods
data = tifffile.imread("./demo.tif")

block_sizes = [32, 32]
max_components = 20

# it's recommended to use masknmf on a machine with a GPU
# device = "cuda" if torch.cuda.is_available() else "cpu"
device = "cpu"
# number of frames used to estimate the spatial basis in PMD
num_frames_for_spatial_fit = data.shape[0]

# perform PMD
pmd_result = masknmf.compression.pmd_decomposition(
    data,
    block_sizes,
    num_frames_for_spatial_fit,
    max_components=max_components,
    device=device,
    frame_batch_size=1024
)

# get the residual
pmd_residual = masknmf.PMDResidualArray(data, pmd_result)

# view the movies, note that all these array are LAZY evaluated, allowing you to view extremely large datasets!
iw = fpl.ImageWidget(
    data=[data, pmd_result, pmd_residual],
    names=["raw", "pmd", "residual"],
    figure_kwargs={"size": (1000, 340), "shape": (1, 3)},
    cmap="gnuplot2",
)

iw.show()

# use the time slider or set the frame index programmatically
iw.current_index = {"t": 1610}

# manually set vmin-vmax to emphasize noise in raw video
# you can also adjust the vmin-vmax using the histogram tool
# reset the vmin-vmax by clicking the buttons under "ImageWidget Controls"
for image in iw.managed_graphics:
    image.vmax = 3_200

# remove toolbar to reduce clutter
for subplot in iw.figure:
    subplot.toolbar = False


# ignore the remaining lines these are just for docs generation
figure = iw.figure
if __name__ == "__main__":
    print(__doc__)
    fpl.loop.run()
