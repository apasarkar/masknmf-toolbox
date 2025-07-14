import masknmf
from pathlib import Path
import fastplotlib as fpl
import tifffile
import numpy as np
import matplotlib.pyplot as plt

try:
    import mbo_utilities as mbo
except ImportError:
    print("uv pip install git+https://github.com/MillerBrainObservatory/mbo_utilities.git@dev")

from mbo_utilities.phasecorr import nd_windowed, phase_offsets_timecourse, apply_patchwise_offsets

raw_files = [x for x in Path(r"D:\tests_bigmem\roi2").glob("*.tif*")]
data_store = {}

for file in raw_files:
    if file.stem == "plane10":
        data_store[file.stem] = tifffile.memmap(file)

zplane_name = "plane10"
data = data_store[zplane_name]

lazy = mbo.imread(r"D:\W2_DATA\kbarber\2025_03_01\mk301\green", fix_phase=False)
mbo.imwrite(lazy, r"D:\phasecorr", planes=[10], ext=".tif", overwrite=True, debug=True)

raw = mbo.imread(r"D:\phasecorr\plane10.tif")

iw = fpl.ImageWidget(raw)
iw.show()
iw.close()

##
subarr = raw[:, 120:170, 270:350]
iw = fpl.ImageWidget(subarr)
iw.show()
fpl.loop.run()

corrected, ofs = nd_windowed(subarr, method="frame", upsample=2, border=0, max_offset=2)
corrected_mean, ofs_mean = nd_windowed(subarr, method="mean", upsample=2, border=0, max_offset=2)

diff = corrected - corrected_mean
iw = fpl.ImageWidget([corrected, corrected_mean, diff])
iw.show()
fpl.loop.run()

##
corrected, raw_offsets = nd_windowed(data, method="mean")
xsplits, patch_offsets = phase_offsets_timecourse(data, n_parts=4, method="mean")
patch_corrected = apply_patchwise_offsets(data, xsplits, patch_offsets)
