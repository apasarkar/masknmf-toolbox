"""Stimulus onsets and epochs drawn over every panel of the demixing viewer's trace dock."""
import numpy as np, fastplotlib as fpl
from masknmf.demixing.demixing_results import DemixingResults
from masknmf.visualization import SingleSessionDemixingVis

dmr = DemixingResults.from_hdf5(r"X:\data\eunji\masknmf-defaults\zplane01\demixing_results.hdf5",
                                device="cpu")
vis = SingleSessionDemixingVis(dmr, device="cpu")

# stand-in for real stimulus times: an onset every 600 frames, each lasting 60
onsets = np.arange(300, dmr.shape[0], 600)
vis.traces.mark("stim onset", onsets, (1.0, 0.9, 0.2))
vis.traces.span("stim", onsets, onsets + 60, (1.0, 0.9, 0.2))

# something in the dock before the first double-click
box = (slice(None), slice(200, 220), slice(200, 220))
vis.traces.set("compressed trace", [("pmd", np.asarray(dmr.pmd_array[box]).mean(axis=(1, 2)), None)])

vis.show()
fpl.loop.run()
