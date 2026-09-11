import numpy as np, fastplotlib as fpl
from masknmf.demixing.demixing_results import DemixingResults
from masknmf.visualization import SingleSessionDemixingVis
from masknmf.visualization.imgui import TracePlot

dmr = DemixingResults.from_hdf5(r"X:\data\eunji\masknmf-defaults\zplane01\demixing_results.hdf5",
                          device="cpu")
vis = SingleSessionDemixingVis(dmr, device="cpu")

plot = TracePlot(("pmd", "residual"), dmr.shape[0])
plot.dock(vis.fov_widget.figure, size=320)
plot.link(vis.reference_index)

box = (slice(None), slice(200, 220), slice(200, 220))
plot.set("pmd", [("pmd", np.asarray(dmr.pmd_array[box]).mean(axis=(1, 2)), (1.0, 0.5, 0.0))])
plot.set("residual", [("residual", np.asarray(dmr.residual_array[box]).mean(axis=(1, 2)),
                       None)])

vis.show()
fpl.loop.run()