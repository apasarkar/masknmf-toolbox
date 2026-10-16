"""Open the multisession viewer on a tracking folder."""

import fastplotlib as fpl

import masknmf
from masknmf.multisession import RoicatTrackingResults

FOLDER = "X:/data/yu/multiDay_population_imaging/tracking/roicat_tracking_take1"

SESSION_FILES = [
    "X:/data/yu/multiDay_population_imaging/tracking/results.hdf5",
    "X:/data/yu/multiDay_population_imaging/tracking/day_2_Yu_results.hdf5",
]
tracking = RoicatTrackingResults.from_roicat_dir(FOLDER, session_files=SESSION_FILES)

print(tracking)
viewer = masknmf.MultiSessionDemixingVis(tracking, device="cuda")
viewer.show()

fpl.loop.run()

x = 2