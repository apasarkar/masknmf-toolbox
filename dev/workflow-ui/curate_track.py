"""
Curate two sessions in the demixing viewer, track them with ROICaT and open the multisession viewer, all from Python.
Runs top to bottom, under the debugger too: each viewer blocks until its windows are closed, then the script goes on.
"""

# %% paths
import glob
import os

import fastplotlib as fpl
import h5py

import masknmf
from masknmf.demixing import latest_results
from masknmf.demixing.labels import SIDECAR_SUFFIX
from masknmf.multisession import RoicatDataAdapter, RoicatTracker, RoicatTrackingResults

import roicat

ROOT = "X:/data/yu/masknmf_testing_0-2-0"
DEVICE = "cuda"
UM_PER_PIXEL = 1.2

# %% day1: curate
# a draws a region that selects, r adds a drawn roi, d marks the selected signals for deletion, then Demix writes
# results.<stamp>.curated.hdf5 beside results.hdf5; close the windows once the status line names the new file
path = f"{ROOT}/day1/results.hdf5"
viewer = masknmf.SingleSessionDemixingVis(
    masknmf.DemixingResults.from_hdf5(path, device=DEVICE), device=DEVICE, results_path=path
)
viewer.show()
fpl.loop.run()
print(sorted(glob.glob(f"{ROOT}/day1/*.curated.hdf5"))[-1])

# %% day2: curate
path = f"{ROOT}/day2/results.hdf5"
viewer = masknmf.SingleSessionDemixingVis(
    masknmf.DemixingResults.from_hdf5(path, device=DEVICE), device=DEVICE, results_path=path
)
viewer.show()
fpl.loop.run()
print(sorted(glob.glob(f"{ROOT}/day2/*.curated.hdf5"))[-1])

# %% the curated files: what a glob takes, their roi counts and descriptions
matches = sorted(p for p in glob.glob(f"{ROOT}/day*/*.hdf5") if not p.endswith(SIDECAR_SUFFIX))
files = latest_results(matches)
for file in files:
    assert file.endswith(".curated.hdf5"), f"{file} is not curated"
    original = os.path.join(os.path.dirname(file), "results.hdf5")
    with h5py.File(original, "r") as f:
        before = f["DemixingResults/temporal_demixed"].shape[1]
    with h5py.File(file, "r") as f:
        after = f["DemixingResults/temporal_demixed"].shape[1]
        assert "RigidRegistrationArray" not in f and "RigidMotionCorrector" not in f
        print(f"{file}\n  {before} -> {after} rois\n  {f.attrs['description']}")

# %% track the curated files
tracker = RoicatTracker()
tracker.params["general"]["use_GPU"] = DEVICE != "cpu"
# dataloader workers are spawned processes that re-run this script; in-process keeps it debuggable
tracker.params["ROInet"]["dataloader"]["numWorkers_dataloader"] = 0
tracker.params["ROInet"]["dataloader"]["persistentWorkers_dataloader"] = False
tracking = tracker.run_tracking(RoicatDataAdapter.from_masknmf(files, um_per_pixel=UM_PER_PIXEL))
folder = tracking.to_roicat_dir(f"{ROOT}/tracking")
print(tracking)
print(folder)

# %% the tracking refuses the uncurated files when curation changed a session's roi count
originals = [os.path.join(os.path.dirname(file), "results.hdf5") for file in files]
try:
    RoicatTrackingResults.from_roicat_dir(folder, session_files=originals)
    print("not refused: every curated file has as many rois as its results.hdf5")
except ValueError as error:
    print(f"refused: {error}")

# %% multisession viewer on the files the tracking recorded
tracking = RoicatTrackingResults.from_roicat_dir(folder)
assert [os.path.normcase(f) for f in tracking.session_files] == [os.path.normcase(os.path.abspath(f)) for f in files]
viewer = masknmf.MultiSessionDemixingVis(tracking, device=DEVICE)
viewer.show()
fpl.loop.run()

# %% curate a curated file again: Demix writes the next results.<stamp>.curated.hdf5, the stem stays results
path = files[0]
viewer = masknmf.SingleSessionDemixingVis(
    masknmf.DemixingResults.from_hdf5(path, device=DEVICE), device=DEVICE, results_path=path
)
viewer.show()
fpl.loop.run()
newest = sorted(glob.glob(os.path.join(os.path.dirname(path), "*.curated.hdf5")))[-1]
assert newest != path and os.path.basename(newest).startswith("results."), "closed without a finished Demix"
print(newest)
