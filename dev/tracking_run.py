"""Track ROIs across sessions with ROICaT and save the tracking folder, what `masknmf track` does."""

import glob

from masknmf.multisession import RoicatDataAdapter, RoicatTracker

RESULTS = "X:/data/yu/multiDay_population_imaging/masknmf-multisession-testdata/day*/*/results.hdf5"
OUT = "X:/data/yu/multiDay_population_imaging/masknmf-multisession-testdata/tracking_dev"

# roicat's worker processes re-import this file (spawn on windows): everything below runs only in the parent
if __name__ == "__main__":
    files = sorted(glob.glob(RESULTS))
    for session, path in enumerate(files):
        print(session, path)
    adapter = RoicatDataAdapter.from_masknmf(files, um_per_pixel=1.2)
    tracker = RoicatTracker()
    tracker.params["general"]["use_GPU"] = True
    tracking = tracker.run_tracking(adapter)
    print(tracking)
    print("saved to", tracking.to_roicat_dir(OUT))
