"""Load a tracking folder, put the results files in session order by roi count, and print what the tracking holds."""

import glob

import h5py

from masknmf.multisession import RoicatTrackingResults

FOLDER = "X:/data/yu/multiDay_population_imaging/tracking/roicat_tracking_take1"
# None takes the files the tracking recorded; a glob is matched to the sessions by roi count
RESULTS = "X:/data/yu/multiDay_population_imaging/tracking/*.hdf5"

files = None if RESULTS is None else sorted(glob.glob(RESULTS))
tracking = RoicatTrackingResults.from_roicat_dir(FOLDER, session_files=files)

if files is not None:
    counts = {}
    for path in files:
        with h5py.File(path, "r") as f:
            counts[path] = f["DemixingResults/temporal_demixed"].shape[1]
    ordered = []
    for session, n in enumerate(tracking.num_roi_per_session):
        matches = [path for path, count in counts.items() if count == n]
        if len(matches) != 1:
            raise ValueError(f"session {session} has {n} rois and {len(matches)} results files do")
        ordered.append(matches[0])
    tracking.session_files = ordered

clustered = sum(int((labels >= 0).sum()) for labels in tracking.labels_by_session)
print(tracking)
print("sessions", tracking.num_sessions)
print("rois", tracking.num_roi_total, "per session", tracking.num_roi_per_session)
print("clusters", tracking.num_clusters, "in every session", int((tracking.num_sessions_per_cluster == tracking.num_sessions).sum()))
print(f"clustered {clustered / tracking.num_roi_total:.0%} ({clustered})")
for session, path in enumerate(tracking.session_files):
    print(session, path)

print("aligned footprints", [rois.shape for rois in tracking.aligned_rois])
print("aligned fovs", [image.shape for image in tracking.aligned_fov_images])
print("cluster 0 members", tracking.find_members(0))

#%%
