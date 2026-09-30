"""A tracking's session files must hold the ROIs it clustered: a file curated after tracking is refused."""

import h5py
import numpy as np
import pytest
import scipy.sparse

pytest.importorskip("roicat")
from masknmf.multisession import RoicatTrackingResults


def test_a_session_file_with_other_rois_than_the_tracking_is_refused(tmp_path):
    num_rois = [3, 2]
    files = []
    for session, count in enumerate(num_rois):
        path = tmp_path / f"day{session}.hdf5"
        with h5py.File(path, "w") as f:
            f.create_dataset("DemixingResults/temporal_demixed", data=np.zeros((10, count), np.float32))
        files.append(str(path))
    results = {
        "clusters": {"labels_bySession": [np.arange(n) for n in num_rois]},
        "ROIs": {"ROIs_raw": [scipy.sparse.csr_matrix((n, 16)) for n in num_rois], "frame_height": 4, "frame_width": 4},
    }
    tracking = RoicatTrackingResults(results=results, run_data={}, session_files=tuple(files))

    curated = tmp_path / "day1.curated.hdf5"
    with h5py.File(curated, "w") as f:
        f.create_dataset("DemixingResults/temporal_demixed", data=np.zeros((10, 1), np.float32))
    with pytest.raises(ValueError, match="holds 1 ROIs, the tracking 2"):
        tracking.session_files = (files[0], str(curated))
    # files not on disk are the caller's to report
    tracking.session_files = (files[0], str(tmp_path / "moved.hdf5"))
