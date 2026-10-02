"""A tracking's session files must hold the ROIs it clustered: a file curated after tracking is refused."""

from pathlib import Path

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


def test_view_places_files_given_out_of_order_by_roi_count_and_refuses_a_curated_one(tmp_path, capsys):
    from masknmf import cli

    num_rois = [3, 2]
    files = []
    for session, count in enumerate(num_rois):
        # named so a glob sorts them against session order
        path = tmp_path / "sessions" / f"day{1 - session}" / "results.hdf5"
        path.parent.mkdir(parents=True)
        with h5py.File(path, "w") as f:
            f.create_dataset("DemixingResults/temporal_demixed", data=np.zeros((10, count), np.float32))
        files.append(str(path))
    results = {
        "clusters": {"labels_bySession": [np.arange(n) for n in num_rois]},
        "ROIs": {"ROIs_raw": [scipy.sparse.csr_matrix((n, 16)) for n in num_rois], "frame_height": 4, "frame_width": 4},
    }
    folder = str(tmp_path / "tracking")
    RoicatTrackingResults(results=results, run_data={}, session_files=tuple(files)).to_roicat_dir(folder)

    cli.main(["view", folder, str(tmp_path / "sessions" / "*" / "results.hdf5"), "--list"])
    rows = [line.split()[-1] for line in capsys.readouterr().out.splitlines() if line.strip().endswith("results.hdf5")]
    assert rows == [str(Path("day1", "results.hdf5")), str(Path("day0", "results.hdf5"))]

    curated = tmp_path / "sessions" / "day0" / "results.20261001T120000.curated.hdf5"
    with h5py.File(curated, "w") as f:
        f.create_dataset("DemixingResults/temporal_demixed", data=np.zeros((10, 1), np.float32))
    with pytest.raises(SystemExit):
        cli.main(["view", folder, str(tmp_path / "sessions" / "*" / "*.hdf5"), "--list"])
    assert "session 1: " in (err := capsys.readouterr().err) and "holds 1 ROIs, the tracking 2" in err
