"""A tracking run saves a timestamped folder and a manifest beside it; masknmf view opens a manifest, or a folder's newest."""

import json

import h5py
import numpy as np
import pytest
import scipy.sparse

pytest.importorskip("roicat")
from masknmf import cli
from masknmf.multisession import RoicatTrackingResults, roicat_tracking


def test_each_run_saves_a_folder_and_a_manifest_and_view_opens_the_newest(tmp_path, capsys, monkeypatch):
    num_rois = [3, 2]
    files = []
    for session, count in enumerate(num_rois):
        path = tmp_path / "experiment" / f"day{session}" / "results.hdf5"
        path.parent.mkdir(parents=True)
        with h5py.File(path, "w") as f:
            f.create_dataset("DemixingResults/temporal_demixed", data=np.zeros((10, count), np.float32))
        files.append(str(path))
    results = {
        "clusters": {"labels_bySession": [np.arange(n) for n in num_rois]},
        "ROIs": {"ROIs_raw": [scipy.sparse.csr_matrix((n, 16)) for n in num_rois], "frame_height": 4, "frame_width": 4},
    }
    tracking = RoicatTrackingResults(results=results, run_data={}, session_files=tuple(files))

    folder = tmp_path / "experiment" / "tracking"
    stamps = ["2026-10-01-12-00-00", "2026-10-02-09-30-00"]
    monkeypatch.setattr(roicat_tracking, "get_timestamp", iter([*stamps, "2026-10-04-00-00-00", "2026-10-05-00-00-00"]).__next__)
    runs = [tracking.to_roicat_dir(folder) for _ in stamps]
    assert runs == [folder / f"{stamp}_roicat-tracking" for stamp in stamps]
    manifests = [folder / f"{stamp}_roicat-tracking-manifest.json" for stamp in stamps]
    assert json.loads(manifests[0].read_text()) == {
        "tracking": f"{stamps[0]}_roicat-tracking",
        "sessions": ["../day0/results.hdf5", "../day1/results.hdf5"],
    }

    # the tracking folder, and the experiment folder holding it, open the newest run; a manifest opens its own
    for entry, opened in ((folder, manifests[1]), (folder.parent, manifests[1]), (manifests[0], manifests[0])):
        cli.main(["view", str(entry), "--list"])
        assert capsys.readouterr().out.splitlines()[0] == f"tracking {opened}"

    # the manifest's paths are relative to it, so the experiment folder still opens once moved
    moved = tmp_path / "moved"
    (tmp_path / "experiment").rename(moved)
    tracking = RoicatTrackingResults.from_manifest(moved / "tracking" / manifests[1].name)
    assert tracking.session_files == tuple(str(moved / f"day{session}" / "results.hdf5") for session in range(2))

    # only a tracking manifest's name is looked for: a later-stamped manifest of another kind is passed over
    (moved / "tracking" / "2026-10-03-00-00-00_other-manifest.json").write_text("{}")
    cli.main(["view", str(moved / "tracking"), "--list"])
    assert capsys.readouterr().out.splitlines()[0] == f"tracking {moved / 'tracking' / manifests[1].name}"

    # a session file beside the experiment is still relative, walking up; sessions of several files get no manifest
    outside = tmp_path / "elsewhere" / "results.hdf5"
    outside.parent.mkdir()
    (moved / "day1" / "results.hdf5").rename(outside)
    tracking.session_files = (tracking.session_files[0], str(outside))
    run = tracking.to_roicat_dir(moved / "tracking")
    assert json.loads(run.with_name(f"{run.name}-manifest.json").read_text())["sessions"] == [
        "../day0/results.hdf5", "../../elsewhere/results.hdf5",
    ]
    tracking.session_files = ((tracking.session_files[0], str(outside)), str(outside))
    run = tracking.to_roicat_dir(moved / "tracking")
    assert run.is_dir() and not run.with_name(f"{run.name}-manifest.json").exists()
