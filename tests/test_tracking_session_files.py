"""Tracking sessions are the results files a manifest lists; a tracking folder without a manifest takes files in the order given."""

from pathlib import Path

import h5py
import numpy as np
import pytest
import scipy.sparse

pytest.importorskip("roicat")
import masknmf
from masknmf.multisession import RoicatTrackingResults


@pytest.fixture
def tracked(tmp_path):
    """Two sessions, both with 2 rois, tracked into tmp_path/tracking; the session files are returned in order."""
    files = []
    for session in range(2):
        path = tmp_path / "sessions" / f"day{session}" / "results.hdf5"
        path.parent.mkdir(parents=True)
        with h5py.File(path, "w") as f:
            f.create_dataset("DemixingResults/temporal_demixed", data=np.zeros((10, 2), np.float32))
        files.append(str(path))
    results = {
        "clusters": {"labels_bySession": [np.arange(2), np.arange(2)]},
        "ROIs": {"ROIs_raw": [scipy.sparse.csr_matrix((2, 16))] * 2, "frame_height": 4, "frame_width": 4},
    }
    run = RoicatTrackingResults(results=results, run_data={}, session_files=tuple(files)).to_roicat_dir(tmp_path / "tracking")
    return run, files


def test_a_manifest_opens_exactly_the_files_it_names_and_refuses_others(tracked, tmp_path, capsys):
    from masknmf import cli

    run, files = tracked
    cli.main(["view", str(run.parent), "--list"])
    folders = [Path(line.strip()).name for line in capsys.readouterr().out.splitlines() if line.strip().endswith(("day0", "day1"))]
    assert folders == ["day0", "day1"]
    # files after a manifest are refused, even when the sessions have equal roi counts
    with pytest.raises(SystemExit):
        cli.main(["view", str(run.parent), files[1], files[0], "--list"])
    assert "lists the sessions' results files" in capsys.readouterr().err


def test_an_old_folder_takes_its_files_in_the_order_given(tracked, tmp_path, capsys):
    from masknmf import cli

    run, files = tracked
    run.with_name(f"{run.name}-manifest.json").unlink()
    cli.main(["view", str(run), files[1], files[0], "--list"])
    folders = [Path(line.strip()).name for line in capsys.readouterr().out.splitlines() if line.strip().endswith(("day0", "day1"))]
    assert folders == ["day1", "day0"]

    compression_only = tmp_path / "compression.hdf5"
    with h5py.File(compression_only, "w") as f:
        f.create_group("CompressionArray")
    with pytest.raises(SystemExit):
        cli.main(["view", str(run), files[0], str(compression_only), "--list"])
    assert "holds no DemixingResults" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        cli.main(["view", str(run), files[0], str(tmp_path / "missing.hdf5"), "--list"])
    assert "no such file" in capsys.readouterr().err


def test_demixing_results_are_not_replaced_in_place(tmp_path):
    path = tmp_path / "results.hdf5"
    with h5py.File(path, "w") as f:
        f.create_group("DemixingResults")
        f.create_group("global/DemixingResults")
    # export raises before it reads the instance, so an empty one is enough
    results = masknmf.DemixingResults.__new__(masknmf.DemixingResults)
    with pytest.raises(FileExistsError, match="already holds DemixingResults"):
        results.export(path)
    with pytest.raises(FileExistsError, match="already holds global/DemixingResults"):
        results.export(path, prefix="global")
