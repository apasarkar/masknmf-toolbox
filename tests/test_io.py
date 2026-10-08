"""masknmf.io: opening results files of any stage with their movies, and the run folder a run writes them in."""

import json
import logging

import h5py
import numpy as np
import pytest
import tifffile
import torch

import masknmf
from masknmf.io import OpenedResults, create_run_folder, drop_group, has_group, log_to, stage_groups, write_run_config


@pytest.fixture(scope="module")
def movie():
    rng = np.random.default_rng(0)
    yy, xx = np.mgrid[:32, :32]
    cell = np.exp(-((yy - 16) ** 2 + (xx - 16) ** 2) / 10.0)
    trace = np.convolve((rng.random(300) < 0.03) * 3.0, np.exp(-np.arange(30) / 6.0))[:300]
    return (10 + trace[:, None, None] * cell + rng.normal(0, 0.2, (300, 32, 32))).astype(np.float32)


@pytest.fixture(scope="module")
def run(movie, tmp_path_factory):
    """A run folder written without a pipeline: registration and compression in one results.hdf5, the movie in the same folder."""
    folder = tmp_path_factory.mktemp("run")
    corrector = masknmf.RigidMotionCorrector(max_shifts=(2, 2), device="cpu")
    corrector.compute_template(movie)
    registration = corrector.motion_correct(movie)
    registration.export(folder / "results.hdf5")
    strategy = masknmf.CompressStrategy(block_sizes=(16, 16), frame_batch_size=100, device="cpu")
    strategy.compress(registration).export(folder / "results.hdf5")
    tifffile.imwrite(folder / "movie.tif", movie)
    return folder


def test_a_results_file_opens_with_the_movie_beside_it_its_registration_and_shifts(run, movie):
    opened = OpenedResults.open(run / "results.hdf5")
    assert isinstance(opened.results, masknmf.CompressionArray)
    assert opened.raw_source == run / "movie.tif"
    assert isinstance(opened.registered, masknmf.RigidRegistrationArray)
    assert opened.shifts.shape == (movie.shape[0], 2)
    assert opened.template.shape == movie.shape[1:]
    assert opened.skipped == []


def test_a_given_raw_movie_that_does_not_line_up_is_skipped_by_open_and_raises_in_resolve(run, movie):
    opened = OpenedResults.open(run / "results.hdf5", raw=movie[:100])
    assert opened.raw_source == run / "movie.tif"
    assert len(opened.skipped) == 1
    with pytest.raises(ValueError, match="raw movie has shape"):
        OpenedResults.resolve(opened.results, run / "results.hdf5", raw=movie[:100])


def test_results_given_as_data_find_nothing_without_a_path(run):
    results = masknmf.CompressionArray.from_hdf5(run / "results.hdf5")
    opened = OpenedResults.resolve(results)
    assert (opened.raw, opened.registered, opened.shifts, opened.template) == (None, None, None, None)


def test_a_registration_only_file_needs_its_movie(movie, tmp_path):
    corrector = masknmf.RigidMotionCorrector(max_shifts=(2, 2), device="cpu")
    corrector.compute_template(movie)
    corrector.motion_correct(movie).export(tmp_path / "results.hdf5")
    with pytest.raises(ValueError, match="holds only RigidRegistrationArray"):
        OpenedResults.open(tmp_path / "results.hdf5")
    opened = OpenedResults.open(tmp_path / "results.hdf5", raw=movie)
    assert isinstance(opened.results, masknmf.RigidRegistrationArray)
    assert opened.registered is None
    assert opened.shifts.shape == (movie.shape[0], 2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a cuda device")
def test_the_registration_is_applied_on_the_device_asked_for(run):
    assert str(OpenedResults.open(run / "results.hdf5", device="cuda").registered.strategy.device).startswith("cuda")


def test_a_registration_replaces_another_kind_but_not_one_later_stages_were_computed_from(run, movie, tmp_path):
    rigid = masknmf.RigidMotionCorrector(max_shifts=(2, 2), device="cpu")
    rigid.compute_template(movie)
    rigid.motion_correct(movie).export(tmp_path / "results.hdf5")
    piecewise = masknmf.PiecewiseRigidMotionCorrector(minimum_patch_sizes=(16, 16), max_rigid_shifts=(2, 2),
                                                      max_deviation_rigid=(1, 1), device="cpu")
    piecewise.compute_template(movie)
    piecewise.motion_correct(movie).export(tmp_path / "results.hdf5")
    with h5py.File(tmp_path / "results.hdf5", "r") as f:
        assert sorted(f) == ["PiecewiseRigidMotionCorrector", "PiecewiseRigidRegistrationArray"]
    with pytest.raises(ValueError, match="holds CompressionArray"):
        rigid.motion_correct(movie).export(run / "results.hdf5")
    assert stage_groups(run / "results.hdf5") == ["RigidRegistrationArray", "CompressionArray"]


def test_only_demixing_results_take_a_prefix(run, tmp_path):
    results = masknmf.CompressionArray.from_hdf5(run / "results.hdf5")
    with pytest.raises(TypeError):
        results.export(tmp_path / "results.hdf5", prefix="global")


def test_drop_group_removes_one_group_and_keeps_the_rest(run, tmp_path):
    path = tmp_path / "results.hdf5"
    path.write_bytes((run / "results.hdf5").read_bytes())
    drop_group(path, "CompressionArray")
    assert not has_group(path, "CompressionArray")
    assert has_group(path, "RigidRegistrationArray")
    assert not has_group(tmp_path / "missing.hdf5", "CompressionArray")


def test_run_folders_started_in_the_same_second_get_a_suffix(tmp_path, monkeypatch):
    monkeypatch.setattr(masknmf.io, "get_timestamp", lambda: "20261007T120000")
    assert create_run_folder(tmp_path, "by-hand") == tmp_path / "20261007T120000_by-hand"
    assert create_run_folder(tmp_path, "by-hand") == tmp_path / "20261007T120000_by-hand_1"


def test_the_run_record_and_log_land_in_the_run_folder(tmp_path):
    folder = create_run_folder(tmp_path, "by-hand")
    handler = log_to(folder)
    logging.getLogger("masknmf").setLevel("INFO")
    logging.getLogger("masknmf").info("by hand")
    handler.flush()
    assert "by hand" in (folder / f"{folder.name}.log").read_text()
    path = write_run_config(folder, "by-hand", {"frame_rate": 30.0, "compress": masknmf.CompressConfig()},
                            timings={"compression": {"seconds": 1.0}},
                            default=masknmf.pipelines.scraper.config_json_value)
    record = json.loads(path.read_text())
    assert record["pipeline"] == "by-hand"
    assert record["configs"]["compress"]["kind"] == "compress"
    assert record["timings"] == {"compression": {"seconds": 1.0}}
    later = log_to(create_run_folder(tmp_path, "later"))
    assert handler not in logging.getLogger("masknmf").handlers
    logging.getLogger("masknmf").removeHandler(later)
    later.close()
