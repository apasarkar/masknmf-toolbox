"""Good frames are saved with the results, reused on resume, and infilled to the raw frame axis."""

import h5py
import numpy as np
import pytest
import torch

from masknmf.arrays import OphysArray
from masknmf.compression import CompressionArray
from masknmf.io import OpenedResults, read_good_frames
from masknmf.motion_correction.gradient_strategy import GradientRegistrationArray
from masknmf.motion_correction.piecewise_rigid_strategy import PiecewiseRigidRegistrationArray
from masknmf.motion_correction.rigid_strategy import RigidMotionCorrector, RigidRegistrationArray
from masknmf.pipelines import TwoPhotonCalciumPipeline, WidefieldSinglechannelPipeline
from masknmf.pipelines.subcellular import GlutamateCalciumSpinePipeline

SHAPE = (30, 32, 40)


@pytest.fixture(scope="module")
def movie():
    return np.random.default_rng(0).random(SHAPE).astype(np.float32) + 1


@pytest.fixture(scope="module")
def good_frames():
    good = np.ones(SHAPE[0], dtype=bool)
    good[:5] = False
    good[12:15] = False
    return good


def registration(kind: str, ophys: OphysArray):
    rng = np.random.default_rng(1)
    num_frames = ophys.shape[0]
    if kind == "rigid":
        return RigidRegistrationArray(ophys, torch.as_tensor(rng.normal(size=(num_frames, 2))), compute_device="cpu")
    if kind == "piecewise":
        return PiecewiseRigidRegistrationArray(ophys, torch.as_tensor(rng.normal(size=(num_frames, 2, 2, 2))), compute_device="cpu")
    gradient = torch.as_tensor(rng.normal(size=(*SHAPE[1:], 2)))
    return GradientRegistrationArray(ophys, torch.as_tensor(rng.normal(size=(num_frames, 2))), gradient, compute_device="cpu")


@pytest.mark.parametrize("negative_indicator", [False, True])
def test_infilled_movie_is_zero_at_the_frames_left_out(movie, good_frames, negative_indicator):
    ophys = OphysArray(movie, negative_indicator=negative_indicator, good_frames=good_frames)
    infilled = ophys.infilled()
    assert infilled.shape == SHAPE
    assert np.array_equal(infilled.frame_indices, np.arange(SHAPE[0]))
    frames = infilled[:]
    assert torch.equal(frames[good_frames], ophys[:])
    assert (frames[~good_frames] == 0).all()
    assert torch.equal(infilled[[0, 6, 13]], frames[[0, 6, 13]])
    assert torch.equal(infilled[3:8, 2:5, 1:3], frames[3:8, 2:5, 1:3])
    assert torch.allclose(infilled.mean_image, ophys.mean_image)


def test_good_frames_survive_a_file(tmp_path, movie, good_frames):
    path = tmp_path / "results.hdf5"
    assert read_good_frames(path) is None
    OphysArray(movie, negative_indicator=False, good_frames=good_frames).export(path)
    assert np.array_equal(read_good_frames(path), good_frames)


def test_a_stored_mean_image_is_reused_instead_of_computed(tmp_path, movie, good_frames, monkeypatch):
    path = tmp_path / "results.hdf5"
    OphysArray(movie, negative_indicator=False, good_frames=good_frames).export(path)
    with h5py.File(path, "r") as f:
        assert "mean_image" not in f["OphysArray"]

    ophys = OphysArray(movie, negative_indicator=True, good_frames=good_frames)
    registration(kind="rigid", ophys=ophys).infill_bad_frames().export(path)
    ophys.export(path)
    with h5py.File(path, "r") as f:
        assert np.allclose(f["OphysArray/mean_image"][()], movie[good_frames].mean(axis=0), atol=1e-5)

    monkeypatch.setattr(OphysArray, "_compute_mean", lambda self: pytest.fail("the mean image was computed again"))
    loaded = RigidRegistrationArray.from_hdf5(path, input_movie=movie)
    assert torch.allclose(loaded.input_movie[:][good_frames], ophys[:], atol=1e-5)


def test_a_gradient_registration_takes_a_movie_that_is_not_an_ophys_array(movie):
    shifts = torch.zeros(SHAPE[0], 2)
    array = GradientRegistrationArray(movie, shifts, torch.zeros(*SHAPE[1:], 2), compute_device="cpu")
    assert torch.equal(array.shifts, shifts)


def test_a_glutamate_resume_from_a_file_without_good_frames_says_so(tmp_path):
    calcium = np.random.default_rng(0).random((300, 16, 16)).astype(np.float32)
    earlier = tmp_path / "earlier" / "results.calcium.hdf5"
    earlier.parent.mkdir()
    RigidRegistrationArray(calcium[200:], torch.zeros(100, 2), compute_device="cpu").export(earlier)
    with pytest.raises(ValueError, match="stores no good frames"):
        GlutamateCalciumSpinePipeline(output_folder=str(tmp_path), device="cpu").run(None, calcium, resume=earlier)


@pytest.mark.parametrize("kind", ["rigid", "piecewise", "gradient"])
@pytest.mark.parametrize("include_mean", [True, False])
def test_infilled_registration_is_zero_at_the_frames_left_out(movie, good_frames, kind, include_mean):
    if kind != "gradient" and not include_mean:
        pytest.skip("only the gradient array has a mean-subtracted view")
    kept = registration(kind=kind, ophys=OphysArray(movie, negative_indicator=False, good_frames=good_frames))
    infilled = kept.infill_bad_frames()
    if kind == "gradient":
        kept.include_mean = infilled.include_mean = include_mean
    assert infilled.shape == SHAPE
    frames = infilled[:]
    assert torch.allclose(frames[good_frames], kept[:], atol=1e-5)
    assert (frames[~good_frames] == 0).all()
    assert (infilled.shifts[~torch.as_tensor(good_frames)] == 0).all()
    assert infilled.infill_bad_frames() is infilled


def test_a_registration_of_every_frame_is_returned_as_it_is(movie):
    whole = registration(kind="rigid", ophys=OphysArray(movie, negative_indicator=False))
    assert whole.infill_bad_frames() is whole


def test_a_glutamate_run_stores_its_good_frames_and_a_resumed_run_registers_the_same_ones(tmp_path):
    rng = np.random.default_rng(0)
    yy, xx = np.mgrid[:40, :60]
    dendrite = np.exp(-((yy - 20) ** 2) / 8.0)
    calcium = (10 + 5 * dendrite + rng.normal(0, 0.3, (600, 40, 60))).astype(np.float32)
    good_frames = np.arange(600) >= 200

    pipeline = GlutamateCalciumSpinePipeline(output_folder=str(tmp_path), device="cpu")
    registered = pipeline.run(None, calcium, stop_after="registration") / "results.calcium.hdf5"
    assert np.array_equal(read_good_frames(registered), good_frames)
    with h5py.File(registered, "r") as f:
        assert "retained_frames" not in f
        shifts = f["RigidRegistrationArray/shifts"][()]
    assert shifts.shape[0] == 600 and (shifts[:200] == 0).all()

    opened = OpenedResults.open(registered, raw=calcium)
    assert tuple(opened.raw.shape) == tuple(opened.results.shape) == (600, 40, 60)
    assert (opened.results[:2] == 0).all()
    assert np.array_equal(opened.raw[:2], calcium[:2])

    compressed = pipeline.run(None, calcium, stop_after="compression", resume=registered) / "results.calcium.hdf5"
    assert np.array_equal(read_good_frames(compressed), good_frames)
    with h5py.File(registered, "r") as earlier, h5py.File(compressed, "r") as resumed:
        assert np.array_equal(earlier["RigidRegistrationArray/shifts"][()], resumed["RigidRegistrationArray/shifts"][()])
    assert CompressionArray.from_hdf5(compressed).shape[0] == 400


@pytest.mark.parametrize("negative_indicator", [False, True])
def test_a_stored_registration_loads_on_the_whole_recording(tmp_path, movie, good_frames, negative_indicator):
    path = tmp_path / "results.hdf5"
    ophys = OphysArray(movie, negative_indicator=negative_indicator, good_frames=good_frames)
    kept = registration(kind="gradient", ophys=ophys)
    kept.infill_bad_frames().export(path)
    ophys.export(path)

    on_recording = GradientRegistrationArray.from_hdf5(path, input_movie=movie)
    assert on_recording.shape == SHAPE
    assert on_recording.input_movie.negative_indicator == negative_indicator
    frames = on_recording[:]
    assert torch.allclose(frames[good_frames], kept[:], atol=1e-5)
    assert (frames[~good_frames] == 0).all()

    on_good_frames = GradientRegistrationArray.from_hdf5(path, input_movie=ophys)
    assert on_good_frames.input_movie is ophys
    assert torch.allclose(on_good_frames[:], kept[:], atol=1e-5)


@pytest.mark.parametrize("leaves_frames_out", [False, True])
def test_a_negative_indicator_registration_opens_in_the_polarity_the_run_registered(tmp_path, movie, good_frames, leaves_frames_out):
    path = tmp_path / "results.hdf5"
    good = good_frames if leaves_frames_out else None
    ophys = OphysArray(movie, negative_indicator=True, good_frames=good)
    corrector = RigidMotionCorrector(max_shifts=(2, 2), device="cpu")
    corrector.compute_template(ophys)
    kept = corrector.motion_correct(ophys)
    kept.infill_bad_frames().export(path)
    ophys.export(path)

    opened = OpenedResults.open(path, raw=movie)
    assert tuple(opened.results.shape) == SHAPE
    assert opened.results.input_movie.negative_indicator
    kept_frames = slice(None) if good is None else good
    assert torch.allclose(torch.as_tensor(opened.results[:])[kept_frames], kept[:], atol=1e-5)
    assert np.array_equal(opened.raw[:2], movie[:2])


def test_a_resumed_run_registers_the_frames_the_earlier_run_kept(tmp_path):
    rng = np.random.default_rng(0)
    yy, xx = np.mgrid[:32, :32]
    blob = np.exp(-((yy - 16) ** 2 + (xx - 16) ** 2) / 10.0)
    recording = (10 + 5 * blob + rng.normal(0, 0.3, (300, 32, 32))).astype(np.float32)
    good = np.arange(300) >= 50

    pipeline = TwoPhotonCalciumPipeline(output_folder=str(tmp_path), device="cpu")
    earlier = pipeline.run(recording, frame_rate=30, stop_after="registration", good_frames=good) / "results.hdf5"
    resumed = pipeline.run(recording, frame_rate=30, stop_after="compression", resume=earlier) / "results.hdf5"
    assert np.array_equal(read_good_frames(resumed), good)
    assert CompressionArray.from_hdf5(resumed).shape[0] == 250
    with h5py.File(resumed, "r") as f:
        shifts = f["RigidRegistrationArray/shifts"][()]
    assert shifts.shape[0] == 300 and (shifts[:50] == 0).all() and (shifts[50:] != 0).any()
    with pytest.raises(ValueError, match="differs from the good frames"):
        pipeline.run(recording, frame_rate=30, stop_after="compression", resume=earlier, good_frames=np.roll(good, 10))


def test_good_frame_indicator_still_works_and_warns_that_it_is_deprecated(tmp_path):
    recording = np.random.default_rng(0).random((60, 32, 32)).astype(np.float32) + 1
    good = np.arange(60) >= 10
    pipeline = WidefieldSinglechannelPipeline(output_folder=str(tmp_path), device="cpu")
    with pytest.warns(DeprecationWarning, match="use good_frames"):
        results = pipeline.run(recording, stop_after="registration", good_frame_indicator=good) / "results.hdf5"
    assert np.array_equal(read_good_frames(results), good)
    with pytest.raises(TypeError, match="pass good_frames only"):
        pipeline.run(recording, stop_after="registration", good_frames=good, good_frame_indicator=good)


def test_a_resumed_widefield_run_infills_its_compression_with_the_earlier_runs_good_frames(tmp_path):
    rng = np.random.default_rng(0)
    yy, xx = np.mgrid[:32, :32]
    blob = np.exp(-((yy - 16) ** 2 + (xx - 16) ** 2) / 10.0)
    recording = (10 + 5 * blob + rng.normal(0, 0.3, (300, 32, 32))).astype(np.float32)
    good = np.arange(300) >= 50

    pipeline = WidefieldSinglechannelPipeline(output_folder=str(tmp_path), device="cpu")
    earlier = pipeline.run(recording, stop_after="registration", good_frames=good) / "results.hdf5"
    resumed = pipeline.run(recording, resume=earlier) / "results.hdf5"
    compressed = CompressionArray.from_hdf5(resumed)
    assert compressed.shape[0] == 300
    # an infilled compression has zero temporal components at the frames left out, so those frames are its mean image
    assert torch.allclose(compressed[:50], compressed[:1].expand(50, -1, -1))
    assert not torch.allclose(compressed[50:52], compressed[:2])
