"""Interleaved channel arrays, compressed-domain hemocorrection and the widefield hemocorrection pipeline."""

import json

import h5py
import numpy as np
import pytest
import torch

import masknmf
from masknmf.arrays import InterleavedChannelArray, load_interleaved_binary
from masknmf.hemocorrection import (align_temporal, hemocorrect, interleaved_frame_positions,
                                    pixelwise_coefficients, HemocorrectionResults)


# ---------------------------------------------------------------- arrays
def test_interleaved_views_of_4d_and_3d_data_agree_and_drop_an_incomplete_cycle():
    rng = np.random.default_rng(0)
    four_d = rng.integers(0, 1000, size=(7, 2, 5, 6)).astype(np.uint16)
    three_d = np.concatenate([four_d.reshape(14, 5, 6), four_d[:1, 0]])  # one stray frame at the end
    for channel in (0, 1):
        a = InterleavedChannelArray(four_d, channel=channel)
        b = InterleavedChannelArray(three_d, channel=channel)
        assert a.shape == b.shape == (7, 5, 6)
        np.testing.assert_array_equal(a[:], four_d[:, channel])
        np.testing.assert_array_equal(b[:], four_d[:, channel])
        np.testing.assert_array_equal(b[2:6:2], four_d[2:6:2, channel])
        np.testing.assert_array_equal(b[[5, 1, 3]], four_d[[5, 1, 3], channel])
        np.testing.assert_array_equal(a[3], four_d[3:4, channel])  # an int keeps the frame axis
        np.testing.assert_array_equal(a[1:4, 2:4, 1], four_d[1:4, channel, 2:4, 1:2])
        np.testing.assert_array_equal(a.raw_frame_indices(), np.arange(7) * 2 + channel)


def test_interleaved_array_checks_its_arguments():
    data = np.zeros((4, 2, 3, 3))
    with pytest.raises(ValueError):
        InterleavedChannelArray(data, channel=2)
    with pytest.raises(ValueError):
        InterleavedChannelArray(data, channel=0, num_channels=3)
    assert InterleavedChannelArray(data, channel=0, num_frames=3).shape == (3, 3, 3)


def test_load_interleaved_binary_reads_the_layout_from_a_wfield_file_name(tmp_path):
    data = np.arange(3 * 2 * 4 * 5, dtype=np.uint16).reshape(3, 2, 4, 5)
    path = tmp_path / "motioncorrect_2_4_5_uint16.bin"
    data.tofile(path)
    with open(path, "ab") as f:
        f.write(b"\x00\x01")  # a partial cycle is ignored
    loaded = load_interleaved_binary(path)
    assert loaded.shape == (3, 2, 4, 5) and loaded.dtype == np.uint16
    np.testing.assert_array_equal(loaded, data)
    with pytest.raises(ValueError):
        load_interleaved_binary(tmp_path / "frames.bin")


# ---------------------------------------------------------------- hemocorrection backend
def random_sparse(num_pixels, rank, density, generator):
    dense = torch.randn(num_pixels, rank, generator=generator)
    dense *= torch.rand(num_pixels, rank, generator=generator) < density
    return dense.to_sparse_coo()


@pytest.mark.parametrize("center", [False, True])
def test_coefficients_from_the_factors_match_regression_on_reconstructed_pixels(center):
    g = torch.Generator().manual_seed(0)
    uf, ur = random_sparse(250, 12, 0.3, g), random_sparse(250, 9, 0.3, g)
    vf, vr = torch.randn(12, 300, generator=g), torch.randn(9, 300, generator=g)
    coefficients = pixelwise_coefficients(uf, vf, ur, vr, center=center, eps=0.0, pixel_batch_size=17)
    f, r = (uf.to_dense() @ vf).double(), (ur.to_dense() @ vr).double()
    if center:
        f, r = f - f.mean(1, keepdim=True), r - r.mean(1, keepdim=True)
    denominator = (r * r).sum(1)
    supported = denominator > 0
    expected = (f * r).sum(1)[supported] / denominator[supported]
    torch.testing.assert_close(coefficients.double()[supported], expected, rtol=1e-4, atol=1e-4)
    assert torch.all(coefficients[~supported] == 0)


def test_align_temporal_interpolates_and_extrapolates_linearly():
    temporal = torch.tensor([[0.0, 2.0, 4.0], [1.0, 1.0, 1.0]])
    aligned = align_temporal(temporal, np.array([1.0, 3.0, 5.0]), np.array([0.0, 2.0, 4.0]))
    torch.testing.assert_close(aligned, torch.tensor([[-1.0, 1.0, 3.0], [1.0, 1.0, 1.0]]))
    np.testing.assert_array_equal(interleaved_frame_positions(3, 1, 2), [1.0, 3.0, 5.0])


def compression_of(movie: np.ndarray) -> masknmf.CompressionArray:
    """An exact CompressionArray of a small movie: identity spatial basis, normalized fluctuations as temporal factors."""
    num_frames, height, width = movie.shape
    flat = torch.from_numpy(movie.reshape(num_frames, -1).T.astype(np.float32))
    mean = flat.mean(1)
    scale = flat.std(1).clamp_min(1e-6)
    identity = torch.eye(height * width).to_sparse_coo()
    return masknmf.CompressionArray.from_tensors(
        shape=movie.shape, spatial_compressed=identity, temporal_compressed=(flat - mean[:, None]) / scale[:, None],
        mean_image=mean.reshape(height, width), noise_variance_image=scale.reshape(height, width),
        spatial_compressed_local_projector=identity)


@pytest.mark.parametrize("projection", ["local", "least_squares"])
def test_hemocorrect_recovers_the_coefficients_and_the_neural_signal(projection):
    rng = np.random.default_rng(1)
    num_frames, height, width = 400, 6, 7
    true_coefficients = rng.uniform(0.2, 1.5, (height, width))
    t = np.arange(num_frames)
    blood = np.sin(2 * np.pi * t / 37)[:, None, None] * rng.uniform(5, 10, (1, height, width))
    neural = (rng.random((num_frames, 1, 1)) < 0.05) * rng.uniform(1, 3, (1, height, width))
    functional = 100 + neural + true_coefficients * blood
    reference = 80 + blood
    corrected, results = hemocorrect(compression_of(functional), compression_of(reference), projection=projection,
                                     device="cpu")
    f, r = compression_of(functional), compression_of(reference)
    raw_coefficients = results.coefficients * (f.noise_variance_image / r.noise_variance_image).numpy()
    # the neural events are uncorrelated with the blood signal, so the fit is close to the truth
    np.testing.assert_allclose(raw_coefficients, true_coefficients, atol=0.05)
    out = corrected[:].numpy()
    remaining = out - out.mean(0) - (neural - neural.mean(0))
    assert np.abs(remaining).max() < 0.15 * np.abs(true_coefficients * blood).max()
    assert results.projection_relative_residual < 1e-3
    assert isinstance(corrected, masknmf.CompressionArray) and corrected.shape == functional.shape


def test_hemocorrect_needs_frame_times_when_the_frame_counts_differ():
    a = compression_of(np.random.default_rng(0).normal(size=(10, 3, 3)))
    b = compression_of(np.random.default_rng(1).normal(size=(9, 3, 3)))
    with pytest.raises(ValueError):
        hemocorrect(a, b, device="cpu")
    with pytest.raises(ValueError):
        hemocorrect(a, a, fit_highpass_hz=0.1, device="cpu")  # filtering needs a frame rate


def test_hemocorrection_results_round_trip_through_hdf5(tmp_path):
    results = HemocorrectionResults(coefficients=np.ones((3, 4)), projection_relative_residual=0.1,
                                    projection_retained_fraction=0.9, num_frames_fit=12)
    results.export(tmp_path / "r.hdf5")
    loaded = HemocorrectionResults.from_hdf5(str(tmp_path / "r.hdf5"))
    np.testing.assert_array_equal(loaded.coefficients, results.coefficients)
    assert loaded.num_frames_fit == 12 and loaded.projection_relative_residual == pytest.approx(0.1)


# ---------------------------------------------------------------- pipeline
def interleaved_movie(num_cycles=150, size=32, seed=0):
    """Functional channel 0 = neural blobs + 0.5 x the blood signal; channel 1 = the blood signal, sampled in between."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[:size, :size]
    raw_t = np.arange(2 * num_cycles)
    blood = np.sin(2 * np.pi * raw_t / 25)[:, None, None] * (20 + yy / 2)[None]
    events = np.convolve((rng.random(2 * num_cycles) < 0.03).astype(float), np.exp(-np.arange(20) / 4))[:2 * num_cycles]
    neural = 60 * events[:, None, None] * np.exp(-((yy - size / 2) ** 2 + (xx - size / 2) ** 2) / 20)[None]
    movie = np.empty((num_cycles, 2, size, size), np.float32)
    movie[:, 0] = 1000 + neural[0::2] + 0.5 * blood[0::2] + rng.normal(0, 1, (num_cycles, size, size))
    movie[:, 1] = 800 + blood[1::2] + rng.normal(0, 1, (num_cycles, size, size))
    return movie


def test_the_pipeline_writes_each_stage_and_removes_the_blood_signal(tmp_path):
    movie = interleaved_movie()
    config = masknmf.CompressConfig(block_sizes=(16, 16), max_components=8)
    pipeline = masknmf.WidefieldHemocorrectionPipeline(motion_correct_config="skip", compress_config=config,
                                                       reference_compress_config=config, output_folder=str(tmp_path),
                                                       device="cpu")
    mask = np.ones(movie.shape[2:])
    folder = pipeline.run(movie, functional_channel=0, pixel_weighting=mask)
    with h5py.File(folder / "results.hdf5") as f:
        assert {"CompressionArray", "HemocorrectionResults"} <= set(f)
    record = json.loads((folder / "config.json").read_text())
    assert record["run"]["status"] == "done"
    assert list(record["timings"]) == ["functional compression", "reference compression", "hemocorrection"]
    assert record["configs"]["functional_channel"] == 0 and record["configs"]["reference_channel"] == 1

    functional = masknmf.CompressionArray.from_hdf5(str(folder / "functional.hdf5"))
    reference = masknmf.CompressionArray.from_hdf5(str(folder / "reference.hdf5"))
    corrected = masknmf.CompressionArray.from_hdf5(str(folder / "results.hdf5"))
    results = HemocorrectionResults.from_hdf5(str(folder / "results.hdf5"))
    raw_coefficients = results.coefficients * (functional.noise_variance_image / reference.noise_variance_image).numpy()
    assert np.median(np.abs(raw_coefficients - 0.5)) < 0.05
    # a corner far from the neural blob holds only blood + noise: correction should flatten it
    before = functional[:, :6, :6].numpy().std(0).mean()
    after = corrected[:, :6, :6].numpy().std(0).mean()
    assert after < 0.2 * before


def test_the_pipeline_registers_both_channels_with_the_same_shifts(tmp_path):
    movie = interleaved_movie(num_cycles=60, size=32, seed=2)
    config = masknmf.CompressConfig(block_sizes=(16, 16), max_components=5)
    pipeline = masknmf.WidefieldHemocorrectionPipeline(compress_config=config, reference_compress_config=config,
                                                       output_folder=str(tmp_path), device="cpu",
                                                       frame_batch_size=20)
    folder = pipeline.run(movie, functional_channel=0, stop_after="registration")
    with h5py.File(folder / "functional.hdf5") as f, h5py.File(folder / "reference.hdf5") as r:
        np.testing.assert_array_equal(f["RigidRegistrationArray"]["shifts"][()], r["RigidRegistrationArray"]["shifts"][()])
    assert not (folder / "results.hdf5").exists() or "CompressionArray" not in h5py.File(folder / "results.hdf5")


def test_the_default_compression_trains_the_denoiser_for_30_epochs():
    pipeline = masknmf.WidefieldHemocorrectionPipeline()
    for config in (pipeline.compress_config, pipeline.reference_compress_config):
        assert isinstance(config, masknmf.CompressDenoiseConfig)
        assert config.num_epochs == 30 and tuple(config.block_sizes) == (100, 100)
