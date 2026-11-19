"""The raw traces the glutamate/calcium pipeline stores beside each demixing result."""

import numpy as np
import pytest
import torch

import masknmf
from masknmf.pipelines.subcellular import GlutamateCalciumSpinePipeline


@pytest.fixture(scope="module")
def channels():
    """Three spines above a dendrite, slow transients that survive the compression's temporal averaging."""
    rng = np.random.default_rng(0)
    frames, height, width = 1500, 40, 60
    yy, xx = np.mgrid[:height, :width]
    dendrite = np.exp(-((yy - 20) ** 2) / 8.0)
    spines = np.array([np.exp(-((yy - 14) ** 2 + (xx - x) ** 2) / 6.0) for x in (12, 30, 48)])
    kernel = np.exp(-np.arange(200) / 40.0)
    traces = np.stack(
        [np.convolve((rng.random(frames) < 0.01) * rng.uniform(2, 4, frames), kernel)[:frames] for _ in spines], axis=1
    )
    global_trace = np.convolve((rng.random(frames) < 0.01) * 2.0, kernel)[:frames]
    spine_movie = np.einsum("tk,kyx->tyx", traces, spines)
    glutamate = 10 + 5 * dendrite + 3 * spine_movie + rng.normal(0, 0.3, (frames, height, width))
    calcium = 10 + 5 * dendrite + 3 * global_trace[:, None, None] * dendrite + 3 * spine_movie + rng.normal(0, 0.3, (frames, height, width))
    return glutamate.astype(np.float32), calcium.astype(np.float32)


@pytest.mark.parametrize("with_glutamate", [True, False], ids=["both channels", "calcium only"])
def test_every_demixing_result_stores_raw_traces_that_the_compressed_movie_turns_back_into_its_traces(
        channels, with_glutamate, tmp_path):
    glutamate, calcium = channels
    run = GlutamateCalciumSpinePipeline(output_folder=str(tmp_path), device="cpu").run(
        glutamate if with_glutamate else None, calcium, compute_raw_correlation=True)
    for channel in ["glutamate", "calcium"] if with_glutamate else ["calcium"]:
        path = run / f"results.{channel}.hdf5"
        compressed = np.asarray(masknmf.CompressionArray.from_hdf5(path)[:])  # good frames only
        retained = np.flatnonzero(masknmf.io.read_good_frames(path))
        for prefix in ["", "global"]:
            results = masknmf.DemixingResults.from_hdf5(path, prefix=prefix)  # infilled to the raw frame count
            assert results.temporal_demixed_raw.shape == results.temporal_demixed.shape
            assert results.raw_residual_correlation_image.shape == results.fov_shape
            assert np.isfinite(results.temporal_demixed_raw.numpy()).all()
            # put the compressed frames back at their raw frame indices so movie, background and traces line up
            movie = np.zeros((results.temporal_demixed.shape[0],) + compressed.shape[1:], dtype=np.float32)
            movie[retained] = compressed
            untouched = movie.copy()
            # the raw movie differs from the compressed one only by the noise compression removed
            from_compressed = masknmf.demixing.estimate_temporal_demixed_raw(results, movie, device="cpu").numpy()
            np.testing.assert_array_equal(movie, untouched)  # the caller's movie is not modified
            # an explicit noise image is the same as the results' own
            with_noise = masknmf.demixing.estimate_temporal_demixed_raw(
                results, movie, device="cpu", noise=results.noise_variance_image).numpy()
            np.testing.assert_array_equal(with_noise, from_compressed)
            # raw traces carry each signal's share of the mean image: the least-squares projection of
            # mean_image / noise_variance_image onto the footprints, one constant per signal
            footprints = results.spatial_demixed.to_dense().numpy()
            normalizer = results.noise_variance_image.numpy().ravel().copy()
            normalizer[normalizer <= 0] = 1.0
            baseline = np.linalg.lstsq(footprints, results.mean_image.numpy().ravel() / normalizer, rcond=None)[0]
            np.testing.assert_allclose(from_compressed[retained] - baseline, results.temporal_demixed.numpy()[retained],
                                       atol=1e-3)


def test_results_without_a_noise_image_are_estimated_at_the_scale_of_the_movie(tmp_path):
    from masknmf.io import OpenedResults
    from test_adapter_sources import FRAMES, write_suite2p_plane

    write_suite2p_plane(tmp_path / "plane0")
    results = OpenedResults.open(tmp_path / "plane0").results
    assert results.noise_variance_image is None
    movie = np.asarray(results.motion_corrected_data[:], dtype=np.float32)
    traces = masknmf.demixing.estimate_temporal_demixed_raw(results, movie, device="cpu")
    assert traces.shape == (FRAMES, results.temporal_demixed.shape[1]) and torch.isfinite(traces).all()
    ones = torch.ones(movie.shape[1] * movie.shape[2])
    with_ones = masknmf.demixing.estimate_temporal_demixed_raw(results, movie, device="cpu", noise=ones)
    np.testing.assert_array_equal(traces.numpy(), with_ones.numpy())
