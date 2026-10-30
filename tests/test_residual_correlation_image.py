"""The global residual correlation image comes from the base class, for any results with a residual movie."""

import numpy as np
import pytest
import torch

from masknmf.demixing._base_results import BaseResults
from masknmf.demixing.local_correlation import local_correlation_image
from masknmf.demixing.signal_demixer import get_local_correlation_structure
from masknmf.io import OpenedResults
from test_adapter_sources import FRAMES, HEIGHT, WIDTH, write_caiman_file, write_suite2p_plane

MAD_THRESHOLD = 1


def movie_with_a_correlated_blob(frames: int = 60, height: int = 11, width: int = 13) -> np.ndarray:
    rng = np.random.default_rng(0)
    movie = rng.standard_normal((frames, height, width)).astype(np.float32)
    shared = rng.standard_normal(frames).astype(np.float32) * 3
    movie[:, 3:7, 4:9] += shared[:, None, None]
    return movie


def test_the_tiled_image_of_a_movie_matches_the_demixers_image_of_its_identity_factorization():
    movie = movie_with_a_correlated_blob()
    frames, height, width = movie.shape
    noise_std = torch.full((height, width), 0.7)
    u_identity = torch.eye(height * width).to_sparse_csr()
    v = torch.from_numpy(movie.reshape(frames, height * width).T.copy())
    expected = get_local_correlation_structure(u_identity, v, (height, width, frames), MAD_THRESHOLD, noise_std,
                                               batch_size=25)

    image = local_correlation_image(movie, noise_std=noise_std, mad_threshold=MAD_THRESHOLD, pixel_batch_size=25)
    assert image.shape == (height, width)
    np.testing.assert_allclose(image.numpy(), expected, atol=1e-5)
    # one tile and many overlapping tiles agree, since repeated pairs are averaged
    one_tile = local_correlation_image(movie, noise_std=noise_std, mad_threshold=MAD_THRESHOLD, pixel_batch_size=10000)
    np.testing.assert_allclose(one_tile.numpy(), image.numpy(), atol=1e-5)
    assert image[3:7, 4:9].min() > image[np.ix_(range(0, 2), range(0, 2))].max()


def test_without_a_noise_term_the_tenth_percentile_of_the_pixel_stds_is_used_everywhere():
    movie = movie_with_a_correlated_blob()
    baseline = np.quantile(movie.std(axis=0, dtype=np.float64), 0.1)
    expected = local_correlation_image(movie, noise_std=torch.full(movie.shape[1:], float(baseline)))
    np.testing.assert_allclose(local_correlation_image(movie).numpy(), expected.numpy(), atol=1e-4)


class ResultsWithAResidual(BaseResults):
    def __init__(self, residual):
        self._residual = residual

    @property
    def residual_array(self):
        return self._residual


def test_the_base_class_computes_the_image_from_the_residual_once_and_gives_none_without_one():
    movie = movie_with_a_correlated_blob()
    results = ResultsWithAResidual(movie)
    image = results.global_residual_correlation_image
    assert image.shape == movie.shape[1:] and results.global_residual_correlation_image is image
    np.testing.assert_allclose(image.numpy(), local_correlation_image(movie).numpy())
    assert ResultsWithAResidual(None).global_residual_correlation_image is None


def test_suite2p_and_caiman_results_have_the_image_when_they_have_a_residual(tmp_path):
    write_suite2p_plane(tmp_path / "plane0")
    image = OpenedResults.open(tmp_path / "plane0").results.global_residual_correlation_image
    assert image.shape == (HEIGHT, WIDTH) and torch.isfinite(image).all() and image.abs().max() <= 1
    write_suite2p_plane(tmp_path / "plane1", with_bin=False)
    assert OpenedResults.open(tmp_path / "plane1").results.global_residual_correlation_image is None

    write_caiman_file(tmp_path / "cnmfe.hdf5", with_ring_model=True)
    opened = OpenedResults.open(tmp_path / "cnmfe.hdf5", raw=np.zeros((FRAMES, HEIGHT, WIDTH), np.float32))
    assert opened.results.global_residual_correlation_image.shape == (HEIGHT, WIDTH)
    write_caiman_file(tmp_path / "cnmf.hdf5")
    assert OpenedResults.open(tmp_path / "cnmf.hdf5").results.global_residual_correlation_image is None
