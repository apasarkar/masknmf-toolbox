"""A denoiser saved after training compresses another recording without being trained again."""

import numpy as np
import pytest
import torch

import masknmf
from masknmf.pipelines.subcellular import GlutamateCalciumSpinePipeline
from masknmf.pipelines.configs import CompressDenoiseConfig


def raise_if_trained(*args, **kwargs):
    raise AssertionError("the denoiser was trained although pretrained weights were given")


@pytest.fixture(scope="module")
def movie():
    """Two blobs with slow transients over noise."""
    rng = np.random.default_rng(0)
    frames, height, width = 1000, 32, 32
    yy, xx = np.mgrid[:height, :width]
    blobs = np.array([np.exp(-((yy - 16) ** 2 + (xx - x) ** 2) / 8.0) for x in (8, 24)])
    kernel = np.exp(-np.arange(100) / 20.0)
    traces = np.stack([np.convolve((rng.random(frames) < 0.02) * 3.0, kernel)[:frames] for _ in blobs], axis=1)
    return (np.einsum("tk,kyx->tyx", traces, blobs) + rng.normal(0, 0.3, (frames, height, width))).astype(np.float32)


def test_a_saved_denoiser_loads_with_the_same_weights(tmp_path):
    model = masknmf.compression.denoising.TotalVarianceTemporalDenoiser()
    filepath = str(tmp_path / "denoiser.pt")
    masknmf.compression.save_denoiser(model, filepath)
    loaded = masknmf.compression.load_denoiser(filepath)
    for name, weights in model.state_dict().items():
        assert torch.equal(weights, loaded.state_dict()[name])


def test_the_strategy_compresses_with_saved_weights_and_does_not_train(movie, tmp_path, monkeypatch):
    filepath = str(tmp_path / "denoiser.pt")
    masknmf.compression.save_denoiser(masknmf.compression.denoising.TotalVarianceTemporalDenoiser(), filepath)
    monkeypatch.setattr(masknmf.compression.denoising, "train_total_variance_denoiser", raise_if_trained)
    strategy = masknmf.compression.CompressDenoiseStrategy(block_sizes=(16, 16), device="cpu", filepath_denoiser=filepath)
    results = strategy.compress(movie)
    assert results.shape == movie.shape
    assert np.isfinite(np.asarray(results[:])).all()


def test_a_pipeline_passes_the_config_weights_to_its_strategy(tmp_path):
    pipeline = GlutamateCalciumSpinePipeline(output_folder=str(tmp_path), device="cpu")
    strategy = pipeline.compress_strategy(CompressDenoiseConfig(filepath_denoiser="denoiser.pt"))
    assert strategy.filepath_denoiser == "denoiser.pt"
