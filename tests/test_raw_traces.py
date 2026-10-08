"""The raw traces the glutamate/calcium pipeline stores beside each demixing result."""

import numpy as np
import pytest

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
        glutamate if with_glutamate else None, calcium)
    for channel in ["glutamate", "calcium"] if with_glutamate else ["calcium"]:
        path = run / f"results.{channel}.hdf5"
        compressed = np.asarray(masknmf.CompressionArray.from_hdf5(path)[:])
        for prefix in ["", "global"]:
            results = masknmf.DemixingResults.from_hdf5(path, prefix=prefix)
            assert results.temporal_demixed_raw.shape == results.temporal_demixed.shape
            assert np.isfinite(results.temporal_demixed_raw.numpy()).all()
            # the raw movie differs from the compressed one only by the noise compression removed
            from_compressed = masknmf.demixing.estimate_temporal_demixed_raw(results, compressed, device="cpu")
            np.testing.assert_allclose(from_compressed.numpy(), results.temporal_demixed.numpy(), atol=1e-4)
