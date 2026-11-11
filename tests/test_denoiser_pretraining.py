"""
A denoiser trained on several recordings' compressions and saved compresses another recording without being
trained again: from Python, from the command line, and from the launcher.
"""

import json

import numpy as np
import pytest
import torch

import masknmf
from masknmf import cli, launcher
from masknmf.pipelines import scraper
from masknmf.pipelines.configs import CompressConfig, CompressDenoiseConfig
from masknmf.pipelines.twophoton_calcium import TwoPhotonCalciumPipeline

from test_launcher_defaults import render_frames


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


@pytest.fixture(scope="module")
def filepaths_results(movie, tmp_path_factory):
    """Two recordings of different lengths, compressed without a denoiser, as results files."""
    folder = tmp_path_factory.mktemp("compressions")
    filepaths = []
    for i, num_frames in enumerate((1000, 800)):
        compression = masknmf.compression.CompressStrategy(block_sizes=(16, 16), device="cpu").compress(movie[:num_frames])
        filepaths.append(str(folder / f"results_{i}.hdf5"))
        compression.export(filepaths[-1])
    return filepaths


def test_a_saved_denoiser_loads_with_the_same_weights(tmp_path):
    model = masknmf.compression.denoising.TotalVarianceTemporalDenoiser()
    filepath = str(tmp_path / "denoiser.pt")
    masknmf.compression.save_denoiser(model, filepath)
    loaded = masknmf.compression.load_denoiser(filepath)
    for name, weights in model.state_dict().items():
        assert torch.equal(weights, loaded.state_dict()[name])


def test_pretraining_on_results_files_saves_weights_the_strategy_compresses_with(movie, filepaths_results, tmp_path, monkeypatch):
    filepath = str(tmp_path / "denoiser.pt")
    model = masknmf.compression.pretrain_denoiser(filepaths_results, filepath, num_epochs=1)
    loaded = masknmf.compression.load_denoiser(filepath)
    for name, weights in model.state_dict().items():
        assert torch.equal(weights.cpu(), loaded.state_dict()[name])

    monkeypatch.setattr(masknmf.compression.denoising, "train_total_variance_denoiser", raise_if_trained)
    strategy = masknmf.compression.CompressDenoiseStrategy(block_sizes=(16, 16), device="cpu", filepath_denoiser=filepath)
    results = strategy.compress(movie)
    assert results.shape == movie.shape
    assert np.isfinite(np.asarray(results[:])).all()


def test_a_pipeline_takes_the_weights_from_its_config_or_its_denoiser_property(tmp_path):
    pipeline = TwoPhotonCalciumPipeline(compress_config=CompressDenoiseConfig(filepath_denoiser="a.pt"),
                                        output_folder=str(tmp_path), device="cpu")
    assert pipeline.denoiser == "a.pt"
    pipeline.denoiser = "b.pt"
    assert pipeline.compress_strategy(pipeline.compress_config).filepath_denoiser == "b.pt"

    without_denoiser = TwoPhotonCalciumPipeline(compress_config=CompressConfig(), output_folder=str(tmp_path), device="cpu")
    assert without_denoiser.denoiser is None
    with pytest.raises(ValueError, match="compress-denoise"):
        without_denoiser.denoiser = "b.pt"


def test_the_command_line_trains_on_results_files_and_run_takes_the_weights(filepaths_results, tmp_path):
    filepath = str(tmp_path / "denoiser.pt")
    cli.main(["train-denoiser", *filepaths_results, "--out", filepath, "--epochs", "1"])
    assert masknmf.compression.load_denoiser(filepath) is not None

    spec = cli.spec_for(slug="two-photon-calcium")
    assert "denoiser" in [param.field for param in spec.properties]
    args = cli.build_parser(spec=spec).parse_args(["run", "--pipeline", "two-photon-calcium", "movie.tif", "--denoiser", filepath])
    assert args.denoiser == filepath


def test_the_command_line_refuses_results_compressed_with_a_denoiser(filepaths_results, tmp_path, capsys):
    record = {"pipeline": "TwoPhotonCalciumPipeline", "configs": {"compress_config": {"kind": "compress-denoise"}}}
    (tmp_path / "config.json").write_text(json.dumps(record))
    filepath = tmp_path / "results.hdf5"
    filepath.write_bytes(open(filepaths_results[0], "rb").read())
    with pytest.raises(SystemExit):
        cli.main(["train-denoiser", str(filepath), "--out", str(tmp_path / "denoiser.pt")])
    assert "compressed with a denoiser" in capsys.readouterr().err


def test_the_launcher_page_builds_a_train_denoiser_command(tmp_path):
    window = launcher.Launcher()
    window.page = "denoiser"
    window.paths_denoiser["results_0"] = str(tmp_path / "*" / "results.hdf5")
    assert window.problems() == ["choose the file the weights are saved in"]
    render_frames(window=window)
    assert window.paths_denoiser["results_1"] == ""
    window.paths_denoiser["out"] = str(tmp_path / "denoiser.pt")
    window.epochs_denoiser = 50
    assert window.problems() == []
    args = cli.build_parser(spec=None).parse_args(window.build_argv())
    assert args.handler is cli.command_train_denoiser
    assert args.results == [str(tmp_path / "*" / "results.hdf5")]
    assert args.out == str(tmp_path / "denoiser.pt") and args.epochs == 50


def test_the_launcher_run_page_passes_the_weights_in_its_config(tmp_path, monkeypatch):
    monkeypatch.setattr(launcher, "FILEPATH_RUN_CONFIGS", tmp_path / "launcher_configs.json")
    window = launcher.Launcher(slug_initial="two-photon-calcium")
    window.page = "run"
    param = next(p for p in scraper.scrape_dataclass(cls_config=CompressDenoiseConfig, name_section="compress_config")
                 if p.field == "filepath_denoiser")
    assert param.settable
    window.values["compress_config"].filepath_denoiser = str(tmp_path / "denoiser.pt")
    render_frames(window=window)
    argv = window.build_argv()
    configs = json.loads(open(argv[argv.index("--config") + 1]).read())["configs"]
    assert configs["compress_config"]["filepath_denoiser"] == str(tmp_path / "denoiser.pt")
