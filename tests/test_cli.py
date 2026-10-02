"""masknmf run: the log file a run leaves and the log level it runs at."""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Literal

import h5py
import numpy as np
import pytest
import tifffile

from masknmf import cli, pipelines
from masknmf.pipelines._base import BasePipeline


class FolderPipeline(BasePipeline):
    """Makes its run folder, logs one line at each level and fails when asked to."""

    def __init__(self,
                 output_folder: str | Path | None = None,
                 frame_batch_size: int = 300,
                 device: Literal["auto", "cuda", "cpu"] = "auto",
                 log_level: Literal["debug", "info", "warning"] = "info"):
        super().__init__(output_folder=output_folder, frame_batch_size=frame_batch_size, device=device,
                         log_level=log_level)

    @classmethod
    def default_configs(cls) -> dict:
        return {}

    def run(self, data: np.ndarray | None = None, frame_rate: float = 1.0, fail: bool = False) -> Path:
        self.run_config = {"frame_rate": frame_rate}
        folder = self.create_run_folder()
        logging.getLogger("masknmf").debug("a debug line")
        logging.getLogger("masknmf").info("an info line")
        logging.getLogger("masknmf").warning("a warning line")
        # a five frame movie stands for one a run breaks on
        if fail or (data is not None and data.shape[0] == 5):
            raise RuntimeError("the run broke")
        return self.finish()


@pytest.fixture
def folder_pipeline(monkeypatch):
    monkeypatch.setattr(pipelines, "FolderPipeline", FolderPipeline, raising=False)
    monkeypatch.setattr(pipelines, "__all__", [*pipelines.__all__, "FolderPipeline"])
    yield
    for handler in [h for h in logging.getLogger("masknmf").handlers if isinstance(h, logging.FileHandler)]:
        logging.getLogger("masknmf").removeHandler(handler)
        handler.close()
    logging.getLogger("masknmf").setLevel(logging.INFO)


def test_a_failed_run_keeps_its_log_where_its_folder_was(folder_pipeline, tmp_path):
    with pytest.raises(SystemExit):
        cli.main(["run", "--pipeline", "folder", "--output-folder", str(tmp_path), "--fail", "true"])
    log, = tmp_path.iterdir()
    assert log.suffix == ".log"
    text = log.read_text()
    assert "run failed" in text and "RuntimeError: the run broke" in text


def test_log_level_selects_what_the_log_holds_and_round_trips_through_config(folder_pipeline, tmp_path):
    cli.main(["run", "--pipeline", "folder", "--output-folder", str(tmp_path / "first"), "--log-level", "warning"])
    first, = (tmp_path / "first").iterdir()
    text = (first / f"{first.name}.log").read_text()
    assert "a warning line" in text and "an info line" not in text and "a debug line" not in text
    assert json.loads((first / "config.json").read_text())["configs"]["log_level"] == "warning"

    cli.main(["run", "--config", str(first / "config.json"), "--output-folder", str(tmp_path / "second")])
    second, = (tmp_path / "second").iterdir()
    text = (second / f"{second.name}.log").read_text()
    assert "a warning line" in text and "an info line" not in text

    cli.main(["run", "--config", str(first / "config.json"), "--output-folder", str(tmp_path / "third"),
              "--log-level", "debug"])
    third, = (tmp_path / "third").iterdir()
    text = (third / f"{third.name}.log").read_text()
    assert "a debug line" in text and "an info line" in text and "a warning line" in text
    assert "FolderPipeline" in text and "done in 0:00:00" in text


def test_the_movie_a_run_read_is_kept_in_its_config(folder_pipeline, tmp_path):
    movie = tmp_path / "movie.tif"
    tifffile.imwrite(movie, np.zeros((2, 8, 8), dtype=np.uint16))
    cli.main(["run", "--pipeline", "folder", str(movie), "--output-folder", str(tmp_path / "out")])
    folder, = (tmp_path / "out").iterdir()
    inputs = json.loads((folder / "config.json").read_text())["inputs"]
    modified = datetime.fromtimestamp(movie.stat().st_mtime).isoformat(timespec="seconds")
    assert inputs == {"data": {"path": str(movie.resolve()), "name": "movie.tif", "bytes": movie.stat().st_size,
                               "modified": modified, "shape": [2, 8, 8], "dtype": "uint16"}}


def test_an_hdf5_movie_is_kept_with_its_dataset(folder_pipeline, tmp_path):
    movie = tmp_path / "movie.h5"
    with h5py.File(movie, "w") as file:
        file.create_dataset("mov", data=np.zeros((2, 8, 8), dtype=np.float32))
    cli.main(["run", "--pipeline", "folder", str(movie), "--dataset", "mov", "--output-folder", str(tmp_path / "out")])
    folder, = (tmp_path / "out").iterdir()
    inputs = json.loads((folder / "config.json").read_text())["inputs"]
    assert inputs["data"]["dataset"] == "mov" and inputs["data"]["dtype"] == "float32"


def test_a_run_records_its_command_device_and_end(folder_pipeline, tmp_path):
    cli.main(["run", "--pipeline", "folder", "--output-folder", str(tmp_path)])
    folder, = tmp_path.iterdir()
    run = json.loads((folder / "config.json").read_text())["run"]
    assert run["command"].startswith("masknmf run --pipeline folder --output-folder ") and str(tmp_path) in run["command"]
    assert run["device"] in ("cpu", "cuda") and run["status"] == "done"
    assert run["finished"] >= run["started"] and run["seconds"] >= 0


def test_several_movies_run_one_after_another_and_a_failed_one_does_not_stop_the_rest(folder_pipeline, tmp_path, capsys):
    movies = tmp_path / "movies"
    movies.mkdir()
    for name, frames in [("a.tif", 2), ("b.tif", 5), ("c.tif", 2)]:
        tifffile.imwrite(movies / name, np.zeros((frames, 8, 8), dtype=np.uint16))
    with pytest.raises(SystemExit):
        cli.main(["run", "--pipeline", "folder", str(movies / "*.tif")])
    folders = sorted(p for p in movies.iterdir() if p.is_dir())
    names = [json.loads((f / "config.json").read_text())["inputs"]["data"]["name"] for f in folders]
    assert sorted(names) == ["a.tif", "c.tif"]
    assert len(list(movies.glob("*.log"))) == 1
    out = capsys.readouterr().out
    assert "2 of 3 runs done" in out and f"failed  {movies / 'b.tif'}" in out


def test_a_config_reruns_with_its_frame_rate_unless_one_is_given_and_writes_beside_the_new_movie(folder_pipeline,
                                                                                                 tmp_path):
    for name in ["first", "second", "third"]:
        (tmp_path / name).mkdir()
        tifffile.imwrite(tmp_path / name / "movie.tif", np.zeros((2, 8, 8), dtype=np.uint16))
    cli.main(["run", "--pipeline", "folder", str(tmp_path / "first" / "movie.tif"), "--fs", "7.5"])
    first, = (p for p in (tmp_path / "first").iterdir() if p.is_dir())

    cli.main(["run", "--config", str(first / "config.json"), str(tmp_path / "second" / "movie.tif")])
    second, = (p for p in (tmp_path / "second").iterdir() if p.is_dir())
    assert json.loads((second / "config.json").read_text())["configs"]["frame_rate"] == 7.5

    cli.main(["run", "--config", str(first / "config.json"), str(tmp_path / "third" / "movie.tif"), "--fs", "30"])
    third, = (p for p in (tmp_path / "third").iterdir() if p.is_dir())
    assert json.loads((third / "config.json").read_text())["configs"]["frame_rate"] == 30


def test_results_globs_expand_without_sidecars_or_zarr_stores(tmp_path, capsys):
    for relative in ["a/results.hdf5", "a/results.labels.hdf5", "a/movie.tif", "b/results.hdf5",
                     "b/deep/results.hdf5", "movie.zarr/0/results.hdf5", "b/raw.zarr/0/0/results.hdf5"]:
        (tmp_path / relative).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / relative).touch()

    assert cli.expand_results([str(tmp_path / "*" / "*.hdf5")]) == [
        str(tmp_path / "a" / "results.hdf5"), str(tmp_path / "b" / "results.hdf5")
    ]
    assert cli.expand_results([str(tmp_path / "*" / "*" / "*.hdf5")]) == [str(tmp_path / "b" / "deep" / "results.hdf5")]
    assert "skipped 2 .zarr store(s)" in capsys.readouterr().out

    assert cli.expand_results([str(tmp_path / "**" / "results.hdf5")]) == [
        str(tmp_path / "a" / "results.hdf5"), str(tmp_path / "b" / "results.hdf5"), str(tmp_path / "b" / "deep" / "results.hdf5")
    ]
    assert "skipped 2 .zarr store(s)" in capsys.readouterr().out

    with pytest.raises(SystemExit):
        cli.expand_results([str(tmp_path / "*" / "*.h5")])


def test_a_glob_takes_each_results_files_newest_curated_file_and_a_named_file_as_given(tmp_path, capsys):
    for relative in ["a/results.hdf5", "a/results.2026-09-30-10-00-00.curated.hdf5", "a/results.2026-09-30-11-00-00.curated.hdf5",
                     "a/results.2026-09-30-11-00-00.curated.labels.hdf5", "b/results.calcium.hdf5",
                     "b/results.glutamate.hdf5", "b/results.calcium.2026-09-30-10-00-00.curated.hdf5"]:
        (tmp_path / relative).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / relative).touch()

    assert cli.expand_results([str(tmp_path / "*" / "*.hdf5")]) == [
        str(tmp_path / "a" / "results.2026-09-30-11-00-00.curated.hdf5"),
        str(tmp_path / "b" / "results.calcium.2026-09-30-10-00-00.curated.hdf5"),
        str(tmp_path / "b" / "results.glutamate.hdf5"),
    ]
    assert "3 file(s) left out" in capsys.readouterr().out
    assert cli.expand_results([str(tmp_path / "*" / "results.hdf5")]) == [str(tmp_path / "a" / "results.hdf5")]
    named = str(tmp_path / "a" / "results.2026-09-30-10-00-00.curated.hdf5")
    assert cli.expand_results([named]) == [named]


def test_view_takes_a_folders_results_files_each_ones_newest_curated_file(tmp_path, capsys):
    # brackets in the folder's name are glob characters
    run = tmp_path / "run [day 1]"
    run.mkdir()
    for name in ["results.hdf5", "results.2026-09-30-10-00-00.curated.hdf5", "results.2026-09-30-11-00-00.curated.hdf5",
                 "results.2026-09-30-11-00-00.curated.labels.hdf5", "movie.hdf5"]:
        h5py.File(run / name, "w").close()

    cli.main(["view", str(run), "--list"])
    out = capsys.readouterr().out
    assert "2 file(s) left out" in out
    assert [line for line in out.splitlines() if line.endswith(".hdf5")] == [str(run / "results.2026-09-30-11-00-00.curated.hdf5")]

    (tmp_path / "empty").mkdir()
    with pytest.raises(SystemExit):
        cli.main(["view", str(tmp_path / "empty"), "--list"])
    assert "no results file matches" in capsys.readouterr().err


def test_view_opens_several_results_only_to_classify_them(tmp_path, capsys):
    for name in ["a", "b"]:
        (tmp_path / name).mkdir()
        h5py.File(tmp_path / name / "results.hdf5", "w").close()
    pattern = str(tmp_path / "*" / "results.hdf5")

    with pytest.raises(SystemExit):
        cli.main(["view", pattern])
    assert "the demixing viewer opens one results file, got 2" in capsys.readouterr().err

    with pytest.raises(SystemExit):
        cli.main(["view", pattern, "--classify"])
    assert "no results file holds DemixingResults" in capsys.readouterr().err
