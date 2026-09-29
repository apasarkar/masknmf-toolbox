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

    def run(self, data: np.ndarray | None = None, fail: bool = False) -> Path:
        folder = self.create_run_folder()
        logging.getLogger("masknmf").debug("a debug line")
        logging.getLogger("masknmf").info("an info line")
        logging.getLogger("masknmf").warning("a warning line")
        if fail:
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
