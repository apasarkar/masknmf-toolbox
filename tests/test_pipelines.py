"""The contract every pipeline keeps with BasePipeline."""

import inspect
import json
import logging
import time
from pathlib import Path

import h5py
import numpy as np
import pytest

import masknmf
from masknmf.pipelines import scraper
from masknmf.pipelines._base import BasePipeline

SLUGS = sorted(scraper.pipeline_registry())


class BrokenPipeline(BasePipeline):
    """Makes its run folder and breaks."""

    def __init__(self, output_folder: str | Path | None = None):
        super().__init__(output_folder=output_folder)

    @classmethod
    def default_configs(cls) -> dict:
        return {}

    def run(self) -> None:
        self.create_run_folder()
        raise RuntimeError("the run broke")


def shifted_blob(num_frames: int = 40) -> np.ndarray:
    """A bright blob wandering a couple of pixels over frames of noise: something to register in a fraction of a second."""
    rng = np.random.default_rng(0)
    yy, xx = np.mgrid[:48, :48]
    frames = []
    for t in range(num_frames):
        dy, dx = round(2 * np.sin(t / 5)), round(2 * np.cos(t / 7))
        frames.append(100 * np.exp(-((yy - 24 - dy) ** 2 + (xx - 24 - dx) ** 2) / 30) + rng.normal(0, 1, (48, 48)))
    return np.stack(frames).astype(np.float32)


def close_log(pipeline: BasePipeline) -> None:
    logging.getLogger("masknmf").removeHandler(pipeline.log_handler)
    pipeline.log_handler.close()


@pytest.mark.parametrize("slug", SLUGS)
def test_config_reports_every_init_argument_with_defaults_filled_in(slug):
    cls = scraper.pipeline_registry()[slug]
    pipeline = cls()
    names = [name for name in inspect.signature(cls.__init__).parameters if name != "self"]
    assert list(pipeline.config) == names
    defaults = cls.default_configs()
    assert set(defaults) <= set(names)
    for name, default in defaults.items():
        assert getattr(pipeline, name) == default


@pytest.mark.parametrize("slug", SLUGS)
def test_a_given_config_is_kept_and_output_folder_is_resolved(slug, tmp_path):
    cls = scraper.pipeline_registry()[slug]
    name, default = next(iter(cls.default_configs().items()))
    pipeline = cls(**{name: default, "output_folder": str(tmp_path)})
    assert getattr(pipeline, name) is default
    assert pipeline.output_folder == tmp_path.resolve()


@pytest.mark.parametrize("slug", SLUGS)
def test_log_level_sets_the_logger_and_a_run_folder_gets_a_log_beside_its_config(slug, tmp_path):
    cls = scraper.pipeline_registry()[slug]
    pipeline = cls(output_folder=str(tmp_path), log_level="debug")
    assert logging.getLogger("masknmf").level == logging.DEBUG
    folder = pipeline.create_run_folder()
    assert json.loads((folder / "config.json").read_text())["configs"]["log_level"] == "debug"
    logging.getLogger("masknmf").debug("a debug line")
    text = (folder / f"{folder.name}.log").read_text()
    assert masknmf.__version__ in text and cls.__name__ in text
    assert "DEBUG" in text and "a debug line" in text
    logging.getLogger("masknmf").removeHandler(pipeline.log_handler)
    pipeline.log_handler.close()
    cls()
    assert logging.getLogger("masknmf").level == logging.INFO


def test_step_logs_its_start_and_how_long_it_took_or_that_it_failed(tmp_path):
    cls = scraper.pipeline_registry()[SLUGS[0]]
    pipeline = cls(output_folder=str(tmp_path))
    folder = pipeline.create_run_folder()
    with pipeline.step("a quick step"):
        pass
    with pytest.raises(RuntimeError):
        with pipeline.step("a broken step"):
            raise RuntimeError("broken")
    text = (folder / f"{folder.name}.log").read_text()
    assert "INFO masknmf.pipelines._base: a quick step\n" in text
    assert "a quick step done in 0:00:00" in text
    assert "ERROR masknmf.pipelines._base: a broken step failed after 0:00:00" in text
    logging.getLogger("masknmf").removeHandler(pipeline.log_handler)
    pipeline.log_handler.close()


def test_a_resumed_run_copies_what_it_reuses_into_a_new_folder_and_keeps_the_earlier_runs_record(tmp_path):
    cls = scraper.pipeline_registry()[SLUGS[0]]
    earlier_folder = tmp_path / "earlier"
    earlier_folder.mkdir()
    with h5py.File(earlier_folder / "results.hdf5", "w") as file:
        for name in ("RigidMotionCorrector", "RigidRegistrationArray", masknmf.CompressionArray.__name__,
                     masknmf.DemixingResults.__name__):
            file.create_group(name)
        file["RigidRegistrationArray"]["shifts"] = np.zeros((3, 2))
    earlier = {"inputs": {"data": {"path": "C:/movies/movie.tif", "name": "movie.tif"}},
               "configs": {"compress_config": "*", "frame_batch_size": 7},
               "timings": {"motion correction": {"seconds": 2.0}, "compression": {"seconds": 1.0}, "demixing": {}}}
    (earlier_folder / "config.json").write_text(json.dumps(earlier))
    pipeline = cls(output_folder=str(tmp_path))
    resume_from, base = pipeline.resume_source(None, reuse_compression=False)
    assert (resume_from, base) == (None, None)
    pipeline.output_folder = earlier_folder
    resume_from, base = pipeline.resume_source(None, reuse_compression=True)
    assert resume_from == (earlier_folder / "results.hdf5").resolve() and base == tmp_path.resolve()
    results_path = pipeline.results_path(base)
    stored = pipeline.resume(resume_from, results_path, reuse_compression=True)
    assert stored is masknmf.RigidRegistrationArray
    folder = pipeline.run_folder
    assert folder.parent == tmp_path.resolve() and folder != earlier_folder
    with h5py.File(results_path) as file:
        assert set(file) == {"RigidMotionCorrector", "RigidRegistrationArray", masknmf.CompressionArray.__name__}
    with h5py.File(earlier_folder / "results.hdf5") as file:
        assert masknmf.DemixingResults.__name__ in file
    written = json.loads((folder / "config.json").read_text())
    assert written["inputs"] == earlier["inputs"]
    assert written["configs"]["compress_config"] == "*" and written["configs"]["frame_batch_size"] == 300
    assert list(written["timings"]) == ["motion correction", "compression"]
    assert json.loads((earlier_folder / "config.json").read_text()) == earlier
    close_log(pipeline)


def test_resume_source_refuses_a_file_without_what_is_reused(tmp_path):
    pipeline = scraper.pipeline_registry()[SLUGS[0]](output_folder=str(tmp_path))
    with h5py.File(tmp_path / "results.hdf5", "w") as file:
        file.create_group("RigidRegistrationArray")
    with pytest.raises(ValueError):
        pipeline.resume_source(tmp_path / "results.hdf5", reuse_compression=True)
    assert pipeline.resume_source(tmp_path / "results.hdf5", reuse_compression=False)[0] == (tmp_path / "results.hdf5").resolve()
    with pytest.raises(FileNotFoundError):
        pipeline.resume_source(tmp_path / "missing.hdf5", reuse_compression=False)


def test_a_run_that_breaks_from_python_records_that_it_failed_and_why(tmp_path):
    pipeline = BrokenPipeline(output_folder=str(tmp_path))
    with pytest.raises(RuntimeError):
        pipeline.run()
    folder, = tmp_path.iterdir()
    assert json.loads((folder / "config.json").read_text())["run"]["status"] == "failed"
    log = (folder / f"{folder.name}.log").read_text()
    assert "BrokenPipeline failed" in log and "RuntimeError: the run broke" in log
    close_log(pipeline)


def test_finish_records_when_the_run_ended_and_how(tmp_path):
    cls = scraper.pipeline_registry()[SLUGS[0]]
    pipeline = cls(output_folder=str(tmp_path))
    folder = pipeline.create_run_folder()
    assert json.loads((folder / "config.json").read_text())["run"]["status"] == "running"
    assert pipeline.finish("failed") == folder
    run = json.loads((folder / "config.json").read_text())["run"]
    assert run["status"] == "failed" and run["finished"] >= run["started"] and run["seconds"] >= 0
    assert run["command"] is None and run["device"] == str(pipeline.torch_device)
    logging.getLogger("masknmf").removeHandler(pipeline.log_handler)
    pipeline.log_handler.close()


def test_a_long_step_logs_that_it_is_still_running(tmp_path):
    cls = scraper.pipeline_registry()[SLUGS[0]]
    pipeline = cls(output_folder=str(tmp_path))
    pipeline.heartbeat_seconds = 0.05
    folder = pipeline.create_run_folder()
    with pipeline.step("a slow step"):
        time.sleep(0.2)
    text = (folder / f"{folder.name}.log").read_text()
    assert "a slow step still running after 0:00:00" in text
    logging.getLogger("masknmf").removeHandler(pipeline.log_handler)
    pipeline.log_handler.close()


def test_a_step_records_its_peak_cuda_memory_only_on_a_cuda_device(tmp_path):
    cls = scraper.pipeline_registry()[SLUGS[0]]
    for device in ("cpu", "auto"):
        pipeline = cls(output_folder=str(tmp_path / device), device=device)
        folder = pipeline.create_run_folder()
        with pipeline.step("a step"):
            pass
        timing = json.loads((folder / "config.json").read_text())["timings"]["a step"]
        assert ("peak_cuda_gb" in timing) == str(pipeline.torch_device).startswith("cuda")
        logging.getLogger("masknmf").removeHandler(pipeline.log_handler)
        pipeline.log_handler.close()


def test_stop_after_registration_writes_the_shifts_and_resume_from_replays_them(tmp_path):
    movie = shifted_blob()
    pipeline = masknmf.TwoPhotonCalciumPipeline(output_folder=str(tmp_path), device="cpu")
    folder = pipeline.run(movie, frame_rate=30, stop_after="registration")
    with h5py.File(folder / "results.hdf5") as file:
        assert set(file) == {"RigidMotionCorrector", "RigidRegistrationArray"}
        shifts = file["RigidRegistrationArray/shifts"][()]
    config = json.loads((folder / "config.json").read_text())
    assert config["configs"]["stop_after"] == "registration" and config["run"]["status"] == "done"
    assert list(config["timings"]) == ["motion correction"]

    replayed = pipeline.run(movie, frame_rate=30, stop_after="registration", resume_from=folder / "results.hdf5")
    assert replayed != folder
    with h5py.File(replayed / "results.hdf5") as file:
        assert np.array_equal(file["RigidRegistrationArray/shifts"][()], shifts)
    assert json.loads((replayed / "config.json").read_text())["configs"]["resume_from"] == str(folder / "results.hdf5")
    close_log(pipeline)


def test_one_photon_saves_its_gradient_registration_and_replays_it(tmp_path):
    # its template is the mean of the first 300 frames
    movie = shifted_blob(num_frames=320)
    active = np.ones(movie.shape[0], dtype=bool)
    pipeline = masknmf.OnePhotonCulturePipeline(output_folder=str(tmp_path), device="cpu")
    folder = pipeline.run(movie, frame_rate=30, indicator_sign="positive", active_frames=active, stop_after="registration")
    with h5py.File(folder / "results.hdf5") as file:
        assert set(file) == {"GradientMotionCorrector", "GradientRegistrationArray"}
        steps = file["GradientRegistrationArray/gradient_steps"][()]
    replayed = pipeline.run(movie, frame_rate=30, indicator_sign="positive", active_frames=active,
                            stop_after="registration", resume_from=folder / "results.hdf5")
    with h5py.File(replayed / "results.hdf5") as file:
        assert np.array_equal(file["GradientRegistrationArray/gradient_steps"][()], steps)
    close_log(pipeline)


def test_stop_after_refuses_runs_with_nothing_to_write(tmp_path):
    movie = np.zeros((4, 16, 16), dtype=np.float32)
    skipped = masknmf.TwoPhotonCalciumPipeline(output_folder=str(tmp_path), motion_correct_config="skip")
    with pytest.raises(ValueError):
        skipped.run(movie, frame_rate=30, stop_after="registration")
    resumed = masknmf.TwoPhotonCalciumPipeline(output_folder=str(tmp_path), compress_config="skip")
    with pytest.raises(ValueError):
        resumed.run(None, frame_rate=30, stop_after="compression")
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("slug", SLUGS)
def test_from_config_builds_the_pipeline_a_config_json_describes(slug, tmp_path):
    cls = scraper.pipeline_registry()[slug]
    pipeline = cls(output_folder=str(tmp_path), frame_batch_size=123, device="cpu")
    folder = pipeline.create_run_folder()
    logging.getLogger("masknmf").removeHandler(pipeline.log_handler)
    pipeline.log_handler.close()
    rebuilt = cls.from_config(folder / "config.json", device="auto")
    assert rebuilt.frame_batch_size == 123 and rebuilt.device == "auto"
    for name, default in cls.default_configs().items():
        assert getattr(rebuilt, name) == default, name
    other = next(c for s, c in scraper.pipeline_registry().items() if s != slug)
    with pytest.raises(ValueError):
        other.from_config(folder / "config.json")
