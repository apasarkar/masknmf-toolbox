"""The contract every pipeline keeps with BasePipeline."""

import inspect
import json
import logging

import h5py
import pytest

import masknmf
from masknmf.pipelines import scraper

SLUGS = sorted(scraper.pipeline_registry())


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


def test_a_resumed_run_keeps_the_inputs_of_the_run_whose_compression_it_reuses(tmp_path):
    cls = scraper.pipeline_registry()[SLUGS[0]]
    with h5py.File(tmp_path / "results.hdf5", "w") as file:
        file.create_group(masknmf.CompressionArray.__name__)
    earlier = {"inputs": {"data": {"path": "C:/movies/movie.tif", "name": "movie.tif"}},
               "configs": {"compress_config": "*"}, "timings": {"compression": {"seconds": 1.0}}}
    (tmp_path / "config.json").write_text(json.dumps(earlier))
    pipeline = cls(output_folder=str(tmp_path))
    pipeline.results_path(resume=True)
    written = json.loads((tmp_path / "config.json").read_text())
    assert written["inputs"] == earlier["inputs"]
    assert written["configs"]["compress_config"] == "*" and written["timings"] == earlier["timings"]
    logging.getLogger("masknmf").removeHandler(pipeline.log_handler)
    pipeline.log_handler.close()


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
