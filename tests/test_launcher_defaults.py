"""
The launcher and the command line against every pipeline's default_configs(): an untouched window
shows exactly the defaults and passes no --config, resets bring them back, and awkward defaults
render without breaking or being mistaken for changes.
"""

import contextlib
import copy
import dataclasses
import io
import json
import logging
from functools import partial
from pathlib import Path
from typing import Literal, Optional

import h5py
import numpy as np
import pytest
from imgui_bundle import imgui

from masknmf import cli, launcher
from masknmf.pipelines import scraper
from masknmf.pipelines._base import BasePipeline

SLUGS = sorted(scraper.pipeline_registry())


@dataclasses.dataclass
class OddInitConfig:
    threshold: float = 1
    size: int = None
    pair: tuple[int, int] = dataclasses.field(default_factory=lambda: [3, 4])
    schedule: tuple[float, float] = (0.9, 0.8, 0.7)
    sign: Literal["positive", "negative"] = "positive"
    template: Optional[np.ndarray] = None


@dataclasses.dataclass
class OtherInitConfig:
    sigma: float = 2.0


@dataclasses.dataclass
class OddPassConfig:
    InitConfig: OddInitConfig | OtherInitConfig
    extra: Optional[OtherInitConfig] = None


@dataclasses.dataclass
class OddMultipassConfig:
    passes: list[OddPassConfig]


class OddPipeline(BasePipeline):
    def __init__(self,
                 stage_config: OddMultipassConfig | Literal["skip"] | None = None,
                 empty_config: OddMultipassConfig | None = None,
                 output_folder: str | Path | None = None,
                 frame_batch_size: int = 300,
                 device: Literal["auto", "cuda", "cpu"] = "auto",
                 log_level: Literal["debug", "info", "warning"] = "info"):
        super().__init__(output_folder=output_folder, frame_batch_size=frame_batch_size, device=device,
                         log_level=log_level, stage_config=stage_config, empty_config=empty_config)

    @classmethod
    def default_configs(cls) -> dict:
        with_template = OddInitConfig(template=np.ones((4, 4)))
        return {'stage_config': OddMultipassConfig([OddPassConfig(with_template), OddPassConfig(OtherInitConfig())]),
                'empty_config': OddMultipassConfig([])}

    def run(self, data: np.ndarray | None, frame_rate: float) -> Path:
        return Path(".")


@pytest.fixture
def odd_registry(monkeypatch):
    registry = {**scraper.pipeline_registry(), "odd": OddPipeline}
    monkeypatch.setattr(scraper, "pipeline_registry", lambda: registry)
    return registry


def render_frames(window: launcher.Launcher, frames: int = 4) -> None:
    """Draw the launcher for a few frames on hello_imgui's null backend."""
    import imgui_data_loader as idl
    from imgui_bundle import hello_imgui, imgui

    idl.ensure_assets()
    dialog = idl.FileDialog(window.config)
    count = {"n": 0}

    def gui():
        dialog.render()
        count["n"] += 1
        if count["n"] >= frames:
            hello_imgui.get_runner_params().app_shall_exit = True

    def enable_textures():
        imgui.get_io().backend_flags |= imgui.BackendFlags_.renderer_has_textures

    params = hello_imgui.RunnerParams()
    params.platform_backend_type = hello_imgui.PlatformBackendType.null
    params.renderer_backend_type = hello_imgui.RendererBackendType.null
    params.ini_folder_type = hello_imgui.IniFolderType.temp_folder
    params.ini_filename = "masknmf_launcher_test.ini"
    params.callbacks.post_init = enable_textures
    params.callbacks.show_gui = gui
    hello_imgui.run(params)
    assert count["n"] >= frames


def assert_untouched(window: launcher.Launcher) -> None:
    """An untouched window holds exactly the pipeline's defaults and passes nothing for them."""
    defaults = window.spec.cls.default_configs()
    for section in window.spec.sections:
        assert launcher.same(window.values[section.argument], defaults[section.argument]), section.argument
    assert window.modified() == []
    assert window.errors == {}
    assert "--config" not in window.argv_run()


@pytest.mark.parametrize("slug", SLUGS)
def test_untouched_window_holds_default_configs(slug):
    assert_untouched(window=launcher.Launcher(slug_initial=slug))


@pytest.mark.parametrize("slug", SLUGS)
def test_drawing_the_untouched_window_changes_nothing(slug):
    window = launcher.Launcher(slug_initial=slug)
    render_frames(window=window)
    assert_untouched(window=window)


@pytest.mark.parametrize("slug", SLUGS)
def test_params_json_through_config_reproduces_the_defaults(slug, tmp_path, monkeypatch):
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        cli.main(["params", "--pipeline", slug, "--json"])
    spec = cli.spec_for(slug=slug)
    written = json.loads(out.getvalue())
    for section in spec.sections:
        rebuilt = cli.section_value(section=section, kind=None, value_file=written["configs"][section.argument])[1]
        assert launcher.same(rebuilt, section.default), section.argument


def test_resets_restore_the_defaults():
    window = launcher.Launcher(slug_initial="two-photon-calcium")
    spec = window.spec
    window.select_kind(section=spec.section("motion-correct"), kind="piecewise-rigid")
    window.values["compress_config"].max_components = 3
    passes = window.values["filtered_demixing_config"].DemixingConfigs
    passes[0].NMFConfig.maxiter = 1
    passes.append(copy.deepcopy(passes[-1]))
    window.texts["device"] = "cpu"
    assert len(window.modified()) == 5

    window.reset_field(config=window.values["compress_config"], name="max_components",
                       default=spec.section("compress").default.max_components, path="compress_config.max_components")
    assert window.values["compress_config"] == spec.section("compress").default
    window.reset_section(section=spec.section("filtered-demixing"))
    assert window.values["filtered_demixing_config"] == spec.section("filtered-demixing").default
    window.texts["frame-rate"] = "30"
    window.reset_all()
    assert window.texts["frame-rate"] == "30"
    assert_untouched(window=window)


def test_awkward_defaults_render_unchanged_and_unmodified(odd_registry):
    window = launcher.Launcher(slug_initial="odd")
    assert_untouched(window=window)
    render_frames(window=window)
    assert_untouched(window=window)
    template = window.values["stage_config"].passes[0].InitConfig.template
    assert isinstance(template, np.ndarray) and template.shape == (4, 4)


def test_switching_a_nested_config_and_back_restores_its_default(odd_registry):
    window = launcher.Launcher(slug_initial="odd")
    config_pass = window.values["stage_config"].passes[0]
    default_pass = window.spec.section("stage").default.passes[0]
    window.select_nested(config=config_pass, name="InitConfig", kind="other-init", default=default_pass.InitConfig,
                         path="stage_config.passes[0].InitConfig")
    assert isinstance(config_pass.InitConfig, OtherInitConfig)
    assert window.modified() != []
    window.select_nested(config=config_pass, name="InitConfig", kind="odd-init", default=default_pass.InitConfig,
                         path="stage_config.passes[0].InitConfig")
    assert launcher.same(config_pass.InitConfig, default_pass.InitConfig)
    assert window.modified() == []


def test_load_run_fills_the_window_from_a_runs_config_json(tmp_path):
    cls = scraper.pipeline_registry()["two-photon-calcium"]
    pipeline = cls(output_folder=str(tmp_path), frame_batch_size=123, device="cpu")
    pipeline.run_config = {"frame_rate": 7.5, "stop_after": "registration"}
    folder = pipeline.create_run_folder()
    pipeline.inputs = {"data": {"path": str(tmp_path / "movie.h5"), "dataset": "mov"}}
    pipeline.write_config()
    logging.getLogger("masknmf").removeHandler(pipeline.log_handler)
    pipeline.log_handler.close()

    window = launcher.Launcher(slug_initial="widefield-singlechannel")
    assert window.load_run(source=str(folder / "results.hdf5")) is None
    assert window.spec.slug == "two-photon-calcium" and window.loaded_run == str(folder)
    assert window.paths["data"] == str(tmp_path / "movie.h5") and window.dataset == "mov"
    assert window.texts["frame-rate"] == "7.5" and window.texts["stop-after"] == "registration"
    assert window.texts["frame-batch-size"] == "123" and window.texts["device"] == "cpu"
    assert window.texts["output-folder"] == str(tmp_path.resolve())
    assert {row[0] for row in window.modified()} == {"stop_after", "output_folder", "frame_batch_size", "device"}
    for section in window.spec.sections:
        assert launcher.same(window.values[section.argument], cls.default_configs()[section.argument]), section.argument
    argv = window.argv_run()
    assert argv[:3] == ["run", "--pipeline", "two-photon-calcium"] and str(tmp_path / "movie.h5") in argv
    assert "--dataset" in argv and "--config" not in argv
    assert window.load_run(source=str(tmp_path)) == f"no config.json in {tmp_path}"


def test_load_run_reads_a_config_json_from_before_configs_were_nested(tmp_path):
    flat = {"masknmf_version": "0.1.0", "pipeline": "TwoPhotonCalciumPipeline", "frame_batch_size": 77,
            "compress_config": {"kind": "compress", "max_components": 9}, "output_folder": str(tmp_path)}
    (tmp_path / "config.json").write_text(json.dumps(flat))
    window = launcher.Launcher(slug_initial="widefield-singlechannel")
    assert window.load_run(source=str(tmp_path / "config.json")) is None
    assert window.spec.slug == "two-photon-calcium" and window.texts["frame-batch-size"] == "77"
    assert scraper.kind_of(value=window.values["compress_config"]) == "compress"
    assert window.values["compress_config"].max_components == 9


def test_start_from_a_loaded_runs_compression_resumes_it_and_back_to_the_movie_restores_the_config(tmp_path):
    cls = scraper.pipeline_registry()["two-photon-calcium"]
    pipeline = cls(output_folder=str(tmp_path), device="cpu")
    folder = pipeline.create_run_folder()
    logging.getLogger("masknmf").removeHandler(pipeline.log_handler)
    pipeline.log_handler.close()
    with h5py.File(folder / "results.hdf5", "w") as file:
        for name in ("RigidMotionCorrector", "RigidRegistrationArray", "CompressionArray"):
            file.create_group(name)
    window = launcher.Launcher()
    assert window.load_run(source=str(folder)) is None
    assert window.starts_available() == ["registration", "compression"]
    window.select_start(start="compression")
    argv = window.argv_run()
    assert argv[argv.index("--resume-from") + 1] == str(folder / "results.hdf5") and "--config" in argv
    assert window.values["compress_config"] == "skip"
    window.select_start(start="movie")
    assert launcher.same(window.values["compress_config"], cls.default_configs()["compress_config"])
    assert "--resume-from" not in window.argv_run()


def test_tracking_rows_grow_and_build_a_track_command(tmp_path):
    window = launcher.Launcher()
    window.page = "track"
    window.paths_tracking["results_0"] = str(tmp_path / "*" / "results.hdf5")
    assert window.problems() == ["choose the folder the tracking is saved in"]
    render_frames(window=window)
    assert window.paths_tracking["results_1"] == ""
    window.paths_tracking["out"] = str(tmp_path / "tracking")
    assert window.problems() == []
    args = cli.build_parser(spec=None).parse_args(window.build_argv())
    assert args.handler is cli.command_track
    assert args.results == [str(tmp_path / "*" / "results.hdf5")]
    assert args.out == str(tmp_path / "tracking") and args.um_per_pixel == 1.2


def click_card(original, slug: str, label: str, *args, **kwargs) -> bool:
    """imgui.invisible_button reporting a click on one pipeline's card."""
    return original(label, *args, **kwargs) or label == f"##card_pipeline_{slug}"


def test_switching_pipeline_from_a_card_mid_frame_draws_on(monkeypatch):
    window = launcher.Launcher(slug_initial="two-photon-calcium")
    window.page = "run"
    monkeypatch.setattr(imgui, "invisible_button", partial(click_card, imgui.invisible_button, "widefield-singlechannel"))
    render_frames(window=window)
    assert window.spec.slug == "widefield-singlechannel"
    assert_untouched(window=window)
