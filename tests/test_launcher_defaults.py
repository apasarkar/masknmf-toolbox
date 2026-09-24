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
from pathlib import Path
from typing import Literal, Optional

import numpy as np
import pytest

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
                 device: Literal["auto", "cuda", "cpu"] = "auto"):
        defaults = self.default_configs()
        self._stage_config = defaults['stage_config'] if stage_config is None else stage_config
        self._empty_config = defaults['empty_config'] if empty_config is None else empty_config
        super().__init__(output_folder, frame_batch_size, device)

    @classmethod
    def default_configs(cls) -> dict:
        with_template = OddInitConfig(template=np.ones((4, 4)))
        return {'stage_config': OddMultipassConfig([OddPassConfig(with_template), OddPassConfig(OtherInitConfig())]),
                'empty_config': OddMultipassConfig([])}

    @property
    def config(self):
        return {'stage_config': self._stage_config,
                'empty_config': self._empty_config,
                'output_folder': self.output_folder,
                'frame_batch_size': self.frame_batch_size,
                'device': self.device}

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
    assert "--config" not in window.build_argv()


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
        rebuilt = cli.section_value(section=section, kind=None, value_file=written[section.argument])[1]
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
