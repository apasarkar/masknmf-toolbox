"""
The Parameters popup on its own, as another imgui app hosts it: config() round-trips through load() for every
pipeline, a changed value reaches config() and BasePipeline.from_config, and the popup draws open offscreen.
"""

import dataclasses
import json
import logging

import numpy as np
import pytest
from rendercanvas.offscreen import RenderCanvas
import wgpu
from wgpu.utils.imgui import ImguiRenderer

from masknmf.pipelines import scraper
from masknmf.visualization.imgui import parameters


SLUGS = sorted(scraper.pipeline_registry())


def first_number_field(popup: parameters.Parameters) -> tuple[str, str]:
    """Returns the section argument and field of the first int or float field of any stage config, for editing."""
    for section in popup.spec.sections:
        config = popup.values[section.argument]
        if not dataclasses.is_dataclass(config):
            continue
        for field in dataclasses.fields(config):
            value = getattr(config, field.name)
            if field.init and parameters.is_number(value=value):
                return section.argument, field.name
    raise ValueError(f"{popup.slug} has no numeric config field")


@pytest.mark.parametrize("slug", SLUGS)
def test_untouched_config_is_empty_and_round_trips(slug):
    popup = parameters.Parameters(slug=slug)
    config = popup.config()
    assert config == {"pipeline": popup.spec.cls.__name__, "configs": {}}
    popup.load(config=config)
    assert popup.modified() == [] and popup.problems() == [] and popup.runtime() == {}


@pytest.mark.parametrize("slug", SLUGS)
def test_a_changed_value_reaches_config_load_and_from_config(slug, tmp_path):
    popup = parameters.Parameters(slug=slug)
    argument, name = first_number_field(popup=popup)
    changed = getattr(popup.values[argument], name) + 3
    setattr(popup.values[argument], name, changed)
    config = popup.config()
    assert list(config["configs"]) == [argument] and config["configs"][argument][name] == changed
    assert [row[0] for row in popup.modified()] == [f"{argument}.{name}"]

    loaded = parameters.Parameters(slug=slug)
    loaded.load(config=json.loads(json.dumps(config)))
    assert parameters.same(loaded.values[argument], popup.values[argument])
    assert loaded.config() == config

    filepath = tmp_path / "config.json"
    filepath.write_text(json.dumps(config))
    pipeline = popup.spec.cls.from_config(filepath)
    assert getattr(getattr(pipeline, argument), name) == changed


def test_runtime_carries_changed_constructor_arguments_and_load_reads_them():
    popup = parameters.Parameters(slug="two-photon-calcium")
    popup.texts["device"] = "cpu"
    popup.texts["frame-batch-size"] = "77"
    assert popup.runtime() == {"device": "cpu", "frame_batch_size": 77}
    loaded = parameters.Parameters(slug="two-photon-calcium")
    loaded.load(config={"pipeline": "TwoPhotonCalciumPipeline", "configs": popup.runtime()})
    assert loaded.runtime() == popup.runtime()


def test_run_args_carry_changed_run_arguments_and_a_typed_frame_rate_but_never_resume():
    popup = parameters.Parameters(slug="two-photon-calcium")
    assert popup.run_args() == {}
    popup.texts["frame-rate"] = "30"
    popup.texts["stop-after"] = "compression"
    popup.texts["exclude-border-radius"] = "4"
    popup.texts["resume"] = "/somewhere/results.hdf5"
    assert popup.run_args() == {"frame_rate": 30.0, "stop_after": "compression", "exclude_border_radius": 4}


def test_load_restores_run_arguments_from_a_run_folders_config_json(tmp_path):
    cls = scraper.pipeline_registry()["two-photon-calcium"]
    pipeline = cls(output_folder=str(tmp_path), device="cpu")
    pipeline.run_config = {"frame_rate": 7.5, "stop_after": "registration", "remove_intermediates": True}
    folder = pipeline.create_run_folder()
    pipeline.write_config()
    logging.getLogger("masknmf").removeHandler(pipeline.log_handler)
    pipeline.log_handler.close()

    popup = parameters.Parameters(slug="two-photon-calcium")
    popup.load(config=json.loads((folder / "config.json").read_text()))
    assert popup.run_args() == {"frame_rate": 7.5, "stop_after": "registration", "remove_intermediates": True}
    assert popup.runtime() == {"device": "cpu"}


def test_load_refuses_another_pipelines_config():
    config = parameters.Parameters(slug="two-photon-calcium").config()
    with pytest.raises(ValueError, match="TwoPhotonCalciumPipeline"):
        parameters.Parameters(slug="widefield-singlechannel").load(config=config)


def test_select_switches_pipeline_at_its_defaults():
    popup = parameters.Parameters(slug="two-photon-calcium")
    argument, name = first_number_field(popup=popup)
    setattr(popup.values[argument], name, getattr(popup.values[argument], name) + 3)
    popup.select(slug="widefield-singlechannel")
    assert popup.slug == "widefield-singlechannel" and popup.spec.slug == "widefield-singlechannel"
    assert popup.modified() == []
    with pytest.raises(ValueError):
        popup.select(slug="no-such-pipeline")


def test_problems_name_a_value_that_does_not_parse():
    popup = parameters.Parameters(slug="two-photon-calcium")
    popup.texts["frame-batch-size"] = "many"
    assert len(popup.problems()) == 1


def test_the_popup_draws_open_in_a_host_frame_with_its_own_font():
    canvas = RenderCanvas(size=(1200, 800), pixel_ratio=1)
    renderer = ImguiRenderer(wgpu.utils.get_default_device(), canvas)
    popup = parameters.Parameters(slug="two-photon-calcium")
    popup.open()
    renderer.set_gui(popup.draw)
    canvas.request_draw(renderer.render)
    for _ in range(3):
        image = np.asarray(canvas.draw())
    # set only once the open popup has drawn its columns and buttons
    assert popup.height_chrome > 0 and not popup.requested
    assert image[..., :3].std() > 0
    assert popup.modified() == []
