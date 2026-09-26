"""The contract every pipeline keeps with BasePipeline."""

import inspect

import pytest

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
