"""
The Parameters popup's hints and columns: every hint names a field that exists, and every registered pipeline's
sections land in a named stage column.
"""

import dataclasses
import inspect

import pytest

from masknmf.pipelines import scraper
from masknmf.pipelines._base import BasePipeline
from masknmf.visualization.imgui import param_hints

SLUGS = sorted(scraper.pipeline_registry())


@pytest.mark.parametrize("cls_config", list(param_hints.HINTS), ids=lambda cls: cls.__name__)
def test_every_hint_names_a_field_of_its_config(cls_config):
    fields = {field.name for field in dataclasses.fields(cls_config) if field.init}
    assert set(param_hints.HINTS[cls_config]) <= fields, set(param_hints.HINTS[cls_config]) - fields


def test_every_runtime_hint_names_an_argument_of_the_base_pipeline():
    arguments = set(inspect.signature(BasePipeline.__init__).parameters)
    assert set(param_hints.HINTS_RUNTIME) <= arguments, set(param_hints.HINTS_RUNTIME) - arguments


@pytest.mark.parametrize("slug", SLUGS)
def test_every_section_lands_in_a_named_stage(slug):
    spec = scraper.scrape(cls_pipeline=scraper.pipeline_registry()[slug])
    stages = {section.name: param_hints.stage_of(section=section) for section in spec.sections}
    assert param_hints.STAGE_OTHER not in stages.values(), stages


def test_a_bad_tier_is_refused():
    with pytest.raises(ValueError, match="tier"):
        param_hints.Hint("Label", tier="hidden")
