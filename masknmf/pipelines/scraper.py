"""
Introspection of the pipeline classes and their config dataclasses.

Turns a pipeline class into a "PipelineSpec" with each value a caller can set.
"""

from typing import Any, Literal, Optional, Union

import dataclasses
import inspect
import re
import types
import typing

import numpy as np

from masknmf.pipelines import configs


SUFFIX_PIPELINE = "Pipeline"

NAMES_MOVIE_ARGUMENT = ("data", "glutamate_channel", "calcium_channel")

TYPES_SCALAR = (int, float, bool, str)

KINDS_VARIADIC = (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)


@dataclasses.dataclass
class Param:
    """One settable value, either a config dataclass field or a plain argument."""

    name: str
    section: Optional[str]
    field: str
    annotation: Any
    default: Any
    choices: Optional[tuple]
    settable: bool
    required: bool
    kind: str


@dataclasses.dataclass
class Section:
    """A pipeline __init__ argument that takes one of several config dataclasses."""

    name: str
    argument: str
    configs_by_kind: dict[str, type]
    allows_skip: bool
    allows_none: bool

    @property
    def kinds(self) -> tuple[str, ...]:
        """Every value the --<section>-kind flag accepts."""
        names = list(self.configs_by_kind)
        if self.allows_skip:
            names.append("skip")
        return tuple(names)


@dataclasses.dataclass
class PipelineSpec:
    """Everything the command line needs to know about one pipeline."""

    slug: str
    cls: type
    sections: list[Section]
    scalars: list[Param]
    run_params: list[Param]

    @property
    def movie_params(self) -> list[Param]:
        """The run arguments that take an imaging movie."""
        return [p for p in self.run_params if p.kind == "movie"]

    @property
    def array_params(self) -> list[Param]:
        """The run arguments that take an array loaded from a .npy file."""
        return [p for p in self.run_params if p.kind == "array"]

    @property
    def run_scalars(self) -> list[Param]:
        """The run arguments that take a plain value."""
        return [p for p in self.run_params if p.kind == "scalar"]

    def section(self, name: str) -> Section:
        """
        Look a section up by name.

        Args:
            name (str): The section name, as scraped from the constructor argument
        Returns:
            Section: The matching section
        Raises:
            KeyError: If the pipeline has no such section
        """
        for section in self.sections:
            if section.name == name:
                return section
        raise KeyError(f"{self.slug} has no section {name!r}")

    def params_for(self, section: Section, kind: str) -> list[Param]:
        """
        Every field of the config dataclass that a kind selects, settable or not.

        Args:
            section (Section): The section being inspected
            kind (str): One of the section's kinds
        Returns:
            list[Param]: One per field, empty when the kind names no dataclass
        """
        if kind not in section.configs_by_kind:
            return []
        return scrape_dataclass(
            cls_config=section.configs_by_kind[kind], name_section=section.name
        )


def slugify(name_class: str) -> str:
    """
    Turn a pipeline class name into its command line name.

    Args:
        name_class (str): For example "TwoPhotonCalciumPipeline"
    Returns:
        str: For example "two-photon-calcium"
    """
    stem = (
        name_class[: -len(SUFFIX_PIPELINE)]
        if name_class.endswith(SUFFIX_PIPELINE)
        else name_class
    )
    return re.sub(r"(?<!^)(?=[A-Z])", "-", stem).lower()


def pipeline_registry() -> dict[str, type]:
    """
    Every pipeline masknmf exports, keyed by its command line name.

    Returns:
        dict[str, type]: Slug to pipeline class
    """
    from masknmf import pipelines

    return {
        slugify(name_class=name): getattr(pipelines, name) for name in pipelines.__all__
    }


def annotation_members(annotation: Any) -> tuple[list, bool]:
    """
    Split a possibly-optional annotation into its members.

    Args:
        annotation (Any): A type, a Union, or an X | None
    Returns:
        list: The members with NoneType removed
        bool: Whether None was one of them
    """
    origin = typing.get_origin(annotation)
    if origin not in (Union, types.UnionType):
        return [annotation], False
    members = list(typing.get_args(annotation))
    allows_none = type(None) in members
    return [m for m in members if m is not type(None)], allows_none


def literal_choices(annotation: Any) -> Optional[tuple]:
    """
    The values a Literal annotation allows.

    Args:
        annotation (Any): The annotation to inspect
    Returns:
        tuple | None: The allowed values, or None for anything that is not a Literal
    """
    if typing.get_origin(annotation) is Literal:
        return typing.get_args(annotation)
    return None


def is_settable(annotation: Any) -> bool:
    """
    Whether a value of this type can be given on a command line.

    Scalars, strings, tuples of scalars and Literal choices can. Arrays, detrenders,
    nested config dataclasses and lists of them cannot.

    Args:
        annotation (Any): The annotation to inspect
    Returns:
        bool: Whether every member of the annotation is command line expressible
    """
    members, _ = annotation_members(annotation=annotation)
    for member in members:
        if literal_choices(annotation=member) is not None:
            continue
        origin = typing.get_origin(member)
        if origin is tuple:
            continue
        if member in TYPES_SCALAR:
            continue
        return False
    return len(members) > 0


def scrape_dataclass(cls_config: type, name_section: str) -> list[Param]:
    """
    Describe every field of a config dataclass.

    Args:
        cls_config (type): A dataclass from masknmf.pipelines.configs
        name_section (str): The section the fields are reported under
    Returns:
        list[Param]: One per field, in declaration order
    """
    hints = resolve_hints(cls=cls_config)
    params = []
    for field in dataclasses.fields(cls_config):
        annotation = hints.get(field.name, field.type)
        has_default = field.default is not dataclasses.MISSING
        default = field.default if has_default else None
        params.append(
            Param(
                name=f"{name_section}.{field.name}",
                section=name_section,
                field=field.name,
                annotation=annotation,
                default=default,
                choices=first_literal(annotation=annotation),
                settable=is_settable(annotation=annotation) and field.init,
                required=not has_default
                and field.default_factory is dataclasses.MISSING,
                kind="config",
            )
        )
    return params


def resolve_hints(cls: type) -> dict:
    """
    Type hints for a class, falling back to raw annotations when they cannot resolve.

    Args:
        cls (type): The dataclass to inspect
    Returns:
        dict: Field name to annotation
    """
    try:
        return typing.get_type_hints(cls)
    except Exception:
        return {f.name: f.type for f in dataclasses.fields(cls)}


def first_literal(annotation: Any) -> Optional[tuple]:
    """
    The choices of the first Literal member of an annotation.

    Args:
        annotation (Any): The annotation to inspect
    Returns:
        tuple | None: The allowed values, or None when no member is a Literal
    """
    members, _ = annotation_members(annotation=annotation)
    for member in members:
        choices = literal_choices(annotation=member)
        if choices is not None:
            return choices
    return None


def config_kind(cls_config: type) -> str:
    """
    The command line name for a config dataclass.

    Args:
        cls_config (type): For example PiecewiseRigidMotionCorrectionConfig
    Returns:
        str: For example "piecewise-rigid"
    """
    name = cls_config.__name__
    for suffix in ("MotionCorrectionConfig", "DemixingConfig", "Config"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
            break
    return re.sub(r"(?<!^)(?=[A-Z])", "-", name).lower() or "default"


def section_name(name_argument: str) -> str:
    """
    Turn a constructor argument name into a section name.

    Args:
        name_argument (str): For example "motion_correct_config"
    Returns:
        str: For example "motion-correct"
    """
    stem = name_argument
    for suffix in ("_config", "_configs"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    return stem.replace("_", "-")


def classify_run_param(name: str, annotation: Any) -> str:
    """
    Whether a run argument takes a movie, a .npy array, or a plain value.

    Args:
        name (str): The argument name
        annotation (Any): Its annotation
    Returns:
        str: One of "movie", "array" or "scalar"
    """
    if name in NAMES_MOVIE_ARGUMENT:
        return "movie"
    members, _ = annotation_members(annotation=annotation)
    if any(member is np.ndarray for member in members):
        return "array"
    return "scalar"


def scrape(cls_pipeline: type) -> PipelineSpec:
    """
    Describe a pipeline class.

    Args:
        cls_pipeline (type): A pipeline from masknmf.pipelines
    Returns:
        PipelineSpec: Its config sections, plain constructor arguments and run arguments
    """
    sections = []
    scalars = []

    signature_init = inspect.signature(cls_pipeline.__init__)
    for name, parameter in signature_init.parameters.items():
        if name == "self" or parameter.kind in KINDS_VARIADIC:
            continue
        annotation = parameter.annotation
        members, allows_none = annotation_members(annotation=annotation)

        configs_by_kind = {
            config_kind(cls_config=member): member
            for member in members
            if dataclasses.is_dataclass(member)
        }
        allows_skip = any(
            literal_choices(annotation=member) == ("skip",) for member in members
        )

        if len(configs_by_kind) > 0 or allows_skip:
            sections.append(
                Section(
                    name=section_name(name_argument=name),
                    argument=name,
                    configs_by_kind=configs_by_kind,
                    allows_skip=allows_skip,
                    allows_none=allows_none,
                )
            )
            continue

        has_default = parameter.default is not inspect.Parameter.empty
        scalars.append(
            Param(
                name=name.replace("_", "-"),
                section=None,
                field=name,
                annotation=annotation,
                default=parameter.default if has_default else None,
                choices=first_literal(annotation=annotation),
                settable=is_settable(annotation=annotation),
                required=not has_default,
                kind="init",
            )
        )

    run_params = []
    signature_run = inspect.signature(cls_pipeline.run)
    for name, parameter in signature_run.parameters.items():
        if name == "self" or parameter.kind in KINDS_VARIADIC:
            continue
        annotation = parameter.annotation
        has_default = parameter.default is not inspect.Parameter.empty
        run_params.append(
            Param(
                name=name.replace("_", "-"),
                section=None,
                field=name,
                annotation=annotation,
                default=parameter.default if has_default else None,
                choices=first_literal(annotation=annotation),
                settable=True,
                required=not has_default,
                kind=classify_run_param(name=name, annotation=annotation),
            )
        )

    return PipelineSpec(
        slug=slugify(name_class=cls_pipeline.__name__),
        cls=cls_pipeline,
        sections=sections,
        scalars=scalars,
        run_params=run_params,
    )


def coerce(param: Param, text: str) -> Any:
    """
    Parse a command line string into the type a parameter wants.

    Args:
        param (Param): The parameter being set
        text (str): The string given on the command line. Tuples are comma separated
    Returns:
        Any: The parsed value
    Raises:
        ValueError: If the text does not parse as the parameter's type
    """
    if param.choices is not None and text in param.choices:
        return text

    members, allows_none = annotation_members(annotation=param.annotation)
    if allows_none and text.lower() in ("none", "null", ""):
        return None

    for member in members:
        if typing.get_origin(member) is tuple:
            return coerce_tuple(member=member, text=text)

    for member in members:
        if member is bool:
            return coerce_bool(text=text)

    for member in (int, float, str):
        if member in members:
            try:
                return member(text)
            except ValueError as error:
                raise ValueError(f"{param.name}: {error}") from error

    raise ValueError(f"{param.name} cannot be set from the command line")


def coerce_tuple(member: Any, text: str) -> tuple:
    """
    Parse "15,15" into a tuple, converting each element to the annotated type.

    Args:
        member (Any): The tuple annotation, for example tuple[int, int]
        text (str): Comma separated elements
    Returns:
        tuple: The parsed elements
    """
    args = typing.get_args(member)
    type_element = args[0] if len(args) > 0 and args[0] is not Ellipsis else str
    if type_element not in TYPES_SCALAR:
        type_element = str
    pieces = [piece.strip() for piece in text.split(",") if piece.strip() != ""]
    return tuple(type_element(piece) for piece in pieces)


def coerce_bool(text: str) -> bool:
    """
    Parse a command line boolean.

    Args:
        text (str): One of true/1/yes/on or false/0/no/off, in any case
    Returns:
        bool: The parsed value
    Raises:
        ValueError: If the text is not a recognised boolean
    """
    lowered = text.strip().lower()
    if lowered in ("true", "1", "yes", "on"):
        return True
    if lowered in ("false", "0", "no", "off"):
        return False
    raise ValueError(f"expected true or false, got {text!r}")
