"""
The Parameters popup: one column per stage of a pipeline with every config field, the runtime under them, and
Defaults and Close. It holds the values typed in and gives them back as the json `masknmf run --config` takes, so
the launcher and any other imgui app can draw it inside their own frame.
"""

from typing import Any, Callable, Optional

import copy
import dataclasses
import json
import typing
from pathlib import Path

import numpy as np
import imgui_data_loader as idl
from imgui_bundle import hello_imgui, imgui, imgui_ctx
from imgui_bundle import icons_fontawesome_6 as fa

from masknmf import cli
from masknmf.io import config_record
from masknmf.pipelines import scraper
from masknmf.pipelines._base import BasePipeline
from masknmf.visualization.imgui import guide, param_hints
from masknmf.visualization.imgui.theme import button_colors, tooltip
from masknmf.visualization.imgui.widgets import draw_min_brightness


COLOR_TITLE = guide.KEY
COLOR_SUBSECTION = guide.ACCENT
COLOR_DIM = guide.DIM
COLOR_MODIFIED = imgui.ImVec4(1.0, 0.72, 0.40, 1.0)
COLOR_ERROR = imgui.ImVec4(1.0, 0.4, 0.4, 1.0)
COLORS_DEFAULTS = (
    imgui.ImVec4(0.60, 0.35, 0.10, 1.0),
    imgui.ImVec4(0.70, 0.42, 0.14, 1.0),
    imgui.ImVec4(0.50, 0.28, 0.08, 1.0),
)
COLORS_CLOSE = (
    imgui.ImVec4(0.55, 0.13, 0.13, 1.0),
    imgui.ImVec4(0.65, 0.18, 0.18, 1.0),
)

WIDTH_INPUT_EM = 7.5

# the popup's id, and the narrowest and widest a stage column gets
POPUP_PARAMETERS = "###parameters"
WIDTH_COLUMN_EM = (19.0, 27.0)

NO_DEFAULT = object()


def text_value(value: Any) -> str:
    """
    A run or constructor argument's value spelled as the command line takes it.

    Parameters
    ----------
    value : Any
        The value; None gives the empty text.

    Returns
    -------
    str
        true or false for a bool, comma separated items for a tuple or list, str(value) otherwise.
    """
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (tuple, list)):
        return ",".join(str(v) for v in value)
    return str(value)


def text_default(param: scraper.Param) -> str:
    """The text a parameter's input starts with: its default, spelled as the command line takes it."""
    return "" if param.required else text_value(value=param.default)


def text_of(value: Any) -> str:
    """A config field's value as text: numbers and lists of them in full, configs by kind, anything else by type."""
    if value is None:
        return ""
    if isinstance(value, np.ndarray):
        return f"array {' x '.join(str(n) for n in value.shape)}"
    if dataclasses.is_dataclass(value):
        return scraper.kind_of(value=value)
    if isinstance(value, float):
        return f"{value:.6g}"
    if isinstance(value, (list, tuple)):
        return ", ".join(text_of(value=v) for v in value)
    if isinstance(value, (bool, int, str)):
        return str(value)
    return type(value).__name__


def kind_or_none(value: Any) -> str:
    """The kind of a config, "none" for None."""
    return "none" if value is None else scraper.kind_of(value=value)


def same(a: Any, b: Any) -> bool:
    """Whether two config values are equal, comparing arrays by content instead of failing on them."""
    if dataclasses.is_dataclass(a) or dataclasses.is_dataclass(b):
        return type(a) is type(b) and all(same(getattr(a, f.name), getattr(b, f.name)) for f in dataclasses.fields(a))
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        return isinstance(a, np.ndarray) and isinstance(b, np.ndarray) and a.shape == b.shape and bool(np.array_equal(a, b))
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    try:
        return bool(a == b)
    except (TypeError, ValueError):
        return a is b


def widget_for(param: scraper.Param) -> str:
    """
    Which widget edits a parameter.

    Args:
        param (Param): The parameter
    Returns:
        str: "bool", "choice", "int", "float", "int2", "folder" or "text". Values that may be
            None or have no default are typed as text, so leaving them empty stays possible.
    """
    members, allows_none = scraper.annotation_members(annotation=param.annotation)
    if bool in members:
        return "bool"
    if param.choices is not None:
        return "choice"
    if Path in members:
        return "folder"
    if allows_none or param.required or len(members) != 1:
        return "text"
    member = members[0]
    if member is int:
        return "int"
    if member is float:
        return "float"
    if typing.get_origin(member) is tuple and typing.get_args(member) == (int, int):
        return "int2"
    return "text"


def is_number(value: Any) -> bool:
    """Whether a value is an int or a float, booleans excluded."""
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def widget_for_field(param: scraper.Param, value: Any) -> str:
    """
    Which widget edits a config field, judged first by the value it holds and then by its annotation,
    so a default of another type than annotated (None in an int field, a list in a tuple field) still
    gets a widget that shows it.

    Args:
        param (Param): The field
        value (Any): Its current value
    Returns:
        str: "bool", "choice", "int", "float", "int2", "float2", "sequence" or "text"
    """
    members, _ = scraper.annotation_members(annotation=param.annotation)
    if value is None:
        return "text"
    if isinstance(value, bool):
        return "bool" if bool in members else "text"
    if param.choices is not None and value in param.choices:
        return "choice"
    if isinstance(value, int) and int in members:
        return "int"
    if is_number(value=value) and float in members:
        return "float"
    if isinstance(value, (list, tuple)) and len(value) == 2 and all(isinstance(v, int) and not isinstance(v, bool) for v in value):
        return "int2"
    if isinstance(value, (list, tuple)) and len(value) == 2 and all(is_number(value=v) for v in value):
        return "float2"
    if isinstance(value, (list, tuple)):
        return "sequence"
    return "text"


def title_of(section: scraper.Section) -> str:
    """A section's heading, e.g. "Motion correct" for motion-correct."""
    return section.name.replace("-", " ").capitalize()


def label_of(kind: str, section: scraper.Section) -> str:
    """A config's name in a stage's dropdown, marking the pipeline's default."""
    return f"{kind} (default)" if kind == section.default_kind else kind


def draw_wrapped(text: str, color: Optional[imgui.ImVec4] = None) -> None:
    """Text wrapped at the current window's right edge, in color when given."""
    if color is None:
        imgui.text_wrapped(text)
    else:
        idl.text_wrapped_colored(color, text)


def same_line_if_fits(width: float) -> None:
    """Continue the line when an item width wide still fits before the right edge of the window or table cell."""
    spacing = imgui.get_style().item_spacing.x
    ## NOTE: after an item the cursor sits at the next line's start, so cursor + avail is the right edge of the work
    ## rect, which in a table is the cell rather than the window
    right = imgui.get_cursor_screen_pos().x + imgui.get_content_region_avail().x
    if imgui.get_item_rect_max().x + spacing + width <= right:
        imgui.same_line()


def draw_boxed(text: str, color: Optional[imgui.ImVec4] = None) -> None:
    """Text in a thin box, marking a value to look at first."""
    if color is None:
        imgui.text(text)
    else:
        imgui.text_colored(color, text)
    pad = imgui.ImVec2(0.25 * imgui.get_font_size(), 0.1 * imgui.get_font_size())
    corner_min, corner_max = imgui.get_item_rect_min(), imgui.get_item_rect_max()
    imgui.get_window_draw_list().add_rect(
        imgui.ImVec2(corner_min.x - pad.x, corner_min.y - pad.y),
        imgui.ImVec2(corner_max.x + pad.x, corner_max.y + pad.y),
        guide.u32(color or guide.TEXT, 0.7),
        3.0,
    )




def draw_divider() -> None:
    """Space, a separator, and space again, between sections."""
    imgui.spacing()
    imgui.separator()
    imgui.spacing()
    imgui.spacing()


def diff_config(current: Any, default: Any, path: str, rows: list) -> None:
    """
    Append a row for every field of current that differs from default, descending into nested
    configs and lists of configs.

    Args:
        current (Any): A config field's value
        default (Any): The pipeline default's value at the same place, or NO_DEFAULT when it has none
        path (str): Where the value sits, e.g. filtered_demixing_config.DemixingConfigs[0].NMFConfig
        rows (list): Receives (path, current, default) text triples
    """
    if default is NO_DEFAULT:
        return
    if dataclasses.is_dataclass(current) and type(current) is type(default):
        for field in dataclasses.fields(current):
            if field.init:
                diff_config(
                    current=getattr(current, field.name),
                    default=getattr(default, field.name),
                    path=f"{path}.{field.name}",
                    rows=rows,
                )
        return
    if dataclasses.is_dataclass(current) or dataclasses.is_dataclass(default):
        rows.append((path, kind_or_none(value=current), kind_or_none(value=default)))
        return
    holds_configs = any(isinstance(v, list) and len(v) > 0 and dataclasses.is_dataclass(v[0]) for v in (current, default))
    if holds_configs and isinstance(current, list) and isinstance(default, list):
        if len(current) != len(default):
            rows.append((f"{path} passes", str(len(current)), str(len(default))))
        for i, item in enumerate(current):
            diff_config(current=item, default=default[i] if i < len(default) else NO_DEFAULT, path=f"{path}[{i}]", rows=rows)
        return
    if not same(current, default):
        rows.append((path, text_of(value=current) or "none", text_of(value=default) or "none"))



def width_frame(text: str) -> float:
    """The width of a framed widget showing text, padding included."""
    return imgui.calc_text_size(text).x + 2 * imgui.get_style().frame_padding.x

def width_stepped(text: str) -> float:
    """An integer field wide enough for its digits beside its - and + buttons."""
    style = imgui.get_style()
    return width_frame(text=text) + hello_imgui.em_size(1) + 2 * (imgui.get_frame_height() + style.item_inner_spacing.x)

def width_combo(items: list[str]) -> float:
    """A combo wide enough for its longest item and its arrow."""
    return max(width_frame(text=item) for item in items) + imgui.get_frame_height()


class Parameters:
    """
    The Parameters popup and the values behind it: every stage config, run argument and constructor argument of one
    pipeline, which start at the pipeline's defaults.

    Draw it with draw() every frame inside an imgui frame and open it with open(). config() gives the stage configs
    that differ from the defaults as the json `masknmf run --config` and BasePipeline.from_config take, and
    runtime() the constructor arguments the Runtime section changed, which `masknmf run` takes as flags.

    Args:
        slug (str): The pipeline, a key of scraper.pipeline_registry()
        on_reset (Callable | None): Called after reset_all, e.g. by the Defaults button, so a host can reset its own state
    """

    def __init__(self, slug: str, on_reset: Optional[Callable[[], None]] = None):
        self.on_reset = on_reset
        # set to open the popup on the next draw; the popup's height outside its columns (title, runtime, buttons) as
        # last drawn, which caps how tall a column grows before it scrolls
        self.requested = False
        self.height_chrome = 0.0
        self.select(slug=slug)

    def select(self, slug: str) -> None:
        """Switch pipelines, every value back at the pipeline's default."""
        if slug not in scraper.pipeline_registry():
            raise ValueError(f"{slug} is not a masknmf pipeline; the pipelines are {', '.join(sorted(scraper.pipeline_registry()))}")
        self.slug = slug
        self.spec: scraper.PipelineSpec = cli.spec_for(slug=slug)
        # run and constructor arguments as typed, by name; stage configs by argument; config fields' text kept while
        # it does not parse, and why, by path
        self.texts: dict[str, str] = {param.name: text_default(param=param) for param in (*self.spec.run_scalars, *self.spec.scalars)}
        self.values: dict[str, Any] = {section.argument: section.value_for(kind=section.default_kind) for section in self.spec.sections}
        self.buffers: dict[str, str] = {}
        self.errors: dict[str, str] = {}
        # the uniform min brightness control: while checked, every change of its value is set in every demixing pass
        self.uniform_min_brightness_on = False
        self.uniform_min_brightness = self.min_brightness_first_pass()
        # the cell radius control: while checked, every change of its value is set through the pipeline's cell_radius
        self.cell_radius_on = False
        self.cell_radius = 10
        self.downsampling_before_cell_radius = []

    def open(self) -> None:
        """Open the popup on the next draw()."""
        self.requested = True

    def config(self) -> dict:
        """
        Returns the stage configs that differ from the pipeline's defaults as {"pipeline": class name, "configs": {...}}, in
        json types: what the launcher writes for --config, and what BasePipeline.from_config reads.
        """
        configs = {section.argument: self.values[section.argument] for section in self.sections_changed()}
        return json.loads(json.dumps({"pipeline": self.spec.cls.__name__, "configs": configs}, default=scraper.config_json_value))

    def runtime(self) -> dict:
        """
        Returns the Runtime section's constructor arguments that differ from their defaults, by field, in json types.
        `masknmf run` takes them as flags; BasePipeline.from_config takes them as overrides.

        Raises:
            ValueError: If a changed value does not parse; problems() names it
        """
        values = {p.field: scraper.coerce(param=p, text=self.texts[p.name]) for p in self.params_runtime() if self.is_changed(param=p)}
        return json.loads(json.dumps(values, default=scraper.config_json_value))

    def run_args(self) -> dict:
        """
        Returns the run arguments that differ from their defaults, by field, in json types, for the pipeline's run():
        a required one such as frame_rate once typed in, never resume.

        Raises:
            ValueError: If one of them does not parse
        """
        params = [p for p in self.spec.run_scalars if p.field != "resume"]
        given = [p for p in params if (self.texts[p.name] != "" if p.required else self.is_changed(param=p))]
        values = {p.field: scraper.coerce(param=p, text=self.texts[p.name]) for p in given}
        return json.loads(json.dumps(values, default=scraper.config_json_value))

    def load(self, config: dict) -> None:
        """
        Set every value from a config as config() returns it or a run folder's config.json holds it, the rest back at
        the defaults. A run's resume is left out, since a new run chooses where it starts anew.

        Args:
            config (dict): {"pipeline": class name, "configs": {...}}; the older flat config.json is read as well
        Raises:
            ValueError: If the config is for another pipeline, or names a config or field the pipeline does not take
        """
        record = config_record(loaded=config)
        if record is None:
            raise ValueError("the config holds no pipeline configs")
        if record.get("pipeline", self.spec.cls.__name__) != self.spec.cls.__name__:
            raise ValueError(f"the config is for {record['pipeline']}, not {self.spec.cls.__name__}")
        self.select(slug=self.slug)
        configs = record["configs"]
        try:
            for section in self.spec.sections:
                if section.argument in configs:
                    passes, value = cli.section_value(section=section, kind=None, value_file=configs[section.argument])
                    if passes:
                        self.values[section.argument] = value
        except SystemExit:
            self.select(slug=self.slug)
            raise ValueError(f"the config does not fit {self.slug}")
        for param in (*self.spec.run_scalars, *self.spec.scalars):
            # a None the parameter does not take was recorded before it existed; where to start is chosen anew
            if param.field not in configs or param.field == "resume":
                continue
            if configs[param.field] is None and param.default is not None:
                continue
            self.texts[param.name] = text_value(value=configs[param.field])

    def problems(self) -> list[str]:
        """Returns why the constructor arguments and the config fields typed as text would be rejected."""
        errors = [self.error_for(param=p) for p in self.spec.scalars]
        return [e for e in errors if e is not None] + list(self.errors.values())

    def min_brightness_first_pass(self) -> float:
        """Returns the min brightness of the pipeline's first demixing pass, where the uniform control starts; 0 when it is off."""
        if self.spec.cls.demixing_configs is BasePipeline.demixing_configs:
            return 0.0
        pipeline = self.spec.cls(**{section.argument: self.values[section.argument] for section in self.spec.sections})
        passes = pipeline.demixing_passes()
        if len(passes) == 0 or passes[0].NMFConfig.min_brightness is None:
            return 0.0
        return float(passes[0].NMFConfig.min_brightness)

    def select_kind(self, section: scraper.Section, kind: str) -> None:
        """Give a section another config, at the pipeline's default when it is the default's kind."""
        self.values[section.argument] = section.value_for(kind=kind)
        self.forget(path=section.argument)

    def forget(self, path: str) -> None:
        """Drop the typed text and errors of every field under path, so they are read again from the values."""
        self.buffers = {k: v for k, v in self.buffers.items() if not k.startswith(path)}
        self.errors = {k: v for k, v in self.errors.items() if not k.startswith(path)}

    def sections_changed(self) -> list[scraper.Section]:
        """The sections whose config differs from the pipeline's default."""
        return [s for s in self.spec.sections if not same(self.values[s.argument], s.default)]

    def reset_field(self, config: Any, name: str, default: Any, path: str) -> None:
        """Put one config field back to the pipeline's default."""
        setattr(config, name, copy.deepcopy(default))
        self.forget(path=path)

    def reset_section(self, section: scraper.Section) -> None:
        """Put a stage back to the pipeline's default config."""
        self.values[section.argument] = section.value_for(kind=section.default_kind)
        self.forget(path=section.argument)

    def reset_all(self) -> None:
        """Put every stage, run argument and constructor argument back to its default; paths and required values stay."""
        for param in (*self.spec.run_scalars, *self.spec.scalars):
            if not param.required:
                self.texts[param.name] = text_default(param=param)
        for section in self.spec.sections:
            self.reset_section(section=section)
        self.cell_radius_on = False
        self.uniform_min_brightness_on = False
        self.uniform_min_brightness = self.min_brightness_first_pass()
        if self.on_reset is not None:
            self.on_reset()

    def select_nested(self, config: Any, name: str, kind: str, default: Any, path: str) -> None:
        """
        Give a nested config field another config, or None.

        Args:
            config (Any): The config holding the field
            name (str): The field
            kind (str): "none", or the kind of one of the configs the field's annotation allows
            default (Any): The pipeline's default for the field, or NO_DEFAULT; choosing its kind restores it
            path (str): Where the field sits
        """
        if kind == "none":
            value = None
        elif default is not NO_DEFAULT and default is not None and scraper.kind_of(value=default) == kind:
            value = copy.deepcopy(default)
        else:
            annotation = scraper.resolve_hints(cls=type(config)).get(name)
            members, _ = scraper.annotation_members(annotation=annotation)
            by_kind = {scraper.config_kind(cls_config=m): m for m in members if dataclasses.is_dataclass(m)}
            value = scraper.build_default(cls_config=by_kind[kind])
        setattr(config, name, value)
        self.forget(path=path)

    def params_folder(self) -> list[scraper.Param]:
        """Constructor arguments that name a folder, shown with the input."""
        return [p for p in self.spec.scalars if widget_for(param=p) == "folder"]

    def params_runtime(self) -> list[scraper.Param]:
        """Constructor arguments that are neither configs nor folders."""
        return [p for p in self.spec.scalars if widget_for(param=p) != "folder"]

    def is_changed(self, param: scraper.Param) -> bool:
        """Whether a value differs from its default and so has to be passed."""
        return self.texts[param.name] != text_default(param=param)

    def is_modified(self, param: scraper.Param) -> bool:
        """Whether a value was changed from a default it has; filling in a required value is not a modification."""
        return not param.required and self.is_changed(param=param)

    def error_for(self, param: scraper.Param) -> Optional[str]:
        """Why a value would be rejected, or None when it parses."""
        text = self.texts[param.name]
        if not param.required and not self.is_changed(param=param):
            return None
        if text == "" and param.required:
            return f"{param.field} is required"
        try:
            scraper.coerce(param=param, text=text)
        except ValueError as error:
            return str(error)
        return None

    def value_of(self, param: scraper.Param):
        """A parameter's value as its widget shows it: the typed text parsed, or its default if it does not parse."""
        try:
            return scraper.coerce(param=param, text=self.texts[param.name])
        except ValueError:
            return param.default


    def modified(self) -> list[tuple[str, str, str]]:
        """Every value that differs from its default: name, current value, default."""
        rows = []
        for param in (*self.spec.run_scalars, *self.spec.scalars):
            if self.is_modified(param=param):
                rows.append((param.field, self.texts[param.name], text_default(param=param) or "none"))
        for section in self.spec.sections:
            diff_config(current=self.values[section.argument], default=section.default, path=section.argument, rows=rows)
        return rows

    def draw_defaults(self, changed: bool) -> None:
        """An orange Defaults button resetting every parameter, disabled while nothing has changed."""
        with button_colors(COLORS_DEFAULTS[0], COLORS_DEFAULTS[1]):
            imgui.begin_disabled(not changed)
            if imgui.button(f"{fa.ICON_FA_ARROW_ROTATE_LEFT}  Defaults"):
                self.reset_all()
            imgui.end_disabled()
        tooltip("Reset every parameter; movies and paths stay" if changed else "Every value is at its default")

    def stage_columns(self) -> list[tuple[str, list[scraper.Section]]]:
        """Returns the Parameters popup's columns: each stage the pipeline has, in order, with its sections."""
        by_stage: dict[str, list[scraper.Section]] = {}
        for section in self.spec.sections:
            by_stage.setdefault(param_hints.stage_of(section=section), []).append(section)
        return [(stage, by_stage[stage]) for stage in param_hints.STAGES_ORDER if stage in by_stage]

    def draw(self) -> None:
        """The Parameters popup: a column per stage, the runtime under them, then Defaults and Close."""
        if self.requested:
            imgui.open_popup(POPUP_PARAMETERS)
            self.requested = False
        em = imgui.get_font_size()
        style = imgui.get_style()
        viewport = imgui.get_main_viewport()
        columns = self.stage_columns()
        room = viewport.size.x * 0.96 - 2 * style.window_padding.x - (len(columns) - 1) * style.item_spacing.x
        width_column = min(max(room / len(columns), WIDTH_COLUMN_EM[0] * em), WIDTH_COLUMN_EM[1] * em)
        height_column = viewport.size.y * 0.94 - self.height_chrome
        ## NOTE: centered every frame, since an auto-resized popup is 0 wide on the frame it appears
        imgui.set_next_window_pos(viewport.get_center(), imgui.Cond_.always, imgui.ImVec2(0.5, 0.5))
        imgui.set_next_window_size_constraints(imgui.ImVec2(0, 0), imgui.ImVec2(viewport.size.x * 0.96, viewport.size.y * 0.94))
        flags = imgui.WindowFlags_.always_auto_resize | imgui.WindowFlags_.no_saved_settings | imgui.WindowFlags_.no_move
        opened, visible = imgui.begin_popup_modal(f"{self.spec.slug} parameters{POPUP_PARAMETERS}", True, flags)
        if not opened:
            return
        if not visible or (imgui.is_key_pressed(imgui.Key.escape) and not imgui.is_any_item_active()):
            imgui.close_current_popup()
        y_columns = imgui.get_cursor_pos_y()
        for i, (stage, sections) in enumerate(columns):
            if i > 0:
                imgui.same_line()
            self.draw_column(stage=stage, sections=sections, width=width_column, height_max=height_column)
        height_columns = imgui.get_cursor_pos_y() - y_columns
        draw_divider()
        self.draw_runtime(columns=max(2, len(columns)))
        draw_divider()
        self.draw_defaults(changed=len(self.modified()) > 0)
        imgui.same_line()
        imgui.set_cursor_pos_x(imgui.get_window_width() - style.window_padding.x - width_frame(text="Close"))
        with button_colors(COLORS_CLOSE[0], COLORS_CLOSE[1]):
            if imgui.button("Close##parameters"):
                imgui.close_current_popup()
        self.height_chrome = imgui.get_cursor_pos_y() + style.window_padding.y + style.popup_border_size - height_columns
        imgui.end_popup()

    def draw_column(self, stage: str, sections: list[scraper.Section], width: float, height_max: float) -> None:
        """One stage's column: its name, then each of its sections; it scrolls once taller than height_max."""
        flags = imgui.ChildFlags_.borders | imgui.ChildFlags_.auto_resize_y | imgui.ChildFlags_.always_use_window_padding
        padding = imgui.get_style().window_padding
        imgui.set_next_window_size_constraints(imgui.ImVec2(width, 0), imgui.ImVec2(width, height_max))
        ## NOTE: whole-pixel padding, since imgui truncates an auto-fit height and a scaled padding such as 8.4 px
        ## leaves every column 0.8 px short of its content and so scrolling
        with (
            imgui_ctx.push_style_var(imgui.StyleVar_.window_padding, imgui.ImVec2(round(padding.x), round(padding.y))),
            imgui_ctx.begin_child(f"##column_{stage}", imgui.ImVec2(width, 0), flags),
        ):
            imgui.text_colored(COLOR_TITLE, stage)
            if stage == param_hints.STAGE_DEMIXING and self.spec.cls.demixing_configs is not BasePipeline.demixing_configs:
                self.draw_uniform_min_brightness()
                self.draw_cell_radius()
            for section in sections:
                self.draw_stage(section=section)

    def draw_uniform_min_brightness(self) -> None:
        """A checkbox that, while checked, sets its min brightness in every demixing pass."""
        changed_on, self.uniform_min_brightness_on = imgui.checkbox("uniform min brightness", self.uniform_min_brightness_on)
        if imgui.is_item_hovered():
            idl.wrapped_tooltip("Sets this value in every demixing pass; a pass edited afterwards keeps its own")
        imgui.begin_disabled(not self.uniform_min_brightness_on)
        width_box = hello_imgui.em_size(WIDTH_INPUT_EM)
        width_slider = imgui.get_content_region_avail().x - width_box - imgui.get_style().item_inner_spacing.x
        changed_value, self.uniform_min_brightness = draw_min_brightness(
            "uniform_min_brightness", self.uniform_min_brightness, width_slider, width_box
        )
        imgui.end_disabled()
        if self.uniform_min_brightness_on and (changed_on or changed_value):
            self.set_meta_parameter(name="min_brightness", value=self.uniform_min_brightness)

    def draw_cell_radius(self) -> None:
        """A checkbox that, while checked, sets the ring radius and background downsampling of every demixing pass from a soma radius."""
        changed_on, self.cell_radius_on = imgui.checkbox("cell radius", self.cell_radius_on)
        if imgui.is_item_hovered():
            idl.wrapped_tooltip("Typical soma radius in pixels, for somatic data; use 20 for dendritic data. Sets ring_radius "
                                "to 2 * radius + 1 in every demixing pass and raises background_downsampling_factor to "
                                "2 * radius where it is below that; a pass edited afterwards keeps its own. "
                                "Only passes that fit the ring background use either value")
        imgui.same_line(0, imgui.get_style().item_inner_spacing.x)
        imgui.begin_disabled(not self.cell_radius_on)
        imgui.set_next_item_width(hello_imgui.em_size(WIDTH_INPUT_EM))
        changed_value, value = imgui.input_int("##cell_radius", self.cell_radius)
        imgui.end_disabled()
        # the pipeline's cell_radius setter refuses a radius below 1
        self.cell_radius = max(1, value)
        if self.cell_radius_on and (changed_on or changed_value):
            self.set_cell_radius(checked_now=changed_on)

    def set_meta_parameter(self, name: str, value) -> None:
        """Set a settable property of the pipeline, which rewrites the stage configs the window holds in place."""
        pipeline = self.spec.cls(**{section.argument: self.values[section.argument] for section in self.spec.sections})
        setattr(pipeline, name, value)

    def set_cell_radius(self, checked_now: bool) -> None:
        """
        Set the pipeline's cell_radius over the background downsampling factors the passes had when the control was
        checked, so that lowering the radius lowers the factors it had raised.
        """
        pipeline = self.spec.cls(**{section.argument: self.values[section.argument] for section in self.spec.sections})
        passes = pipeline.demixing_passes()
        if checked_now or len(self.downsampling_before_cell_radius) != len(passes):
            self.downsampling_before_cell_radius = [p.NMFConfig.background_downsampling_factor for p in passes]
        for p, factor in zip(passes, self.downsampling_before_cell_radius):
            p.NMFConfig.background_downsampling_factor = factor
        pipeline.cell_radius = self.cell_radius

    def draw_stage(self, section: scraper.Section) -> None:
        """A stage's heading, the config it takes with a reset once it differs from the default, then the config's fields."""
        imgui.spacing()
        imgui.separator_text(title_of(section=section))
        kinds = section.kinds_buildable
        value = self.values[section.argument]
        kind = scraper.kind_of(value=value)
        labels = [label_of(kind=k, section=section) for k in kinds]
        changed_kind = kind != section.default_kind
        if changed_kind:
            imgui.push_style_color(imgui.Col_.text, COLOR_MODIFIED)
        imgui.set_next_item_width(max(width_combo(items=labels), hello_imgui.em_size(WIDTH_INPUT_EM)))
        edited, index = imgui.combo(f"##kind_{section.name}", kinds.index(kind), labels)
        if changed_kind:
            imgui.pop_style_color()
        if edited:
            self.select_kind(section=section, kind=kinds[index])
        imgui.same_line(0, imgui.get_style().item_inner_spacing.x)
        imgui.text_colored(COLOR_MODIFIED if changed_kind else COLOR_DIM, "config")
        if not same(self.values[section.argument], section.default):
            same_line_if_fits(width=width_frame(text="Reset stage"))
            if imgui.small_button(f"Reset stage##{section.name}"):
                self.reset_section(section=section)
            if imgui.is_item_hovered():
                idl.wrapped_tooltip(f"Back to {label_of(kind=section.default_kind, section=section)} with the pipeline's values")
        value = self.values[section.argument]
        if isinstance(value, str):
            draw_wrapped(text="Skipped.", color=COLOR_DIM)
            return
        default = section.default if type(section.default) is type(value) else NO_DEFAULT
        self.draw_config(config=value, default=default, path=section.argument)

    def draw_config(self, config: Any, default: Any, path: str) -> None:
        """
        A config's fields: important and normal ones in order, then folds holding its advanced and its stable ones,
        each fold orange and counting its changes once any differ from the default.

        Args:
            config (Any): The config dataclass being edited
            default (Any): The pipeline's default at the same place, or NO_DEFAULT when it has none
            path (str): Where config sits, naming its fields' widgets and errors
        """
        cls_config = type(config)
        params = {param.field: param for param in scraper.scrape_dataclass(cls_config=cls_config, name_section=path)}
        fields = [field for field in dataclasses.fields(config) if field.init]
        tiers = {field.name: param_hints.tier_of(cls_config=cls_config, name=field.name) for field in fields}
        for field in fields:
            if tiers[field.name] in ("important", "normal"):
                self.draw_config_field(config=config, param=params[field.name], default=default, path=path)
        for tier, title in (("advanced", "Advanced"), ("stable", "Stable")):
            folded = [field.name for field in fields if tiers[field.name] == tier]
            if len(folded) == 0:
                continue
            changed = 0 if default is NO_DEFAULT else sum(not same(getattr(config, n), getattr(default, n)) for n in folded)
            color = COLOR_MODIFIED if changed > 0 else COLOR_DIM if tier == "stable" else None
            if color is not None:
                imgui.push_style_color(imgui.Col_.text, color)
            opened = imgui.tree_node(f"{title}{f' ({changed} changed)' if changed > 0 else ''}###{tier}_{path}")
            if color is not None:
                imgui.pop_style_color()
            if tier == "stable" and imgui.is_item_hovered():
                idl.wrapped_tooltip("Values the masknmf authors mark as stable; change them only with a reason")
            if opened:
                for name in folded:
                    self.draw_config_field(config=config, param=params[name], default=default, path=path)
                imgui.tree_pop()

    def draw_config_field(self, config: Any, param: scraper.Param, default: Any, path: str) -> None:
        """
        One field of a config: a nested config under its name, a list of configs as tabs, an editable value with
        a widget, or a value only Python can set shown read-only when it is set.

        Args:
            config (Any): The config dataclass holding the field
            param (Param): The field, as scraper.scrape_dataclass describes it
            default (Any): The pipeline's default config at the same place as config, or NO_DEFAULT
            path (str): Where config sits
        """
        name = param.field
        current = getattr(config, name)
        default_field = NO_DEFAULT if default is NO_DEFAULT else getattr(default, name)
        path_field = f"{path}.{name}"
        members, allows_none = scraper.annotation_members(annotation=param.annotation)
        configs = [m for m in members if dataclasses.is_dataclass(m)]
        classes_item = [scraper.item_dataclass_of(annotation=m) for m in members]
        classes_item = [c for c in classes_item if c is not None]
        if len(configs) > 0 or dataclasses.is_dataclass(current):
            self.draw_nested(config=config, name=name, configs=configs, allows_none=allows_none,
                             default=default_field, path=path_field)
        elif len(classes_item) > 0:
            self.draw_items(config=config, name=name, cls_item=classes_item[0], default=default_field, path=path_field)
        elif param.settable:
            self.draw_field(config=config, param=param, default=default_field, path=path_field)
        elif current is not None:
            draw_wrapped(text=f"{name}: {text_of(value=current)} (set from Python)", color=COLOR_DIM)

    def draw_nested(self, config: Any, name: str, configs: list, allows_none: bool, default: Any, path: str) -> None:
        """
        A nested config field: a dropdown when it can hold more than one config or None, then the held config's fields.

        Args:
            config (Any): The config holding the field
            name (str): The field
            configs (list): The config classes its annotation allows
            allows_none (bool): Whether its annotation allows None
            default (Any): The pipeline's default for the field, or NO_DEFAULT
            path (str): Where the field sits
        """
        current = getattr(config, name)
        kinds = ["none"] if allows_none else []
        for cls_config in configs:
            try:
                scraper.build_default(cls_config=cls_config)
            except ValueError:
                continue
            kinds.append(scraper.config_kind(cls_config=cls_config))
        kind = kind_or_none(value=current)
        if kind not in kinds:
            kinds.append(kind)
        modified = default is not NO_DEFAULT and kind != kind_or_none(value=default)
        hint = param_hints.hint_for(cls_config=type(config), name=name)
        title = name if hint is None else hint.label
        imgui.spacing()
        if len(kinds) > 1:
            labels = [f"{k} (default)" if default is not NO_DEFAULT and k == kind_or_none(value=default) else k for k in kinds]
            if modified:
                imgui.push_style_color(imgui.Col_.text, COLOR_MODIFIED)
            imgui.set_next_item_width(max(width_combo(items=labels), hello_imgui.em_size(WIDTH_INPUT_EM)))
            edited, index = imgui.combo(f"##{path}", kinds.index(kind), labels)
            if modified:
                imgui.pop_style_color()
            imgui.same_line(0, imgui.get_style().item_inner_spacing.x)
            draw_wrapped(text=title, color=COLOR_MODIFIED if modified else COLOR_SUBSECTION)
            if edited:
                self.select_nested(config=config, name=name, kind=kinds[index], default=default, path=path)
            if modified and self.draw_reset(path=path, text_default=kind_or_none(value=default)):
                self.reset_field(config=config, name=name, default=default, path=path)
        else:
            imgui.text_colored(COLOR_SUBSECTION, title)
        current = getattr(config, name)
        if dataclasses.is_dataclass(current):
            imgui.indent(hello_imgui.em_size(0.8))
            matches = default is not NO_DEFAULT and type(default) is type(current)
            self.draw_config(config=current, default=default if matches else NO_DEFAULT, path=path)
            imgui.unindent(hello_imgui.em_size(0.8))

    def draw_items(self, config: Any, name: str, cls_item: type, default: Any, path: str) -> None:
        """
        A list of configs, such as a multipass config's passes: a tab per item, orange once it differs from the
        default and closable while another is left, then a + tab adding a copy of the last item, or a fresh one
        when the list is empty or None.

        Args:
            config (Any): The config holding the list
            name (str): The list's field
            cls_item (type): The config class the list holds
            default (Any): The pipeline's default list, or NO_DEFAULT
            path (str): Where the list sits
        """
        items = getattr(config, name)
        defaults = default if isinstance(default, list) else []
        if items is None or len(items) == 0:
            draw_wrapped(text=f"{name}: none" if items is None else f"{name}: no passes", color=COLOR_DIM)
        if not imgui.begin_tab_bar(f"##tabs_{path}", imgui.TabBarFlags_.fitting_policy_scroll):
            return
        removed = None
        for i, item in enumerate(items or []):
            added = i >= len(defaults)
            modified = added or not same(item, defaults[i])
            if modified:
                imgui.push_style_color(imgui.Col_.text, COLOR_MODIFIED)
            label = f"Pass {i + 1}{' (added)' if added else ''}###{path}[{i}]"
            selected, kept = imgui.begin_tab_item(label, True if len(items) > 1 else None)
            if modified:
                imgui.pop_style_color()
            if kept is False:
                removed = i
            if selected:
                self.draw_config(config=item, default=NO_DEFAULT if added else defaults[i], path=f"{path}[{i}]")
                imgui.end_tab_item()
        if imgui.tab_item_button(f"{fa.ICON_FA_PLUS}##add_{path}", imgui.TabItemFlags_.trailing | imgui.TabItemFlags_.no_tooltip):
            fresh = copy.deepcopy(items[-1]) if items else scraper.build_default(cls_config=cls_item)
            setattr(config, name, [*(items or []), fresh])
        if imgui.is_item_hovered():
            idl.wrapped_tooltip("Adds a pass copying the last one")
        imgui.end_tab_bar()
        if removed is not None:
            del items[removed]
            self.forget(path=path)

    def draw_field(self, config: Any, param: scraper.Param, default: Any, path: str) -> None:
        """
        A widget for one config field chosen by the value it holds, its name to the right, and once it
        differs from the pipeline's default an orange tint, a reset button and the default beside it.
        """
        name = param.field
        current = getattr(config, name)
        modified = default is not NO_DEFAULT and not same(current, default)
        widget = widget_for_field(param=param, value=current)
        width = hello_imgui.em_size(WIDTH_INPUT_EM)
        if modified:
            imgui.push_style_color(imgui.Col_.text, COLOR_MODIFIED)
        if widget == "bool":
            edited, value = imgui.checkbox(f"##{path}", current)
            if edited:
                setattr(config, name, value)
        elif widget == "choice":
            items = [str(choice) for choice in param.choices]
            imgui.set_next_item_width(max(width, width_combo(items=items)))
            edited, index = imgui.combo(f"##{path}", list(param.choices).index(current), items)
            if edited:
                setattr(config, name, param.choices[index])
        elif widget == "int":
            imgui.set_next_item_width(max(width, width_stepped(text=str(current))))
            edited, value = imgui.input_int(f"##{path}", int(current))
            if edited:
                setattr(config, name, value)
        elif widget == "float":
            imgui.set_next_item_width(max(width, width_frame(text=text_of(value=float(current))) + hello_imgui.em_size(1)))
            edited, value = imgui.input_float(f"##{path}", float(current), 0.0, 0.0, "%.6g")
            if edited:
                setattr(config, name, float(f"{value:.6g}"))
        elif widget in ("int2", "float2"):
            widest = max(width_frame(text=text_of(value=v)) for v in current)
            imgui.set_next_item_width(max(width, 2 * (widest + hello_imgui.em_size(1)) + imgui.get_style().item_inner_spacing.x))
            if widget == "int2":
                edited, values = imgui.input_int2(f"##{path}", list(current))
            else:
                edited, values = imgui.input_float2(f"##{path}", [float(v) for v in current], "%.6g")
                values = [float(f"{v:.6g}") for v in values]
            if edited:
                setattr(config, name, type(current)(values))
        else:
            self.draw_field_text(config=config, param=param, path=path, wrap=widget == "sequence")
        if modified:
            imgui.pop_style_color()

        hint = param_hints.hint_for(cls_config=type(config), name=name)
        error = self.errors.get(path)
        color = COLOR_ERROR if error is not None else COLOR_MODIFIED if modified else None
        if hint is not None and hint.tier == "important":
            imgui.same_line(0, imgui.get_style().item_inner_spacing.x + 0.3 * imgui.get_font_size())
            draw_boxed(text=hint.label, color=color)
        else:
            imgui.same_line(0, imgui.get_style().item_inner_spacing.x)
            draw_wrapped(text=name if hint is None else hint.label, color=color)
        if imgui.is_item_hovered():
            lines = [error] if error is not None else []
            if hint is not None and hint.tooltip != "":
                lines.append(hint.tooltip)
            lines.append(name if default is NO_DEFAULT else f"{name}, pipeline default {text_of(value=default) or 'none'}")
            idl.wrapped_tooltip("\n".join(lines))
        if modified and self.draw_reset(path=path, text_default=text_of(value=default) or "none"):
            self.reset_field(config=config, name=name, default=default, path=path)

    def draw_reset(self, path: str, text_default: str) -> bool:
        """A reset button after a changed value, then its default in dim text; True when the button was clicked."""
        imgui.same_line(0, imgui.get_style().item_inner_spacing.x)
        clicked = imgui.small_button(f"{fa.ICON_FA_ARROW_ROTATE_LEFT}##reset_{path}")
        if imgui.is_item_hovered():
            idl.wrapped_tooltip(f"Reset to {text_default}")
        same_line_if_fits(width=imgui.calc_text_size(f"default {text_default}").x)
        draw_wrapped(text=f"default {text_default}", color=COLOR_DIM)
        return clicked

    def draw_field_text(self, config: Any, param: scraper.Param, path: str, wrap: bool) -> None:
        """
        A text box for a field no numeric widget fits, such as an optional value or a long list,
        keeping what was typed until it parses.

        Args:
            config (Any): The config holding the field
            param (Param): The field
            path (str): Where the field sits
            wrap (bool): Whether to wrap the text over lines filling the row, for long lists
        """
        name = param.field
        current = getattr(config, name)
        text = self.buffers.get(path, text_of(value=current))
        if wrap:
            padding = imgui.get_style().frame_padding
            width = max(imgui.get_content_region_avail().x - hello_imgui.em_size(12), hello_imgui.em_size(WIDTH_INPUT_EM))
            width_wrap = width - 2 * padding.x - imgui.get_style().scrollbar_size
            height = imgui.calc_text_size(text or " ", wrap_width=width_wrap).y + 2 * padding.y
            edited, text = imgui.input_text_multiline(
                f"##{path}", text, imgui.ImVec2(width, height), imgui.InputTextFlags_.word_wrap
            )
            text = text.replace("\n", "").replace("\r", "")
        else:
            imgui.set_next_item_width(max(hello_imgui.em_size(WIDTH_INPUT_EM), width_frame(text=text)))
            edited, text = imgui.input_text_with_hint(f"##{path}", "none", text)
        if not edited:
            return
        self.buffers[path] = text
        try:
            value = scraper.coerce(param=param, text=text)
        except ValueError as error:
            self.errors[path] = f"{path}: {error}"
            return
        self.errors.pop(path, None)
        setattr(config, name, list(value) if isinstance(current, list) and isinstance(value, tuple) else value)

    def draw_runtime(self, columns: int) -> None:
        """The constructor arguments that are neither configs nor folders, in a table columns wide."""
        imgui.text_colored(COLOR_TITLE, "Runtime")
        if not imgui.begin_table("##runtime", columns, imgui.TableFlags_.sizing_stretch_same):
            return
        for param in self.params_runtime():
            imgui.table_next_column()
            self.draw_param(param=param)
        imgui.end_table()

    def draw_param(self, param: scraper.Param, named: bool = True) -> None:
        """A fixed width widget for one parameter, its name to the right unless a form caption names it, orange once changed."""
        key = param.name
        changed_before = self.is_modified(param=param)
        error = self.error_for(param=param)
        widget = widget_for(param=param)
        width = hello_imgui.em_size(WIDTH_INPUT_EM)
        if changed_before:
            imgui.push_style_color(imgui.Col_.text, COLOR_MODIFIED)
        if widget == "bool":
            edited, value = imgui.checkbox(f"##{key}", self.value_of(param=param) is True)
            if edited:
                self.texts[key] = "true" if value else "false"
        elif widget == "choice":
            items = [str(choice) for choice in param.choices]
            index = items.index(self.texts[key]) if self.texts[key] in items else -1
            imgui.set_next_item_width(max(width, width_combo(items=items)))
            edited, index = imgui.combo(f"##{key}", index, items)
            if edited:
                self.texts[key] = items[index]
        elif widget == "int":
            imgui.set_next_item_width(max(width, width_stepped(text=self.texts[key])))
            edited, value = imgui.input_int(f"##{key}", int(self.value_of(param=param)))
            if edited:
                self.texts[key] = str(value)
        elif widget == "float":
            imgui.set_next_item_width(max(width, width_frame(text=self.texts[key]) + hello_imgui.em_size(1)))
            edited, value = imgui.input_float(f"##{key}", float(self.value_of(param=param)), 0.0, 0.0, "%.6g")
            if edited:
                self.texts[key] = f"{value:.6g}"
        elif widget == "int2":
            widest = max(width_frame(text=piece) for piece in self.texts[key].split(","))
            imgui.set_next_item_width(max(width, 2 * (widest + hello_imgui.em_size(1)) + imgui.get_style().item_inner_spacing.x))
            edited, values = imgui.input_int2(f"##{key}", list(self.value_of(param=param)))
            if edited:
                self.texts[key] = f"{values[0]},{values[1]}"
        else:
            hint = "required" if param.required else "none"
            imgui.set_next_item_width(max(width, width_frame(text=self.texts[key]), width_frame(text=hint)))
            _, self.texts[key] = imgui.input_text_with_hint(f"##{key}", hint, self.texts[key])
        if changed_before:
            imgui.pop_style_color()

        hint = param_hints.HINTS_RUNTIME.get(param.field)
        if named:
            imgui.same_line(0, imgui.get_style().item_inner_spacing.x)
            color = COLOR_ERROR if error is not None else COLOR_MODIFIED if self.is_modified(param=param) else None
            draw_wrapped(text=param.field if hint is None else hint.label, color=color)
        if imgui.is_item_hovered():
            lines = [error] if error is not None else []
            if hint is not None and hint.tooltip != "":
                lines.append(hint.tooltip)
            lines.append(f"{param.field}: {param.description}")
            idl.wrapped_tooltip("\n".join(lines))
        if self.is_modified(param=param) and self.draw_reset(path=key, text_default=text_default(param=param) or "none"):
            self.texts[key] = text_default(param=param)

