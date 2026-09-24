"""
The window `masknmf` opens when it is run with no arguments.

Every value in it becomes an argument of `masknmf run`, so a run started here is a
command line run, and the equivalent command is printed before it starts. Stage configs
that differ from the pipeline's defaults are written to a json file passed as --config.
"""

from typing import Any, Optional

import copy
import dataclasses
import json
import typing
from pathlib import Path

import imgui_data_loader as idl
import numpy as np
from imgui_bundle import hello_imgui, imgui, imgui_ctx
from imgui_bundle import icons_fontawesome_6 as fa
from imgui_bundle import portable_file_dialogs as pfd

import masknmf
from masknmf import cli
from masknmf.pipelines import scraper


DIR_CONFIG = Path.home() / ".config" / "masknmf"
FILEPATH_RUN_CONFIGS = DIR_CONFIG / "launcher_configs.json"

PIPELINE_INITIAL = "two-photon-calcium"

FILETYPES_MOVIE = [
    idl.FileType("Movies", "*.tif *.tiff *.h5 *.hdf5"),
    idl.FileType("All Files", "*"),
]
FILETYPES_ARRAY = [idl.FileType("NumPy", "*.npy")]

COLOR_TITLE = imgui.ImVec4(1.0, 0.85, 0.4, 1.0)
COLOR_SUBSECTION = imgui.ImVec4(0.55, 0.75, 1.0, 1.0)
COLOR_DIM = imgui.ImVec4(0.6, 0.6, 0.6, 1.0)
COLOR_WARN = imgui.ImVec4(1.0, 0.75, 0.3, 1.0)
COLOR_MODIFIED = imgui.ImVec4(1.0, 0.72, 0.40, 1.0)
COLOR_ERROR = imgui.ImVec4(1.0, 0.4, 0.4, 1.0)
COLORS_RUN = (
    imgui.ImVec4(0.13, 0.55, 0.13, 1.0),
    imgui.ImVec4(0.18, 0.65, 0.18, 1.0),
    imgui.ImVec4(0.10, 0.45, 0.10, 1.0),
)
COLORS_DEFAULTS = (
    imgui.ImVec4(0.60, 0.35, 0.10, 1.0),
    imgui.ImVec4(0.70, 0.42, 0.14, 1.0),
    imgui.ImVec4(0.50, 0.28, 0.08, 1.0),
)

WIDTH_INPUT_EM = 7.5
WIDTH_MIN_EM = 24
WIDTH_RUN_EM = 15

NO_DEFAULT = object()


def text_default(param: scraper.Param) -> str:
    """The text a parameter's input starts with: its default, spelled as the command line takes it."""
    if param.required or param.default is None:
        return ""
    if isinstance(param.default, bool):
        return "true" if param.default else "false"
    if isinstance(param.default, tuple):
        return ",".join(str(value) for value in param.default)
    return str(param.default)


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


def is_hdf5(filepath: str) -> bool:
    """Whether a path names an hdf5 file."""
    return Path(filepath).suffix.lower() in cli.SUFFIXES_HDF5


def title_of(section: scraper.Section) -> str:
    """A section's heading, e.g. "Motion correct" for motion-correct."""
    return section.name.replace("-", " ").capitalize()


def label_of(kind: str, section: scraper.Section) -> str:
    """A config's name in a stage's dropdown, marking the pipeline's default."""
    return f"{kind} (default)" if kind == section.default_kind else kind


def width_visible() -> float:
    """The width of the current window's visible area, scrollbars excluded."""
    rect = imgui.internal.get_current_window().inner_rect
    return rect.max.x - rect.min.x


def draw_wrapped(text: str, color: Optional[imgui.ImVec4] = None) -> None:
    """Text wrapped at the current window's right edge, in color when given."""
    if color is None:
        imgui.text_wrapped(text)
    else:
        idl.text_wrapped_colored(color, text)


def same_line_if_fits(width: float) -> None:
    """Continue the line when an item width wide still fits before the window's right edge."""
    spacing = imgui.get_style().item_spacing.x
    right = imgui.get_window_pos().x + imgui.get_window_size().x - imgui.get_style().window_padding.x
    if imgui.get_item_rect_max().x + spacing + width <= right:
        imgui.same_line()


def draw_hint(text: str) -> None:
    """A dim (?) after the previous item that shows text when hovered."""
    imgui.same_line()
    imgui.text_disabled("(?)")
    if imgui.is_item_hovered():
        idl.wrapped_tooltip(text)


def draw_subsection(text: str, hint: str) -> None:
    """A light blue subsection heading with a (?) hint."""
    imgui.text_colored(COLOR_SUBSECTION, text)
    draw_hint(text=hint)


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


class Launcher:
    """
    State behind the launcher window: the chosen pipeline and every value typed into it.

    Args:
        slug_initial (str): The pipeline selected when the window opens
    """

    def __init__(self, slug_initial: str = PIPELINE_INITIAL):
        self.slugs = sorted(scraper.pipeline_registry())
        self.index_pipeline = self.slugs.index(slug_initial) if slug_initial in self.slugs else 0
        self.spec: Optional[scraper.PipelineSpec] = None
        self.paths: dict[str, str] = {}
        self.texts: dict[str, str] = {}
        self.values: dict[str, Any] = {}
        self.buffers: dict[str, str] = {}
        self.errors: dict[str, str] = {}
        self.dataset = ""
        self.picker = None
        self.target_picker: Optional[tuple[dict, str]] = None
        self.argv: Optional[list[str]] = None

        self.store = idl.JsonPreferenceStore(path=str(DIR_CONFIG / "recent.json"))
        self.config = idl.FileDialogConfig(
            title="masknmf",
            buttons=[],
            header_draw=self.draw_header,
            body_draw=self.draw_body,
            footer_draw=self.draw_footer,
            show_options_button=False,
            close_on_select=False,
            window_title="masknmf",
            window_size=(600, 900),
            ini_path=str(DIR_CONFIG / "launcher.ini"),
            on_cancel=self.quit,
        )
        self.select_pipeline(index=self.index_pipeline)

    def select_pipeline(self, index: int) -> None:
        """Switch pipelines, resetting every value but the movie paths."""
        self.index_pipeline = index
        self.spec = cli.spec_for(slug=self.slugs[index])
        self.texts = {
            param.name: text_default(param=param)
            for param in (*self.spec.run_scalars, *self.spec.scalars)
        }
        self.values = {section.argument: section.value_for(kind=section.default_kind) for section in self.spec.sections}
        self.buffers = {}
        self.errors = {}
        for param in (*self.spec.movie_params, *self.spec.array_params):
            self.paths.setdefault(param.field, "")

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
        """Constructor arguments that name a folder, shown with the output section."""
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

    def movies_given(self) -> list[str]:
        """The movie paths filled in, in run argument order."""
        return [
            self.paths[param.field].strip()
            for param in self.spec.movie_params
            if self.paths[param.field].strip() != ""
        ]

    def problems(self) -> list[str]:
        """Everything that has to be fixed before the pipeline can run."""
        problems = []
        movies = self.movies_given()
        if len(movies) == 0:
            problems.append("choose a movie")
        problems += [f"not found: {p}" for p in movies if not Path(p).expanduser().exists()]
        if any(is_hdf5(filepath=p) for p in movies) and self.dataset.strip() == "":
            problems.append("name the dataset holding the movie in the hdf5 file")
        for param in self.spec.array_params:
            path = self.paths[param.field].strip()
            if path == "" and param.required:
                problems.append(f"{param.field} needs a .npy file")
            elif path != "" and not Path(path).expanduser().exists():
                problems.append(f"not found: {path}")
        params = [*self.spec.run_scalars, *self.spec.scalars]
        problems += [e for e in (self.error_for(param=p) for p in params) if e is not None]
        problems += list(self.errors.values())
        return problems

    def modified(self) -> list[tuple[str, str, str]]:
        """Every value that differs from its default: name, current value, default."""
        rows = []
        for param in (*self.spec.run_scalars, *self.spec.scalars):
            if self.is_modified(param=param):
                rows.append((param.field, self.texts[param.name], text_default(param=param) or "none"))
        for section in self.spec.sections:
            diff_config(current=self.values[section.argument], default=section.default, path=section.argument, rows=rows)
        return rows

    def build_argv(self) -> list[str]:
        """The `masknmf` arguments the window's values amount to, writing changed configs to a json file."""
        spec = self.spec
        argv = ["run", "--pipeline", spec.slug]
        for param in spec.movie_params:
            path = self.paths[param.field].strip()
            if path == "":
                continue
            argv += [path] if len(spec.movie_params) == 1 else [cli.flag_for(param), path]
        if any(is_hdf5(filepath=p) for p in self.movies_given()):
            argv += ["--dataset", self.dataset.strip()]
        for param in spec.array_params:
            path = self.paths[param.field].strip()
            if path != "":
                argv += [cli.flag_for(param), path]
        for param in spec.run_scalars:
            if param.required or self.is_changed(param=param):
                argv += [cli.flag_for(param), self.texts[param.name]]
        for param in spec.scalars:
            if self.is_changed(param=param):
                argv += [f"--{param.name}", self.texts[param.name]]
        sections = self.sections_changed()
        if len(sections) > 0:
            configs = {section.argument: self.values[section.argument] for section in sections}
            FILEPATH_RUN_CONFIGS.write_text(
                json.dumps({"pipeline": spec.cls.__name__, **configs}, indent=2, default=scraper.config_json_value)
            )
            argv += ["--config", str(FILEPATH_RUN_CONFIGS)]
        return argv

    def quit(self) -> None:
        """Close the window without running; Esc and Quit cancel the dialog, which leaves exiting to its host."""
        hello_imgui.get_runner_params().app_shall_exit = True

    def open_picker(self, target: dict, key: str, filetypes: Optional[list] = None) -> None:
        """Open a native picker whose choice lands in target[key]; a folder picker without filetypes."""
        if self.picker is not None:
            return
        start = self.store.default_dir() or str(Path.home())
        if filetypes is None:
            self.picker = pfd.select_folder("Select folder", start)
        else:
            self.picker = pfd.open_file("Select file", start, idl.flatten_filters(filetypes), pfd.opt.none)
        self.target_picker = (target, key)

    def poll_picker(self) -> None:
        """Land a finished picker's choice and remember its folder for the next picker."""
        if self.picker is None or not self.picker.ready():
            return
        chosen = self.picker.result()
        if isinstance(chosen, list):
            chosen = chosen[0] if len(chosen) > 0 else ""
        if chosen:
            target, key = self.target_picker
            target[key] = chosen
            self.store.record_selection(idl.DialogResult(paths=[chosen]))
        self.picker = None
        self.target_picker = None

    def draw_header(self) -> None:
        """The name, version and what masknmf does, left aligned."""
        imgui.dummy(hello_imgui.em_to_vec2(0, 0.2))
        imgui.text_colored(COLOR_TITLE, "masknmf")
        imgui.same_line()
        imgui.text_disabled(f"v{masknmf.__version__}")
        imgui.push_text_wrap_pos(imgui.get_cursor_pos_x() + width_visible())
        imgui.text_colored(COLOR_DIM, "Motion correction, compression and demixing of functional imaging data")
        imgui.pop_text_wrap_pos()

    def draw_body(self) -> None:
        """
        Every section, one under another, in a child as wide as the visible area.

        Sizing against that width rather than the scrolling content's keeps wrapping and
        full width rows inside the window; only narrower than WIDTH_MIN_EM does it scroll.
        """
        width = max(width_visible(), hello_imgui.em_size(WIDTH_MIN_EM))
        flags_window = imgui.WindowFlags_.no_scrollbar | imgui.WindowFlags_.no_scroll_with_mouse
        with (
            imgui_ctx.begin_child("##launcher", imgui.ImVec2(width, 0), imgui.ChildFlags_.auto_resize_y, flags_window),
            imgui_ctx.push_style_var(imgui.StyleVar_.item_spacing, hello_imgui.em_to_vec2(0.55, 0.3)),
            imgui_ctx.push_style_var(imgui.StyleVar_.frame_padding, hello_imgui.em_to_vec2(0.35, 0.18)),
        ):
            self.draw_pipeline()
            draw_divider()
            self.draw_input()
            draw_divider()
            self.draw_output()
            draw_divider()
            self.draw_run_parameters()
            draw_divider()
            self.draw_stages()
            draw_divider()
            self.draw_runtime()
            draw_divider()
            self.draw_modified()
        self.poll_picker()

    def draw_pipeline(self) -> None:
        """Which pipeline runs."""
        draw_subsection(text="Pipeline", hint="Which masknmf pipeline runs. Switching resets every value but the movie.")
        imgui.spacing()
        imgui.set_next_item_width(max(self.width_combo(items=self.slugs), hello_imgui.em_size(WIDTH_INPUT_EM)))
        changed, index = imgui.combo("##pipeline", self.index_pipeline, self.slugs)
        if changed:
            self.select_pipeline(index=index)
        same_line_if_fits(width=imgui.calc_text_size(self.spec.cls.__name__).x)
        imgui.text_colored(COLOR_DIM, self.spec.cls.__name__)

    def draw_input(self) -> None:
        """The movie(s), the hdf5 dataset, and any .npy inputs the pipeline takes."""
        draw_subsection(
            text="Input movie" if len(self.spec.movie_params) == 1 else "Input movies",
            hint="A .tif/.tiff file, a folder of tiffs read in name order, or an .h5/.hdf5 file. "
            "Type or paste a path, or browse with the file or folder button.",
        )
        imgui.spacing()
        for param in self.spec.movie_params:
            if len(self.spec.movie_params) > 1:
                imgui.text(param.field.replace("_", " "))
            self.draw_path(target=self.paths, key=param.field, filetypes=FILETYPES_MOVIE, folders=True)
        if any(is_hdf5(filepath=p) for p in self.movies_given()):
            imgui.set_next_item_width(max(self.width_frame(text=self.dataset), hello_imgui.em_size(WIDTH_INPUT_EM)))
            _, self.dataset = imgui.input_text_with_hint("##dataset", "required", self.dataset)
            imgui.same_line(0, imgui.get_style().item_inner_spacing.x)
            if self.dataset.strip() == "":
                imgui.text_colored(COLOR_ERROR, "hdf5 dataset")
            else:
                imgui.text("hdf5 dataset")
        for param in self.spec.array_params:
            imgui.spacing()
            imgui.text(f"{param.field.replace('_', ' ')} (.npy)")
            self.draw_path(target=self.paths, key=param.field, filetypes=FILETYPES_ARRAY, folders=False)

    def draw_output(self) -> None:
        """Where the run folder is made."""
        draw_subsection(
            text="Output folder",
            hint="Each run writes a timestamped folder here holding results.hdf5 and config.json. "
            "Empty uses the folder masknmf was started from.",
        )
        imgui.spacing()
        for param in self.params_folder():
            self.draw_path(target=self.texts, key=param.name, filetypes=None, folders=True, hint="working directory")

    def draw_run_parameters(self) -> None:
        """The pipeline's run arguments that are not movies or arrays."""
        draw_subsection(text="Recording", hint="Arguments of the pipeline's run(): what the recording is.")
        imgui.spacing()
        for param in self.spec.run_scalars:
            self.draw_param(param=param)

    def draw_stages(self) -> None:
        """One bordered box per pipeline stage."""
        draw_subsection(
            text="Pipeline stages",
            hint="Every config a stage accepts is listed; the pipeline's own is selected and shows the values it "
            "runs with. Changed values show orange and are listed under Modified parameters.",
        )
        imgui.spacing()
        for section in self.spec.sections:
            self.draw_stage(section=section)

    def draw_stage(self, section: scraper.Section) -> None:
        """A stage's box: which config it takes, a reset once it differs from the default, then the config's fields."""
        flags = imgui.ChildFlags_.borders | imgui.ChildFlags_.auto_resize_y | imgui.ChildFlags_.always_use_window_padding
        with imgui_ctx.begin_child(f"##stage_{section.name}", imgui.ImVec2(0, 0), flags):
            imgui.text_colored(COLOR_TITLE, title_of(section=section))
            if not same(self.values[section.argument], section.default):
                same_line_if_fits(width=self.width_frame(text="Reset stage"))
                if imgui.small_button(f"Reset stage##{section.name}"):
                    self.reset_section(section=section)
                if imgui.is_item_hovered():
                    idl.wrapped_tooltip(f"Back to {label_of(kind=section.default_kind, section=section)} with the pipeline's values")
            kinds = cli.kinds_buildable(section=section)
            value = self.values[section.argument]
            kind = scraper.kind_of(value=value)
            labels = [label_of(kind=k, section=section) for k in kinds]
            changed_kind = kind != section.default_kind
            if changed_kind:
                imgui.push_style_color(imgui.Col_.text, COLOR_MODIFIED)
            imgui.set_next_item_width(max(self.width_combo(items=labels), hello_imgui.em_size(WIDTH_INPUT_EM)))
            edited, index = imgui.combo(f"##kind_{section.name}", kinds.index(kind), labels)
            if changed_kind:
                imgui.pop_style_color()
            if edited:
                self.select_kind(section=section, kind=kinds[index])
            imgui.same_line(0, imgui.get_style().item_inner_spacing.x)
            draw_wrapped(text="config", color=COLOR_MODIFIED if changed_kind else None)
            imgui.separator()
            value = self.values[section.argument]
            if isinstance(value, str):
                draw_wrapped(text="Skipped.", color=COLOR_DIM)
                return
            default = section.default if type(section.default) is type(value) else NO_DEFAULT
            self.draw_config(config=value, default=default, path=section.argument)

    def draw_config(self, config: Any, default: Any, path: str) -> None:
        """
        A config's fields: nested configs under their name, lists of configs one box per item, editable
        values with a widget, and values only Python can set shown read-only when they are set.

        Args:
            config (Any): The config dataclass being edited
            default (Any): The pipeline's default at the same place, or NO_DEFAULT when it has none
            path (str): Where config sits, naming its fields' widgets and errors
        """
        hints = scraper.resolve_hints(cls=type(config))
        params = {param.field: param for param in scraper.scrape_dataclass(cls_config=type(config), name_section=path)}
        for field in dataclasses.fields(config):
            if not field.init:
                continue
            annotation = hints.get(field.name, field.type)
            current = getattr(config, field.name)
            default_field = NO_DEFAULT if default is NO_DEFAULT else getattr(default, field.name)
            path_field = f"{path}.{field.name}"
            members, allows_none = scraper.annotation_members(annotation=annotation)
            configs = [m for m in members if dataclasses.is_dataclass(m)]
            classes_item = [scraper.item_dataclass_of(annotation=m) for m in members]
            classes_item = [c for c in classes_item if c is not None]
            if len(configs) > 0 or dataclasses.is_dataclass(current):
                self.draw_nested(config=config, name=field.name, configs=configs, allows_none=allows_none,
                                 default=default_field, path=path_field)
            elif len(classes_item) > 0:
                self.draw_items(config=config, name=field.name, cls_item=classes_item[0], default=default_field, path=path_field)
            elif params[field.name].settable:
                self.draw_field(config=config, param=params[field.name], default=default_field, path=path_field)
            elif current is not None:
                draw_wrapped(text=f"{field.name}: {text_of(value=current)} (set from Python)", color=COLOR_DIM)

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
        imgui.spacing()
        if len(kinds) > 1:
            labels = [f"{k} (default)" if default is not NO_DEFAULT and k == kind_or_none(value=default) else k for k in kinds]
            if modified:
                imgui.push_style_color(imgui.Col_.text, COLOR_MODIFIED)
            imgui.set_next_item_width(max(self.width_combo(items=labels), hello_imgui.em_size(WIDTH_INPUT_EM)))
            edited, index = imgui.combo(f"##{path}", kinds.index(kind), labels)
            if modified:
                imgui.pop_style_color()
            imgui.same_line(0, imgui.get_style().item_inner_spacing.x)
            draw_wrapped(text=name, color=COLOR_MODIFIED if modified else COLOR_SUBSECTION)
            if edited:
                self.select_nested(config=config, name=name, kind=kinds[index], default=default, path=path)
            if modified and self.draw_reset(path=path, text_default=kind_or_none(value=default)):
                self.reset_field(config=config, name=name, default=default, path=path)
        else:
            imgui.text_colored(COLOR_SUBSECTION, name)
        current = getattr(config, name)
        if dataclasses.is_dataclass(current):
            imgui.indent(hello_imgui.em_size(0.8))
            matches = default is not NO_DEFAULT and type(default) is type(current)
            self.draw_config(config=current, default=default if matches else NO_DEFAULT, path=path)
            imgui.unindent(hello_imgui.em_size(0.8))

    def draw_items(self, config: Any, name: str, cls_item: type, default: Any, path: str) -> None:
        """
        A list of configs, such as a multipass config's passes: a box per item with a remove button,
        then a button adding a copy of the last item, or a fresh one when the list is empty or None.

        Args:
            config (Any): The config holding the list
            name (str): The list's field
            cls_item (type): The config class the list holds
            default (Any): The pipeline's default list, or NO_DEFAULT
            path (str): Where the list sits
        """
        items = getattr(config, name)
        defaults = default if isinstance(default, list) else []
        flags = imgui.ChildFlags_.borders | imgui.ChildFlags_.auto_resize_y | imgui.ChildFlags_.always_use_window_padding
        if items is None or len(items) == 0:
            draw_wrapped(text=f"{name}: none" if items is None else f"{name}: no passes", color=COLOR_DIM)
        removed = None
        for i, item in enumerate(items or []):
            with imgui_ctx.begin_child(f"##{path}[{i}]", imgui.ImVec2(0, 0), flags):
                added = i >= len(defaults)
                imgui.text_colored(COLOR_MODIFIED if added else COLOR_SUBSECTION, f"Pass {i + 1}{' (added)' if added else ''}")
                if len(items) > 1:
                    same_line_if_fits(width=self.width_frame(text="Remove"))
                    if imgui.small_button(f"Remove##{path}[{i}]"):
                        removed = i
                self.draw_config(config=item, default=NO_DEFAULT if added else defaults[i], path=f"{path}[{i}]")
        if removed is not None:
            del items[removed]
            self.forget(path=path)
        if imgui.button(f"{fa.ICON_FA_PLUS}  Add pass##{path}"):
            fresh = copy.deepcopy(items[-1]) if items else scraper.build_default(cls_config=cls_item)
            setattr(config, name, [*(items or []), fresh])
        if imgui.is_item_hovered():
            idl.wrapped_tooltip("Adds a pass copying the last one")

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
            imgui.set_next_item_width(max(width, self.width_combo(items=items)))
            edited, index = imgui.combo(f"##{path}", list(param.choices).index(current), items)
            if edited:
                setattr(config, name, param.choices[index])
        elif widget == "int":
            imgui.set_next_item_width(max(width, self.width_stepped(text=str(current))))
            edited, value = imgui.input_int(f"##{path}", int(current))
            if edited:
                setattr(config, name, value)
        elif widget == "float":
            imgui.set_next_item_width(max(width, self.width_frame(text=text_of(value=float(current))) + hello_imgui.em_size(1)))
            edited, value = imgui.input_float(f"##{path}", float(current), 0.0, 0.0, "%.6g")
            if edited:
                setattr(config, name, float(f"{value:.6g}"))
        elif widget in ("int2", "float2"):
            widest = max(self.width_frame(text=text_of(value=v)) for v in current)
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

        imgui.same_line(0, imgui.get_style().item_inner_spacing.x)
        error = self.errors.get(path)
        if error is not None:
            draw_wrapped(text=name, color=COLOR_ERROR)
        elif modified:
            draw_wrapped(text=name, color=COLOR_MODIFIED)
        else:
            draw_wrapped(text=name)
        if imgui.is_item_hovered():
            idl.wrapped_tooltip(error or name if default is NO_DEFAULT else error or f"pipeline default: {text_of(value=default) or 'none'}")
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
            imgui.set_next_item_width(max(hello_imgui.em_size(WIDTH_INPUT_EM), self.width_frame(text=text)))
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

    def draw_runtime(self) -> None:
        """The constructor arguments that are neither configs nor folders."""
        draw_subsection(text="Runtime", hint="Where and how the pipeline computes.")
        imgui.spacing()
        for param in self.params_runtime():
            self.draw_param(param=param)

    def draw_modified(self) -> None:
        """Every changed value beside its default."""
        rows = self.modified()
        imgui.text(f"Modified parameters ({len(rows)})")
        if len(rows) == 0:
            imgui.text_disabled("All parameters at defaults")
            return
        flags = (
            imgui.TableFlags_.row_bg
            | imgui.TableFlags_.borders_inner_h
            | imgui.TableFlags_.borders_outer
            | imgui.TableFlags_.sizing_stretch_prop
        )
        if not imgui.begin_table("##modified", 3, flags):
            return
        for heading, weight in (("Parameter", 4.0), ("Current", 2.5), ("Default", 2.5)):
            imgui.table_setup_column(heading, imgui.TableColumnFlags_.width_stretch, weight)
        imgui.table_headers_row()
        for name, current, default in rows:
            imgui.table_next_row()
            imgui.table_next_column()
            draw_wrapped(text=name, color=COLOR_MODIFIED)
            imgui.table_next_column()
            draw_wrapped(text=current)
            imgui.table_next_column()
            draw_wrapped(text=default, color=COLOR_DIM)
        imgui.end_table()

    def width_frame(self, text: str) -> float:
        """The width of a framed widget showing text, padding included."""
        return imgui.calc_text_size(text).x + 2 * imgui.get_style().frame_padding.x

    def width_stepped(self, text: str) -> float:
        """An integer field wide enough for its digits beside its - and + buttons."""
        style = imgui.get_style()
        return self.width_frame(text=text) + hello_imgui.em_size(1) + 2 * (imgui.get_frame_height() + style.item_inner_spacing.x)

    def width_combo(self, items: list[str]) -> float:
        """A combo wide enough for its longest item and its arrow."""
        return max(self.width_frame(text=item) for item in items) + imgui.get_frame_height()

    def draw_path(self, target: dict, key: str, filetypes: Optional[list], folders: bool, hint: str = "type or browse") -> None:
        """
        Browse buttons, then a path field filling the row that wraps a long path onto more lines.

        Args:
            target (dict): Holds the path
            key (str): The path's key in target
            filetypes (list | None): Filters for a file button; None for no file button
            folders (bool): Whether to show a folder button
            hint (str): Shown while the field is empty
        """
        inner = imgui.get_style().item_inner_spacing.x
        if filetypes is not None:
            if imgui.button(f"{fa.ICON_FA_FILE}##file_{key}"):
                self.open_picker(target=target, key=key, filetypes=filetypes)
            if imgui.is_item_hovered():
                idl.wrapped_tooltip("Browse for a file")
            imgui.same_line(0, inner)
        if folders:
            if imgui.button(f"{fa.ICON_FA_FOLDER_OPEN}##folder_{key}"):
                self.open_picker(target=target, key=key, filetypes=None)
            if imgui.is_item_hovered():
                idl.wrapped_tooltip("Browse for a folder")
            imgui.same_line(0, inner)

        padding = imgui.get_style().frame_padding
        width = max(imgui.get_content_region_avail().x, hello_imgui.em_size(12))
        width_wrap = width - 2 * padding.x - imgui.get_style().scrollbar_size
        height = imgui.calc_text_size(target[key] or " ", wrap_width=width_wrap).y + 2 * padding.y
        _, text = imgui.input_text_multiline(
            f"##path_{key}", target[key], imgui.ImVec2(width, height), imgui.InputTextFlags_.word_wrap
        )
        target[key] = text.replace("\n", "").replace("\r", "")
        if target[key] == "":
            corner = imgui.get_item_rect_min()
            imgui.get_window_draw_list().add_text(
                imgui.ImVec2(corner.x + padding.x, corner.y + padding.y),
                imgui.get_color_u32(imgui.Col_.text_disabled),
                hint,
            )
        path = target[key].strip()
        if path != "" and not Path(path).expanduser().exists():
            imgui.text_colored(COLOR_ERROR, "not found on this machine")

    def draw_param(self, param: scraper.Param) -> None:
        """A fixed width widget for one parameter with its name to the right, orange once changed."""
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
            imgui.set_next_item_width(max(width, self.width_combo(items=items)))
            edited, index = imgui.combo(f"##{key}", index, items)
            if edited:
                self.texts[key] = items[index]
        elif widget == "int":
            imgui.set_next_item_width(max(width, self.width_stepped(text=self.texts[key])))
            edited, value = imgui.input_int(f"##{key}", int(self.value_of(param=param)))
            if edited:
                self.texts[key] = str(value)
        elif widget == "float":
            imgui.set_next_item_width(max(width, self.width_frame(text=self.texts[key]) + hello_imgui.em_size(1)))
            edited, value = imgui.input_float(f"##{key}", float(self.value_of(param=param)), 0.0, 0.0, "%.6g")
            if edited:
                self.texts[key] = f"{value:.6g}"
        elif widget == "int2":
            widest = max(self.width_frame(text=piece) for piece in self.texts[key].split(","))
            imgui.set_next_item_width(max(width, 2 * (widest + hello_imgui.em_size(1)) + imgui.get_style().item_inner_spacing.x))
            edited, values = imgui.input_int2(f"##{key}", list(self.value_of(param=param)))
            if edited:
                self.texts[key] = f"{values[0]},{values[1]}"
        else:
            hint = "required" if param.required else "none"
            imgui.set_next_item_width(max(width, self.width_frame(text=self.texts[key]), self.width_frame(text=hint)))
            _, self.texts[key] = imgui.input_text_with_hint(f"##{key}", hint, self.texts[key])
        if changed_before:
            imgui.pop_style_color()

        imgui.same_line(0, imgui.get_style().item_inner_spacing.x)
        if error is not None:
            draw_wrapped(text=param.field, color=COLOR_ERROR)
        elif self.is_modified(param=param):
            draw_wrapped(text=param.field, color=COLOR_MODIFIED)
        else:
            draw_wrapped(text=param.field)
        if imgui.is_item_hovered():
            idl.wrapped_tooltip(error or cli.describe(param=param))
        if self.is_modified(param=param) and self.draw_reset(path=key, text_default=text_default(param=param) or "none"):
            self.texts[key] = text_default(param=param)

    def draw_footer(self, dialog: idl.FileDialog) -> None:
        """
        Run, Defaults and Quit centered on one row, or Run above the other two when the row does not fit,
        with anything blocking the run under them.
        """
        problems = self.problems()
        nothing_modified = len(self.modified()) == 0
        label_run = f"{fa.ICON_FA_PLAY}  Run {self.spec.slug}"
        label_defaults = f"{fa.ICON_FA_ARROW_ROTATE_LEFT}  Defaults"
        label_quit = "Quit"
        spacing = imgui.get_style().item_spacing.x
        width_run = max(self.width_frame(text=label_run) + hello_imgui.em_size(1), hello_imgui.em_size(WIDTH_RUN_EM))
        width_defaults = self.width_frame(text=label_defaults) + hello_imgui.em_size(1)
        width_quit = self.width_frame(text=label_quit) + hello_imgui.em_size(1.5)
        width_rest = width_defaults + spacing + width_quit
        one_row = width_run + spacing + width_rest <= imgui.get_content_region_avail().x
        height = hello_imgui.em_size(1.6)

        imgui.separator()
        imgui.spacing()
        idl.center_next_item(width_run + spacing + width_rest if one_row else width_run)
        for color, colors in zip((imgui.Col_.button, imgui.Col_.button_hovered, imgui.Col_.button_active), COLORS_RUN):
            imgui.push_style_color(color, colors)
        imgui.begin_disabled(len(problems) > 0)
        if imgui.button(label_run, imgui.ImVec2(width_run, height)):
            self.argv = self.build_argv()
            self.quit()
        imgui.end_disabled()
        imgui.pop_style_color(3)

        if one_row:
            imgui.same_line()
        else:
            idl.center_next_item(width_rest)
        for color, colors in zip((imgui.Col_.button, imgui.Col_.button_hovered, imgui.Col_.button_active), COLORS_DEFAULTS):
            imgui.push_style_color(color, colors)
        imgui.begin_disabled(nothing_modified)
        if imgui.button(label_defaults, imgui.ImVec2(width_defaults, height)):
            self.reset_all()
        imgui.end_disabled()
        imgui.pop_style_color(3)
        if imgui.is_item_hovered(imgui.HoveredFlags_.allow_when_disabled):
            idl.wrapped_tooltip("Every value is at its default" if nothing_modified else
                                "Reset every stage, run and runtime value to its default; movies, paths and required values stay")
        imgui.same_line()
        if imgui.button(label_quit, imgui.ImVec2(width_quit, height)):
            dialog.cancel()
        for problem in problems:
            idl.text_wrapped_colored(COLOR_WARN, problem)


def run_launcher() -> Optional[list[str]]:
    """
    Open the launcher and block until the user runs or quits.

    Returns:
        list[str] | None: The `masknmf` arguments to run, or None when the user quit
    """
    DIR_CONFIG.mkdir(parents=True, exist_ok=True)
    launcher = Launcher()
    idl.run_file_dialog(launcher.config)
    return launcher.argv
