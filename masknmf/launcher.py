"""
The window `masknmf` opens when it is run with no arguments.

Every value in it becomes an argument of `masknmf run`, so a run started here is a
command line run, and the equivalent command is printed before it starts.
"""

from typing import Optional

import typing
from pathlib import Path

import imgui_data_loader as idl
from imgui_bundle import hello_imgui, imgui, imgui_ctx
from imgui_bundle import icons_fontawesome_6 as fa
from imgui_bundle import portable_file_dialogs as pfd

import masknmf
from masknmf import cli
from masknmf.pipelines import scraper


DIR_CONFIG = Path.home() / ".config" / "masknmf"

PIPELINE_INITIAL = "two-photon-calcium"

KIND_DEFAULT = "pipeline default"

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

WIDTH_INPUT_EM = 7.5
WIDTH_MIN_EM = 24
WIDTH_RUN_EM = 15


def text_default(param: scraper.Param) -> str:
    """The text a parameter's input starts with: its default, spelled as the command line takes it."""
    if param.required or param.default is None:
        return ""
    if isinstance(param.default, bool):
        return "true" if param.default else "false"
    if isinstance(param.default, tuple):
        return ",".join(str(value) for value in param.default)
    return str(param.default)


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


def is_hdf5(filepath: str) -> bool:
    """Whether a path names an hdf5 file."""
    return Path(filepath).suffix.lower() in cli.SUFFIXES_HDF5


def title_of(section: scraper.Section) -> str:
    """A section's heading, e.g. "Motion correct" for motion-correct."""
    return section.name.replace("-", " ").capitalize()


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
        self.kinds: dict[str, str] = {}
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
        self.kinds = {section.name: KIND_DEFAULT for section in self.spec.sections}
        for param in (*self.spec.movie_params, *self.spec.array_params):
            self.paths.setdefault(param.field, "")

    def select_kind(self, section: scraper.Section, kind: str) -> None:
        """Choose the config a section receives, resetting its fields to that config's defaults."""
        self.kinds[section.name] = kind
        prefix = f"{section.name}."
        self.texts = {k: v for k, v in self.texts.items() if not k.startswith(prefix)}
        for param in self.spec.params_for(section=section, kind=kind):
            self.texts[param.name] = text_default(param=param)

    def kinds_offered(self, section: scraper.Section) -> list[str]:
        """The configs a section can be given from here: those the command line can build."""
        kinds = [
            kind
            for kind in section.kinds
            if kind == "skip" or cli.is_constructible(spec=self.spec, section=section, kind=kind)
        ]
        return [KIND_DEFAULT, *kinds]

    def params_config(self, section: scraper.Section) -> list[scraper.Param]:
        """The settable fields of the config a section currently receives."""
        return [
            param
            for param in self.spec.params_for(section=section, kind=self.kinds[section.name])
            if param.settable
        ]

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
        for section in self.spec.sections:
            params += self.params_config(section=section)
        problems += [e for e in (self.error_for(param=p) for p in params) if e is not None]
        return problems

    def modified(self) -> list[tuple[str, str, str]]:
        """Every value that differs from its default: name, current value, default."""
        rows = []
        for param in (*self.spec.run_scalars, *self.spec.scalars):
            if self.is_modified(param=param):
                rows.append((param.field, self.texts[param.name], text_default(param=param) or "none"))
        for section in self.spec.sections:
            kind = self.kinds[section.name]
            if kind == KIND_DEFAULT:
                continue
            rows.append((f"{section.name} config", kind, KIND_DEFAULT))
            for param in self.params_config(section=section):
                if self.is_changed(param=param):
                    rows.append((param.name, self.texts[param.name], text_default(param=param) or "none"))
        return rows

    def build_argv(self) -> list[str]:
        """The `masknmf` arguments the window's values amount to."""
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
        for section in spec.sections:
            kind = self.kinds[section.name]
            if kind == KIND_DEFAULT:
                continue
            argv += [f"--{section.name}-kind", kind]
            for param in self.params_config(section=section):
                if self.is_changed(param=param):
                    argv += ["--set", f"{param.name}={self.texts[param.name]}"]
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
            hint="Each stage runs with the pipeline's own settings unless you pick a config. "
            "Changed values show orange and are listed under Modified parameters.",
        )
        imgui.spacing()
        for section in self.spec.sections:
            self.draw_stage(section=section)

    def draw_stage(self, section: scraper.Section) -> None:
        """A stage's box: a radio button per config it can take, then that config's fields."""
        flags = imgui.ChildFlags_.borders | imgui.ChildFlags_.auto_resize_y | imgui.ChildFlags_.always_use_window_padding
        with imgui_ctx.begin_child(f"##stage_{section.name}", imgui.ImVec2(0, 0), flags):
            imgui.text_colored(COLOR_TITLE, title_of(section=section))
            kinds = self.kinds_offered(section=section)
            if len(kinds) == 1:
                draw_wrapped(text="Runs with the pipeline's own settings; set it from Python to change it.", color=COLOR_DIM)
                return
            self.draw_kind_radios(section=section, kinds=kinds)
            imgui.separator()
            kind = self.kinds[section.name]
            if kind == KIND_DEFAULT:
                draw_wrapped(text="Runs with the pipeline's own settings.", color=COLOR_DIM)
                return
            if kind == "skip":
                draw_wrapped(text="Skipped.", color=COLOR_DIM)
                return
            for param in self.params_config(section=section):
                self.draw_param(param=param)

    def draw_kind_radios(self, section: scraper.Section, kinds: list[str]) -> None:
        """A radio button per config, continuing on the next line rather than past the box's edge."""
        style = imgui.get_style()
        right = imgui.get_window_pos().x + imgui.get_window_size().x - style.window_padding.x
        for i, kind in enumerate(kinds):
            label = "default" if kind == KIND_DEFAULT else kind
            width = imgui.get_frame_height() + style.item_inner_spacing.x + imgui.calc_text_size(label).x
            if i > 0 and imgui.get_item_rect_max().x + style.item_spacing.x + width <= right:
                imgui.same_line()
            if imgui.radio_button(f"{label}##{section.name}", self.kinds[section.name] == kind):
                self.select_kind(section=section, kind=kind)

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

    def draw_footer(self, dialog: idl.FileDialog) -> None:
        """A green Run button and Quit, centered, with anything blocking the run under them."""
        problems = self.problems()
        label_run = f"{fa.ICON_FA_PLAY}  Run {self.spec.slug}"
        label_quit = "Quit"
        style = imgui.get_style()
        width_run = max(self.width_frame(text=label_run) + hello_imgui.em_size(1), hello_imgui.em_size(WIDTH_RUN_EM))
        width_quit = self.width_frame(text=label_quit) + hello_imgui.em_size(1.5)
        height = hello_imgui.em_size(1.6)

        imgui.separator()
        imgui.spacing()
        idl.center_next_item(width_run + style.item_spacing.x + width_quit)
        for color, colors in zip((imgui.Col_.button, imgui.Col_.button_hovered, imgui.Col_.button_active), COLORS_RUN):
            imgui.push_style_color(color, colors)
        imgui.begin_disabled(len(problems) > 0)
        if imgui.button(label_run, imgui.ImVec2(width_run, height)):
            self.argv = self.build_argv()
            self.quit()
        imgui.end_disabled()
        imgui.pop_style_color(3)
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
