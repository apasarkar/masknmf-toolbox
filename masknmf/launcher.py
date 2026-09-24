"""
The window `masknmf` opens when it is run with no arguments.

Every value in it becomes an argument of `masknmf run`, so a run started here is a
command line run, and the equivalent command is printed before it starts.
"""

from typing import Optional

import functools
from pathlib import Path

import imgui_data_loader as idl
from imgui_bundle import hello_imgui, imgui
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

FORMATS = (
    ("TIFF", ".tif, .tiff"),
    ("TIFF folder", "folder/"),
    ("HDF5", ".h5, .hdf5"),
)

WIDTH_PATH_EM = 18


def text_default(param: scraper.Param) -> str:
    """The text a parameter's input starts with: its default, spelled as the command line takes it."""
    if param.required or param.default is None:
        return ""
    if isinstance(param.default, bool):
        return "true" if param.default else "false"
    if isinstance(param.default, tuple):
        return ",".join(str(value) for value in param.default)
    return str(param.default)


def is_bool(param: scraper.Param) -> bool:
    """Whether a parameter is a boolean, drawn as a checkbox."""
    members, _ = scraper.annotation_members(annotation=param.annotation)
    return bool in members


def is_folder(param: scraper.Param) -> bool:
    """Whether a parameter takes a path, drawn with a folder picker."""
    members, _ = scraper.annotation_members(annotation=param.annotation)
    return Path in members


def is_hdf5(filepath: str) -> bool:
    """Whether a path names an hdf5 file."""
    return Path(filepath).suffix.lower() in cli.SUFFIXES_HDF5


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
            subtitle="motion correction, compression, demixing",
            buttons=[
                idl.ButtonSpec(
                    "Open Movie",
                    idl.PickKind.OPEN_FILE,
                    icon=fa.ICON_FA_FILE_IMAGE,
                    tooltip="A .tif/.tiff movie, or an .h5/.hdf5 file holding one",
                ),
                idl.ButtonSpec(
                    "Select Tiff Folder",
                    idl.PickKind.SELECT_FOLDER,
                    icon=fa.ICON_FA_FOLDER_OPEN,
                    tooltip="A folder of tiff files, read in name order as one movie",
                ),
            ],
            filetypes=FILETYPES_MOVIE,
            top_draw=self.draw_pipeline_selector,
            info=self.draw_card,
            info_title="About",
            buttons_title="Open",
            options_draw=self.draw_options,
            footer_draw=self.draw_footer,
            options_label="Options",
            close_on_select=False,
            window_title="masknmf",
            window_size=(960, 820),
            ini_path=str(DIR_CONFIG / "launcher.ini"),
            persistence=self.store,
            on_select=self.accept_selection,
            on_cancel=self.quit,
        )
        self.select_pipeline(index=self.index_pipeline)

    @property
    def theme(self) -> idl.Theme:
        return self.config.theme

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
        panels = [idl.Panel("Movie", self.draw_data), idl.Panel("Run", self.draw_run)]
        panels += [
            idl.Panel(section.name, functools.partial(self.draw_section, section))
            for section in self.spec.sections
            if len(self.kinds_offered(section=section)) > 1
        ]
        self.config.panels = panels

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

    def is_changed(self, param: scraper.Param) -> bool:
        """Whether a value differs from its default and so has to be passed."""
        return self.texts[param.name] != text_default(param=param)

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

    def accept_selection(self, result: idl.DialogResult) -> None:
        """Put a picked movie in the first empty movie slot, or the first slot when all are full."""
        fields = [param.field for param in self.spec.movie_params]
        empty = [field for field in fields if self.paths[field].strip() == ""]
        self.paths[empty[0] if len(empty) > 0 else fields[0]] = result.path

    def quit(self) -> None:
        """Close the window without running; Quit and Esc cancel the dialog, which leaves exiting to its host."""
        hello_imgui.get_runner_params().app_shall_exit = True

    def dir_start(self) -> str:
        """Where the native pickers open."""
        return self.store.default_dir() or str(Path.home())

    def open_picker(self, target: dict, key: str, filetypes: Optional[list] = None) -> None:
        """Open a native picker whose choice lands in target[key]; a folder picker without filetypes."""
        if self.picker is not None:
            return
        if filetypes is None:
            self.picker = pfd.select_folder("Select folder", self.dir_start())
        else:
            self.picker = pfd.open_file(
                "Select file", self.dir_start(), idl.flatten_filters(filetypes), pfd.opt.none
            )
        self.target_picker = (target, key)

    def poll_picker(self) -> None:
        """Land a finished picker's choice."""
        if self.picker is None or not self.picker.ready():
            return
        chosen = self.picker.result()
        if isinstance(chosen, list):
            chosen = chosen[0] if len(chosen) > 0 else ""
        if chosen:
            target, key = self.target_picker
            target[key] = chosen
        self.picker = None
        self.target_picker = None

    def width_frame(self, text: str) -> float:
        """The width of a framed widget showing text, padding included."""
        return imgui.calc_text_size(text).x + 2 * imgui.get_style().frame_padding.x

    def width_combo(self, items: list[str]) -> float:
        """A combo wide enough for its longest item and its arrow."""
        return max(self.width_frame(text=item) for item in items) + imgui.get_frame_height()

    def width_value(self, param: scraper.Param) -> float:
        """The natural width of a parameter's widget: its current text, its hint, or its longest choice."""
        if is_bool(param=param):
            return imgui.get_frame_height()
        if param.choices is not None:
            return self.width_combo(items=[str(choice) for choice in param.choices])
        hint = "required" if param.required else "none"
        return max(
            self.width_frame(text=self.texts[param.name]),
            self.width_frame(text=hint),
            hello_imgui.em_size(5),
        )

    def width_path(self) -> float:
        """A path field and its browse button."""
        return hello_imgui.em_size(WIDTH_PATH_EM) + imgui.get_style().item_spacing.x + imgui.get_frame_height()

    def draw_pipeline_selector(self, dialog: idl.FileDialog) -> None:
        """The pipeline combo above the picker buttons."""
        imgui.align_text_to_frame_padding()
        imgui.text_colored(idl.to_vec4(self.theme.text_dim), "pipeline")
        imgui.same_line()
        imgui.set_next_item_width(self.width_combo(items=self.slugs))
        changed, index = imgui.combo("##pipeline", self.index_pipeline, self.slugs)
        if imgui.is_item_hovered():
            idl.wrapped_tooltip(f"Pipeline: {self.spec.cls.__name__}")
        if changed:
            self.select_pipeline(index=index)

    def begin_table(self, id_table: str) -> bool:
        """A name / value table whose columns fit their widest cell; pair a True return with imgui.end_table."""
        flags = (
            imgui.TableFlags_.row_bg
            | imgui.TableFlags_.borders_inner_v
            | imgui.TableFlags_.sizing_fixed_fit
            | imgui.TableFlags_.no_host_extend_x
        )
        if not imgui.begin_table(id_table, 2, flags):
            return False
        imgui.table_setup_column("name", imgui.TableColumnFlags_.width_fixed)
        imgui.table_setup_column("value", imgui.TableColumnFlags_.width_fixed)
        return True

    def draw_label(self, text: str, color, tooltip: str) -> None:
        """The name cell of a table row."""
        imgui.table_next_row()
        imgui.table_next_column()
        imgui.align_text_to_frame_padding()
        imgui.text_colored(idl.to_vec4(color), text)
        if imgui.is_item_hovered():
            idl.wrapped_tooltip(tooltip)
        imgui.table_next_column()

    def draw_path_row(self, label: str, target: dict, key: str, filetypes: Optional[list], tooltip: str) -> None:
        """A path field that wraps a long path onto more lines, and a browse button; filetypes None browses for a folder."""
        path = target[key].strip()
        missing = path != "" and not Path(path).expanduser().exists()
        self.draw_label(
            text=label,
            color=self.theme.err if missing else self.theme.text,
            tooltip=f"not found: {path}" if missing else tooltip,
        )
        padding = imgui.get_style().frame_padding
        width = hello_imgui.em_size(WIDTH_PATH_EM)
        height_text = imgui.calc_text_size(target[key] or " ", False, width - 2 * padding.x).y
        size = imgui.ImVec2(width, height_text + 2 * padding.y)
        _, text = imgui.input_text_multiline(f"##path_{key}", target[key], size, imgui.InputTextFlags_.word_wrap)
        target[key] = text.replace("\n", "").replace("\r", "")
        if target[key] == "":
            position = imgui.get_item_rect_min()
            imgui.get_window_draw_list().add_text(
                imgui.ImVec2(position.x + padding.x, position.y + padding.y),
                imgui.get_color_u32(imgui.Col_.text_disabled),
                "type or browse",
            )
        imgui.same_line()
        if imgui.button(f"{fa.ICON_FA_FOLDER_OPEN}##browse_{key}", imgui.ImVec2(imgui.get_frame_height(), 0)):
            self.open_picker(target=target, key=key, filetypes=filetypes)

    def draw_param_row(self, param: scraper.Param, width: float) -> None:
        """One parameter at the table's value width: a checkbox, a combo of its choices, or a text input."""
        if is_folder(param=param):
            self.draw_path_row(
                label=param.field, target=self.texts, key=param.name, filetypes=None, tooltip=cli.describe(param=param)
            )
            return
        error = self.error_for(param=param)
        color = self.theme.err if error is not None else (
            self.theme.text if self.is_changed(param=param) else self.theme.text_dim
        )
        self.draw_label(text=param.field, color=color, tooltip=error or cli.describe(param=param))
        key = param.name
        if is_bool(param=param):
            changed, value = imgui.checkbox(f"##{key}", self.texts[key] == "true")
            if changed:
                self.texts[key] = "true" if value else "false"
            return
        imgui.set_next_item_width(width)
        if param.choices is not None:
            items = [str(choice) for choice in param.choices]
            index = items.index(self.texts[key]) if self.texts[key] in items else -1
            changed, index = imgui.combo(f"##{key}", index, items)
            if changed:
                self.texts[key] = items[index]
        else:
            hint = "required" if param.required else "none"
            _, self.texts[key] = imgui.input_text_with_hint(f"##{key}", hint, self.texts[key])

    def draw_param_table(self, id_table: str, params: list[scraper.Param]) -> None:
        """Parameters in a table whose value widgets all take the widest one's natural width."""
        if len(params) == 0:
            return
        width = max(self.width_path() if is_folder(param=p) else self.width_value(param=p) for p in params)
        if self.begin_table(id_table=id_table):
            for param in params:
                self.draw_param_row(param=param, width=width)
            imgui.end_table()

    def draw_section(self, section: scraper.Section) -> None:
        """A config section's panel: which config it receives, then that config's fields."""
        kinds = self.kinds_offered(section=section)
        if self.begin_table(id_table=f"##kind_{section.name}"):
            self.draw_label(text="config", color=self.theme.text, tooltip=f"which config {section.argument} receives")
            imgui.set_next_item_width(self.width_combo(items=kinds))
            index = kinds.index(self.kinds[section.name])
            changed, index = imgui.combo(f"##kind_{section.name}", index, kinds)
            if changed:
                self.select_kind(section=section, kind=kinds[index])
            imgui.end_table()
        self.draw_param_table(id_table=f"##fields_{section.name}", params=self.params_config(section=section))

    def draw_card(self, dialog: idl.FileDialog) -> None:
        """The info panel: version and the movie formats masknmf reads."""
        imgui.text_colored(idl.to_vec4(self.theme.text_dim), f"v{masknmf.__version__}")
        imgui.dummy(hello_imgui.em_to_vec2(0, 0.2))
        imgui.text_colored(idl.to_vec4(self.theme.accent), "Supported Formats")
        flags = (
            imgui.TableFlags_.borders_inner_v
            | imgui.TableFlags_.row_bg
            | imgui.TableFlags_.sizing_fixed_fit
            | imgui.TableFlags_.no_host_extend_x
        )
        if imgui.begin_table("##formats", 2, flags):
            imgui.table_setup_column("Format", imgui.TableColumnFlags_.width_fixed)
            imgui.table_setup_column("Extensions", imgui.TableColumnFlags_.width_fixed)
            imgui.table_headers_row()
            for name, extensions in FORMATS:
                imgui.table_next_row()
                imgui.table_next_column()
                imgui.text(name)
                imgui.table_next_column()
                imgui.text_colored(idl.to_vec4(self.theme.text_dim), extensions)
            imgui.end_table()

    def draw_data(self) -> None:
        """The movie panel: a path per movie, the hdf5 dataset, and any .npy inputs."""
        if self.begin_table(id_table="##data"):
            for param in self.spec.movie_params:
                label = "movie" if len(self.spec.movie_params) == 1 else param.field
                self.draw_path_row(
                    label=label, target=self.paths, key=param.field, filetypes=FILETYPES_MOVIE,
                    tooltip=f"imaging movie for {param.field}",
                )
            if any(is_hdf5(filepath=p) for p in self.movies_given()):
                self.draw_label(text="dataset", color=self.theme.text, tooltip="the hdf5 dataset holding the movie")
                imgui.set_next_item_width(max(self.width_frame(text=self.dataset), self.width_frame(text="required")))
                _, self.dataset = imgui.input_text_with_hint("##dataset", "required", self.dataset)
            for param in self.spec.array_params:
                self.draw_path_row(
                    label=param.field, target=self.paths, key=param.field, filetypes=FILETYPES_ARRAY,
                    tooltip=f".npy file holding {param.field}",
                )
            imgui.end_table()
        self.poll_picker()

    def draw_run(self) -> None:
        """The run panel: the pipeline's run arguments that are not movies or arrays."""
        self.draw_param_table(id_table="##run", params=self.spec.run_scalars)

    def draw_options(self, dialog: idl.FileDialog) -> None:
        """The options popup: the pipeline's constructor arguments that are not configs."""
        self.draw_param_table(id_table="##options", params=self.spec.scalars)
        self.poll_picker()

    def draw_footer(self, dialog: idl.FileDialog) -> None:
        """Run, Options and Quit, with the first problem blocking Run above them."""
        problems = self.problems()
        if len(problems) > 0:
            idl.center_text(problems[0], self.theme.text_dim)
        imgui.dummy(hello_imgui.em_to_vec2(0, 0.2))

        label_run = f"{fa.ICON_FA_PLAY}  Run"
        label_options = f"{fa.ICON_FA_GEARS}  Options"
        label_quit = f"{fa.ICON_FA_XMARK}  Quit"
        width_run, width_options, width_quit = (
            max(self.width_frame(text=label) + hello_imgui.em_size(1), hello_imgui.em_size(6))
            for label in (label_run, label_options, label_quit)
        )
        spacing = imgui.get_style().item_spacing.x
        idl.center_next_item(width_run + width_options + width_quit + 2 * spacing)
        height = hello_imgui.em_size(1.5)

        idl.push_button_style(self.theme, primary=True)
        imgui.begin_disabled(len(problems) > 0)
        if imgui.button(label_run, imgui.ImVec2(width_run, height)):
            self.argv = self.build_argv()
            hello_imgui.get_runner_params().app_shall_exit = True
        imgui.end_disabled()
        idl.pop_button_style()
        if len(problems) > 0 and imgui.is_item_hovered(imgui.HoveredFlags_.allow_when_disabled):
            idl.wrapped_tooltip("\n".join(problems))

        imgui.same_line()
        idl.push_button_style(self.theme, primary=False)
        if imgui.button(label_options, imgui.ImVec2(width_options, height)):
            dialog.open_options()
        if imgui.is_item_hovered():
            idl.wrapped_tooltip("Output folder, device, batch size")
        imgui.same_line()
        if imgui.button(label_quit, imgui.ImVec2(width_quit, height)):
            dialog.cancel()
        idl.pop_button_style()


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
