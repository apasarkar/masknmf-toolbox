"""
Command line entry point for masknmf.

Every option a pipeline accepts is discovered at runtime by
masknmf.pipelines.scraper, so --pipeline decides which flags exist and nothing
here enumerates a parameter by hand:

    masknmf
    masknmf pipelines
    masknmf params --pipeline two-photon-calcium
    masknmf run --pipeline two-photon-calcium movie.tif --fs 30
    masknmf run --pipeline two-photon-calcium movie.tif --fs 30 --motion-correct-kind piecewise-rigid
    masknmf params --pipeline two-photon-calcium --json > configs.json
    masknmf run movie.tif --fs 30 --config configs.json
    masknmf view results.hdf5 --raw movie.tif
"""

from typing import Any, Optional

import argparse
import dataclasses
import json
import shutil
import sys
from pathlib import Path

import h5py
import numpy as np

import masknmf
from masknmf.pipelines import scraper


SUFFIXES_TIFF = (".tif", ".tiff")
SUFFIXES_HDF5 = (".h5", ".hdf5")

NAMES_ALIAS = {"frame_rate": "--fs"}

CHARACTERS_NEEDING_QUOTES = set(" \t\\'&|;<>()$`!*?[]{}~#")


def group_names_registration() -> tuple[str, ...]:
    """The hdf5 group names a registration stage can be stored under."""
    return (
        masknmf.RigidRegistrationArray.__name__,
        masknmf.PiecewiseRigidRegistrationArray.__name__,
        masknmf.GradientRegistrationArray.__name__,
    )


def group_name_compression() -> str:
    """The hdf5 group name the compression stage is stored under."""
    return masknmf.CompressionArray.__name__


def group_name_demixing() -> str:
    """The hdf5 group name the demixing stage is stored under."""
    return masknmf.DemixingResults.__name__


def load_movie(filepath_movie: str, name_dataset: Optional[str] = None):
    """
    Open a raw movie with the loader matching its path.

    Args:
        filepath_movie (str): A tiff file, a directory of tiff files, or an hdf5 file
        name_dataset (str | None): For hdf5 input, the dataset holding the movie
    Returns:
        LazyFrameLoader: A (frames, height, width) lazy array
    Raises:
        SystemExit: If the path is not one masknmf can read, a tiff directory is empty,
            or an hdf5 dataset was not named
    """
    path_movie = Path(filepath_movie).expanduser()
    if not path_movie.exists():
        fail(f"no such file or directory: {path_movie}")

    if path_movie.is_dir():
        filepaths_tiffs = sorted(
            str(p) for p in path_movie.iterdir() if p.suffix.lower() in SUFFIXES_TIFF
        )
        if len(filepaths_tiffs) == 0:
            fail(f"no tiff files in {path_movie}")
        if len(filepaths_tiffs) == 1:
            return masknmf.TiffArray(filepaths_tiffs[0])
        return masknmf.TiffSeriesLoader(filepaths_tiffs)

    suffix = path_movie.suffix.lower()
    if suffix in SUFFIXES_TIFF:
        return masknmf.TiffArray(str(path_movie))
    if suffix in SUFFIXES_HDF5:
        if name_dataset is None:
            fail(f"{path_movie.name} is hdf5; name the movie dataset with --dataset")
        return masknmf.Hdf5Array(str(path_movie), name_dataset)

    fail(
        f"masknmf cannot read {path_movie.name}; expected a .tif/.tiff file, a directory "
        "of them, or a .h5/.hdf5 file"
    )


def has_stage(filepath_results: str, name_group: str) -> bool:
    """
    Whether a results file holds a stage group.

    masknmf.utils.has_group opens any path it is handed and raises a bare OSError
    on a non-hdf5 file, so the suffix is checked here first.
    """
    path_results = Path(filepath_results)
    if path_results.suffix.lower() not in SUFFIXES_HDF5:
        fail(f"{path_results.name} is not a .h5/.hdf5 results file")
    return masknmf.utils.has_group(str(path_results), name_group)


def groups_present(filepath_results: str) -> list[str]:
    """The masknmf stage groups a results file holds, in pipeline order, then demixing results under a prefix."""
    names_known = (
        *group_names_registration(),
        group_name_compression(),
        group_name_demixing(),
    )
    names = [
        name
        for name in names_known
        if has_stage(filepath_results=filepath_results, name_group=name)
    ]
    if len(names) > 0:
        with h5py.File(filepath_results, "r") as f:
            names += [
                f"{key}/{group_name_demixing()}"
                for key in f
                if isinstance(f[key], h5py.Group) and group_name_demixing() in f[key]
            ]
    return names


def format_command(argv: list[str]) -> str:
    """
    Spell arguments as a command line, double quoting any that a shell would split or expand.

    Args:
        argv (list[str]): The arguments, without the program name
    Returns:
        str: The arguments joined by spaces
    """
    pieces = []
    for arg in argv:
        needs_quotes = arg == "" or any(c in CHARACTERS_NEEDING_QUOTES for c in arg)
        pieces.append(f'"{arg}"' if needs_quotes else arg)
    return " ".join(pieces)


def fail(message: str) -> None:
    """Print an error and exit non-zero."""
    print(f"error: {message}", file=sys.stderr)
    raise SystemExit(2)


def spec_for(slug: str) -> scraper.PipelineSpec:
    """
    Scrape the pipeline a --pipeline value names.

    Args:
        slug (str): The command line name of a pipeline
    Returns:
        PipelineSpec: The scraped pipeline
    Raises:
        SystemExit: If the slug is not a pipeline masknmf exports
    """
    registry = scraper.pipeline_registry()
    if slug not in registry:
        fail(f"unknown pipeline {slug!r}; choose from {', '.join(sorted(registry))}")
    return scraper.scrape(cls_pipeline=registry[slug])


def kinds_buildable(section: scraper.Section) -> list[str]:
    """The kinds of a section that can be built without Python, which leaves out configs requiring arrays."""
    kinds = []
    for kind in section.kinds:
        try:
            section.value_for(kind=kind)
        except ValueError:
            continue
        kinds.append(kind)
    return kinds


def read_config_file(filepath: str) -> dict:
    """
    Read a --config file: the json `masknmf params --json` prints, or a run folder's config.json.

    Args:
        filepath (str): The file
    Returns:
        dict: Argument name to value
    Raises:
        SystemExit: If the file is missing or is not a json object
    """
    path = Path(filepath).expanduser()
    if not path.is_file():
        fail(f"no such config file: {path}")
    try:
        loaded = json.loads(path.read_text())
    except json.JSONDecodeError as error:
        fail(f"{path.name} is not valid json: {error}")
    if not isinstance(loaded, dict):
        fail(f"{path.name} should hold a json object of argument names to values")
    return loaded


def slug_of(name_class: str) -> str:
    """The --pipeline value of the pipeline class a config file names."""
    for slug, cls in scraper.pipeline_registry().items():
        if cls.__name__ == name_class:
            return slug
    fail(f"the config file names {name_class!r}, which is not a masknmf pipeline")


def section_value(section: scraper.Section, kind: Optional[str], value_file: Any) -> tuple[bool, Any]:
    """
    The value a pipeline's config argument receives from --<section>-kind and a --config file.

    The file's value builds on the pipeline's default, so a file may give only the fields it
    changes. A --<section>-kind naming another config than the file's replaces it with that
    config's defaults.

    Args:
        section (Section): The section
        kind (str | None): The --<section>-kind value, or None when not given
        value_file (Any): The file's value for the argument, or None when it has none
    Returns:
        bool: Whether the argument should be passed at all
        Any: The value, when it should be passed
    Raises:
        SystemExit: If the file's value names a config the section does not take, or a field the config does not have
    """
    if value_file is None:
        return (False, None) if kind is None else (True, section.value_for(kind=kind))
    kind_file = value_file if isinstance(value_file, str) else value_file.get("kind", section.default_kind)
    if kind_file not in section.kinds:
        fail(f"{section.argument} in the config file is {kind_file!r}; {section.name} takes {', '.join(section.kinds)}")
    if kind is not None and kind != kind_file:
        return True, section.value_for(kind=kind)
    try:
        return True, scraper.config_from_json(
            value=value_file, annotation=section.annotation, base=section.value_for(kind=kind_file)
        )
    except ValueError as error:
        fail(f"{section.argument} in the config file: {error}")


def flag_for(param: scraper.Param) -> str:
    """The long flag a run argument is exposed under."""
    return f"--{param.field.replace('_', '-')}"


def option_name(flag: str) -> str:
    """The argparse destination a long flag lands in."""
    return flag.lstrip("-").replace("-", "_")


def build_bootstrap_parser() -> argparse.ArgumentParser:
    """A parser that reads only --pipeline and --config, so the real parser can be built from them."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--pipeline", default=None)
    parser.add_argument("--config", default=None)
    return parser


def add_pipeline_options(parser: argparse.ArgumentParser, spec: scraper.PipelineSpec) -> None:
    """
    Add every flag the scraped pipeline accepts to a run parser.

    Args:
        parser (argparse.ArgumentParser): The run subparser
        spec (PipelineSpec): The scraped pipeline
    """
    for param in spec.movie_params:
        if len(spec.movie_params) > 1:
            parser.add_argument(flag_for(param), help=f"imaging movie for {param.field}", default=None)
            continue
        _, allows_none = scraper.annotation_members(annotation=param.annotation)
        parser.add_argument(
            param.field,
            nargs="?" if allows_none else None,
            help=f"imaging movie for {param.field}",
            default=None,
        )

    for param in spec.array_params:
        parser.add_argument(
            flag_for(param),
            default=None,
            metavar="NPY",
            help=f".npy file holding {param.field}",
        )

    for param in spec.run_scalars:
        flags = [flag_for(param)]
        if param.field in NAMES_ALIAS:
            flags.append(NAMES_ALIAS[param.field])
        parser.add_argument(
            *flags,
            default=None,
            metavar="VALUE",
            help=describe(param=param),
        )

    for param in spec.scalars:
        parser.add_argument(
            f"--{param.name}",
            default=None,
            metavar="VALUE",
            help=describe(param=param),
        )

    for section in spec.sections:
        parser.add_argument(
            f"--{section.name}-kind",
            choices=kinds_buildable(section=section),
            default=None,
            help=f"which config {section.argument} receives, at its defaults; default {section.default_kind}",
        )

    parser.add_argument(
        "--config",
        default=None,
        metavar="JSON",
        help="config values to run with: `masknmf params --json` output, or a run folder's config.json",
    )


def describe(param: scraper.Param) -> str:
    """A one line description of a parameter for argparse help."""
    pieces = []
    if param.choices is not None:
        pieces.append("one of " + ", ".join(str(c) for c in param.choices))
    if param.required:
        pieces.append("required")
    else:
        pieces.append(f"default {param.default!r}")
    return "; ".join(pieces)


def command_pipelines(args: argparse.Namespace) -> None:
    """List the pipelines masknmf exports."""
    for slug, cls in sorted(scraper.pipeline_registry().items()):
        spec = scraper.scrape(cls_pipeline=cls)
        sections = ", ".join(s.name for s in spec.sections)
        print(f"{slug}\n    {cls.__name__}\n    sections: {sections}")


def print_config(value: Any, depth: int) -> None:
    """
    Print a config's fields, nested configs and lists of configs indented under their field.

    Args:
        value (Any): A config dataclass
        depth (int): How many levels to indent
    """
    pad = "  " * depth
    hints = scraper.resolve_hints(cls=type(value))
    for field in dataclasses.fields(value):
        if not field.init:
            continue
        current = getattr(value, field.name)
        annotation = hints.get(field.name, field.type)
        if dataclasses.is_dataclass(current):
            print(f"{pad}  {field.name}")
            print_config(value=current, depth=depth + 1)
        elif scraper.item_dataclass_of(annotation=annotation) is not None:
            for i, item in enumerate(current):
                print(f"{pad}  {field.name}[{i}]")
                print_config(value=item, depth=depth + 1)
        elif scraper.is_settable(annotation=annotation):
            print(f"{pad}  {field.name:34} {current!r}")
        else:
            shown = "None" if current is None else type(current).__name__
            print(f"{pad}* {field.name:34} {shown}")


def command_params(args: argparse.Namespace) -> None:
    """List every parameter the scraper found for one pipeline, with the values it uses by default."""
    spec = spec_for(slug=args.pipeline)
    if args.json:
        configs = {section.argument: section.default for section in spec.sections}
        print(json.dumps({"pipeline": spec.cls.__name__, **configs}, indent=2, default=scraper.config_json_value))
        return

    print(f"{spec.slug}  ({spec.cls.__name__})\n")

    print("run arguments")
    for param in spec.run_params:
        note = {"movie": "imaging movie", "array": ".npy file"}.get(param.kind, "")
        print(f"  {param.field:28} {describe(param=param)}{'  [' + note + ']' if note else ''}")

    print("\nconstructor arguments")
    for param in spec.scalars:
        print(f"  {param.field:28} {describe(param=param)}")

    for section in spec.sections:
        print(f"\n[{section.name}] --{section.name}-kind {' | '.join(kinds_buildable(section=section))}")
        for kind in section.kinds:
            if kind == "skip":
                continue
            try:
                value = section.value_for(kind=kind)
            except ValueError:
                print(f"  {kind}   (not constructible from the CLI)")
                continue
            print(f"  {kind}{'   (default)' if kind == section.default_kind else ''}")
            print_config(value=value, depth=1)
    print("\n(*) set from Python only")


def command_run(args: argparse.Namespace) -> None:
    """Build the pipeline the scraper described and run it."""
    spec = spec_for(slug=args.pipeline)
    values_file = read_config_file(filepath=args.config) if args.config is not None else {}

    if values_file.get("pipeline", spec.cls.__name__) != spec.cls.__name__:
        fail(f"the config file is for {values_file['pipeline']}, not {spec.cls.__name__}")
    names_known = {s.argument for s in spec.sections} | {p.field for p in spec.scalars}
    unknown = set(values_file) - names_known - {"pipeline", "masknmf_version"}
    if len(unknown) > 0:
        fail(f"the config file sets {', '.join(sorted(unknown))}, which {spec.slug} does not take")

    kwargs_init = {}
    for param in spec.scalars:
        if param.field in values_file:
            kwargs_init[param.field] = values_file[param.field]
        text = getattr(args, option_name(param.name), None)
        if text is not None:
            try:
                kwargs_init[param.field] = scraper.coerce(param=param, text=text)
            except ValueError as error:
                fail(str(error))

    for section in spec.sections:
        kind = getattr(args, option_name(f"{section.name}-kind"), None)
        passes, value = section_value(section=section, kind=kind, value_file=values_file.get(section.argument))
        if passes:
            kwargs_init[section.argument] = value

    kwargs_run = {}
    for param in spec.movie_params:
        name = param.field if len(spec.movie_params) == 1 else option_name(flag_for(param))
        filepath = getattr(args, name, None)
        if filepath is None:
            kwargs_run[param.field] = None
        else:
            kwargs_run[param.field] = load_movie(
                filepath_movie=filepath, name_dataset=args.dataset
            )

    for param in spec.array_params:
        filepath = getattr(args, option_name(flag_for(param)), None)
        if filepath is None:
            if param.required:
                fail(f"--{param.field.replace('_', '-')} is required for {spec.slug}")
        else:
            kwargs_run[param.field] = np.load(filepath)

    for param in spec.run_scalars:
        text = getattr(args, option_name(flag_for(param)), None)
        if text is None:
            if param.required:
                fail(f"--{param.field.replace('_', '-')} is required for {spec.slug}")
            continue
        try:
            kwargs_run[param.field] = scraper.coerce(param=param, text=text)
        except ValueError as error:
            fail(str(error))

    filepaths_movie = [
        getattr(args, p.field if len(spec.movie_params) == 1 else option_name(flag_for(p)), None)
        for p in spec.movie_params
    ]
    filepaths_movie = [Path(f).expanduser().resolve() for f in filepaths_movie if f is not None]
    if kwargs_init.get("output_folder") is None and len(filepaths_movie) > 0:
        first = filepaths_movie[0]
        kwargs_init["output_folder"] = str(first if first.is_dir() else first.parent)

    pipeline = spec.cls(**kwargs_init)
    shapes = ", ".join(
        str(kwargs_run[p.field].shape)
        for p in spec.movie_params
        if kwargs_run.get(p.field) is not None
    )
    print(f"{spec.cls.__name__} on {shapes or 'stored results'}")
    try:
        run_folder = pipeline.run(**kwargs_run)
    except BaseException:
        # a failed run keeps its folder only when a later run can resume from its compression
        folder = pipeline.run_folder
        if folder is not None and not has_stage(
            filepath_results=str(folder / "results.hdf5"), name_group=group_name_compression()
        ):
            shutil.rmtree(folder)
            print(f"removed {folder}", file=sys.stderr)
        raise
    print(f"done: {run_folder}")


def command_view(args: argparse.Namespace) -> None:
    """Open the viewers for whatever stages a results file holds."""
    names_present = groups_present(filepath_results=args.results)
    if len(names_present) == 0:
        fail(f"{args.results} holds no masknmf results")

    print(Path(args.results).resolve())
    for name in names_present:
        print(f"  {name}")
    if args.list:
        return

    name_demixing = f"{args.prefix}/{group_name_demixing()}" if args.prefix else group_name_demixing()
    if name_demixing not in names_present and args.prefix:
        fail(f"{args.results} holds no {name_demixing}")

    import fastplotlib as fpl

    device = (
        str(masknmf.utils.torch_select_device()) if args.device == "auto" else args.device
    )
    viewers = []
    raw = None if args.raw is None else load_movie(filepath_movie=args.raw, name_dataset=args.dataset)

    if group_name_compression() in names_present and raw is not None:
        compressed = masknmf.CompressionArray.from_hdf5(args.results)
        # with registration skipped, the raw movie is what was compressed
        viewers.append(
            masknmf.CompressionVis(
                moco_stack=raw,
                pmd_stack=compressed,
                frame_timings=timings(compressed.shape[0], args.fs),
                device=device,
            )
        )
    elif group_name_compression() in names_present:
        print(f"skipping the {group_name_compression()} viewer; it needs --raw")

    if name_demixing in names_present:
        results = masknmf.DemixingResults.from_hdf5(args.results, prefix=args.prefix, device=device)
    elif group_name_compression() in names_present:
        results = masknmf.CompressionArray.from_hdf5(args.results)
    else:
        results = None
    if results is not None:
        # a raw movie the pipeline trimmed (the glutamate pipeline drops its first frames) no longer lines up
        if raw is not None and tuple(raw.shape) != tuple(results.shape):
            print(f"raw movie is {tuple(raw.shape)}, the results {tuple(results.shape)}; no raw panel")
        viewers.append(
            masknmf.SingleSessionDemixingVis(
                demixing_results=results,
                frame_timings=timings(results.shape[0], args.fs),
                device=device,
                results_path=args.results,
                raw=raw if raw is not None and tuple(raw.shape) == tuple(results.shape) else None,
            )
        )

    if len(viewers) == 0:
        fail(f"nothing to show for {args.results}")
    for viewer in viewers:
        viewer.show()
    fpl.loop.run()


def timings(num_frames: int, frame_rate: Optional[float]):
    """Frame times in seconds, or None when no acquisition rate was given."""
    if frame_rate is None:
        return None
    return np.arange(num_frames) / float(frame_rate)


def build_parser(spec: Optional[scraper.PipelineSpec]) -> argparse.ArgumentParser:
    """
    Build the full parser, adding pipeline specific flags when a pipeline was named.

    Args:
        spec (PipelineSpec | None): The scraped pipeline, or None when --pipeline was
            not given
    Returns:
        argparse.ArgumentParser: The parser
    """
    parser = argparse.ArgumentParser(
        prog="masknmf",
        description="Motion correct, compress and demix functional imaging data.",
    )
    parser.add_argument("--version", action="version", version=masknmf.__version__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    parser_pipelines = subparsers.add_parser("pipelines", help="list the available pipelines")
    parser_pipelines.set_defaults(handler=command_pipelines)

    parser_params = subparsers.add_parser(
        "params", help="list every parameter a pipeline accepts"
    )
    parser_params.add_argument("--pipeline", required=True)
    parser_params.add_argument(
        "--json", action="store_true", help="print the default configs as a --config file instead"
    )
    parser_params.set_defaults(handler=command_params)

    parser_run = subparsers.add_parser("run", help="run a pipeline")
    parser_run.add_argument(
        "--pipeline",
        required=True,
        choices=sorted(scraper.pipeline_registry()),
        help="which pipeline to run; decides the rest of the flags",
    )
    parser_run.add_argument(
        "--dataset", default=None, help="for hdf5 input, the dataset holding the movie"
    )
    if spec is not None:
        add_pipeline_options(parser=parser_run, spec=spec)
    parser_run.set_defaults(handler=command_run)

    parser_view = subparsers.add_parser("view", help="open the viewers for a results file")
    parser_view.add_argument("results")
    parser_view.add_argument("--raw", default=None, help="the raw movie the results came from")
    parser_view.add_argument("--dataset", default=None)
    parser_view.add_argument("--fs", default=None, type=float, help="acquisition rate in Hz")
    parser_view.add_argument(
        "--prefix", default="", help="group the demixing results sit under, e.g. global for the glutamate pipeline's whole-dendrite result"
    )
    parser_view.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser_view.add_argument(
        "--list", action="store_true", help="print what the file holds and exit"
    )
    parser_view.set_defaults(handler=command_view)

    return parser


def main(argv: Optional[list[str]] = None) -> None:
    """Parse the command line and dispatch."""
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) == 0:
        from masknmf import launcher

        argv = launcher.run_launcher()
        if argv is None:
            return
        print(f"masknmf {format_command(argv=argv)}")
    else:
        print("\r\033[K", end="", file=sys.stderr, flush=True)

    bootstrap, _ = build_bootstrap_parser().parse_known_args(argv)
    if argv[:1] == ["run"] and bootstrap.pipeline is None and bootstrap.config is not None:
        name_class = read_config_file(filepath=bootstrap.config).get("pipeline")
        if name_class is None:
            fail("the config file names no pipeline; pass --pipeline")
        bootstrap.pipeline = slug_of(name_class=name_class)
        argv = ["run", "--pipeline", bootstrap.pipeline, *argv[1:]]

    spec = None
    if bootstrap.pipeline is not None and bootstrap.pipeline in scraper.pipeline_registry():
        spec = spec_for(slug=bootstrap.pipeline)

    args = build_parser(spec=spec).parse_args(argv)
    args.handler(args)


if __name__ == "__main__":
    main()
