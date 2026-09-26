"""
Command line entry point for masknmf.

Every option a pipeline accepts is discovered at runtime by
masknmf.pipelines.scraper, so --pipeline decides which flags exist and nothing
here enumerates a parameter by hand:

    masknmf pipelines
    masknmf params --pipeline two-photon-calcium
    masknmf run --pipeline two-photon-calcium movie.tif --fs 30
    masknmf run --pipeline two-photon-calcium movie.tif --fs 30 \\
        --motion-correct-kind piecewise-rigid --set compress.max_components=30
    masknmf view results.hdf5 --raw movie.tif
"""

from typing import Any, Optional

import argparse
import sys
from pathlib import Path

import numpy as np

import masknmf
from masknmf.pipelines import scraper


SUFFIXES_TIFF = (".tif", ".tiff")
SUFFIXES_HDF5 = (".h5", ".hdf5")

NAMES_ALIAS = {"frame_rate": "--fs"}


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
    """The masknmf stage groups a results file holds, in pipeline order."""
    names_known = (
        *group_names_registration(),
        group_name_compression(),
        group_name_demixing(),
    )
    return [
        name
        for name in names_known
        if has_stage(filepath_results=filepath_results, name_group=name)
    ]


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


def is_constructible(spec: scraper.PipelineSpec, section: scraper.Section, kind: str) -> bool:
    """Whether every required field of a section's config can be set from the command line."""
    params = spec.params_for(section=section, kind=kind)
    return all(param.settable for param in params if param.required)


def flag_for(param: scraper.Param) -> str:
    """The long flag a run argument is exposed under."""
    return f"--{param.field.replace('_', '-')}"


def option_name(flag: str) -> str:
    """The argparse destination a long flag lands in."""
    return flag.lstrip("-").replace("-", "_")


def build_bootstrap_parser() -> argparse.ArgumentParser:
    """A parser that reads only --pipeline, so the real parser can be built from it."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--pipeline", default=None)
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
            choices=list(section.kinds),
            default=None,
            help=f"which config {section.argument} receives",
        )

    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="SECTION.FIELD=VALUE",
        help="set any parameter `masknmf params` lists; repeatable",
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


def parse_overrides(texts: list[str]) -> dict[str, str]:
    """
    Read --set section.field=value strings into a mapping.

    Args:
        texts (list[str]): The raw --set entries
    Returns:
        dict[str, str]: Dotted name to unparsed value
    Raises:
        SystemExit: If an entry has no "="
    """
    overrides = {}
    for text in texts:
        if "=" not in text:
            fail(f"--set expects SECTION.FIELD=VALUE, got {text!r}")
        key, value = text.split("=", 1)
        overrides[key.strip()] = value.strip()
    return overrides


def build_section_value(
    spec: scraper.PipelineSpec,
    section: scraper.Section,
    kind: Optional[str],
    overrides: dict[str, str],
) -> tuple[bool, Any]:
    """
    Build the value a pipeline's config argument should receive.

    Args:
        spec (PipelineSpec): The scraped pipeline
        section (Section): The section being built
        kind (str | None): The --<section>-kind value, or None when the user did not choose
        overrides (dict[str, str]): Every --set entry, keyed by dotted name
    Returns:
        bool: Whether the caller should pass this value at all
        Any: The value, when it should be passed
    Raises:
        SystemExit: If a kind was chosen whose config cannot be built from the command
            line, or an override names a field the chosen config does not have
    """
    mine = {k: v for k, v in overrides.items() if k.startswith(f"{section.name}.")}

    if kind == "skip":
        return True, "skip"
    if kind is None and len(mine) == 0:
        return False, None

    if kind is None:
        kinds_real = [k for k in section.kinds if k != "skip"]
        if len(kinds_real) != 1:
            fail(
                f"--set touches {section.name} but several configs fit it; "
                f"pass --{section.name}-kind ({', '.join(kinds_real)})"
            )
        kind = kinds_real[0]

    if not is_constructible(spec=spec, section=section, kind=kind):
        required = [
            p.field
            for p in spec.params_for(section=section, kind=kind)
            if p.required and not p.settable
        ]
        fail(
            f"--{section.name}-kind {kind} cannot be built from the command line; "
            f"it requires {', '.join(required)}. Drive it from Python."
        )

    params = {p.field: p for p in spec.params_for(section=section, kind=kind)}
    kwargs = {}
    for key, text in mine.items():
        field = key.split(".", 1)[1]
        if field not in params:
            fail(
                f"{key} is not a field of --{section.name}-kind {kind}; "
                f"try `masknmf params --pipeline {spec.slug}`"
            )
        param = params[field]
        if not param.settable:
            fail(f"{key} cannot be set from the command line")
        try:
            kwargs[field] = scraper.coerce(param=param, text=text)
        except ValueError as error:
            fail(str(error))

    return True, section.configs_by_kind[kind](**kwargs)


def command_pipelines(args: argparse.Namespace) -> None:
    """List the pipelines masknmf exports."""
    for slug, cls in sorted(scraper.pipeline_registry().items()):
        spec = scraper.scrape(cls_pipeline=cls)
        sections = ", ".join(s.name for s in spec.sections)
        print(f"{slug}\n    {cls.__name__}\n    sections: {sections}")


def command_params(args: argparse.Namespace) -> None:
    """List every parameter the scraper found for one pipeline."""
    spec = spec_for(slug=args.pipeline)
    print(f"{spec.slug}  ({spec.cls.__name__})\n")

    print("run arguments")
    for param in spec.run_params:
        note = {"movie": "imaging movie", "array": ".npy file"}.get(param.kind, "")
        print(f"  {param.field:28} {describe(param=param)}{'  [' + note + ']' if note else ''}")

    print("\nconstructor arguments")
    for param in spec.scalars:
        print(f"  {param.field:28} {describe(param=param)}")

    for section in spec.sections:
        for kind in section.kinds:
            if kind == "skip":
                continue
            buildable = is_constructible(spec=spec, section=section, kind=kind)
            head = f"\n[{section.name}] --{section.name}-kind {kind}"
            print(head if buildable else f"{head}   (not constructible from the CLI)")
            for param in spec.params_for(section=section, kind=kind):
                mark = " " if param.settable else "*"
                print(f"  {mark} {param.name:34} {describe(param=param)}")
        if section.allows_skip:
            print(f"\n[{section.name}] --{section.name}-kind skip")
    print("\n(*) cannot be set from the command line")


def command_run(args: argparse.Namespace) -> None:
    """Build the pipeline the scraper described and run it."""
    spec = spec_for(slug=args.pipeline)
    overrides = parse_overrides(texts=args.overrides)

    known = {s.name for s in spec.sections}
    for key in overrides:
        if "." not in key or key.split(".", 1)[0] not in known:
            fail(
                f"--set {key} does not name a section of {spec.slug}; "
                f"sections are {', '.join(sorted(known))}"
            )

    kwargs_init = {}
    for param in spec.scalars:
        text = getattr(args, option_name(param.name), None)
        if text is not None:
            try:
                kwargs_init[param.field] = scraper.coerce(param=param, text=text)
            except ValueError as error:
                fail(str(error))

    for section in spec.sections:
        kind = getattr(args, option_name(f"{section.name}-kind"), None)
        passes, value = build_section_value(
            spec=spec, section=section, kind=kind, overrides=overrides
        )
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

    try:
        pipeline = spec.cls(**kwargs_init)
    except ValueError as error:
        fail(str(error))
    shapes = ", ".join(
        str(kwargs_run[p.field].shape)
        for p in spec.movie_params
        if kwargs_run.get(p.field) is not None
    )
    print(f"{spec.cls.__name__} on {shapes or 'stored results'}")
    try:
        run_folder = pipeline.run(**kwargs_run)
    except ValueError as error:
        fail(str(error))
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

    import fastplotlib as fpl

    device = (
        str(masknmf.utils.torch_select_device()) if args.device == "auto" else args.device
    )
    viewers = []
    registered = None

    name_registration = next(
        (n for n in group_names_registration() if n in names_present), None
    )
    if name_registration is not None and args.raw is not None:
        raw = load_movie(filepath_movie=args.raw, name_dataset=args.dataset)
        registered = getattr(masknmf, name_registration).from_hdf5(
            args.results, input_movie=raw
        )
        viewers.append(
            masknmf.MotionCorrectionVis(
                registration_array=registered,
                frame_timings=timings(registered.shape[0], args.fs),
                mean_subtract=True,
            )
        )
    elif name_registration is not None:
        print(f"skipping the {name_registration} viewer; it needs --raw")

    if group_name_compression() in names_present and registered is not None:
        compressed = masknmf.CompressionArray.from_hdf5(args.results)
        viewers.append(
            masknmf.CompressionVis(
                moco_stack=registered,
                pmd_stack=compressed,
                frame_timings=timings(compressed.shape[0], args.fs),
                device=device,
            )
        )

    if group_name_demixing() in names_present:
        results = masknmf.DemixingResults.from_hdf5(args.results, device=device)
        viewers.append(
            masknmf.SingleSessionDemixingVis(
                demixing_results=results,
                frame_timings=timings(results.shape[0], args.fs),
                device=device,
                results_path=args.results,
            )
        )
    elif group_name_compression() in names_present:
        compressed = masknmf.CompressionArray.from_hdf5(args.results)
        viewers.append(
            masknmf.SingleSessionDemixingVis(
                demixing_results=compressed,
                frame_timings=timings(compressed.shape[0], args.fs),
                device=device,
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
    parser_view.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser_view.add_argument(
        "--list", action="store_true", help="print what the file holds and exit"
    )
    parser_view.set_defaults(handler=command_view)

    return parser


def main(argv: Optional[list[str]] = None) -> None:
    """Parse the command line and dispatch."""
    argv = list(sys.argv[1:] if argv is None else argv)
    bootstrap, _ = build_bootstrap_parser().parse_known_args(argv)

    spec = None
    if bootstrap.pipeline is not None and bootstrap.pipeline in scraper.pipeline_registry():
        spec = spec_for(slug=bootstrap.pipeline)

    args = build_parser(spec=spec).parse_args(argv)
    args.handler(args)


if __name__ == "__main__":
    main()
