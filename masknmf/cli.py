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
    masknmf run --pipeline two-photon-calcium movie.tif --fs 30 --log-level debug
    masknmf params --pipeline two-photon-calcium --json > configs.json
    masknmf run movie.tif --fs 30 --config configs.json
    masknmf run "sessions/*/movie.tif" --config run_folder/config.json
    masknmf view results.hdf5 --raw movie.tif
    masknmf view run_folder
    masknmf view results.hdf5 --raw movie.tif --compression
    masknmf view results.hdf5 --classify --labels soma,dendrite,junk
    masknmf view "sessions/*/results.hdf5" --classify --classifier cells.roicat_classifier
    masknmf view tracking_folder
    masknmf view tracking_folder/20261001T180415_roicat-tracking-manifest.json
    masknmf view tracking_folder day1/results.hdf5 day2/results.hdf5
    masknmf train-classifier "sessions/*/results.hdf5" --out cells
    masknmf classify "new_sessions/**/results.hdf5" --classifier cells.roicat_classifier
    masknmf track "sessions/*/results.hdf5" --out tracking
"""

from typing import Any, Optional

import argparse
import dataclasses
import glob
import json
import logging
import os
import shutil
import sys
import time
import warnings
from datetime import datetime, timedelta
from pathlib import Path

import h5py
import numpy as np

import masknmf
from masknmf.classification import RoicatClassifier
from masknmf.demixing.curation import latest_results
from masknmf.demixing.labels import SIDECAR_SUFFIX, read_labels
from masknmf.io import (
    SUFFIXES_HDF5,
    SUFFIXES_TIFF,
    group_name_compression,
    group_name_demixing,
    group_names_registration,
    OpenedResults,
    stage_groups,
)
from masknmf.multisession import RoicatDataAdapter, RoicatTracker
from masknmf.pipelines import scraper


GLOB_TRACKING_MANIFEST = "*_roicat-tracking-manifest.json"

NAMES_ALIAS = {"frame_rate": "--fs"}

CHARACTERS_NEEDING_QUOTES = set(" \t\\'&|;<>()$`!*?[]{}~#")

logger = logging.getLogger("masknmf")


def load_movie(filepath_movie: str, name_dataset: Optional[str] = None):
    """masknmf.io.load_movie, a path it cannot read ending the command."""
    try:
        return masknmf.io.load_movie(filepath_movie, name_dataset)
    except (OSError, ValueError) as e:
        fail(f"{e}; an hdf5 movie's dataset is named with --dataset")


def has_stage(filepath_results: str, name_group: str) -> bool:
    """Whether a results file holds a stage group; a path that is not a results file ends the command."""
    return name_group in groups_present(filepath_results)


def groups_present(filepath_results: str) -> list[str]:
    """masknmf.io.stage_groups, a path that is not a results file ending the command."""
    try:
        return stage_groups(filepath_results)
    except (OSError, ValueError) as e:
        fail(str(e))


def expand_results(entries: list[str]) -> list[str]:
    """
    The .h5/.hdf5 results files a list of paths and globs names, labels sidecars left out. A named file is used as
    given; of what a glob matches, each results file stands in for itself only when no curated file of it matched,
    else the newest curated file does (``masknmf.demixing.latest_results``).

    Powershell and cmd hand globs over unexpanded, so they are expanded here, one path
    level at a time; ** walks every folder below. Neither ever enters a .zarr store,
    whose chunk folders can number in the millions, and each store passed over is reported.
    """
    files = []
    stores = []
    for entry in entries:
        parts = Path(entry).expanduser().parts
        if not any(c in entry for c in "*?["):
            if not os.path.isfile(Path(*parts)):
                fail(f"no such file: {entry}")
            files.append(str(Path(*parts)))
            continue
        index = next(i for i, part in enumerate(parts) if any(c in part for c in "*?["))
        candidates = [str(Path(*parts[:index])) if index > 0 else "."]
        for depth, part in enumerate(parts[index:], start=index):
            found = []
            for base in candidates:
                if part == "**":
                    for folder, names_folder, _ in os.walk(base):
                        stores += [os.path.join(folder, n) for n in names_folder if n.endswith(".zarr")]
                        names_folder[:] = sorted(n for n in names_folder if not n.endswith(".zarr"))
                        found.append(folder)
                    continue
                for path in sorted(glob.glob(os.path.join(glob.escape(base), part))):
                    if depth < len(parts) - 1 and path.endswith(".zarr"):
                        stores.append(path)
                    elif depth < len(parts) - 1 and os.path.isdir(path):
                        found.append(path)
                    elif depth == len(parts) - 1 and os.path.isfile(path):
                        found.append(path)
            candidates = found
        matches = [
            os.path.normpath(path)
            for path in candidates
            if path.lower().endswith(SUFFIXES_HDF5) and not path.endswith(SIDECAR_SUFFIX)
        ]
        if len(matches) == 0:
            fail(f"no results file matches {entry}")
        kept = latest_results(matches)
        if len(kept) < len(matches):
            print(f"{entry}: {len(matches) - len(kept)} file(s) left out, each replaced by a newer curated file")
        files.extend(kept)
    if len(stores) > 0:
        print(f"warning: skipped {len(stores)} .zarr store(s) without looking inside, e.g. {stores[0]}")
    return list(dict.fromkeys(files))


def find_tracking(entry: str) -> Optional[Path]:
    """
    The tracking run a view entry names: a ``*_roicat-tracking-manifest.json`` file as given; for a folder, the
    manifest with the latest timestamp in it or, with none there, in its ``tracking`` subfolder; else the folder
    itself when it holds ROICaT's files, as RoicatTrackingResults.to_roicat_dir wrote them before manifests. None
    for anything else.
    """
    path = Path(entry).expanduser()
    if path.is_file():
        return path if path.match(GLOB_TRACKING_MANIFEST) else None
    if not path.is_dir():
        return None
    # a manifest's name starts with its timestamp, so the last by name is the newest
    manifests = sorted(path.glob(GLOB_TRACKING_MANIFEST)) or sorted(path.glob(f"tracking/{GLOB_TRACKING_MANIFEST}"))
    if len(manifests) > 0:
        return manifests[-1]
    return path if any(path.glob("*.tracking.results_all.*")) else None


def demixing_sessions(files: list[str]) -> list[str]:
    """The results files holding top-level demixing results, each one a classification session; the rest are reported and skipped."""
    sessions = [filepath for filepath in files if has_stage(filepath_results=filepath, name_group=group_name_demixing())]
    for filepath in files:
        if filepath not in sessions:
            print(f"skipped {filepath}: it holds no {group_name_demixing()}")
    if len(sessions) == 0:
        fail(f"no results file holds {group_name_demixing()}")
    return sessions


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


KEYS_RECORD = ("masknmf_version", "pipeline", "run", "inputs", "timings")


def config_record(loaded: Any) -> Optional[dict]:
    """
    A config file's object with the argument values under "configs". Run folders from before they were nested
    there hold them at the top level, beside "pipeline".

    Args:
        loaded (Any): The parsed json
    Returns:
        dict | None: The object, or None when it holds no argument values
    """
    if not isinstance(loaded, dict):
        return None
    if isinstance(loaded.get("configs"), dict):
        return loaded
    if "pipeline" not in loaded:
        return None
    return {**{k: loaded[k] for k in KEYS_RECORD if k in loaded},
            "configs": {k: v for k, v in loaded.items() if k not in KEYS_RECORD}}


def read_config_file(filepath: str) -> dict:
    """
    Read a --config file: the json `masknmf params --json` prints, or a run folder's config.json, an object
    naming the pipeline under "pipeline" and holding its argument values under "configs".

    Args:
        filepath (str): The file
    Returns:
        dict: The object
    Raises:
        SystemExit: If the file is missing, is not a json object, or its "configs" is not one
    """
    path = Path(filepath).expanduser()
    if not path.is_file():
        fail(f"no such config file: {path}")
    try:
        loaded = json.loads(path.read_text())
    except json.JSONDecodeError as error:
        fail(f"{path.name} is not valid json: {error}")
    record = config_record(loaded=loaded)
    if record is None:
        fail(f"{path.name} should hold a json object with argument names to values under \"configs\"")
    return record


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
            parser.add_argument(param.flag, help=f"imaging movie for {param.field}", default=None)
            continue
        parser.add_argument(
            param.field,
            nargs="*",
            help=f"imaging movie(s) for {param.field}; several movies or a glob run one after another, each in its "
                 f"own run folder; left out, the movie a --config run folder recorded is used",
            default=None,
        )

    for param in spec.array_params:
        parser.add_argument(
            param.flag,
            default=None,
            metavar="NPY",
            help=f".npy file holding {param.field}",
        )

    for param in spec.run_scalars:
        flags = [param.flag]
        if param.field in NAMES_ALIAS:
            flags.append(NAMES_ALIAS[param.field])
        parser.add_argument(
            *flags,
            default=None,
            metavar="VALUE",
            help=param.description,
        )

    for param in spec.scalars:
        parser.add_argument(
            param.flag,
            default=None,
            metavar="VALUE",
            help=param.description,
        )

    for section in spec.sections:
        parser.add_argument(
            f"--{section.name}-kind",
            choices=section.kinds_buildable,
            default=None,
            help=f"which config {section.argument} receives, at its defaults; default {section.default_kind}",
        )

    parser.add_argument(
        "--config",
        default=None,
        metavar="JSON",
        help="config values to run with: `masknmf params --json` output, or a run folder's config.json",
    )


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
        print(json.dumps({"pipeline": spec.cls.__name__, "configs": configs}, indent=2, default=scraper.config_json_value))
        return

    print(f"{spec.slug}  ({spec.cls.__name__})\n")

    print("run arguments")
    for param in spec.run_params:
        note = {"movie": "imaging movie", "array": ".npy file"}.get(param.kind, "")
        print(f"  {param.field:28} {param.description}{'  [' + note + ']' if note else ''}")

    print("\nconstructor arguments")
    for param in spec.scalars:
        print(f"  {param.field:28} {param.description}")

    for section in spec.sections:
        print(f"\n[{section.name}] --{section.name}-kind {' | '.join(section.kinds_buildable)}")
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
    loaded = read_config_file(filepath=args.config) if args.config is not None else {"configs": {}}

    if loaded.get("pipeline", spec.cls.__name__) != spec.cls.__name__:
        fail(f"the config file is for {loaded['pipeline']}, not {spec.cls.__name__}")
    values_file = loaded["configs"]
    names_known = {s.argument for s in spec.sections} | {p.field for p in spec.scalars} | {p.field for p in spec.run_scalars}
    unknown = set(values_file) - names_known
    if len(unknown) > 0:
        fail(f"the config file sets {', '.join(sorted(unknown))}, which {spec.slug} does not take")

    # a run folder's config.json records the files its run read; a movie or array not given comes from there
    inputs_file = loaded.get("inputs", {})
    for param in (*spec.movie_params, *spec.array_params):
        positional = param.kind == "movie" and len(spec.movie_params) == 1
        if getattr(args, param.field, None) or param.field not in inputs_file:
            continue
        recorded = inputs_file[param.field]
        path = Path(recorded["path"]).expanduser()
        if path.is_file() and path.stat().st_size != recorded.get("bytes", path.stat().st_size):
            print(f"{path.name} is not the size the config's run recorded; it may have changed", file=sys.stderr)
        setattr(args, param.field, [str(path)] if positional else str(path))
        if args.dataset is None and "dataset" in recorded:
            args.dataset = recorded["dataset"]
    if len(spec.movie_params) == 1 and not getattr(args, spec.movie_params[0].field):
        _, allows_none = scraper.annotation_members(annotation=spec.movie_params[0].annotation)
        if not allows_none:
            fail(f"{spec.slug} needs a movie: pass one, or a --config from a run that recorded it")

    kwargs_init = {}
    for param in spec.scalars:
        if param.field in values_file:
            kwargs_init[param.field] = values_file[param.field]
        text = getattr(args, param.field, None)
        if text is not None:
            try:
                kwargs_init[param.field] = scraper.coerce(param=param, text=text)
            except ValueError as error:
                fail(str(error))

    for section in spec.sections:
        kind = getattr(args, f"{section.name}-kind".replace("-", "_"), None)
        passes, value = section_value(section=section, kind=kind, value_file=values_file.get(section.argument))
        if passes:
            kwargs_init[section.argument] = value

    movies = [None]
    if len(spec.movie_params) == 1 and getattr(args, spec.movie_params[0].field):
        movies = []
        for entry in getattr(args, spec.movie_params[0].field):
            # powershell and cmd hand globs over unexpanded
            matches = sorted(glob.glob(str(Path(entry).expanduser()))) if any(c in entry for c in "*?[") else [entry]
            if len(matches) == 0:
                fail(f"no movie matches {entry}")
            movies.extend(matches)
        # a movie that cannot be opened stops the batch before any run starts
        for movie in movies:
            load_movie(filepath_movie=movie, name_dataset=args.dataset)
        # a config file's output_folder is where its own movie's runs went; this run's movies decide where theirs go
        if getattr(args, "output_folder", None) is None:
            kwargs_init.pop("output_folder", None)

    finished = []
    for movie in movies:
        if len(spec.movie_params) == 1:
            # the loading below reads the one movie this run takes from args
            setattr(args, spec.movie_params[0].field, movie)
        kwargs_pipeline = {**kwargs_init}
        kwargs_run = {}
        filepaths_input = {}
        for param in spec.movie_params:
            filepath = getattr(args, param.field, None)
            if filepath is None:
                kwargs_run[param.field] = None
            else:
                kwargs_run[param.field] = load_movie(
                    filepath_movie=filepath, name_dataset=args.dataset
                )
                filepaths_input[param.field] = filepath

        for param in spec.array_params:
            filepath = getattr(args, param.field, None)
            if filepath is None:
                if param.required:
                    fail(f"{param.flag} is required for {spec.slug}")
            else:
                kwargs_run[param.field] = np.load(filepath)
                filepaths_input[param.field] = filepath

        inputs = {}
        for field, filepath in filepaths_input.items():
            path = Path(filepath).expanduser().resolve()
            if path.is_dir():
                size = sum(p.stat().st_size for p in path.iterdir() if p.suffix.lower() in SUFFIXES_TIFF)
            else:
                size = path.stat().st_size
            inputs[field] = {"path": str(path), "name": path.name, "bytes": size,
                             "modified": datetime.fromtimestamp(path.stat().st_mtime).strftime(masknmf.utils.TIMESTAMP_FORMAT),
                             "shape": list(kwargs_run[field].shape), "dtype": str(kwargs_run[field].dtype)}
            if isinstance(kwargs_run[field], masknmf.Hdf5Array):
                inputs[field]["dataset"] = args.dataset

        for param in spec.run_scalars:
            if param.field in values_file:
                kwargs_run[param.field] = values_file[param.field]
            text = getattr(args, param.field, None)
            if text is None:
                if param.required and param.field not in values_file:
                    fail(f"{param.flag} is required for {spec.slug}")
                continue
            try:
                kwargs_run[param.field] = scraper.coerce(param=param, text=text)
            except ValueError as error:
                fail(str(error))

        filepaths_movie = [
            getattr(args, p.field, None)
            for p in spec.movie_params
        ]
        filepaths_movie = [Path(f).expanduser().resolve() for f in filepaths_movie if f is not None]
        if kwargs_pipeline.get("output_folder") is None and len(filepaths_movie) > 0:
            first = filepaths_movie[0]
            kwargs_pipeline["output_folder"] = str(first if first.is_dir() else first.parent)

        pipeline = spec.cls(**kwargs_pipeline)
        pipeline.inputs = inputs
        pipeline.command = args.command
        shapes = ", ".join(
            str(kwargs_run[p.field].shape)
            for p in spec.movie_params
            if kwargs_run.get(p.field) is not None
        )
        print(f"{spec.cls.__name__} on {shapes or 'stored results'}")
        start = time.monotonic()
        try:
            run_folder = pipeline.run(**kwargs_run)
        except BaseException as error:
            # the pipeline logged the error and recorded the failure in config.json
            logger.error("run failed" if movie is None else f"run failed on {movie}")
            # a failed run keeps its folder only when one of its results files holds a finished stage; its log file,
            # closed first so windows lets the folder go, moves up to where the folder was
            folder = pipeline.run_folder
            if folder is not None and not any(
                has_stage(filepath_results=str(filepath), name_group=name)
                for filepath in folder.glob("*.hdf5")
                for name in (*group_names_registration(), group_name_compression())
            ):
                logger.removeHandler(pipeline.log_handler)
                pipeline.log_handler.close()
                shutil.move(pipeline.log_handler.baseFilename, folder.parent)
                shutil.rmtree(folder)
                print(f"removed {folder}, its log is in {folder.parent}", file=sys.stderr)
            # one failed movie of several lets the rest run; ctrl+c stops them all
            if len(movies) == 1 or not isinstance(error, Exception):
                raise SystemExit(1)
            finished.append((movie, None))
            continue
        logger.info(f"done in {timedelta(seconds=round(time.monotonic() - start))}: {run_folder}")
        argv_view = ["view", str(run_folder)]
        if len(spec.movie_params) == 1 and spec.movie_params[0].field in filepaths_input:
            argv_view += ["--raw", filepaths_input[spec.movie_params[0].field]]
            if args.dataset is not None:
                argv_view += ["--dataset", args.dataset]
        print(f"open it with: masknmf {format_command(argv=argv_view)}")
        finished.append((movie, run_folder))

    if len(movies) > 1:
        print(f"{sum(folder is not None for _, folder in finished)} of {len(movies)} runs done")
        for movie, folder in finished:
            print(f"  {'done' if folder is not None else 'failed'}  {movie}" + ("" if folder is None else f"  ->  {folder}"))
        if any(folder is None for _, folder in finished):
            raise SystemExit(1)


def print_tracking(tracking: "masknmf.multisession.RoicatTrackingResults", folder: str) -> list[str]:
    """Summarize a tracking run and list its sessions' results files; returns the files that do not exist."""
    clustered = sum(int((labels >= 0).sum()) for labels in tracking.labels_by_session)
    spans = tracking.num_sessions_per_cluster
    print(f"tracking {Path(folder).resolve()}")
    print(f"  sessions   {tracking.num_sessions}")
    print(f"  rois       {tracking.num_roi_total}, every session's together")
    print(f"  clusters   {tracking.num_clusters}, each one cell matched across sessions; "
          f"{int((spans == tracking.num_sessions).sum())} found in every session")
    print(f"  clustered  {clustered / max(tracking.num_roi_total, 1):.0%} of the rois ({clustered}) belong to a cluster; "
          f"the other {tracking.num_roi_total - clustered} matched no roi of another session")
    missing = [filepath for filepath in tracking.session_files if not os.path.isfile(filepath)]
    root = os.path.commonpath([os.path.dirname(filepath) for filepath in tracking.session_files])
    print(f"  results files under {root}:")
    print("    session  rois  clustered  file")
    for session, filepath in enumerate(tracking.session_files):
        labels = tracking.labels_by_session[session]
        print(f"    {session:>7}  {len(labels):>4}  {int((labels >= 0).sum()):>9}  {os.path.relpath(filepath, root)}"
              + ("  (missing)" if filepath in missing else ""))
    return missing


def view_tracking(args: argparse.Namespace, source: Path) -> None:
    """
    Open the multisession viewer on a tracking run, its manifest or a folder from before manifests; results files
    after it replace the sessions it recorded.
    """
    if args.classify or args.raw is not None or args.compression or args.prefix or args.fs is not None:
        fail("a tracking folder opens only the multisession viewer; drop --classify, --raw, --compression, --prefix and --fs")
    folder, entries_sessions = args.results[0], args.results[1:]
    files = expand_results(entries=entries_sessions) if len(entries_sessions) > 0 else None
    # richfile and roicat warn about their own metadata on every load, nothing the user can act on
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            if source.is_file():
                tracking = masknmf.multisession.RoicatTrackingResults.from_manifest(source)
            else:
                tracking = masknmf.multisession.RoicatTrackingResults.from_roicat_dir(source)
        except ValueError as error:
            fail(str(error))
    if files is not None:
        if len(files) != tracking.num_sessions:
            fail(f"the tracking has {tracking.num_sessions} sessions, got {len(files)} results files")
        counts = {}
        for filepath in files:
            with h5py.File(filepath, "r") as f:
                counts[filepath] = f[f"{group_name_demixing()}/temporal_demixed"].shape[1]
        if list(counts.values()) != list(tracking.num_roi_per_session):
            # globs sort by name, so put each file at the session with its roi count
            ordered = []
            for session, count in enumerate(tracking.num_roi_per_session):
                matches = [filepath for filepath, n in counts.items() if n == count]
                if len(matches) > 1:
                    fail(f"session {session} has {count} rois and {len(matches)} results files do; pass the files in session order")
                ordered.append(matches[0] if len(matches) == 1 else None)
            # a file that fits no session takes an open one, so the check below names it
            unplaced = iter([filepath for filepath in files if filepath not in ordered])
            files = [next(unplaced) if filepath is None else filepath for filepath in ordered]
        try:
            tracking.session_files = files
        except ValueError as error:
            fail(str(error))
    missing = print_tracking(tracking=tracking, folder=str(source))
    if args.list:
        return
    if len(missing) > 0:
        fail(f"results files not found; pass them in session order after the folder: masknmf view {folder} day1.hdf5 day2.hdf5 ...")

    import fastplotlib as fpl

    device = str(masknmf.utils.torch_select_device()) if args.device == "auto" else args.device
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        masknmf.MultiSessionDemixingVis(tracking, device=device).show()
    fpl.loop.run()


def command_view(args: argparse.Namespace) -> None:
    """
    Open the viewers for whatever stages the results files hold; --classify opens only the classification viewer.
    A folder opens its newest tracking manifest in the multisession viewer, else its results files.
    """
    tracking = find_tracking(entry=args.results[0])
    if tracking is not None:
        view_tracking(args=args, source=tracking)
        return
    # a folder without a manifest stands for the results files in it, each one's newest curated file in its place
    entries = [
        # glob characters in the folder's own name are escaped, so only the file pattern matches
        str(Path(glob.escape(entry)) / "*results*.hdf5") if Path(entry).expanduser().is_dir() else entry
        for entry in args.results
    ]
    files = expand_results(entries=entries)
    present = {filepath: groups_present(filepath_results=filepath) for filepath in files}
    for filepath, names in present.items():
        print(Path(filepath).resolve())
        for name in names or ["no masknmf results"]:
            print(f"  {name}")
    if args.list:
        return

    if args.classify:
        if args.prefix:
            fail("--classify reads only the top-level demixing results; drop --prefix")
        if args.raw is not None or args.compression:
            fail("--classify opens only the classification viewer; drop --raw and --compression")
        import fastplotlib as fpl

        classification = masknmf.ClassificationVis.from_masknmf(
            demixing_sessions(files=files), label_names=args.labels.split(",") if args.labels else ()
        )
        if args.classifier is not None:
            classification.classifier_path = args.classifier
            if Path(args.classifier).is_file():
                classification.select_classifier(args.classifier)
        classification.show()
        fpl.loop.run()
        return

    if len(files) > 1:
        fail(f"the demixing viewer opens one results file, got {len(files)}; --classify opens several")
    filepath_results = files[0]
    names_present = present[filepath_results]
    if len(names_present) == 0:
        fail(f"{filepath_results} holds no masknmf results")

    name_demixing = f"{args.prefix}/{group_name_demixing()}" if args.prefix else group_name_demixing()
    if name_demixing not in names_present and args.prefix:
        fail(f"{filepath_results} holds no {name_demixing}")

    import fastplotlib as fpl

    device = (
        str(masknmf.utils.torch_select_device()) if args.device == "auto" else args.device
    )
    raw = None if args.raw is None else load_movie(filepath_movie=args.raw, name_dataset=args.dataset)
    # the demixing results, else the compression, else the registration replayed on the raw movie
    try:
        opened = OpenedResults.open(filepath_results, raw=raw, prefix=args.prefix, device=device)
    except (OSError, ValueError) as e:
        fail(f"{e}; --raw names the movie")
    for note in opened.skipped:
        print(note)
    results, raw, registered = opened.results, opened.raw, opened.registered
    if args.compression and isinstance(results, masknmf.BaseRegistrationArray):
        fail(f"{filepath_results} holds no compression")
    if args.compression and raw is None:
        fail("--compression needs --raw, the movie the compression saw, with the results' frame count")
    viewer = masknmf.SingleSessionDemixingVis(
        demixing_results=results,
        frame_timings=timings(results.shape[0], args.fs),
        device=device,
        results_path=filepath_results,
        raw=raw,
        registered=registered,
    )
    if args.compression:
        viewer.compute_lag1_acf()
    viewer.show()
    fpl.loop.run()


def command_train_classifier(args: argparse.Namespace) -> None:
    """Train a ROICaT classifier on the labels the classification viewer saved beside each results file."""
    sessions = demixing_sessions(files=expand_results(entries=args.results))
    incomplete = []
    for filepath in sessions:
        labels, _ = read_labels(filepath)
        if labels is None or (labels < 0).any():
            incomplete.append(filepath)
    if len(incomplete) > 0:
        fail("every ROI must be labeled before training; label these with masknmf view --classify:\n  " + "\n  ".join(incomplete))

    classifier = RoicatClassifier.from_masknmf(sessions)
    try:
        classifier.train(num_workers=0)
    except ValueError as error:
        fail(str(error))
    for name, count in sorted(classifier.class_counts.items()):
        print(f"  {name}: {count}")
    classifier.save(args.out)


def command_classify(args: argparse.Namespace) -> None:
    """Classify every ROI in each results file; predictions go to its labels sidecar, unlabeled ROIs take them."""
    sessions = demixing_sessions(files=expand_results(entries=args.results))
    if not Path(args.classifier).is_file():
        fail(f"no such classifier: {args.classifier}")
    classifier = RoicatClassifier.from_disk(args.classifier, device=args.device)
    _, names, _ = classifier.classify(sessions, write=True)
    for filepath, names_session in zip(sessions, names):
        counts = ", ".join(f"{name}: {names_session.count(name)}" for name in classifier.label_names)
        print(f"{filepath}  {counts}")


def command_track(args: argparse.Namespace) -> None:
    """Track ROIs across sessions with ROICaT, one session per results file in the order given, and save the tracking folder."""
    sessions = demixing_sessions(files=expand_results(entries=args.results))
    if len(sessions) < 2:
        fail(f"tracking needs at least two sessions, got {len(sessions)}")
    for session, filepath in enumerate(sessions):
        print(f"  {session}  {filepath}")
    tracker = RoicatTracker()
    tracker.params["general"]["use_GPU"] = args.device != "cpu"
    # richfile and roicat warn about their own metadata, nothing the user can act on; roicat's progress still prints
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tracking = tracker.run_tracking(RoicatDataAdapter.from_masknmf(sessions, um_per_pixel=args.um_per_pixel))
        folder = tracking.to_roicat_dir(args.out)
    print_tracking(tracking=tracking, folder=str(folder))


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
    parser_view.add_argument(
        "results", nargs="+", help="results .hdf5 files or globs, e.g. \"sessions/*/results.hdf5\"; several need --classify. "
        "Or a tracking folder (its newest run) or one run's *-manifest.json, optionally followed by its results files in session order. "
        "Any other folder stands for the *results*.hdf5 files in it, the newest curated one of each",
    )
    parser_view.add_argument("--raw", default=None, help="the raw movie the results came from")
    parser_view.add_argument("--dataset", default=None)
    parser_view.add_argument("--fs", default=None, type=float, help="acquisition rate in Hz")
    parser_view.add_argument(
        "--prefix", default="", help="group the demixing results sit under, e.g. global for the glutamate pipeline's whole-dendrite result"
    )
    parser_view.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser_view.add_argument(
        "--compression",
        action="store_true",
        help="also compute the lag-1 autocorrelation images of the registered (else raw), compressed and residual movies into the Static images window (needs --raw and a compression)",
    )
    parser_view.add_argument(
        "--classify",
        action="store_true",
        help="open only the ROI labeling and classification viewer, one session per results file; labels are saved next to each",
    )
    parser_view.add_argument(
        "--labels", default=None, help="with --classify, comma-separated class names, e.g. soma,dendrite,junk"
    )
    parser_view.add_argument(
        "--classifier",
        default=None,
        help="with --classify, the .roicat_classifier path; train saves here, and an existing file is selected for classify",
    )
    parser_view.add_argument(
        "--list", action="store_true", help="print what the file holds and exit"
    )
    parser_view.set_defaults(handler=command_view)

    parser_train = subparsers.add_parser(
        "train-classifier", help="train a ROI classifier on the labels saved with masknmf view --classify"
    )
    parser_train.add_argument("results", nargs="+", help="labeled results .hdf5 files or globs")
    parser_train.add_argument(
        "--out", required=True, help="where the classifier is saved, as <out>.roicat_classifier and <out>.training.json"
    )
    parser_train.set_defaults(handler=command_train_classifier)

    parser_classify = subparsers.add_parser(
        "classify", help="classify the ROIs in results files; predictions are saved next to each"
    )
    parser_classify.add_argument("results", nargs="+", help="results .hdf5 files or globs")
    parser_classify.add_argument("--classifier", required=True, help="a .roicat_classifier from masknmf train-classifier")
    parser_classify.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser_classify.set_defaults(handler=command_classify)

    parser_track = subparsers.add_parser("track", help="track ROIs across sessions with ROICaT")
    parser_track.add_argument(
        "results", nargs="+", help="results .hdf5 files or globs, one per session; sessions are numbered in this order"
    )
    parser_track.add_argument(
        "--out", required=True,
        help="the tracking folder; each run adds <timestamp>_roicat-tracking/ and its -manifest.json there",
    )
    parser_track.add_argument("--um-per-pixel", default=1.2, type=float, help="imaging resolution; default 1.2")
    parser_track.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser_track.set_defaults(handler=command_track)

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
        bootstrap.pipeline = scraper.slugify(name_class=name_class)
        if bootstrap.pipeline not in scraper.pipeline_registry():
            fail(f"the config file names {name_class!r}, which is not a masknmf pipeline")
        argv = ["run", "--pipeline", bootstrap.pipeline, *argv[1:]]

    spec = None
    if bootstrap.pipeline is not None and bootstrap.pipeline in scraper.pipeline_registry():
        spec = spec_for(slug=bootstrap.pipeline)

    args = build_parser(spec=spec).parse_args(argv)
    args.command = f"masknmf {format_command(argv=argv)}"
    args.handler(args)


if __name__ == "__main__":
    main()
