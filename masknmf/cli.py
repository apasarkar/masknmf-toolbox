"""
Command line entry point for masknmf.

Each processing stage is its own command. They compose through one results file, which holds
a group per stage, so motion correction, compression and demixing are independently optional
and a partial run can be picked up where it stopped::

    masknmf register raw.tif -o results.hdf5
    masknmf compress results.hdf5
    masknmf demix    results.hdf5
    masknmf view     results.hdf5

``masknmf run`` does the same end to end through one of the shipped pipelines. Both paths call
the same stage functions in :mod:`masknmf.pipelines.stages`, so they cannot drift apart.

Every parameter of every stage is settable with ``-p key=value``; the keys are scraped off the
config dataclasses, so ``masknmf params`` lists them and nothing here enumerates them.

Imports here stay light; fastplotlib and the pipelines load inside the command that needs them.
"""

from __future__ import annotations

import sys
from pathlib import Path

import click

RUN_GROUP = "MaskNMFRun"

RESULTS_SUFFIXES = (".h5", ".hdf5")


class PathAwareGroup(click.Group):
    """A group that reads a leading file path as ``view PATH``, so ``masknmf results.hdf5`` works."""

    def resolve_command(self, ctx, args):
        if args:
            first = args[0]
            if first not in self.commands and (
                "/" in first or "\\" in first or "." in first or Path(first).exists()
            ):
                return "view", self.commands["view"], args
        return super().resolve_command(ctx, args)


def _version_callback(ctx, param, value):
    if not value or ctx.resilient_parsing:
        return
    from masknmf import __version__

    click.echo(f"masknmf {__version__}\nPython: {sys.executable}")
    ctx.exit()


@click.group(cls=PathAwareGroup)
@click.option(
    "-V",
    "--version",
    is_flag=True,
    callback=_version_callback,
    expose_value=False,
    is_eager=True,
    help="Show the version and exit.",
)
def main():
    """
    Motion correct, compress and demix functional imaging data.

    \b
    Stage by stage, into one results file:
      masknmf register raw.tif -o results.hdf5
      masknmf compress results.hdf5
      masknmf demix    results.hdf5

    \b
    Or end to end:
      masknmf init raw.tif            # write a masknmf.toml to edit
      masknmf run  raw.tif --fs 30

    \b
    Then look at it:
      masknmf results.hdf5
    """


# --------------------------------------------------------------------------- shared plumbing


def common_options(func):
    """The options every stage command shares. Defaults are None so the file still wins."""
    for option in reversed(
        [
            click.option("-c", "--config", "config_path", type=click.Path(exists=True, dir_okay=False),
                         default=None, help="masknmf.toml to read; command line flags override it."),
            click.option("-o", "--output", default=None,
                         help="Results file. Default: [io] output, else results.hdf5 beside the input."),
            click.option("-p", "--param", "overrides", multiple=True, metavar="KEY=VALUE",
                         help="Set any parameter `masknmf params` lists. Repeatable."),
            click.option("--fs", "frame_rate", type=float, default=None,
                         help="Acquisition rate in Hz."),
            click.option("--dataset", default=None,
                         help="For hdf5 input, the dataset holding the movie."),
            click.option("--device", type=click.Choice(["auto", "cuda", "cpu"]), default=None,
                         help="Device pytorch runs on."),
            click.option("--batch-size", "frame_batch_size", type=int, default=None,
                         help="Frames held on the GPU at a time."),
            click.option("--dry-run", is_flag=True, default=False,
                         help="Print the resolved configuration and exit without computing."),
        ]
    ):
        func = option(func)
    return func


def _load_config(config_path, overrides=(), kinds=None, **flags):
    """
    Read the TOML when given, then lay the command line over it.

    Args:
        config_path: A masknmf.toml, or None for the built-in defaults.
        overrides: ``key=value`` strings from ``-p``.
        kinds (dict | None): Stage to kind, from the ``--kind`` flags; applied first, since
            the kind decides which fields the stage has.
        **flags: Named options, as dotted parameter names with ``__`` for the dot. None means
            "not passed". These win over ``-p``.

    Returns:
        RunConfig: The resolved configuration.
    """
    from masknmf.pipelines import params
    from masknmf.pipelines.config_io import RunConfig

    try:
        cfg = RunConfig.from_toml(config_path) if config_path else RunConfig()
        for stage, kind in (kinds or {}).items():
            if kind is not None:
                cfg.set_kind(stage, kind)
        for dotted, value in params.parse(overrides).items():
            cfg.set(dotted, value)
        for dotted, value in flags.items():
            if value is not None:
                cfg.set(dotted.replace("__", "."), value)
        cfg.validate()
    except (KeyError, ValueError, OSError) as e:
        raise click.ClickException(str(e).strip("'")) from e
    return cfg


def _resolve_output(cfg, input_path, output) -> str:
    if output:
        return output
    if cfg.io.output and cfg.io.output != "results.hdf5":
        return cfg.io.output
    base = Path(input_path)
    parent = base if base.is_dir() else base.parent
    return str(parent / "results.hdf5")


def _is_results_file(path: str | Path) -> bool:
    """Whether the path is an hdf5 file that already holds a masknmf stage."""
    return Path(path).suffix.lower() in RESULTS_SUFFIXES and bool(groups_in(path))


def groups_in(path: str | Path) -> list[str]:
    """The masknmf stage groups present in a results file, in pipeline order."""
    from masknmf.utils import has_group
    from masknmf.pipelines.stages import STAGE_GROUPS

    return [
        group
        for groups in STAGE_GROUPS.values()
        for group in groups
        if has_group(str(path), group)
    ]


def write_run_info(results_path, raw_path, dataset, frame_rate, config_toml=""):
    """
    Record what a stage was run on.

    Registration results store only their shifts, so a later stage has to reopen the raw movie
    to rebuild the registered array. Keeping the raw path in the results file is what lets the
    stage commands compose without restating it every time.
    """
    from masknmf import __version__
    from masknmf.utils import get_timestamp
    from masknmf.utils._serialization import save_dict

    raw = Path(raw_path).resolve()
    try:
        relative = str(raw.relative_to(Path(results_path).resolve().parent))
    except ValueError:
        relative = ""

    save_dict(
        {
            "raw_path": str(raw),
            "raw_path_relative": relative,
            "dataset": dataset or "",
            "frame_rate": float(frame_rate),
            "version": __version__,
            "timestamp": get_timestamp(),
            "config": config_toml,
        },
        filename=results_path,
        group=RUN_GROUP,
        exists_ok=True,
    )


def read_run_info(results_path) -> dict:
    """The run-info group of a results file, or an empty dict when it has none."""
    from masknmf.utils import has_group
    from masknmf.utils._serialization import load_dict

    if not has_group(str(results_path), RUN_GROUP):
        return {}
    out = {}
    for key, value in load_dict(str(results_path), RUN_GROUP).items():
        out[key] = value.decode() if isinstance(value, bytes) else value
    return out


def _raw_for(results_path, run_info, raw_override, dataset):
    """
    Reopen the raw movie a results file was built from.

    Raises:
        click.ClickException: If no raw path is recorded, or the recorded one has moved.
    """
    from masknmf import imread

    if raw_override:
        path = Path(raw_override)
    else:
        recorded = run_info.get("raw_path", "")
        relative = run_info.get("raw_path_relative", "")
        candidates = [Path(recorded)] if recorded else []
        if relative:
            candidates.append(Path(results_path).resolve().parent / relative)
        path = next((p for p in candidates if p.exists()), None)
        if path is None:
            hint = f" (recorded as {recorded})" if recorded else ""
            raise click.ClickException(
                f"{results_path} needs its raw movie to rebuild the registered array{hint}; "
                "point at it with --raw"
            )
    return imread(path, dataset=dataset or run_info.get("dataset") or None)


def _echo_config(cfg, stages: tuple):
    from dataclasses import fields, is_dataclass

    for name in ("io", "compute", *stages):
        section = getattr(cfg, name)
        click.secho(f"[{name}]", fg="cyan")
        if name in cfg.kinds():
            click.echo(f"  {'kind':<26} {cfg.kinds()[name]}")
        for f in fields(section):
            value = getattr(section, f.name)
            if is_dataclass(value):
                continue
            if hasattr(value, "DemixingConfigs"):
                value = f"{len(value.DemixingConfigs)} pass(es)"
            elif value is None and f.name in ("filtered", "unfiltered"):
                value = "built-in schedule"
            click.echo(f"  {f.name:<26} {value}")


# --------------------------------------------------------------------------- commands


@main.command("init")
@click.argument("data_path", required=False, type=click.Path())
@click.option("-o", "--config", "config_path", type=click.Path(), default=None,
              help="Where to write it. Default: masknmf.toml in the current directory.")
@click.option("-O", "--output", "output_path", default=None,
              help="Prefill [io] output with this results file.")
@click.option("--overwrite", is_flag=True, default=False, help="Replace an existing file.")
def init(data_path, config_path, output_path, overwrite):
    """Write a commented masknmf.toml carrying every default."""
    from masknmf.pipelines.config_io import render_template

    target = Path(config_path) if config_path else Path("masknmf.toml")
    if target.exists() and not overwrite:
        raise click.ClickException(f"{target} exists; pass --overwrite to replace it")

    default_output = output_path or (
        str(Path(data_path).with_suffix("").parent / "results.hdf5") if data_path else "results.hdf5"
    )
    try:
        target.write_text(render_template(str(data_path) if data_path else "", default_output))
    except OSError as e:
        raise click.ClickException(f"could not write {target}: {e}") from e
    click.echo(f"wrote {target}")
    click.echo(f"edit it, then:  masknmf run {data_path or '<data>'} -c {target}")


@main.command("register")
@click.argument("input_path", type=click.Path(exists=True))
@common_options
@click.option("--kind", type=click.Choice(["rigid", "piecewise_rigid", "gradient"]), default=None,
              help="Motion correction model.")
def register(input_path, config_path, output, overrides, frame_rate, dataset, device,
             frame_batch_size, dry_run, kind):
    """
    Motion correct a raw movie.

    INPUT_PATH is a .tif, a directory of them, or a .h5/.hdf5.
    """
    cfg = _load_config(
        config_path, overrides, {"register": kind},
        io__input=input_path, io__dataset=dataset, io__frame_rate=frame_rate,
        compute__device=device, compute__frame_batch_size=frame_batch_size,
    )
    out = _resolve_output(cfg, input_path, output)

    if dry_run:
        _echo_config(cfg, ("register",))
        click.echo(f"\nwould write {out}")
        return

    from masknmf import imread
    from masknmf.pipelines.stages import register as register_stage

    if cfg.register.skip:
        raise click.ClickException("[register] skip is set; nothing for this command to do")

    movie = imread(input_path, dataset=cfg.io.dataset or None)
    click.secho(f"registering {movie.shape} from {input_path}", fg="cyan")
    register_stage(
        movie,
        cfg.register,
        device=cfg.compute.device,
        batch_size=cfg.compute.frame_batch_size,
        outpath=out,
    )
    write_run_info(out, input_path, cfg.io.dataset, cfg.io.frame_rate)
    click.secho(f"wrote {out}", fg="green")


@main.command("compress")
@click.argument("input_path", type=click.Path(exists=True))
@common_options
@click.option("--raw", "raw_override", type=click.Path(exists=True), default=None,
              help="The raw movie behind a results file, when it has moved since registration.")
@click.option("--kind", type=click.Choice(["plain", "denoise"]), default=None,
              help="Compression with or without denoising.")
def compress(input_path, config_path, output, overrides, frame_rate, dataset, device,
             frame_batch_size, dry_run, raw_override, kind):
    """
    Compress a movie, optionally denoising as it goes.

    INPUT_PATH is either a results file holding a registration stage, or a raw movie to
    compress without motion correction.
    """
    cfg = _load_config(
        config_path, overrides, {"compress": kind},
        io__input=input_path, io__dataset=dataset, io__frame_rate=frame_rate,
        compute__device=device, compute__frame_batch_size=frame_batch_size,
    )
    from_results = _is_results_file(input_path)
    out = output or (str(input_path) if from_results else _resolve_output(cfg, input_path, None))

    if dry_run:
        _echo_config(cfg, ("register", "compress"))
        click.echo(f"\nsource: {'registered stage in ' + str(input_path) if from_results else 'raw ' + str(input_path)}")
        click.echo(f"would write {out}")
        return

    if cfg.compress.skip:
        raise click.ClickException("[compress] skip is set; nothing for this command to do")

    from masknmf import imread
    from masknmf.pipelines.stages import compress as compress_stage
    from masknmf.pipelines.stages import find_stage, load_stage

    run_info = {}
    if from_results:
        run_info = read_run_info(input_path)
        if find_stage(input_path, "register") is None:
            raise click.ClickException(
                f"{input_path} holds no registration stage; pass the raw movie instead"
            )
        raw = _raw_for(input_path, run_info, raw_override, cfg.io.dataset)
        movie = load_stage(input_path, "register", movie=raw, device=cfg.compute.device)
        if frame_rate is None and run_info.get("frame_rate"):
            cfg.io.frame_rate = float(run_info["frame_rate"])
        raw_source = run_info.get("raw_path", input_path)
    else:
        movie = imread(input_path, dataset=cfg.io.dataset or None)
        raw_source = input_path

    click.secho(f"compressing {movie.shape}", fg="cyan")
    compress_stage(
        movie,
        cfg.compress,
        frame_rate=cfg.io.frame_rate,
        device=cfg.compute.device,
        outpath=out,
    )
    write_run_info(out, raw_source, cfg.io.dataset or run_info.get("dataset", ""), cfg.io.frame_rate)
    click.secho(f"wrote {out}", fg="green")


@main.command("demix")
@click.argument("input_path", type=click.Path(exists=True))
@common_options
def demix(input_path, config_path, output, overrides, frame_rate, dataset, device,
          frame_batch_size, dry_run):
    """
    Demix a compressed movie.

    INPUT_PATH is a results file holding a PMDArray. Runs over a high-pass filtered copy
    first, then over the unfiltered movie seeded from what that found.
    """
    cfg = _load_config(
        config_path, overrides, None,
        io__input=input_path, io__dataset=dataset, io__frame_rate=frame_rate,
        compute__device=device, compute__frame_batch_size=frame_batch_size,
    )
    out = output or str(input_path)

    if dry_run:
        _echo_config(cfg, ("demix",))
        click.echo(f"\nwould write {out}")
        return

    if cfg.demix.skip:
        raise click.ClickException("[demix] skip is set; nothing for this command to do")

    from masknmf.pipelines.stages import demix_two_phase, find_stage, load_stage

    if find_stage(input_path, "compress") is None:
        raise click.ClickException(
            f"{input_path} holds no PMDArray; run `masknmf compress` on it first"
        )

    run_info = read_run_info(input_path)
    if frame_rate is None and run_info.get("frame_rate"):
        cfg.io.frame_rate = float(run_info["frame_rate"])

    pmd = load_stage(input_path, "compress", device=cfg.compute.device)
    click.secho(f"demixing {pmd.shape} at {cfg.io.frame_rate} Hz", fg="cyan")
    results = demix_two_phase(
        pmd,
        cfg.io.frame_rate,
        cfg.demix,
        device=cfg.compute.device,
        frame_batch_size=cfg.compute.frame_batch_size,
        outpath=out,
    )
    if run_info:
        write_run_info(out, run_info.get("raw_path", input_path),
                         run_info.get("dataset", ""), cfg.io.frame_rate)
    click.secho(f"wrote {out} ({results.shape})", fg="green")


PIPELINES = {
    "twophoton": "TwoPhotonCalciumPipeline",
    "widefield": "WidefieldSinglechannelPipeline",
    "onephoton": "OnePhotonCulturePipeline",
    "spines": "GlutamateCalciumSpinePipeline",
}


@main.command("run")
@click.argument("input_path", type=click.Path(exists=True))
@common_options
@click.option("--pipeline", type=click.Choice(sorted(PIPELINES)), default="twophoton",
              show_default=True, help="Which shipped pipeline to run.")
@click.option("--skip", default=None,
              help="Comma-separated stages to leave out, e.g. register,compress. A skipped "
                   "stage is read back from the results file instead.")
@click.option("--remove-intermediates", is_flag=True, default=False,
              help="Delete the registration and compression stages once demixing is done. "
                   "Off by default so the stages stay resumable and viewable.")
def run(input_path, config_path, output, overrides, frame_rate, dataset, device,
        frame_batch_size, dry_run, pipeline, skip, remove_intermediates):
    """
    Run a whole pipeline end to end.

    INPUT_PATH is a .tif, a directory of them, or a .h5/.hdf5.
    """
    from masknmf.pipelines.configs import STAGES

    skipped = {s.strip() for s in skip.split(",")} if skip else set()
    unknown = skipped - set(STAGES)
    if unknown:
        raise click.ClickException(
            f"--skip does not know {', '.join(sorted(unknown))}; choose from {', '.join(STAGES)}"
        )

    cfg = _load_config(
        config_path, (*overrides, *(f"{stage}.skip=true" for stage in skipped)), None,
        io__input=input_path, io__dataset=dataset, io__frame_rate=frame_rate,
        compute__device=device, compute__frame_batch_size=frame_batch_size,
    )
    out = _resolve_output(cfg, input_path, output)

    if dry_run:
        _echo_config(cfg, tuple(STAGES))
        click.echo(f"\npipeline: {PIPELINES[pipeline]}")
        click.echo(f"would write {out}")
        return

    import masknmf
    from masknmf import imread

    movie = imread(input_path, dataset=cfg.io.dataset or None)
    shared = {
        "motion_correct_config": cfg.register,
        "compress_config": cfg.compress,
        "outpath_motion_correction": out,
        "outpath_compression": out,
        "frame_batch_size": cfg.compute.frame_batch_size,
        "device": cfg.compute.device,
    }
    cls = getattr(masknmf, PIPELINES[pipeline])

    click.secho(f"{PIPELINES[pipeline]} on {movie.shape} at {cfg.io.frame_rate} Hz", fg="cyan")
    if pipeline == "widefield":
        pipe = cls(**shared)
        pipe.run(movie)
    elif pipeline == "twophoton":
        pipe = cls(demix_config=cfg.demix, outpath_demixing=out, **shared)
        pipe.run(movie, cfg.io.frame_rate, remove_intermediates=remove_intermediates)
    else:
        raise click.ClickException(
            f"--pipeline {pipeline} takes inputs this command cannot supply yet "
            "(active frames for onephoton, two channels for spines); drive it from Python"
        )

    write_run_info(out, input_path, cfg.io.dataset, cfg.io.frame_rate)
    click.secho(f"wrote {out}", fg="green")


@main.command("view")
@click.argument("path", type=click.Path(exists=True))
@click.option("--stage", type=click.Choice(["register", "compress", "demix"]), default=None,
              help="Open only this stage's viewer. Default: every stage the file holds.")
@click.option("--raw", "raw_override", type=click.Path(exists=True), default=None,
              help="The raw movie, when the recorded one has moved.")
@click.option("--fs", "frame_rate", type=float, default=None, help="Acquisition rate in Hz.")
@click.option("--device", type=click.Choice(["auto", "cuda", "cpu"]), default="auto",
              show_default=True, help="Device pytorch runs on.")
@click.option("--list", "list_only", is_flag=True, default=False,
              help="Print what the file holds and exit, without opening a window.")
def view(path, stage, raw_override, frame_rate, device, list_only):
    """
    Open the viewers for whatever stages a results file holds.

    PATH is a results file. `masknmf PATH` is the same thing.
    """
    from masknmf.pipelines.stages import STAGE_GROUPS, find_stage

    present = groups_in(path)
    if not present:
        every = ", ".join(g for groups in STAGE_GROUPS.values() for g in groups)
        raise click.ClickException(f"{path} holds no masknmf results (looked for {every})")

    run_info = read_run_info(path)
    click.secho(str(Path(path).resolve()), bold=True)
    click.secho("\nStages", fg="cyan")
    for name in present:
        click.echo(f"  {name}")
    if run_info:
        click.secho("\nRun", fg="cyan")
        for key in ("raw_path", "frame_rate", "version", "timestamp"):
            if run_info.get(key) not in (None, ""):
                click.echo(f"  {key:<12} {run_info[key]}")
    if list_only:
        return

    import numpy as np

    from masknmf.pipelines.stages import load_stage
    from masknmf.utils import torch_select_device
    from masknmf.visualization import (
        CompressionVis,
        MotionCorrectionVis,
        SingleSessionDemixingVis,
    )

    device = str(torch_select_device(device)) if device == "auto" else device
    if frame_rate is None:
        frame_rate = float(run_info.get("frame_rate") or 0) or None

    wants = {stage} if stage else {"register", "compress", "demix"}
    open_ = []
    raw = registered = None

    if find_stage(path, "register") is not None and wants & {"register", "compress"}:
        try:
            raw = _raw_for(path, run_info, raw_override, run_info.get("dataset"))
            registered = load_stage(path, "register", movie=raw, device=device)
        except (click.ClickException, ValueError) as e:
            if stage in ("register", "compress"):
                raise
            click.secho(f"skipping the registration viewers: {e}", fg="yellow")

    def timings(n):
        return np.arange(n) / frame_rate if frame_rate else None

    if registered is not None and "register" in wants:
        open_.append(MotionCorrectionVis(registered, frame_timings=timings(registered.shape[0]),
                                         mean_subtract=True))

    has_pmd = find_stage(path, "compress") is not None
    has_demix = find_stage(path, "demix") is not None

    if has_pmd and "compress" in wants and registered is not None:
        pmd = load_stage(path, "compress", device=device)
        open_.append(CompressionVis(registered[:].cpu().numpy(), pmd,
                                    frame_timings=timings(pmd.shape[0]), device=device))

    if has_demix and "demix" in wants:
        results = load_stage(path, "demix", device=device)
        open_.append(SingleSessionDemixingVis(results, frame_timings=timings(results.shape[0]),
                                              device=device, results_path=path, raw=raw,
                                              shifts=None if registered is None else registered.shifts))
    elif has_pmd and "demix" in wants:
        pmd = load_stage(path, "compress", device=device)
        open_.append(SingleSessionDemixingVis(pmd, frame_timings=timings(pmd.shape[0]), device=device))

    if not open_:
        raise click.ClickException(f"nothing to show for --stage {stage}")

    import fastplotlib as fpl

    for viewer in open_:
        viewer.show()
    fpl.loop.run()


@main.command("params")
@click.argument("pattern", required=False)
@click.option("--section", default=None, help="Only this section, e.g. demix.nmf.")
@click.option("--json", "as_json", is_flag=True, default=False, help="Machine readable dump.")
def params(pattern, section, as_json):
    """
    List every parameter scraped off the config dataclasses.

    PATTERN keeps only the parameters whose name or help text contains it. A parameter can be
    set by its bare field name unless two sections define it, which is marked (*). Any of them
    can be given to a stage command as `-p name=value`.
    """
    import json

    from masknmf.pipelines import params as registry

    if section and section not in registry.SECTIONS:
        raise click.ClickException(
            f"unknown section {section}; choose from {', '.join(registry.SECTIONS)}"
        )

    claims = registry.bare_names()
    rows = [
        p
        for p in registry.registry().values()
        if (not section or p.section == section)
        and (not pattern or pattern.lower() in f"{p.name} {p.help}".lower())
    ]
    if not rows:
        raise click.ClickException("nothing matched")

    if as_json:
        click.echo(json.dumps(
            [
                {
                    "name": p.name, "section": p.section, "field": p.field,
                    "type": p.type_name, "default": p.default, "choices": p.choices,
                    "settable": p.settable, "owners": p.owners, "note": p.note,
                    "help": p.help, "ambiguous": len(claims[p.field]) > 1,
                }
                for p in rows
            ],
            indent=2, default=str,
        ))
        return

    width = max(len(p.name) for p in rows)
    kind = max(len(p.type_name) for p in rows)
    for name in registry.SECTIONS:
        here = [p for p in rows if p.section == name]
        if not here:
            continue
        click.secho(f"\n[{name}]", fg="cyan")
        for p in here:
            mark = "*" if len(claims[p.field]) > 1 else " "
            default = repr(p.default) if isinstance(p.default, str) else p.default
            trailing = "; ".join(t for t in (p.note, p.help) if t)
            click.echo(
                f"  {p.name:<{width}}{mark} {p.type_name:<{kind}} = {default}"
                + (f"    {trailing}" if trailing else "")
            )
    click.echo(f"\n{len(rows)} parameters. (*) needs its section to disambiguate.")


if __name__ == "__main__":
    main()
