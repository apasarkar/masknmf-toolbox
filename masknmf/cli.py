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

Imports here stay light; fastplotlib and the pipelines load inside the command that needs them.
"""

from __future__ import annotations

import sys
from pathlib import Path

import click

PROVENANCE = "MaskNMFRun"

REGISTRATION_GROUPS = (
    "RigidRegistrationArray",
    "PiecewiseRigidRegistrationArray",
    "GradientRegistrationArray",
)
STAGE_GROUPS = ("PMDArray", "DemixingResults", *REGISTRATION_GROUPS)

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


def _load_config(config_path, **overrides):
    """
    Read the TOML when given, then lay the command line over it.

    Args:
        config_path: A masknmf.toml, or None for the built-in defaults.
        **overrides: Section-qualified values as ``section__key``. None means "not passed".

    Returns:
        RunConfig: The resolved configuration.
    """
    from masknmf.pipelines.config_io import RunConfig

    try:
        cfg = RunConfig.from_toml(config_path) if config_path else RunConfig()
    except (ValueError, OSError) as e:
        raise click.ClickException(str(e)) from e

    for dotted, value in overrides.items():
        if value is None:
            continue
        section, key = dotted.split("__")
        setattr(getattr(cfg, section), key, value)

    try:
        cfg.validate()
    except ValueError as e:
        raise click.ClickException(str(e)) from e
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
    from masknmf.utils import has_group

    path = Path(path)
    if path.suffix.lower() not in RESULTS_SUFFIXES or not path.is_file():
        return False
    return any(has_group(str(path), g) for g in STAGE_GROUPS)


def groups_in(path: str | Path) -> list[str]:
    """The masknmf stage groups present in a results file, in pipeline order."""
    from masknmf.utils import has_group

    return [g for g in (*REGISTRATION_GROUPS, "PMDArray", "DemixingResults") if has_group(str(path), g)]


def write_provenance(results_path, raw_path, dataset, frame_rate, config_toml=""):
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
        group=PROVENANCE,
        exists_ok=True,
    )


def read_provenance(results_path) -> dict:
    """The provenance group of a results file, or an empty dict when it has none."""
    from masknmf.utils import has_group
    from masknmf.utils._serialization import load_dict

    if not has_group(str(results_path), PROVENANCE):
        return {}
    out = {}
    for key, value in load_dict(str(results_path), PROVENANCE).items():
        out[key] = value.decode() if isinstance(value, bytes) else value
    return out


def _raw_for(results_path, provenance, raw_override, dataset):
    """
    Reopen the raw movie a results file was built from.

    Raises:
        click.ClickException: If no raw path is recorded, or the recorded one has moved.
    """
    from masknmf import imread

    if raw_override:
        path = Path(raw_override)
    else:
        recorded = provenance.get("raw_path", "")
        relative = provenance.get("raw_path_relative", "")
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
    return imread(path, dataset=dataset or provenance.get("dataset") or None)


def _registered_from(results_path, movie):
    """Rebuild the registration array stored in a results file, over ``movie``."""
    import masknmf
    from masknmf.utils import has_group

    for name in REGISTRATION_GROUPS:
        if has_group(str(results_path), name):
            return getattr(masknmf, name).from_hdf5(results_path, input_movie=movie)
    return None


def _echo_config(cfg, stage: str):
    click.secho(f"\n{stage}", fg="cyan")
    for line in _config_lines(cfg, stage):
        click.echo(line)


def _config_lines(cfg, stage: str) -> list[str]:
    from dataclasses import fields

    sections = {"register": ["io", "compute", "register"],
                "compress": ["io", "compute", "register", "compress"],
                "demix": ["io", "compute", "demix"],
                "run": ["io", "compute", "register", "compress", "demix"]}[stage]
    lines = []
    for name in sections:
        lines.append(f"[{name}]")
        section = getattr(cfg, name)
        for f in fields(section):
            value = getattr(section, f.name)
            if isinstance(value, list):
                value = f"{len(value)} pass(es)" if value else "built-in schedule"
            lines.append(f"  {f.name:<26} {value}")
    return lines


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
@click.option("--kind", type=click.Choice(["rigid", "piecewise_rigid"]), default=None,
              help="Motion correction model.")
@click.option("--max-shifts", callback=lambda c, p, v: _csv_ints(v), default=None,
              help="Rigid: largest shift searched, e.g. 15,15.")
@click.option("--num-blocks", callback=lambda c, p, v: _csv_ints(v), default=None,
              help="Piecewise rigid: block grid, e.g. 15,15.")
def register(input_path, config_path, output, frame_rate, dataset, device, frame_batch_size,
             dry_run, kind, max_shifts, num_blocks):
    """
    Motion correct a raw movie.

    INPUT_PATH is a .tif, a directory of them, or a .h5/.hdf5.
    """
    cfg = _load_config(
        config_path,
        io__input=input_path, io__dataset=dataset, io__frame_rate=frame_rate,
        compute__device=device, compute__frame_batch_size=frame_batch_size,
        register__kind=kind, register__max_shifts=max_shifts, register__num_blocks=num_blocks,
    )
    out = _resolve_output(cfg, input_path, output)

    if dry_run:
        _echo_config(cfg, "register")
        click.echo(f"\nwould write {out}")
        return

    from masknmf import imread
    from masknmf.pipelines.stages import register as register_stage

    if cfg.register.kind == "skip":
        raise click.ClickException("[register] kind is 'skip'; nothing for this command to do")

    movie = imread(input_path, dataset=cfg.io.dataset or None)
    click.secho(f"registering {movie.shape} from {input_path}", fg="cyan")
    register_stage(
        movie,
        cfg.register_config(),
        device=cfg.compute.device,
        batch_size=cfg.compute.frame_batch_size,
        outpath=out,
    )
    write_provenance(out, input_path, cfg.io.dataset, cfg.io.frame_rate)
    click.secho(f"wrote {out}", fg="green")


@main.command("compress")
@click.argument("input_path", type=click.Path(exists=True))
@common_options
@click.option("--raw", "raw_override", type=click.Path(exists=True), default=None,
              help="The raw movie behind a results file, when it has moved since registration.")
@click.option("--kind", type=click.Choice(["plain", "denoise"]), default=None,
              help="Compression with or without denoising.")
@click.option("--max-components", type=int, default=None, help="Components kept per block.")
@click.option("--block-sizes", callback=lambda c, p, v: _csv_ints(v), default=None,
              help="Spatial block size, e.g. 20,20.")
@click.option("--exclude-border-radius", type=int, default=None,
              help="Zero this many pixels at each edge.")
def compress(input_path, config_path, output, frame_rate, dataset, device, frame_batch_size,
             dry_run, raw_override, kind, max_components, block_sizes, exclude_border_radius):
    """
    Compress a movie, optionally denoising as it goes.

    INPUT_PATH is either a results file holding a registration stage, or a raw movie to
    compress without motion correction.
    """
    cfg = _load_config(
        config_path,
        io__input=input_path, io__dataset=dataset, io__frame_rate=frame_rate,
        compute__device=device, compute__frame_batch_size=frame_batch_size,
        compress__kind=kind, compress__max_components=max_components,
        compress__block_sizes=block_sizes,
        register__exclude_border_radius=exclude_border_radius,
    )
    from_results = _is_results_file(input_path)
    out = output or (str(input_path) if from_results else _resolve_output(cfg, input_path, None))

    if dry_run:
        _echo_config(cfg, "compress")
        click.echo(f"\nsource: {'registered stage in ' + str(input_path) if from_results else 'raw ' + str(input_path)}")
        click.echo(f"would write {out}")
        return

    if cfg.compress.kind == "skip":
        raise click.ClickException("[compress] kind is 'skip'; nothing for this command to do")

    from masknmf import imread
    from masknmf.pipelines.stages import build_pixel_weighting
    from masknmf.pipelines.stages import compress as compress_stage

    provenance = {}
    if from_results:
        provenance = read_provenance(input_path)
        raw = _raw_for(input_path, provenance, raw_override, cfg.io.dataset)
        movie = _registered_from(input_path, raw)
        if movie is None:
            raise click.ClickException(
                f"{input_path} holds no registration stage; pass the raw movie instead"
            )
        if frame_rate is None and provenance.get("frame_rate"):
            cfg.io.frame_rate = float(provenance["frame_rate"])
        raw_source = provenance.get("raw_path", input_path)
    else:
        movie = imread(input_path, dataset=cfg.io.dataset or None)
        raw_source = input_path

    weighting = build_pixel_weighting(movie, cfg.register.exclude_border_radius)
    click.secho(f"compressing {movie.shape}", fg="cyan")
    compress_stage(
        movie,
        cfg.compress_config(),
        pixel_weighting=weighting,
        detrender=None,
        device=cfg.compute.device,
        outpath=out,
    )
    write_provenance(out, raw_source, cfg.io.dataset or provenance.get("dataset", ""), cfg.io.frame_rate)
    click.secho(f"wrote {out}", fg="green")


@main.command("demix")
@click.argument("input_path", type=click.Path(exists=True))
@common_options
@click.option("--highpass-sigma", type=float, default=None,
              help="Sigma of the spatial high-pass filter used for the first phase.")
@click.option("--no-detrend", is_flag=True, default=False,
              help="Do not spline-detrend while demixing.")
def demix(input_path, config_path, output, frame_rate, dataset, device, frame_batch_size,
          dry_run, highpass_sigma, no_detrend):
    """
    Demix a compressed movie.

    INPUT_PATH is a results file holding a PMDArray. Runs over a high-pass filtered copy
    first, then over the unfiltered movie seeded from what that found.
    """
    cfg = _load_config(
        config_path,
        io__input=input_path, io__dataset=dataset, io__frame_rate=frame_rate,
        compute__device=device, compute__frame_batch_size=frame_batch_size,
        demix__spatial_highpass_sigma=highpass_sigma,
        demix__detrend=False if no_detrend else None,
    )
    out = output or str(input_path)

    if dry_run:
        _echo_config(cfg, "demix")
        click.echo(f"\nwould write {out}")
        return

    from masknmf.utils import has_group

    if not has_group(str(input_path), "PMDArray"):
        raise click.ClickException(
            f"{input_path} holds no PMDArray; run `masknmf compress` on it first"
        )

    import masknmf
    from masknmf.pipelines.stages import build_detrender, demix_two_phase

    provenance = read_provenance(input_path)
    if frame_rate is None and provenance.get("frame_rate"):
        cfg.io.frame_rate = float(provenance["frame_rate"])

    pmd = masknmf.PMDArray.from_hdf5(str(input_path))
    detrender = (
        build_detrender(pmd.shape[0], cfg.io.frame_rate, 20.0, 20.0, cfg.compute.device)
        if cfg.demix.detrend
        else None
    )
    highpass, filtered, unfiltered = cfg.demixing_configs(detrender)

    click.secho(f"demixing {pmd.shape} at {cfg.io.frame_rate} Hz", fg="cyan")
    results = demix_two_phase(
        pmd,
        cfg.io.frame_rate,
        filtered_config=filtered,
        unfiltered_config=unfiltered,
        spatial_highpass_config=highpass,
        device=cfg.compute.device,
        frame_batch_size=cfg.compute.frame_batch_size,
        outpath=out,
    )
    if provenance:
        write_provenance(out, provenance.get("raw_path", input_path),
                         provenance.get("dataset", ""), cfg.io.frame_rate)
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
@click.option("--exclude-border-radius", type=int, default=None,
              help="Zero this many pixels at each edge when compressing.")
@click.option("--remove-intermediates", is_flag=True, default=False,
              help="Delete the registration and compression stages once demixing is done. "
                   "Off by default so the stages stay resumable and viewable.")
def run(input_path, config_path, output, frame_rate, dataset, device, frame_batch_size,
        dry_run, pipeline, skip, exclude_border_radius, remove_intermediates):
    """
    Run a whole pipeline end to end.

    INPUT_PATH is a .tif, a directory of them, or a .h5/.hdf5.
    """
    skipped = {s.strip() for s in skip.split(",")} if skip else set()
    unknown = skipped - {"register", "compress"}
    if unknown:
        raise click.ClickException(
            f"--skip does not know {', '.join(sorted(unknown))}; choose from register, compress"
        )

    cfg = _load_config(
        config_path,
        io__input=input_path, io__dataset=dataset, io__frame_rate=frame_rate,
        compute__device=device, compute__frame_batch_size=frame_batch_size,
        register__kind="skip" if "register" in skipped else None,
        register__exclude_border_radius=exclude_border_radius,
        compress__kind="skip" if "compress" in skipped else None,
    )
    out = _resolve_output(cfg, input_path, output)

    if dry_run:
        _echo_config(cfg, "run")
        click.echo(f"\npipeline: {PIPELINES[pipeline]}")
        click.echo(f"would write {out}")
        return

    import masknmf
    from masknmf import imread
    from masknmf.pipelines.stages import build_detrender

    movie = imread(input_path, dataset=cfg.io.dataset or None)
    detrender = (
        build_detrender(movie.shape[0], cfg.io.frame_rate, 20.0, 20.0, cfg.compute.device)
        if cfg.demix.detrend
        else None
    )
    highpass, filtered, unfiltered = cfg.demixing_configs(detrender)

    shared = {
        "motion_correct_config": cfg.register_config(),
        "compress_config": cfg.compress_config(),
        "outpath_compression": out,
        "frame_batch_size": cfg.compute.frame_batch_size,
        "device": cfg.compute.device,
    }
    cls = getattr(masknmf, PIPELINES[pipeline])

    click.secho(f"{PIPELINES[pipeline]} on {movie.shape} at {cfg.io.frame_rate} Hz", fg="cyan")
    if pipeline == "widefield":
        pipe = cls(outpath_motion_correction=out, **shared)
        pipe.run(movie, exclude_border_radius=cfg.register.exclude_border_radius)
    elif pipeline == "twophoton":
        pipe = cls(
            spatial_highpass_config=highpass,
            filtered_demixing_config=filtered,
            unfiltered_demixing_config=unfiltered,
            outpath_motion_correction=out,
            outpath_demixing=out,
            **shared,
        )
        pipe.run(
            movie,
            cfg.io.frame_rate,
            exclude_border_radius=cfg.register.exclude_border_radius,
            remove_intermediates=remove_intermediates,
        )
    else:
        raise click.ClickException(
            f"--pipeline {pipeline} takes inputs this command cannot supply yet "
            "(active frames for onephoton, two channels for spines); drive it from Python"
        )

    write_provenance(out, input_path, cfg.io.dataset, cfg.io.frame_rate)
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
    present = groups_in(path)
    if not present:
        raise click.ClickException(
            f"{path} holds no masknmf results (looked for {', '.join(STAGE_GROUPS)})"
        )

    provenance = read_provenance(path)
    click.secho(str(Path(path).resolve()), bold=True)
    click.secho("\nStages", fg="cyan")
    for name in present:
        click.echo(f"  {name}")
    if provenance:
        click.secho("\nRun", fg="cyan")
        for key in ("raw_path", "frame_rate", "version", "timestamp"):
            if provenance.get(key) not in (None, ""):
                click.echo(f"  {key:<12} {provenance[key]}")
    if list_only:
        return

    import numpy as np

    import masknmf
    from masknmf.utils import torch_select_device
    from masknmf.visualization import (
        CompressionVis,
        MotionCorrectionVis,
        SingleSessionDemixingVis,
    )

    device = str(torch_select_device(device)) if device == "auto" else device
    if frame_rate is None:
        frame_rate = float(provenance.get("frame_rate") or 0) or None

    wants = {stage} if stage else {"register", "compress", "demix"}
    has_registration = any(g in present for g in REGISTRATION_GROUPS)
    open_ = []
    raw = registered = None

    if has_registration and wants & {"register", "compress"}:
        try:
            raw = _raw_for(path, provenance, raw_override, provenance.get("dataset"))
            registered = _registered_from(path, raw)
        except click.ClickException as e:
            if stage in ("register", "compress"):
                raise
            click.secho(f"skipping the registration viewers: {e.message}", fg="yellow")

    def timings(n):
        return np.arange(n) / frame_rate if frame_rate else None

    if registered is not None and "register" in wants:
        open_.append(MotionCorrectionVis(registered, frame_timings=timings(registered.shape[0]),
                                         mean_subtract=True))

    if "PMDArray" in present and "compress" in wants and registered is not None:
        pmd = masknmf.PMDArray.from_hdf5(str(path))
        open_.append(CompressionVis(registered[:].cpu().numpy(), pmd,
                                    frame_timings=timings(pmd.shape[0]), device=device))

    if "DemixingResults" in present and "demix" in wants:
        results = masknmf.DemixingResults.from_hdf5(str(path), device=device)
        open_.append(SingleSessionDemixingVis(results, frame_timings=timings(results.shape[0]),
                                              device=device, results_path=path, raw=raw,
                                              shifts=None if registered is None else registered.shifts))
    elif "PMDArray" in present and "demix" in wants and "DemixingResults" not in present:
        pmd = masknmf.PMDArray.from_hdf5(str(path), device=device)
        open_.append(SingleSessionDemixingVis(pmd, frame_timings=timings(pmd.shape[0]), device=device))

    if not open_:
        raise click.ClickException(f"nothing to show for --stage {stage}")

    import fastplotlib as fpl

    for viewer in open_:
        viewer.show()
    fpl.loop.run()


def _csv_ints(value):
    """Parse ``15,15`` into a tuple of ints; None passes through."""
    if not value:
        return None
    try:
        return tuple(int(v) for v in str(value).split(",") if v.strip())
    except ValueError as e:
        raise click.BadParameter(f"expected comma-separated integers, got {value!r}") from e


if __name__ == "__main__":
    main()
