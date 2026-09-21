"""
TOML front end for the pipeline configs.

The dataclasses here mirror what a ``masknmf.toml`` holds. They are deliberately separate from
the algorithm dataclasses in :mod:`masknmf.pipelines.configs`: these carry only what TOML can
express, and :meth:`RunConfig.to_configs` turns them into the real config objects.

Three fields cannot be written literally and get an encoding of their own:

- ``Optional[int]``, e.g. ``ring_model_start_pt``, takes the string ``"none"``, following the
  ``"skip"`` sentinel the pipelines already use.
- ndarray fields, e.g. ``template``, take the path of a ``.npy`` file; ``""`` means unset.
- the detrenders are absent entirely. They are sized from the frame rate at run time.
"""

from __future__ import annotations

import json
import tomllib
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import *

NONE = "none"

DEVICES = ("auto", "cuda", "cpu")
REGISTER_KINDS = ("rigid", "piecewise_rigid", "skip")
COMPRESS_KINDS = ("plain", "denoise", "skip")
SIGNS = ("positive", "negative", "unconstrained")


@dataclass
class IOConfig:
    input: str = ""
    dataset: str = ""
    output: str = "results.hdf5"
    frame_rate: float = 30.0


@dataclass
class ComputeConfig:
    device: str = "auto"
    frame_batch_size: int = 300


@dataclass
class RegisterConfig:
    kind: str = "rigid"
    max_shifts: tuple[int, int] = (15, 15)
    num_blocks: tuple[int, int] = (15, 15)
    overlaps: tuple[int, int] = (5, 5)
    max_rigid_shifts: tuple[int, int] = (15, 15)
    max_deviation_rigid: tuple[int, int] = (2, 2)
    template: str = ""
    pixel_weighting: str = ""
    exclude_border_radius: int = 0


@dataclass
class CompressSection:
    kind: str = "denoise"
    block_sizes: tuple[int, int] = (20, 20)
    frame_range: Union[int, str] = NONE
    max_components: int = 20
    sim_conf: int = 5
    frame_batch_size: int = 10000
    max_consecutive_failures: int = 1
    spatial_avg_factor: int = 1
    temporal_avg_factor: int = 1
    compute_normalizer: bool = True
    pixel_weighting: str = ""
    noise_variance_quantile: float = 0.3
    num_epochs: int = 10


@dataclass
class InitTable:
    mad_correlation_threshold: float = 0.8
    min_peak_distance: int = 3
    mad_threshold: float = 1.0
    residual_threshold: float = 0.3
    patch_size: tuple[int, int] = (40, 40)
    sign: str = "positive"


@dataclass
class NmfTable:
    maxiter: int = 40
    support_threshold: tuple[float, float] = (0.95, 0.8)
    deletion_threshold: float = 0.2
    min_brightness: float = 1.0
    ring_model_start_pt: Union[int, str] = 0
    ring_radius: int = 10
    background_downsampling_factor: int = 30
    merge_threshold: float = 0.6
    merge_overlap_threshold: float = 0.6
    update_frequency: int = 4
    c_nonneg: bool = True
    denoise: bool = False
    reassign_background: bool = True


@dataclass
class PassTable:
    init: InitTable = field(default_factory=InitTable)
    nmf: NmfTable = field(default_factory=NmfTable)


@dataclass
class DemixSection:
    spatial_highpass_sigma: float = 4.0
    detrend: bool = True
    filtered: list[PassTable] = field(default_factory=list)
    unfiltered: list[PassTable] = field(default_factory=list)


SECTIONS = {
    "io": IOConfig,
    "compute": ComputeConfig,
    "register": RegisterConfig,
    "compress": CompressSection,
    "demix": DemixSection,
}

HELP: dict[str, dict[str, str]] = {
    "io": {
        "input": "raw movie: a .tif, a directory of them, or a .h5/.hdf5",
        "dataset": "hdf5 only: dataset holding the movie; empty picks the sole 3D one",
        "output": "results file; every stage writes its own group into it",
        "frame_rate": "acquisition rate in Hz, used to size the detrenders",
    },
    "compute": {
        "device": f"one of {', '.join(DEVICES)}",
        "frame_batch_size": "frames held on the GPU at a time",
    },
    "register": {
        "kind": f"one of {', '.join(REGISTER_KINDS)}",
        "max_shifts": "rigid only: largest shift searched, in pixels",
        "num_blocks": "piecewise_rigid only: block grid over the field of view",
        "overlaps": "piecewise_rigid only: block overlap, in pixels",
        "max_rigid_shifts": "piecewise_rigid only: largest whole-frame shift",
        "max_deviation_rigid": "piecewise_rigid only: how far a block may stray from it",
        "template": "optional .npy template; empty computes one from the data",
        "pixel_weighting": "optional .npy weight image",
        "exclude_border_radius": "zero this many pixels at each edge before compressing",
    },
    "compress": {
        "kind": f"one of {', '.join(COMPRESS_KINDS)}",
        "block_sizes": "spatial block size the decomposition runs on",
        "frame_range": f"frames used to fit; {NONE!r} uses all of them",
        "max_components": "components kept per block",
        "sim_conf": "significance level for keeping a component",
        "frame_batch_size": "frames per compression batch",
        "max_consecutive_failures": "rejected components before a block stops",
        "spatial_avg_factor": "spatial downsampling before fitting",
        "temporal_avg_factor": "temporal downsampling before fitting",
        "compute_normalizer": "estimate and divide out per-pixel noise",
        "pixel_weighting": "optional .npy weight image, multiplied into the border mask",
        "noise_variance_quantile": "denoise only: quantile taken as the noise floor",
        "num_epochs": "denoise only: training epochs for the denoiser",
    },
    "demix": {
        "spatial_highpass_sigma": "sigma of the high-pass filter used for the first phase",
        "detrend": "spline-detrend during demixing, sized from io.frame_rate",
        "filtered": "passes over the filtered movie; omit for the built-in schedule",
        "unfiltered": "passes over the unfiltered movie; omit for the built-in schedule",
    },
}

PASS_HELP: dict[str, dict[str, str]] = {
    "init": {
        "mad_correlation_threshold": "correlation a superpixel must reach to seed a signal",
        "sign": f"deviations to seed from: {', '.join(SIGNS)}",
        "patch_size": "superpixel search patch",
    },
    "nmf": {
        "maxiter": "NMF iterations in this pass",
        "support_threshold": "(spatial, temporal) support quantiles",
        "ring_model_start_pt": f"iteration the ring background starts; {NONE!r} disables it",
        "merge_threshold": "correlation above which two signals merge",
    },
}


def _coerce(value, default):
    """Bring a parsed TOML value in line with the type of the field's default."""
    if isinstance(default, tuple):
        if not isinstance(value, (list, tuple)):
            raise TypeError(f"expected a list, got {value!r}")
        return tuple(value)
    if isinstance(default, bool):
        if not isinstance(value, bool):
            raise TypeError(f"expected true or false, got {value!r}")
        return value
    if isinstance(default, float) and isinstance(value, int) and not isinstance(value, bool):
        return float(value)
    return value


def _load_section(cls, mapping: dict, name: str):
    """Build one dataclass from a TOML table, rejecting keys it does not define."""
    known = {f.name: f for f in fields(cls)}
    unknown = sorted(set(mapping) - set(known))
    if unknown:
        raise ValueError(f"[{name}] unknown key(s): {', '.join(unknown)}")

    kwargs = {}
    for key, value in mapping.items():
        spec = known[key]
        if spec.name in ("filtered", "unfiltered"):
            kwargs[key] = [_load_pass(entry, f"{name}.{key}") for entry in value]
            continue
        try:
            kwargs[key] = _coerce(value, getattr(cls(), key))
        except (TypeError, ValueError) as e:
            raise ValueError(f"[{name}] {key}: {e}") from e
    return cls(**kwargs)


def _load_pass(mapping: dict, name: str) -> PassTable:
    unknown = sorted(set(mapping) - {"init", "nmf"})
    if unknown:
        raise ValueError(f"[[{name}]] unknown table(s): {', '.join(unknown)}")
    return PassTable(
        init=_load_section(InitTable, mapping.get("init", {}), f"{name}.init"),
        nmf=_load_section(NmfTable, mapping.get("nmf", {}), f"{name}.nmf"),
    )


@dataclass
class RunConfig:
    io: IOConfig = field(default_factory=IOConfig)
    compute: ComputeConfig = field(default_factory=ComputeConfig)
    register: RegisterConfig = field(default_factory=RegisterConfig)
    compress: CompressSection = field(default_factory=CompressSection)
    demix: DemixSection = field(default_factory=DemixSection)

    @classmethod
    def from_dict(cls, mapping: dict) -> RunConfig:
        """
        Build a config from a parsed TOML mapping, layered over the defaults.

        Raises:
            ValueError: On an unknown section, an unknown key within one, or a value whose
                type does not match the field's.
        """
        unknown = sorted(set(mapping) - set(SECTIONS))
        if unknown:
            raise ValueError(f"unknown section(s): {', '.join(unknown)}")
        built = {
            name: _load_section(section_cls, mapping.get(name, {}), name)
            for name, section_cls in SECTIONS.items()
        }
        cfg = cls(**built)
        cfg.validate()
        return cfg

    @classmethod
    def from_toml(cls, path: str | Path) -> RunConfig:
        """Read and validate a ``masknmf.toml``."""
        return cls.from_dict(tomllib.loads(Path(path).read_text()))

    def validate(self) -> None:
        """
        Check the values that have a fixed set of choices.

        Raises:
            ValueError: On an out-of-range choice or a non-positive frame rate.
        """
        if self.compute.device not in DEVICES:
            raise ValueError(f"[compute] device must be one of {', '.join(DEVICES)}")
        if self.register.kind not in REGISTER_KINDS:
            raise ValueError(f"[register] kind must be one of {', '.join(REGISTER_KINDS)}")
        if self.compress.kind not in COMPRESS_KINDS:
            raise ValueError(f"[compress] kind must be one of {', '.join(COMPRESS_KINDS)}")
        if self.io.frame_rate <= 0:
            raise ValueError("[io] frame_rate must be positive")
        for which in ("filtered", "unfiltered"):
            for i, p in enumerate(getattr(self.demix, which)):
                if p.init.sign not in SIGNS:
                    raise ValueError(
                        f"[[demix.{which}]] #{i} init.sign must be one of {', '.join(SIGNS)}"
                    )

    def register_config(self):
        """The motion correction config, or the string ``"skip"``."""
        from masknmf.pipelines.configs.motion_correction_configs import (
            PiecewiseRigidMotionCorrectionConfig,
            RigidMotionCorrectionConfig,
        )

        r = self.register
        if r.kind == "skip":
            return "skip"
        common = {
            "template": _load_npy(r.template),
            "pixel_weighting": _load_npy(r.pixel_weighting),
        }
        if r.kind == "rigid":
            return RigidMotionCorrectionConfig(max_shifts=r.max_shifts, **common)
        return PiecewiseRigidMotionCorrectionConfig(
            num_blocks=r.num_blocks,
            overlaps=r.overlaps,
            max_rigid_shifts=r.max_rigid_shifts,
            max_deviation_rigid=r.max_deviation_rigid,
            **common,
        )

    def compress_config(self):
        """The compression config, or the string ``"skip"``."""
        from masknmf.pipelines.configs.compression_configs import (
            CompressConfig,
            CompressDenoiseConfig,
        )

        c = self.compress
        if c.kind == "skip":
            return "skip"
        shared = {
            "block_sizes": c.block_sizes,
            "frame_range": _maybe_none(c.frame_range),
            "max_components": c.max_components,
            "sim_conf": c.sim_conf,
            "max_consecutive_failures": c.max_consecutive_failures,
            "spatial_avg_factor": c.spatial_avg_factor,
            "temporal_avg_factor": c.temporal_avg_factor,
            "compute_normalizer": c.compute_normalizer,
            "pixel_weighting": _load_npy(c.pixel_weighting),
        }
        if c.kind == "plain":
            return CompressConfig(frame_batch_size=c.frame_batch_size, **shared)
        return CompressDenoiseConfig(
            noise_variance_quantile=c.noise_variance_quantile,
            num_epochs=c.num_epochs,
            **shared,
        )

    def demixing_configs(self, detrender=None) -> tuple:
        """
        The demixing configs described by ``[[demix.filtered]]`` / ``[[demix.unfiltered]]``.

        Args:
            detrender: Stamped into every pass this builds, when ``[demix] detrend`` is on.
                Passes written out in TOML carry no detrender of their own, so without this
                they would silently demix untrended.

        Returns:
            tuple: ``(spatial_highpass_config, filtered, unfiltered)``. Either multipass config
            is None when its table is absent, which leaves the built-in schedule in place.
        """
        from masknmf.pipelines.configs.demixing_configs import (
            MultipassDemixingConfig,
            NMFConfig,
            SinglepassDemixingConfig,
            SpatialHighpassConfig,
            SuperpixelInitConfig,
        )

        if not self.demix.detrend:
            detrender = None

        def build(passes):
            if not passes:
                return None
            out = []
            for p in passes:
                out.append(
                    SinglepassDemixingConfig(
                        SuperpixelInitConfig(
                            mad_correlation_threshold=p.init.mad_correlation_threshold,
                            min_peak_distance=p.init.min_peak_distance,
                            mad_threshold=p.init.mad_threshold,
                            residual_threshold=p.init.residual_threshold,
                            patch_size=p.init.patch_size,
                            sign=p.init.sign,
                            detrender=detrender,
                        ),
                        NMFConfig(
                            maxiter=p.nmf.maxiter,
                            support_threshold=p.nmf.support_threshold,
                            deletion_threshold=p.nmf.deletion_threshold,
                            min_brightness=p.nmf.min_brightness,
                            ring_model_start_pt=_maybe_none(p.nmf.ring_model_start_pt),
                            ring_radius=p.nmf.ring_radius,
                            background_downsampling_factor=p.nmf.background_downsampling_factor,
                            merge_threshold=p.nmf.merge_threshold,
                            merge_overlap_threshold=p.nmf.merge_overlap_threshold,
                            update_frequency=p.nmf.update_frequency,
                            c_nonneg=p.nmf.c_nonneg,
                            denoise=p.nmf.denoise,
                            reassign_background=p.nmf.reassign_background,
                            detrender=detrender,
                        ),
                    )
                )
            return MultipassDemixingConfig(out)

        highpass = SpatialHighpassConfig(filter_sigma=self.demix.spatial_highpass_sigma)
        return highpass, build(self.demix.filtered), build(self.demix.unfiltered)


def _maybe_none(value):
    """Turn the ``"none"`` sentinel back into ``None``."""
    return None if isinstance(value, str) and value.lower() == NONE else value


def _load_npy(path: str):
    """Load a ``.npy`` field, treating the empty string as unset."""
    if not path:
        return None
    import numpy as np

    return np.load(path)


def _toml_value(value) -> str:
    """Render a scalar or list the way TOML spells it."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        # TOML basic strings take JSON's escaping, which Windows paths need: \U in C:\Users... is
        # otherwise read as a unicode escape and rejected
        return json.dumps(value)
    if isinstance(value, (tuple, list)):
        return "[" + ", ".join(_toml_value(v) for v in value) + "]"
    return str(value)


def _render_table(obj, help_for: dict[str, str], indent: str = "") -> list[str]:
    rows = [
        (f.name, _toml_value(getattr(obj, f.name)), help_for.get(f.name, ""))
        for f in fields(obj)
        if not isinstance(getattr(obj, f.name), list) and not is_dataclass(getattr(obj, f.name))
    ]
    width = max((len(f"{k} = {v}") for k, v, _ in rows), default=0)
    lines = []
    for key, value, comment in rows:
        assignment = f"{indent}{key} = {value}"
        lines.append(f"{assignment:<{width + len(indent)}}  # {comment}" if comment else assignment)
    return lines


def render_template(input_path: str = "", output_path: str = "results.hdf5") -> str:
    """
    A commented ``masknmf.toml`` carrying every default.

    Args:
        input_path (str): Prefilled ``[io] input``.
        output_path (str): Prefilled ``[io] output``.

    Returns:
        str: The file's contents.
    """
    cfg = RunConfig()
    cfg.io.input = input_path
    cfg.io.output = output_path

    out = [
        "# masknmf run configuration.",
        "# Every value here is the default; delete what you do not need to change.",
        "# Any of these can be overridden on the command line.",
        "",
    ]
    for name in SECTIONS:
        out.append(f"[{name}]")
        out.extend(_render_table(getattr(cfg, name), HELP.get(name, {})))
        out.append("")

    out += [
        "# Demixing runs in two phases: first over a high-pass filtered copy of the movie,",
        "# then over the unfiltered one seeded from what the first phase found. Each phase is",
        "# a list of passes. Leave both lists out to use the built-in schedule (two filtered",
        "# passes, three unfiltered). Uncomment to take control of it.",
        "#",
    ]
    example = PassTable()
    commented = [
        "[[demix.filtered]]",
        "[demix.filtered.init]",
        *_render_table(example.init, PASS_HELP["init"]),
        "[demix.filtered.nmf]",
        *_render_table(example.nmf, PASS_HELP["nmf"]),
    ]
    out.extend(f"# {line}" for line in commented)
    out.append("")
    return "\n".join(out)
