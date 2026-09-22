"""Open the masknmf viewers on a GlutamateCalciumSpinePipeline results file.

    python examples/pipelines/view_glu_ca_results.py results.hdf5 motion red.tif
    python examples/pipelines/view_glu_ca_results.py results.hdf5 compression red.tif --channel calcium
    python examples/pipelines/view_glu_ca_results.py results.hdf5 demixing --channel glutamate --which spine
    python examples/pipelines/view_glu_ca_results.py results.hdf5 signals --channel glutamate
    python examples/pipelines/view_glu_ca_results.py results.hdf5 all red.tif

The raw tiff is indexed with the run's retained frames so it matches the stored shifts.
It's only needed (and required) for motion/compression/all; demixing and signals read
solely from the results file and don't take a raw tiff.
signals opens SingleSessionDemixingVis on the bare compressed movie for drawing/labeling
and exporting ROIs before demixing has run; demixing already offers the same ROI tools
once a demixed result is loaded.
"""

import argparse
from dataclasses import replace
from pathlib import Path

import fastplotlib as fpl
import numpy as np
import tifffile

import masknmf
from masknmf.visualization import CompressionVis, MotionCorrectionVis, SingleSessionDemixingVis
from masknmf.pipelines.configs.demixing_configs import NMFConfig
from masknmf.pipelines.subcellular.glutamate_calcium_spines import DEFAULT_SPINE_NMF_CONFIG, NMF_JUST_HALS
from masknmf.pipelines.subcellular.glutamate_calcium_spines import CALCIUM_PREFIX, GLOBAL_PREFIX, RUN_GROUP
from masknmf.utils import torch_select_device
from masknmf.utils._serialization import load_dict


def channel_prefix(run_info, channel):
    """Where a channel's groups sit: the primary channel at the top level, the other under calcium/."""
    primary = run_info["primary_channel"]
    primary = primary.decode() if isinstance(primary, bytes) else primary
    return "" if channel == primary else CALCIUM_PREFIX


def registration(results, run_info, prefix, raw_path):
    """Registered array over the raw frames the run kept."""
    raw = tifffile.imread(raw_path)[run_info["retained_frames"]]
    return raw, masknmf.RigidRegistrationArray.from_hdf5(results, input_movie=raw, prefix=prefix)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("results", type=Path, help="results .hdf5 the pipeline wrote")
    p.add_argument("viz", choices=["motion", "compression", "demixing", "signals", "all"])
    p.add_argument("raw", type=Path, nargs="?", default=None, help="raw tiff of --channel; required for motion/compression/all, optional for demixing (adds the raw panel and the shift traces)")
    p.add_argument("--channel", default="calcium", choices=["glutamate", "calcium"])
    p.add_argument("--which", default="spine", choices=["spine", "global_activity"], help="demixing result to open")
    p.add_argument("--cell-stats", type=Path, default=None, help="per-signal stats (.npy/.npz/.csv/.tsv, one row per signal) shown as sortable Signals-table columns; demixing only")
    p.add_argument("--cell-order", type=Path, default=None, help="signal ids in a custom order (.npy or one id per line), shown as an 'order' column; demixing only")
    p.add_argument("--fps", type=float, default=19.66, help="frame rate, for the seconds axis")
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    args = p.parse_args()
    if args.viz in ("motion", "compression", "all") and args.raw is None:
        p.error(f"raw tiff is required for --viz {args.viz}")
    results = args.results
    run_info = load_dict(results, RUN_GROUP)
    prefix = channel_prefix(run_info, args.channel)
    device = str(torch_select_device(args.device))
    open_ = []

    raw = reg = None
    if args.raw is not None:
        raw, reg = registration(results, run_info, prefix, args.raw)
        timings = np.arange(raw.shape[0]) / args.fps

        if args.viz in ("motion", "all"):
            open_.append(MotionCorrectionVis(reg, frame_timings=timings, mean_subtract=True))

        if args.viz in ("compression", "all"):
            pmd = masknmf.PMDArray.from_hdf5(results, prefix=prefix)
            # dense numpy, as in the notebook; the pipeline compressed the masked movie so masked-out pixels show in the residual
            moco = reg[:].cpu().numpy()
            open_.append(CompressionVis(moco, pmd, frame_timings=timings, device=device))

    if args.viz in ("demixing", "all"):
        demix_prefix = prefix + (GLOBAL_PREFIX if args.which == "global_activity" else "")
        res = masknmf.DemixingResults.from_hdf5(results, prefix=demix_prefix, device=device)
        demix_timings = np.arange(res.shape[0]) / args.fps
        # the pass the pipeline ran on this file: ring model off (ring_model_start_pt > maxiter), so Demix adds no background
        nmf_config = (
            replace(DEFAULT_SPINE_NMF_CONFIG)
            if (args.channel, args.which) == ("glutamate", "spine")
            else NMFConfig(**NMF_JUST_HALS)
        )
        open_.append(
            SingleSessionDemixingVis(
                res,
                frame_timings=demix_timings,
                device=device,
                results_path=results,
                nmf_config=nmf_config,
                raw=raw,
                shifts=None if reg is None else reg.shifts,
                cell_stats=args.cell_stats,
                cell_order=args.cell_order,
            )
        )

    if args.viz == "signals":
        pmd = masknmf.PMDArray.from_hdf5(results, prefix=prefix, device=device)
        signal_timings = np.arange(pmd.shape[0]) / args.fps
        open_.append(SingleSessionDemixingVis(pmd, frame_timings=signal_timings, device=device))

    for v in open_:
        v.show()
    fpl.loop.run()


if __name__ == "__main__":
    main()
