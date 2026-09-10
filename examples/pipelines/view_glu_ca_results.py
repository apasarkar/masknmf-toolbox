"""Open the masknmf viewers on a GlutamateCalciumSpinePipeline run folder.

    python examples/pipelines/view_glu_ca_results.py <run_folder> motion red.tif
    python examples/pipelines/view_glu_ca_results.py <run_folder> compression red.tif --channel calcium
    python examples/pipelines/view_glu_ca_results.py <run_folder> demixing --channel glutamate --which spine
    python examples/pipelines/view_glu_ca_results.py <run_folder> signals --channel glutamate
    python examples/pipelines/view_glu_ca_results.py <run_folder> all red.tif

The raw tiff is cropped from the front so its frame count matches the stored shifts.
It's only needed (and required) for motion/compression/all; demixing and signals read
solely from the exported pmd_*.hdf5 / *_demixing.hdf5 and don't take a raw tiff.
signals opens SingleSessionDemixingVis on the bare compressed movie for drawing/labeling
and exporting ROIs before demixing has run; demixing already offers the same ROI tools
once a demixed result is loaded.
"""

import argparse
from pathlib import Path

import fastplotlib as fpl
import h5py
import numpy as np
import tifffile

import masknmf
from masknmf.visualization import CompressionVis, MotionCorrectionVis, SingleSessionDemixingVis
from masknmf.utils import torch_select_device

def registration(run, channel, raw_path):
    """Registered array over the raw tiff, cropped from the front to match the stored shifts."""
    raw = tifffile.imread(raw_path)
    with h5py.File(run / f"{channel}_moco.hdf5") as h:
        num_frames = h["RigidRegistrationArray/shifts"].shape[0]
    raw = raw[raw.shape[0] - num_frames:]
    return raw, masknmf.RigidRegistrationArray.from_hdf5(run / f"{channel}_moco.hdf5", input_movie=raw)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run", type=Path, help="timestamped *_glutamate_calcium_spine_results folder")
    p.add_argument("viz", choices=["motion", "compression", "demixing", "signals", "all"])
    p.add_argument("raw", type=Path, nargs="?", default=None, help="raw tiff of --channel; required for motion/compression/all, unused for demixing")
    p.add_argument("--channel", default="calcium", choices=["glutamate", "calcium"])
    p.add_argument("--which", default="spine", choices=["spine", "global_activity"], help="demixing result to open")
    p.add_argument("--fps", type=float, default=19.66, help="frame rate, for the seconds axis")
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    args = p.parse_args()
    if args.viz in ("motion", "compression", "all") and args.raw is None:
        p.error(f"raw tiff is required for --viz {args.viz}")
    run = args.run
    device = str(torch_select_device(args.device))
    open_ = []

    if args.viz in ("motion", "compression", "all"):
        raw, reg = registration(run, args.channel, args.raw)
        timings = np.arange(raw.shape[0]) / args.fps

        if args.viz in ("motion", "all"):
            open_.append(MotionCorrectionVis(reg, frame_timings=timings, mean_subtract=True))

        if args.viz in ("compression", "all"):
            pmd = masknmf.PMDArray.from_hdf5(run / f"pmd_{args.channel}.hdf5")
            # dense numpy, as in the notebook; the pipeline compressed the masked movie so masked-out pixels show in the residual
            moco = reg[:].cpu().numpy()
            open_.append(CompressionVis(moco, pmd, frame_timings=timings, device=device))

    if args.viz in ("demixing", "all"):
        res = masknmf.DemixingResults.from_hdf5(run / f"{args.channel}_{args.which}_demixing.hdf5", device=device)
        demix_timings = np.arange(res.shape[0]) / args.fps
        open_.append(SingleSessionDemixingVis(res, frame_timings=demix_timings, device=device))

    if args.viz == "signals":
        pmd = masknmf.PMDArray.from_hdf5(run / f"pmd_{args.channel}.hdf5", device=device)
        signal_timings = np.arange(pmd.shape[0]) / args.fps
        open_.append(SingleSessionDemixingVis(pmd, frame_timings=signal_timings, device=device))

    for v in open_:
        v.show()
    fpl.loop.run()


if __name__ == "__main__":
    main()
