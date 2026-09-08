"""Open the masknmf viewers on a GlutamateCalciumSpinePipeline run folder.

    python examples/pipelines/view_glu_ca_results.py <run_folder> motion
    python examples/pipelines/view_glu_ca_results.py <run_folder> compression --channel calcium
    python examples/pipelines/view_glu_ca_results.py <run_folder> demixing --channel glutamate --which spine
    python examples/pipelines/view_glu_ca_results.py <run_folder> all

Raw tiffs are reloaded and cropped the same way run_glu_ca_pipeline.py did (--crop + --exclude-initial-frames).
"""

import argparse
from pathlib import Path

import fastplotlib as fpl
import numpy as np
import tifffile

import masknmf
from masknmf.visualization import CompressionVis, MotionCorrectionVis, SingleSessionDemixingVis

RAW = {
    "glutamate": "X:/data/temp/red_green/kg236_expt2_green.tif",
    "calcium": "X:/data/temp/red_green/kg236_expt2_red.tif",
}


def load_raw(channel, crop):
    return tifffile.imread(RAW[channel])[crop:]


def registration(run, channel, raw):
    return masknmf.RigidRegistrationArray.from_hdf5(run / f"{channel}_moco.hdf5", input_movie=raw)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run", type=Path, help="timestamped *_glutamate_calcium_spine_results folder")
    p.add_argument("viz", choices=["motion", "compression", "demixing", "all"])
    p.add_argument("--channel", default="calcium", choices=["glutamate", "calcium"])
    p.add_argument("--which", default="spine", choices=["spine", "global_activity"], help="demixing result to open")
    p.add_argument("--fps", type=float, default=19.66, help="frame rate, for the seconds axis")
    p.add_argument("--crop", type=int, default=400, help="frames dropped from the raw tiff (script crop + pipeline exclude)")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    run = args.run

    raw = load_raw(args.channel, args.crop)
    timings = np.arange(raw.shape[0]) / args.fps
    reg = registration(run, args.channel, raw)
    open_ = []

    if args.viz in ("motion", "all"):
        open_.append(MotionCorrectionVis(reg, frame_timings=timings, mean_subtract=True))

    if args.viz in ("compression", "all"):
        pmd = masknmf.PMDArray.from_hdf5(run / f"pmd_{args.channel}.hdf5")
        # dense numpy, as in the notebook; the pipeline compressed the masked movie so masked-out pixels show in the residual
        moco = reg[:].cpu().numpy()
        open_.append(CompressionVis(moco, pmd, frame_timings=timings, device=args.device))

    if args.viz in ("demixing", "all"):
        res = masknmf.DemixingResults.from_hdf5(run / f"{args.channel}_{args.which}_demixing.hdf5", device=args.device)
        open_.append(SingleSessionDemixingVis(res, frame_timings=timings, device=args.device))

    for v in open_:
        v.show()
    fpl.loop.run()


if __name__ == "__main__":
    main()
