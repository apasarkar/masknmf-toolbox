"""
Run the glutamate/calcium spine pipeline on raw tiffs.

green = glutamate, red = calcium

First 200 frames optionally cropped

    python scripts/run_spine_pipeline.py
    python scripts/run_spine_pipeline.py --glutamate glu.tif --calcium ca.tif --output ./test_outputs
"""

import argparse
from pathlib import Path

import tifffile

from masknmf.pipelines.subcellular import GlutamateCalciumSpinePipeline

DEFAULT_GLUTAMATE = "X:/data/temp/red_green/kg236_expt2_green.tif"
DEFAULT_CALCIUM = "X:/data/temp/red_green/kg236_expt2_red.tif"
DEFAULT_OUTPUT = "X:/data/temp/red_green/test_outputs"


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--glutamate",
        default=DEFAULT_GLUTAMATE,
        help="glutamate channel tiff, or 'none'",
    )
    parser.add_argument(
        "--calcium", default=DEFAULT_CALCIUM, help="calcium channel tiff, or 'none'"
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help="parent folder; a timestamped run folder is created inside",
    )
    parser.add_argument(
        "--crop",
        type=int,
        default=200,
        help="frames dropped from the start of each tiff, as in the notebook",
    )
    parser.add_argument(
        "--exclude-initial-frames",
        type=int,
        default=200,
        help="passed through to pipe.run",
    )
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    args = parser.parse_args()

    glu = (
        None
        if args.glutamate in (None, "none")
        else tifffile.imread(args.glutamate)[args.crop :]
    )
    calcium = (
        None
        if args.calcium in (None, "none")
        else tifffile.imread(args.calcium)[args.crop :]
    )
    if glu is not None:
        print(f"glutamate {glu.shape} {glu.dtype}")
    if calcium is not None:
        print(f"calcium {calcium.shape} {calcium.dtype}")

    Path(args.output).mkdir(exist_ok=True)
    pipe = GlutamateCalciumSpinePipeline(output_folder=args.output, device=args.device)
    pipe.run(
        glutamate_channel=glu,
        calcium_channel=calcium,
        exclude_initial_frames=args.exclude_initial_frames,
    )
    print(f"done, results under {args.output}")


if __name__ == "__main__":
    main()
