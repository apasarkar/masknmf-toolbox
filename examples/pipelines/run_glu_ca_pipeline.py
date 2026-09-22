"""
Run the glutamate/calcium spine pipeline on raw tiffs.

green = glutamate, red = calcium

The pipeline drops the first --exclude-initial-frames frames.

    python examples/pipelines/run_glu_ca_pipeline.py --glutamate glu.tif --calcium ca.tif --output ./test_outputs/results.hdf5
"""

import argparse

import tifffile

from masknmf.pipelines.subcellular import GlutamateCalciumSpinePipeline


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--glutamate",
        default=None,
        help="glutamate channel tiff, or 'none'",
    )
    parser.add_argument(
        "--calcium", default=None, help="calcium channel tiff, or 'none'"
    )
    parser.add_argument(
        "--output",
        required=True,
        help="results .hdf5 file every stage writes into",
    )
    parser.add_argument(
        "--exclude-initial-frames",
        type=int,
        default=200,
        help="passed through to pipe.run",
    )
    parser.add_argument(
        "--glutamate-sign", default="positive", choices=["positive", "negative"],
        help="'negative' for an indicator that dims with activity",
    )
    parser.add_argument(
        "--calcium-sign", default="positive", choices=["positive", "negative"],
        help="'negative' for an indicator that dims with activity",
    )
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    args = parser.parse_args()

    glu = (
        None
        if args.glutamate in (None, "none")
        else tifffile.imread(args.glutamate)
    )
    calcium = (
        None
        if args.calcium in (None, "none")
        else tifffile.imread(args.calcium)
    )
    if glu is not None:
        print(f"glutamate {glu.shape} {glu.dtype}")
    if calcium is not None:
        print(f"calcium {calcium.shape} {calcium.dtype}")

    pipe = GlutamateCalciumSpinePipeline(outpath=args.output, device=args.device)
    out = pipe.run(
        glutamate_channel=glu,
        calcium_channel=calcium,
        exclude_initial_frames=args.exclude_initial_frames,
        glutamate_indicator_sign=args.glutamate_sign,
        calcium_indicator_sign=args.calcium_sign,
    )
    print(f"done, results in {out}")


if __name__ == "__main__":
    main()
