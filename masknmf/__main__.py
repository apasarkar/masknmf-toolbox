import argparse
import logging
from pathlib import Path

import numpy as np
import tifffile
import torch

from masknmf.run_masknmf import run_array

logger = logging.getLogger(__name__)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

def default_ops():
    """
    Default options for masknmf processing.
    """
    return {
        "do_rigid": True,
        "do_nonrigid": True,
        "overwrite": False,
        "debug": False,
    }

def add_args(parser: argparse.ArgumentParser):
    parser.add_argument("inpath", type=Path, help="Input TIFF directory")
    parser.add_argument("outpath", type=Path, help="Output directory")
    parser.add_argument("--ops", type=str, default="", help="Optional path to ops npy file")

    ops0 = default_ops()
    for k, v in ops0.items():
        t = type(v)
        if isinstance(v, bool):
            parser.add_argument(f"--{k}", type=int, default=int(v), help=f"{k} (bool, default={v})")
        else:
            parser.add_argument(f"--{k}", type=t, default=v, help=f"{k} (default={v})")
    return parser

def parse_args(parser: argparse.ArgumentParser):
    args = parser.parse_args()
    dargs = vars(args)

    ops = default_ops()
    if args.ops:
        ops.update(np.load(args.ops, allow_pickle=True).item())

    for k in default_ops():
        if k in dargs:
            v = dargs[k]
            ops[k] = bool(v) if isinstance(default_ops()[k], bool) else type(default_ops()[k])(v)
    return args, ops


def main():
    args, ops = parse_args(add_args(argparse.ArgumentParser()))

    args.outpath.mkdir(exist_ok=True)
    completed = {x.stem for x in args.outpath.iterdir()}
    files = sorted(args.inpath.glob("*.tif*"))

    for file in files:

        if file.stem in completed and not ops["overwrite"]:
            print(f"Skipping {file.stem}, already processed.")
            continue

        data_arr = tifffile.memmap(file)

        run_array(
            data_array=data_arr,
            data_index=file.stem,
            save_path=args.outpath,
            ops=ops,
            debug=ops["debug"],
            overwrite=ops["overwrite"],
        )

if __name__ == "__main__":
    main()