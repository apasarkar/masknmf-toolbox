import argparse
import logging
from pathlib import Path

import numpy as np
import tifffile
import torch

from masknmf.default_ops import default_ops
from masknmf.run_masknmf import run_array

logger = logging.getLogger(__name__)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

def is_completed(file: Path, outdir: Path, *, required=("pmd_demixer.npy",)) -> bool:
    file = Path(file)
    outdir = Path(outdir).expanduser()
    plane_dir = outdir / file.stem
    return all((plane_dir / name).exists() for name in required)

def add_args(parser: argparse.ArgumentParser):
    parser.add_argument("inpath", type=Path, help="Input TIFF directory")
    parser.add_argument("outpath", type=Path, help="Output directory")
    parser.add_argument("--ops", type=str, default="", help="Optional path to ops npy file")
    parser.add_argument("--pmd-only", action="store_true", help="Return after running PMD."
                                                                "Helpful if you expect demixing to fail.")

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

    inpath = args.inpath
    if inpath.is_file():
        files = [inpath]
    elif inpath.is_dir():
        files = sorted(inpath.glob("*.tif*"))
    else:
        raise ValueError(f"Invalid input path: {inpath}")

    if not files:
        raise ValueError(f"No TIFF files found in {inpath}")

    for file in files:
        if is_completed(args.outpath, file.stem) and not ops["overwrite"]:
            print(f"Skipping {file.stem}, already processed.")
            continue
        print(f"Running: {file.name}")
        try:
            data_arr = tifffile.memmap(file)
        except (ValueError, MemoryError):
            data_arr = tifffile.imread(file)

        run_array(
            data_array=data_arr,
            data_index=file.stem,
            save_path=args.outpath,
            ops=ops,
            debug=ops["debug"],
            overwrite=ops["overwrite"],
            pmd_only=args.pmd_only,
        )

if __name__ == "__main__":
    main()