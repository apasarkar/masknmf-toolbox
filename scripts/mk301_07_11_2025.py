import argparse
import logging
import time
from pathlib import Path
import fastplotlib as fpl

import numpy as np
import tifffile
import torch

from masknmf.default_ops import default_ops
from masknmf.run_masknmf import run_array
import masknmf

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

# data = tifffile.imread(r"D:\W2_DATA\foconnell\2025-07-10_Pollen\plane1\analysis\Zplane1_pollen_2mROIs_224px-USampleON_1ms-1ms_ScanPhase_neg0p9015us_00001_substack.tif")
# std_image = np.std(data, axis=0)
# fpl.ImageWidget([std_image]).show()
# fpl.loop.run()

z1 = Path(r"D:\W2_DATA\kbarber\2025_07_16\m350\assembled_phase_frame\temp\file00000_chan0\pmd_obj.npy")
# z2 = Path(r"D:\W2_DATA\foconnell\2025-07-10_Pollen\Zplane7_pollen_2mROIs_224px-USampleON_1ms-1ms_ScanPhase_neg1p008us_00001\pmd_obj.npy")
# z3 = Path(r"D:\W2_DATA\foconnell\2025-07-10_Pollen\Zplane14_pollen_2mROIs_224px-USampleON_1ms-1ms_ScanPhase_neg0p9015us_00001\pmd_obj.npy")
z1_pmd = np.load(z1, allow_pickle=True).item()
# z2_pmd = np.load(z2, allow_pickle=True).item()
# z3_pmd = np.load(z3, allow_pickle=True).item()
z1_pmd.rescale = False
# z2_pmd.rescale = False
# z3_pmd.rescale = False

fpl.ImageWidget([z1_pmd], names=["Plane 01"]).show()
fpl.loop.run()
