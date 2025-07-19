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
import mbo_utilities as mbo

file = Path(r"D:\W2_DATA\santi\stitched\plane07_stitched/data.bin")
save_path = file.parent / "masknmf"
data_arr = mbo.imread(file)
ops = default_ops()
print(ops)

run_array(
    data_array=data_arr,
    data_index=file.stem,
    save_path=save_path,
    ops=ops,
    debug=ops["debug"],
    overwrite=ops["overwrite"],
    pmd_only=False,
)
x = 4
