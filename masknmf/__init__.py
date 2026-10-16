import logging
import sys
from pathlib import Path

logging.getLogger("torch.utils.flop_counter").setLevel(logging.ERROR)
logging.getLogger("cmap").setLevel(logging.ERROR)
# the masknmf console script imports this package before any of its own code runs
if Path(sys.argv[0]).stem == "masknmf":
    print("Loading packages...", end="", file=sys.stderr, flush=True)

from ._version import __version__, version_info
## TODO: Update the arrays import
from masknmf.arrays import *
from masknmf.utils import display
from masknmf.compression import *
from masknmf.motion_correction import *
from masknmf.demixing import *
from masknmf.visualization import *
from masknmf.diagnostics import *
from masknmf.pipelines import *
from masknmf.pipelines.configs import *