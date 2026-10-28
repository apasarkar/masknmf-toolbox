import datetime
import logging
import sys

from tqdm import tqdm

# iso 8601 basic form, 20261001T183740: safe in file names, and datetime.fromisoformat reads it back
TIMESTAMP_FORMAT = "%Y%m%dT%H%M%S"

logger = logging.getLogger("masknmf")
logger.setLevel(logging.INFO)
logger.propagate = False
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(logging.Formatter("%(asctime)s  %(message)s", datefmt="%H:%M:%S"))
logger.addHandler(handler)


def display(msg):
    """
    Log msg at info: the masknmf logger prints it with a timestamp and, during a pipeline run, writes it to the run's
    log file.
    """
    logger.info(msg)


def get_timestamp() -> str:
    """Now, as yyyymmddTHHMMSS, for file names, log lines and saved records."""
    return datetime.datetime.now().strftime(TIMESTAMP_FORMAT)


def progress(iterable, desc: str) -> tqdm:
    """A progress bar for iterable that clears once done, and stays off when stderr is a log file rather than a terminal."""
    interactive = sys.stderr.isatty() or "ipykernel" in sys.modules
    return tqdm(iterable, desc=desc, leave=False, disable=not interactive)
