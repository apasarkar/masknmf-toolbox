import datetime
import logging
import sys

logger = logging.getLogger("masknmf")
logger.setLevel(logging.INFO)
logger.propagate = False
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(logging.Formatter("[%(asctime)s]: %(message)s", datefmt="%y-%m-%d %H:%M:%S"))
logger.addHandler(handler)


def display(msg):
    """
    Log msg at info: the masknmf logger prints it with a timestamp and, during a pipeline run, writes it to the run's
    log file.
    """
    logger.info(msg)


def get_timestamp() -> str:
    """Now, as yyyy-mm-dd-HH-MM-SS, for file names."""
    return datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
