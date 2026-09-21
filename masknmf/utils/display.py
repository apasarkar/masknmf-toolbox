import datetime
import sys


def display(msg):
    """
    Printing utility that logs time and flushes.
    """
    tag = "[" + datetime.datetime.today().strftime("%y-%m-%d %H:%M:%S") + "]: "
    sys.stdout.write(tag + msg + "\n")
    sys.stdout.flush()


def get_timestamp() -> str:
    """Now, as yyyy-mm-dd-HH-MM-SS, for file names."""
    return datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
