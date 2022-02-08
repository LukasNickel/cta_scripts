import logging
from rich.logging import RichHandler


def setup_logging(logfile=None, verbose=False):

    level = logging.INFO
    if verbose is True:
        level = logging.DEBUG

    log = logging.getLogger()
    log.level = level

    formatter = logging.Formatter(
        fmt="%(asctime)s|%(levelname)s|%(name)s|%(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    rich = RichHandler()
    log.addHandler(rich)

    return log
