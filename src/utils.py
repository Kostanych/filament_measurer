"""Utilities file"""
import logging
import sys

import pandas as pd


def get_logger(name: str = None, level=logging.INFO):
    """
    Sets up the logger handlers for jupyter notebook, ipython or python.

    Separate initialization each time is needed to ensure that logger is set
    when calling from subprocess
    (e.g. joblib.Parallel)

    :param name: name of the logger. If None, will return root logger.
    :param level: Log level (default - INFO)
    :return: logger with correct handlers
    """
    logger = logging.getLogger(name)
    logger.handlers = []
    stdout = logging.StreamHandler(sys.stdout)
    fmt = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%m/%d/%Y %I:%M:%S %p",
    )
    stdout.setFormatter(fmt)
    stdout.setLevel(level)
    logger.addHandler(stdout)
    logger.setLevel(level)
    logger.propagate = False
    return logger


def mean_rolling(data, fps, seconds=1):
    """ Calculate mean rolling value for N seconds."""
    # N for the rolling mean is len of an array, or frames of one second.
    if len(data) < fps*seconds:
        n = len(data)
    else:
        n = fps*seconds

    # calculate moving average
    return pd.Series(data).rolling(window=n).mean().iloc[n - 1:].values[-1]

