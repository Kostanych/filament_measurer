"""Utilities file"""
import logging
import sys

import pandas as pd
import streamlit as st


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
    if len(data) < fps * seconds:
        n = len(data)
    else:
        n = int(fps * seconds)
    # print(f"NNNNNNNN     {n}")
    # calculate moving average
    return pd.Series(data).rolling(window=n).mean().iloc[n - 1:].values[-1]


def check_variables():
    logger = get_logger('VARIABLES CHECKER', level=logging.DEBUG)
    logger.info("session_state variables check")
    if "width_list" not in st.session_state:
        st.session_state["width_list"] = []
    if "source" not in st.session_state:
        st.session_state["source"] = "File"
    if "cap" not in st.session_state:
        st.session_state["cap"] = None
    if "show_mask" not in st.session_state:
        st.session_state["show_mask"] = False
    if "show_every_n_frame" not in st.session_state:
        st.session_state["show_every_n_frame"] = 1
    if "df_points" not in st.session_state:
        st.session_state["df_points"] = pd.DataFrame()
    if "width_pxl" not in st.session_state:
        st.session_state["width_pxl"] = 1
    if "reference" not in st.session_state:
        st.session_state["reference"] = 1.75
    if "width_multiplier" not in st.session_state:
        st.session_state["width_multiplier"] = 1
    if "rolling_1s" not in st.session_state:
        st.session_state["rolling_1s"] = 0
    if "rolling_10s" not in st.session_state:
        st.session_state["rolling_10s"] = 0
    if "mean_1" not in st.session_state:
        st.session_state["mean_1"] = []
    if "mean_2" not in st.session_state:
        st.session_state["mean_2"] = []
    if "fps" not in st.session_state:
        st.session_state["fps"] = 24

    if "width_pxl" not in st.session_state:
        st.session_state["width_pxl"] = 1
    if st.session_state["width_pxl"] == 0:
        st.session_state["width_pxl"] = 1
