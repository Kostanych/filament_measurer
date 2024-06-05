"""Utilities file"""
import sys
import time

import pandas as pd
import numpy as np
import streamlit as st
import logging

logging_level = logging.INFO
logging_level = logging.DEBUG


class AppState:
    def __init__(self):
        # TODO: must be lite this dict
        self.state = {"show_mask": False, "width_multiplier": 1, "width_list": []}
        self.show_mask = False
        self.width_multiplier = 1
        self.width_list = []

    def update(self, show_mask, width_multiplier):
        self.show_mask = show_mask
        self.width_multiplier = width_multiplier

    def add_width(self, width_mm):
        self.width_list.append(width_mm)


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
    """Calculate mean rolling value for N seconds."""
    # N for the rolling mean is len of an array, or frames of one second.
    if len(data) < fps * seconds:
        n = len(data)
    else:
        n = int(fps * seconds)
    # print(f"NNNNNNNN     {n}")
    # calculate moving average
    return pd.Series(data).rolling(window=n).mean().iloc[n - 1 :].values[-1]


def init_variables():
    logger = get_logger("VARIABLES CHECKER", level=logging.DEBUG)
    logger.info("session_state variables check")
    if "play" not in st.session_state:
        st.session_state["play"] = False
    if "status_message" not in st.session_state:
        st.session_state["status_message"] = "Ready to work!"
    if "title_frame" not in st.session_state:
        st.session_state["title_frame"] = np.full((480, 640, 3), 255, dtype=np.uint8)
    if "title_frame_is_blank" not in st.session_state:
        st.session_state["title_frame_is_blank"] = True
    if "last_frame" not in st.session_state:
        st.session_state["last_frame"] = np.full((480, 640, 3), 255, dtype=np.uint8)
    # if "filename" not in st.session_state:
    #     st.session_state["filename"] = ''
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
    if "width_mm" not in st.session_state:
        st.session_state["width_mm"] = 1
    if "reference" not in st.session_state:
        st.session_state["reference"] = 1.75
    if "width_multiplier" not in st.session_state:
        st.session_state["width_multiplier"] = 0.005
    if "rolling_1s" not in st.session_state:
        st.session_state["rolling_1s"] = 0
    if "rolling_10s" not in st.session_state:
        st.session_state["rolling_10s"] = 0
    if "mean_1" not in st.session_state:
        st.session_state["mean_1"] = []
    if "mean_2" not in st.session_state:
        st.session_state["mean_2"] = []
    if "difference" not in st.session_state:
        st.session_state["difference"] = 0
    if "prev_time" not in st.session_state:
        st.session_state["prev_time"] = 0
    if "fps" not in st.session_state:
        st.session_state["fps"] = 24

    # if st.session_state["width_pxl"] == 0:
    #     st.session_state["width_pxl"] = 1


def make_result_df(num_seconds=2) -> pd.DataFrame():
    """
    Consumes dataframe and melt it to display on the Altair plot
    Returns:
        melted dataframe.
    """
    # logger.info(f"MEAN 1: {st.session_state.mean_1}")
    # logger.info(f"MEAN 2: {st.session_state.mean_2}")
    df = pd.DataFrame(
        {
            "Mean 1s": st.session_state.mean_1,
            "Mean 10s": st.session_state.mean_2,
        }
    )
    # logger.info(f"FIRST DF:\n {df}")
    df["frame"] = df.index
    # Cut dataframe to represent X seconds of work.
    max_frame = df.frame.max()
    df = df[df.frame > (max_frame - st.session_state.fps * num_seconds)]
    df = df.melt("frame", var_name="seconds_count", value_name="values")
    # logger.info(f"MELTED DF:\n {df}")
    return df


class FpsCalculator:
    def __init__(self):
        self.frame_timestamps = []
        self.interval = 1

    def tick(self):
        """Update every frame"""
        self.frame_timestamps.append(time.time())
        self._clean_old_timestamps()

    def _clean_old_timestamps(self):
        """Delete timestamps older than 'interval' seconds"""
        current_time = time.time()
        self.frame_timestamps = [
            ts for ts in self.frame_timestamps if current_time - ts <= self.interval
        ]

    def get_fps(self):
        """Return mean FPS for 'interval' seconds"""
        if len(self.frame_timestamps) < 2:
            return 24
        time_passed = self.frame_timestamps[-1] - self.frame_timestamps[0]
        return (len(self.frame_timestamps) - 1) / time_passed
