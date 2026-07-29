"""Utilities file"""

import itertools
import logging
import sys
import time
from collections import deque

import numpy as np
import pandas as pd
import streamlit as st

from config import config

logging_level = logging.DEBUG


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


def blank_frame() -> np.ndarray:
    """Return a white placeholder frame"""
    return np.full(config.IMAGE_SHAPE, 255, dtype=np.uint8)


def big_text(text: str) -> str:
    """Wrap the text into a html span with the enlarged font"""
    return f'<span style="font-size: {config.FONT_SIZE_LARGE}px;">{text}</span>'


def measurement_buffer() -> deque:
    """
    Return an empty buffer for the measured values.

    The buffer is bounded: a long run drops the oldest values instead of
    eating the memory frame by frame.
    """
    return deque(maxlen=config.MAX_MEASUREMENT_HISTORY)


def last_n(data, n: int) -> list:
    """Return up to n last elements of a sequence, deque included"""
    if n <= 0 or not len(data):
        return []
    start = max(len(data) - n, 0)
    if isinstance(data, deque):
        return list(itertools.islice(data, start, len(data)))
    return list(data[start:])


class AppState:
    """Holds default values of the streamlit session state"""

    def __init__(self):
        self.init_variables()

    def init_variables(self):
        """Fill session_state with default values for missing keys"""
        default_values = {
            "play": False,
            "status_message": "Ready to work!",
            "title_frame": blank_frame(),
            "title_frame_is_blank": True,
            "last_frame": blank_frame(),
            "width_list": measurement_buffer(),
            "source": "File",
            "cap": None,
            "show_mask": False,
            "show_every_n_frame": 1,
            "df_points": pd.DataFrame(),
            "width_pxl": 1,
            "width_mm": 1,
            "reference": config.DEFAULT_REFERENCE_WIDTH_MM,
            "width_multiplier": config.DEFAULT_WIDTH_MULTIPLIER,
            "rolling_1s": 0,
            "rolling_10s": 0,
            "mean_1": measurement_buffer(),
            "mean_2": measurement_buffer(),
            "measurements_total": 0,
            "difference": 0,
            "prev_time": 0,
            "fps": config.DEFAULT_FPS,
            "update_interval": "Every Frame",
        }
        for key, value in default_values.items():
            if key not in st.session_state:
                st.session_state[key] = value

    def reset_measurements(self):
        """
        Drop the measurements of the previous run.

        Playback always restarts the video source from the beginning, so old
        values would otherwise be mixed into the new run on the same plot.
        """
        st.session_state.width_list = measurement_buffer()
        st.session_state.mean_1 = measurement_buffer()
        st.session_state.mean_2 = measurement_buffer()
        st.session_state.measurements_total = 0
        st.session_state.df_points = pd.DataFrame()

    def get_logger(self, name: str = None, level=logging.INFO):
        """Return a configured logger. Kept for backward compatibility"""
        return get_logger(name, level)


def mean_rolling(data, fps, seconds=1):
    """
    Calculate mean value over the last N seconds of measurements.

    Args:
        data: sequence of measured values.
        fps: current frames per second, defines the window size.
        seconds: length of the averaging window in seconds.

    Returns:
        Mean of the last `fps * seconds` values, or 0.0 for empty data.
    """
    if not len(data):
        return 0.0
    # N for the rolling mean is len of an array, or frames of one second.
    window = min(len(data), max(int(fps * seconds), 1))
    return float(np.mean(last_n(data, window)))


def make_result_df(num_seconds=config.PLOT_HISTORY_SECONDS) -> pd.DataFrame:
    """
    Build the dataframe of the last seconds of measurements for the plot.

    Returns:
        melted dataframe.
    """
    points = max(int(st.session_state.fps * num_seconds), 1)
    mean_1 = last_n(st.session_state.mean_1, points)
    mean_2 = last_n(st.session_state.mean_2, points)
    size = min(len(mean_1), len(mean_2))
    if not size:
        return pd.DataFrame(columns=["frame", "seconds_count", "values"])

    # Frames are numbered globally, so the plot keeps moving forward even
    # after the oldest measurements are dropped from the buffer.
    first_frame = st.session_state.measurements_total - size
    df = pd.DataFrame(
        {
            "Mean 1s": mean_1[-size:],
            "Mean 10s": mean_2[-size:],
            "frame": range(first_frame, first_frame + size),
        }
    )
    return df.melt("frame", var_name="seconds_count", value_name="values")


class FpsCalculator:
    """Calculates FPS as a number of frames processed during the last interval"""

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
            return config.DEFAULT_FPS
        time_passed = self.frame_timestamps[-1] - self.frame_timestamps[0]
        return (len(self.frame_timestamps) - 1) / time_passed
