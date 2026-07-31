"""Filesystem helpers for the video input"""

import logging

import streamlit as st

from config import config
from utils import get_logger

logging_level = logging.DEBUG


def get_video_filename() -> str:
    """Build a full path to the uploaded video inside the input folder"""
    logger = get_logger("GET VIDEO FILENAME", level=logging_level)
    filename = config.INPUT_PATH / st.session_state["filename"]
    logger.info(f"st.session_state['filename']  {st.session_state['filename']}")
    # cv2.VideoCapture accepts strings only, not Path objects.
    return str(filename)
