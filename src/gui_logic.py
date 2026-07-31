"""GUI logic and callbacks for Filament Measurer"""

import logging

import cv2
import streamlit as st

from files import get_video_filename
from image_processor import update_title_frame
from plot import update_rolling_plot
from utils import get_logger

logging_level = logging.DEBUG


def stop():
    """Stop video playback and release resources"""
    st.session_state.play = False
    logger = get_logger("STOP VIDEO", level=logging_level)
    logger.info("BUTTON Stop")

    update_title_frame(st.session_state["last_frame"])

    if st.session_state.cap:
        st.session_state.cap.release()
        st.session_state.cap = None
        logger.debug("Cap released")

    if "width_list" in st.session_state:
        update_rolling_plot(st.session_state["plot_area"])


def set_play_flag():
    """Set play flag to True to start video processing"""
    st.session_state["play"] = True


def read_first_frame(video_path):
    """Read the very first frame of the video and release the capture"""
    cap = cv2.VideoCapture(video_path)
    try:
        ret, frame = cap.read()
    finally:
        cap.release()
    return frame if ret else None


def change_video_source(video_file):
    """Change video source and update title frame"""
    if not video_file:
        return

    st.session_state["filename"] = video_file.name
    st.session_state["video_path"] = get_video_filename()
    frame = read_first_frame(st.session_state["video_path"])
    if frame is not None:
        update_title_frame(frame)
