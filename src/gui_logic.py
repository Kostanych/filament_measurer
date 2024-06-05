import logging

import cv2
import streamlit as st

from plot import update_rolling_plot
from image_processor import update_title_frame
from files import get_video_filename
from utils import get_logger

logging_level = logging.DEBUG


def stop():
    """Stop the cap"""
    st.session_state.play = False
    logger = get_logger("STOP VIDEO", level=logging_level)
    logger.info(f"BUTTON Stop")

    # Update title frame by the last processed frame
    update_title_frame(st.session_state["last_frame"])
    logger.info(f"st.session_state.play:   {st.session_state.play}")

    if st.session_state.cap:
        st.session_state.cap.release()
        st.session_state.cap = None
        logger.debug("Cap released")

    # st.session_state.width_list = []

    if "width_list" in st.session_state:
        update_rolling_plot(st.session_state["plot_area"])


def set_play_flag():
    st.session_state["play"] = True


def change_video_source(video_file):
    # Close the cap if we change video source
    # print(st.session_state.source)
    # if ('cap' in st.session_state) & (st.session_state.source == 'File'):
    #
    #     st.session_state.cap = None
    if video_file:
        st.session_state["filename"] = video_file.name
        st.session_state["video_path"] = get_video_filename()
        cap = cv2.VideoCapture(st.session_state["video_path"])
        _, frame = cap.read()
        update_title_frame(frame)
        del _, frame
