import logging

import cv2
import streamlit as st

from plot import update_rolling_plot
from utils import check_variables, get_logger

logging_level = logging.DEBUG


def open_video_source():
    """Open a video, return cap into session state"""

    st.session_state["play"] = True

    logger = get_logger("OPEN VIDEO", level=logging_level)

    if st.session_state.cap:
        st.session_state.cap.release()
    if ("video_path" in st.session_state) and (st.session_state["source"] == "File"):
        logger.info("Video from file")
        video_path = st.session_state["video_path"]
        st.session_state.cap = cv2.VideoCapture(video_path)
        launch_video_processing()
    elif st.session_state["source"] == "USB Device":
        logger.info("Video from USB")
        # st.session_state.cap = cv2.VideoCapture(0)
        # connect_camera()
        logger.info("Press 'START' button in the main window")
    else:
        logger.info("Select the video first!")


def launch_video_processing():
    logger = get_logger('File processor')
    if st.session_state.filename:
        # check_variables()
        logger.debug('Got the Video file')
        # Get filename, set title frame
        logger.debug('Start to load the video')
        # load_video(video_file)
        # else:
        #     logger.debug("filename IS in session state")
        st.session_state.vid_area.image(st.session_state.title_frame)


def stop(app_state):
    """Stop the cap"""
    logger = get_logger("STOP VIDEO", level=logging_level)

    check_variables()
    logger.info(f"BUTTON Stop")
    st.session_state.play = False
    logger.info(f"st.session_state.play:   {st.session_state.play}")
    if st.session_state.cap:
        st.session_state.cap.release()
        logger.debug('Cap released')

    # st.session_state.width_list = []

    if app_state.state['width_list']:
        update_rolling_plot(st.session_state["plot_area"])




