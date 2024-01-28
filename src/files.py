import os
import logging
import streamlit as st

from utils import get_logger

logging_level = logging.INFO
logging_level = logging.DEBUG


def get_video_filename():
    """Get file name"""
    logger = get_logger("GET VIDEO FILENAME", level=logging_level)
    logger.info("GET FILENAME...")
    ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    filename = ROOT_DIR + "/data/input/" + st.session_state["filename"]
    logger.info(f"st.session_state['filename']  {st.session_state['filename']}")
    # Return a normalized path.
    return os.path.normpath(filename)


def load_video(video_file):
    """
    Load video
    Args:
        video_file: file object from Streamlit.
    """
    logger = get_logger("LOAD VIDEO")
    st.session_state["filename"] = video_file.name
    st.session_state["video_path"] = get_video_filename()
    logger.info(f"video_path:    {st.session_state.video_path}")
    # open_video_source()
    # update_title_frame(st.session_state.video_path)

