import os
import logging
from pathlib import Path

import cv2
import streamlit as st

from image_processor import update_title_frame
from utils import get_logger

logging_level = logging.INFO
logging_level = logging.DEBUG


def get_video_filename():
    """Get file name"""
    logger = get_logger("GET VIDEO FILENAME", level=logging_level)
    logger.info("GET FILENAME...")
    ROOT_DIR = Path(__file__).resolve().parent.parent
    filename = Path(ROOT_DIR, "data", "input", st.session_state["filename"])
    logger.info(f"st.session_state['filename']  {st.session_state['filename']}")
    # Return a normalized path.
    return filename


def load_video(video_file):
    """
    Load video from the streamlit object.
    Get the filename and path, fill session_state variables.
    Args:
        video_file: file object from Streamlit.
    """
    logger = get_logger("LOAD VIDEO")
    st.session_state["filename"] = video_file.name
    st.session_state["video_path"] = get_video_filename()
    logger.info(f"video_path:    {st.session_state['video_path']}")

    # Update title frame by first frame of the video
    cap = cv2.VideoCapture(st.session_state["video_path"])
    _, frame = cap.read()
    update_title_frame(frame)
    del cap
