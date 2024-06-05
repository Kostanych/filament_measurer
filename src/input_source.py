import time

import streamlit as st
import numpy as np

from video_processor import webcam_callback


def image_input():
    st.session_state.title_frame = np.full((480, 640, 3), 255, dtype=np.uint8)
    return st.session_state.title_frame
