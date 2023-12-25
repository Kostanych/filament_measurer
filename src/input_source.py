
import streamlit as st
import numpy as np
from streamlit_webrtc import webrtc_streamer, VideoHTMLAttributes

from image_processor import add_info_on_the_frame


def image_input():
    st.session_state.title_frame = np.full((480, 640, 3), 255, dtype=np.uint8)
    return st.session_state.title_frame


def webcam_input():
    return webrtc_streamer(
        key="webcam_input",
        video_frame_callback=add_info_on_the_frame,
        rtc_configuration={  # Add this line
            "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
        },
        video_html_attrs=VideoHTMLAttributes(
            autoPlay=True,
            # controls=True,
            # style={"width": "100%"},
            muted=True)
    )

