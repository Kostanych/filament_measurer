import time

import streamlit as st
import numpy as np
from streamlit_webrtc import webrtc_streamer, VideoHTMLAttributes

from video_processor import webcam_callback


def image_input():
    st.session_state.title_frame = np.full((480, 640, 3), 255, dtype=np.uint8)
    return st.session_state.title_frame


def webcam_input(app_state):
    # Инициализация состояния приложения для отслеживания FPS
    if 'prev_time' not in st.session_state:
        st.session_state['prev_time'] = time.time()
        st.session_state['frame_count'] = 0

    webrtc_streamer(
        key="webcam_input",
        video_frame_callback=lambda frame: webcam_callback(frame, app_state),
        rtc_configuration={  # Add this line
            "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}],
            # "iceTransportPolicy": "relay",
        },
        # video_html_attrs=VideoHTM
        #         #     # style={"width": "100%"},
        #         #     muted=True),LAttributes(
        #     autoPlay=True,
        #     # controls=True,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )






