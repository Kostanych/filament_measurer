
import av
import streamlit as st
import numpy as np
from streamlit_webrtc import webrtc_streamer, VideoHTMLAttributes

from image_processor import add_info_on_the_frame


def image_input():
    st.session_state.title_frame = np.full((480, 640, 3), 255, dtype=np.uint8)
    return st.session_state.title_frame


def webcam_input(app_state):
    print("WEEEEEEEEEEEEEEEEEEEEEEEEEBCAM")
    webrtc_streamer(
        key="webcam_input",
        video_frame_callback=lambda frame: webcam_callback(frame, app_state),
        rtc_configuration={  # Add this line
            "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}],
            # "iceTransportPolicy": "relay",
        },
        # video_html_attrs=VideoHTMLAttributes(
        #     autoPlay=True,
        #     # controls=True,
        #     # style={"width": "100%"},
        #     muted=True),
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )


def webcam_callback(frame: av.VideoFrame, app_state) -> av.VideoFrame:
    image = frame.to_ndarray(format="bgr24")
    image, width_pxl, width_mm = add_info_on_the_frame(
        image,
        app_state.show_mask,
        app_state.width_multiplier
    )
    app_state.add_width(width_mm)
    return av.VideoFrame.from_ndarray(image, format="bgr24")


