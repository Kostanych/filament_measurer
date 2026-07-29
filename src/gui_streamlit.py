"""Main UI script for Filament Measurer application"""

import logging

import streamlit as st

from config import config
from gui_logic import change_video_source, read_first_frame, set_play_flag, stop
from image_processor import (
    change_calibration_multiplier,
    mask_switcher,
    to_display,
    update_title_frame,
)
from plot import update_rolling_plot
from utils import AppState, big_text, get_logger
from video_processor import VideoProcessor

logging_level = logging.DEBUG

logger = get_logger("STREAMLIT GUI", level=logging_level)
st.set_page_config(layout="wide")

# Session variables
app_state = AppState()

# Streamlit elements
status_bar = st.empty()
status_bar.text(st.session_state["status_message"])
st.title("Filament Measurer")

# Sidebar
st.sidebar.header("Control Panel")


# Put reference changing into the Fragment. Fragment updates independently
@st.fragment
def set_or_change_reference():
    st.session_state["reference"] = st.number_input(
        "Reference width (mm):",
        value=config.DEFAULT_REFERENCE_WIDTH_MM,
    )


with st.sidebar:
    set_or_change_reference()

change_reference = st.sidebar.button(
    "Change reference standard",
    key="change_reference",
    on_click=change_calibration_multiplier,
)

input_source = st.sidebar.radio(
    "Input Source",
    options=["File", "USB Device"],
)
st.session_state["source"] = input_source

video_file = st.sidebar.file_uploader(
    "Select a video file",
    type=["mp4", "avi", "mov"],
)
change_video_source(video_file)

# Show PLAY/STOP button
play_button = st.sidebar.button(
    "Play",
    key="play_button",
    on_click=set_play_flag,
)

stop_button = st.sidebar.button(
    "Stop",
    key="stop_button",
    on_click=stop,
)

update_interval = st.sidebar.selectbox(
    "Update Interval",
    options=list(config.UPDATE_INTERVALS),
    index=0,
)
st.session_state["update_interval"] = update_interval

mask_radio = st.sidebar.radio(
    "Mask/Image",
    ["Image", "Mask"],
    key="Mask_or_image",
    on_change=mask_switcher,
)

# Image display area
col1, col2, col3 = st.columns(config.COL_WIDTHS_MAIN)
with col1:
    st.header("Video")
    st.session_state.vid_area = st.image(to_display(st.session_state.title_frame))

with col2:
    st.header("Results")
    st.session_state.width_pxl_area = st.markdown(
        big_text("Width, pixels: N/A"),
        unsafe_allow_html=True,
    )
    st.session_state.width_mm_area = st.markdown(
        big_text("Width, mm:     N/A"),
        unsafe_allow_html=True,
    )
with col3:
    st.header("Mean rolling")
    st.session_state.rolling_1s_markdown = st.markdown(
        big_text("1 second:   0"),
        unsafe_allow_html=True,
    )
    st.session_state.rolling_10s_markdown = st.markdown(
        big_text("10 seconds: 0"),
        unsafe_allow_html=True,
    )

# Plot display area
col11, col12 = st.columns(config.COL_WIDTHS_PLOT)
with col11:
    if not st.session_state["width_list"]:
        st.session_state.plot_area = st.empty()
    else:
        update_rolling_plot(st.session_state.plot_area)
with col12:
    st.header("Difference")
    st.markdown(
        big_text(f"Reference:  {st.session_state.reference}"),
        unsafe_allow_html=True,
    )
    difference = round(st.session_state.reference - st.session_state.rolling_1s, 5)
    st.session_state.difference_markdown = st.markdown(
        big_text(f"Difference(1s mean):{difference}"),
        unsafe_allow_html=True,
    )

# Update title frame first time
if st.session_state["title_frame_is_blank"]:
    logger.info(f"Update title frame first time {video_file}")
    if video_file:
        frame = read_first_frame(st.session_state["video_path"])
        if frame is not None:
            update_title_frame(frame)
    st.session_state["title_frame_is_blank"] = False

# Initialize VideoProcessor
video_processor = VideoProcessor()

if st.session_state["play"]:
    video_processor.play_or_continue_video()
