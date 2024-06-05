import stat

from files import *
from gui_logic import *
from image_processor import *
from plot import *
from utils import *
from video_processor import play_or_continue_video
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))


def update_status(new_message):
    st.session_state["status_message"] = new_message
    status_bar.text(new_message)


# logging_level = logging.INFO
logging_level = logging.DEBUG

logger = get_logger("STREAMLIT GUI", level=logging_level)
st.set_page_config(layout="wide")

# Session variables
init_variables()
# app_state = AppState()

# Streamlit elements
status_bar = st.empty()
status_bar.text(st.session_state["status_message"])
st.title("Filament Measurer")

# Sidebar
st.sidebar.header("Control Panel")


# Put reference changing into the Fragment. Fragment updates independently
@st.experimental_fragment
def set_or_change_reference():
    st.session_state["reference"] = st.number_input(
        "Reference width (mm):",
        value=float(1.75),
        # on_change=change_calibration_multiplier
    )
    # Change the reference


with st.sidebar:
    set_or_change_reference()

change_reference = st.sidebar.button(
    "Change reference standard",
    key="change_reference",
    on_click=change_calibration_multiplier,
    # args=(app_state,)
)

input_source = st.sidebar.radio(
    "Input Source",
    options=["File", "USB Device"],
    # on_change=change_video_source
)
st.session_state["source"] = input_source

video_file = st.sidebar.file_uploader(
    "Select a video file",
    type=["mp4", "avi", "mov"],
    # on_change=on_video_file_change,
    # key="video_file"
)
change_video_source(video_file)

# Show PLAY/STOP button
play_button = st.sidebar.button(
    "Play",
    key="play_button",
    on_click=set_play_flag,
    # args=(app_state,)
)

stop_button = st.sidebar.button(
    "Stop",
    key="stop_button",
    on_click=stop,
    # args=(app_state,)
)

mask_radio = st.sidebar.radio(
    "Mask/Image",
    ["Image", "Mask"],
    key="Mask_or_image",
    on_change=mask_switcher,
)

# Image display area
col1, col2, col3 = st.columns([0.3, 0.2, 0.2])
with col1:
    st.header("Video")
    # Switcher videofile\webcamera
    if input_source == "File":
        # print(st.session_state.title_frame)
        st.session_state.vid_area = st.image(st.session_state.title_frame)
    elif input_source == "USB Device":
        # This part is for the streamlit_webrtc library
        # st.session_state.vid_area = webcam_input(app_state)
        # placeholder
        st.session_state.vid_area = st.image(st.session_state.title_frame)

with col2:
    st.header("Results")
    st.session_state.width_pxl_area = st.markdown(
        f'<span style="font-size: 20px;">Width, pixels: N/A</span>',
        unsafe_allow_html=True,
    )
    st.session_state.width_mm_area = st.markdown(
        f'<span style="font-size: 20px;">Width, mm:     N/A</span>',
        unsafe_allow_html=True,
    )
with col3:
    st.header("Mean rolling")
    st.session_state.rolling_1s_markdown = st.markdown(
        f'<span style="font-size: 20px;">1 second:   0</span>', unsafe_allow_html=True
    )
    st.session_state.rolling_10s_markdown = st.markdown(
        f'<span style="font-size: 20px;">10 seconds: 0</span>', unsafe_allow_html=True
    )

# Plot display area
col11, col12 = st.columns([0.8, 0.2])
with col11:
    if not st.session_state["width_list"]:
        st.session_state.plot_area = st.empty()
    else:
        update_rolling_plot(st.session_state.plot_area)
with col12:
    st.header("Difference")
    st.markdown(
        f'<span style="font-size: 20px;">Reference:  {st.session_state.reference}</span>',
        unsafe_allow_html=True,
    )
    st.session_state.difference_markdown = st.markdown(
        f'<span style="font-size: 20px;">Difference(1s mean):{round(st.session_state.reference - st.session_state.rolling_1s, 5)}</span>',
        unsafe_allow_html=True,
    )

# Update title frame first time
if st.session_state["title_frame_is_blank"]:
    # If this is not a video file:
    logger.info(f"Update title frame first time {video_file}")
    if video_file:
        logger.info(video_file)

        # Update title frame by first frame of the video
        cap = cv2.VideoCapture(st.session_state["video_path"])
        _, frame = cap.read()
        update_title_frame(frame)
    st.session_state["title_frame_is_blank"] = False
if st.session_state["play"]:
    play_or_continue_video()
