import time
import os.path

import altair as alt

import pandas as pd
import streamlit as st

from image_processor import *
from utils import mean_rolling, get_logger
import logging

# logging_level = logging.INFO
logging_level = logging.DEBUG

logger = get_logger("STREAMLIT GUI", level=logging_level)
st.set_page_config(layout="wide")

st.title("Filament Measurer")
st.sidebar.header("Control Panel")

# Session variables
st.session_state.play = False
st.session_state.pixel_width_multiplier = 1
if "title_frame" not in st.session_state:
    st.session_state.title_frame = np.full((480, 640, 3), 255, dtype=np.uint8)


# st.session_state.play_button_disabled = False
# st.session_state.stop_button_disabled = True
# st.session_state.disabled = False

check_variables()


# Functions
# @st.cache_resource
def load_video(video_file):
    """
    Load video
    Args:
        video_file: file object from Streamlit.
    """
    st.session_state["filename"] = video_file.name
    st.session_state["video_path"] = get_video_filename()
    logger.info(f"video_path:    {st.session_state.video_path}")
    # open_video_source()
    update_title_frame(st.session_state.video_path)



# Function to open video using OpenCV


# Elements
reference = st.sidebar.number_input(
    "Reference width (mm):",
    value=float(1.75),
    # on_change=change_calibration_multiplier
)
input_source = st.sidebar.selectbox(
    "Input Source", key="input_source", options=["File", "USB Device"], index=0
)
video_file = st.sidebar.file_uploader(
    "Select a video file",
    type=["mp4", "avi", "mov"],
)
play_button = st.sidebar.button("Play", key="play_button", on_click=open_video_source)
stop_button = st.sidebar.button("Stop", key="stop_button", on_click=stop)
# frames_radio = st.sidebar.radio("Show N% of frames", ["100%", "10%"])
# show_10_button = st.sidebar.button('Show 10% of frames', key='show_10_button',
#                                    disabled=st.session_state.disabled)
# show_100_button = st.sidebar.button('Show 100% of frames', key='show_100_button',
#                                     disabled=st.session_state.disabled)
# mask_radio = st.sidebar.radio("Mask/Image", [":rainbow[Image]", "***Mask***"])
mask_radio = st.sidebar.radio(
    "Mask/Image",
    ["Image", "Mask"],
    # on_change=mask_switcher
)

# Image display area
col1, col2, col3 = st.columns([0.3, 0.2, 0.2])
with col1:
    st.header("Video")
    vid_area = st.image(st.session_state.title_frame)
    st.session_state.vid_area = vid_area
with col2:
    st.header("Results")
    width_pxl_area = st.markdown(
        f'<span style="font-size: 20px;">Width, pixels: N/A</span>',
        unsafe_allow_html=True,
    )
    width_mm_area = st.markdown(
        f'<span style="font-size: 20px;">Width, mm:     N/A</span>',
        unsafe_allow_html=True,
    )
with col3:
    st.header("Mean rolling")
    rolling_1s = st.markdown(
        f'<span style="font-size: 20px;">1 second:   0</span>', unsafe_allow_html=True
    )
    rolling_10s = st.markdown(
        f'<span style="font-size: 20px;">10 seconds: 0</span>', unsafe_allow_html=True
    )

# Plot display area
col11, col12 = st.columns([0.8, 0.2])
with col11:
    if not st.session_state["width_list"]:
        st.session_state["plot_area"] = st.empty()
    else:
        update_rolling_plot(st.session_state["plot_area"])
with col12:
    st.header("Difference")
    st.markdown(
        f'<span style="font-size: 20px;">Reference:  {st.session_state.reference}</span>',
        unsafe_allow_html=True,
    )
    difference = st.markdown(
        f'<span style="font-size: 20px;">Difference(1s mean):{round(st.session_state.reference - st.session_state.rolling_1s, 5)}</span>',
        unsafe_allow_html=True,
    )

# Logic
# Change reference multiplier
if reference:
    st.session_state["reference"] = reference
    change_calibration_multiplier()

# """ Switcher mask/image """
if mask_radio == "Image":
    st.session_state.show_mask = False
else:
    st.session_state.show_mask = True

# Input switcher
st.session_state.source = input_source

# if frames_radio == '100%':
#     st.session_state['show_every_n_frame'] = 1
# else:
#     st.session_state['show_every_n_frame'] = 10

# if stop_button:
#     """Stop the cap"""
#     logger = get_logger("STOP VIDEO", level=logging_level)
#
#     check_variables()
#     logger.info(f"BUTTON Stop")
#     st.session_state.play = False
#     logger.info(f"st.session_state.play:   {st.session_state.play}")
#     if st.session_state.cap:
#         st.session_state.cap.release()
#         logger.debug('Cap released')
#
#     # st.session_state.width_list = []
#
#     if st.session_state["width_list"]:
#         update_rolling_plot(st.session_state["plot_area"])

try:
    logger.debug(f"video_file name:        {video_file.name}")
except:
    logger.debug(f"video_file name:        {video_file}")
logger.debug(f"input_source:           {input_source}")


if play_button:
    st.session_state["play"] = True


# if play_button and (input_source == 'USB Device'):
#     check_variables()
#     logger.debug('Got the USB Device')
#     open_video_source()
#     st.session_state["play"] = True

# If we selected any video file
if (video_file != None) & (input_source == 'File'):
    check_variables()
    logger.debug('Got the Video file')
    # Get filename, set title frame
    if ("filename" not in st.session_state) or (
            st.session_state.filename != video_file.name
    ):
        logger.debug('Start to load the video')
        load_video(video_file)
    else:
        logger.debug("filename IS in session state")
    vid_area.image(st.session_state.title_frame)

# sns.set(rc={'figure.figsize': (10, 2)})
# def meanplot():
#     # depricated
#     # Clear the existing legend
#     fig, ax = plt.subplots()
#     chart_data = pd.DataFrame({
#         'Mean 1s': st.session_state.mean_1,
#         'Mean 10s': st.session_state.mean_2,
#     })
#
#     # pl = sns.lineplot(data=chart_data, palette="tab10", linewidth=1.5)
#     ax.plot(chart_data['Mean 1s'])
#     ax.plot(chart_data['Mean 10s'],
#              color='red',
#              linewidth=1.0,
#              linestyle='--'
#              )
#     return fig


logger.debug(f"play_button:               {play_button}")
try:
    logger.debug(f"video_file:                {video_file.name}")
except:
    logger.debug(f"video_file:                {video_file}")
logger.debug(f"st.session_state.play:     {st.session_state.play}")




# if play_button and video_file:
#     check_variables()
#     logger.info(f"play_button and video_file is TRUE")
#     st.session_state["play"] = True
#     # Process first frame
#     ret, frame = st.session_state.cap.read()
#     if ret:
#         _, width = process_image(frame=frame, verbose=0)
#         st.session_state.width_pxl = width
#     change_calibration_multiplier()


logger.debug(f"st.session_state.cap:               {st.session_state.cap}")
logger.debug(f"st.session_state.play:              {st.session_state.play}")

# Play video
if st.session_state.cap and st.session_state.play:
    check_variables()
    print(f"PLAY   {st.session_state['play']}")
    cap = st.session_state.cap
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            # When the video starts
            mask, width_pxl = process_image(frame=frame, verbose=0)
            st.session_state.width_pxl = width_pxl
            # show_mask is missed sometimes. need to fix it
            try:
                if st.session_state.show_mask:
                    source = mask
                else:
                    source = frame
                logger.debug(f"SOURCE IS MASK:   {st.session_state.show_mask}")
            except Exception as e:
                print(repr(e))
                source = frame

            # Process frame
            check_variables()
            source, angle = draw_angle_line(source.copy(), mask)
            angle_multiplier = calculate_pixel_multiplier(angle)
            width_pxl = width_pxl * angle_multiplier
            width_mm = width_pxl * angle_multiplier * st.session_state.width_multiplier
            st.session_state.width_list.append(width_mm)
            # width_multiplier_calibrated = change_calibration_multiplier()

            # Process variables
            fps = cap.get(cv2.CAP_PROP_FPS)
            st.session_state.fps = fps
            st.session_state.rolling_1s = round(
                mean_rolling(st.session_state.width_list, fps), 4
            )
            st.session_state.rolling_10s = round(
                mean_rolling(st.session_state.width_list, fps, 10), 4
            )
            st.session_state.mean_1.append(st.session_state.rolling_1s)
            st.session_state.mean_2.append(st.session_state.rolling_10s)

            width_pxl_area.markdown(
                f'<span style="font-size: 20px;">Width, pixels: {round(width_pxl, 0)}</span>',
                unsafe_allow_html=True,
            )
            width_mm_area.markdown(
                f'<span style="font-size: 20px;">Width, mm:     {round(width_mm, 3)}</span>',
                unsafe_allow_html=True,
            )

            rolling_1s.markdown(
                f'<span style="font-size: 20px;">1 second:   {st.session_state.rolling_1s}</span>',
                unsafe_allow_html=True,
            )
            rolling_10s.markdown(
                f'<span style="font-size: 20px;">10 seconds: {st.session_state.rolling_10s}</span>',
                unsafe_allow_html=True,
            )

            source = draw_fps(source, cap)
            # frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            vid_area.image(source)
            time.sleep(1 / fps)  # keep the fps the same as the original fps

            try:
                chart_data = make_result_df()
            except Exception as e:
                print(repr(e))
                chart_data = pd.DataFrame()


            # Plot results
            try:
                st.session_state.df_points = chart_data
                # Update variables
                update_rolling_plot(st.session_state["plot_area"])
                difference.markdown(
                    f'<span style="font-size: 20px;">Difference(1s mean):{round(st.session_state.reference - st.session_state.rolling_1s, 5)}</span>',
                    unsafe_allow_html=True,
                )
            except Exception as e:
                print(repr(e))

        else:
            # End of video
            logger.info(f"END OF PLAYBACK")
            st.session_state.play = False
            # st.session_state.width_list = []
            cap.release()
            # width_list = []

        # plot_area = st.pyplot(meanplot())

# vid_area = st.image(st.session_state.title_frame)

# else:
#     # End of video reached
#     st.session_state.play = False
