import time

from image_processor import *
from input_source import *
from gui_logic import *
from plot import *
from files import *
from utils import *
import logging


def update_status(new_message):
    st.session_state['status_message'] = new_message
    status_bar.text(new_message)


# logging_level = logging.INFO
logging_level = logging.DEBUG

logger = get_logger("STREAMLIT GUI", level=logging_level)
st.set_page_config(layout="wide")

# Session variables
init_variables()
app_state = AppState()

# Streamlit elements
status_bar = st.empty()
status_bar.text(st.session_state['status_message'])
st.title("Filament Measurer")

# Sidebar
st.sidebar.header("Control Panel")
reference = st.sidebar.number_input(
    "Reference width (mm):",
    value=float(1.75),
    # on_change=change_calibration_multiplier
)
st.session_state["reference"] = reference
input_source = st.sidebar.radio(
    'Input Source',
    options=['File', 'USB Device']
)
st.session_state["source"] = input_source
video_file = st.sidebar.file_uploader(
    "Select a video file",
    type=["mp4", "avi", "mov"],
    # on_change=load_video
)

# Show PLAY button only if you want to play video file
if input_source == 'File':
    play_button = st.sidebar.button("Play",
                                    key="play_button",
                                    on_click=open_video_source,
                                    args=(app_state,)
                                    )

    stop_button = st.sidebar.button("Stop",
                                    key="stop_button",
                                    on_click=stop,
                                    args=(app_state,)
                                    )
# frames_radio = st.sidebar.radio("Show N% of frames", ["100%", "10%"])
# show_10_button = st.sidebar.button('Show 10% of frames', key='show_10_button',
#                                    disabled=st.session_state.disabled)
# show_100_button = st.sidebar.button('Show 100% of frames', key='show_100_button',
#                                     disabled=st.session_state.disabled)
# mask_radio = st.sidebar.radio("Mask/Image", [":rainbow[Image]", "***Mask***"])
mask_radio = st.sidebar.radio(
    "Mask/Image",
    ["Image", "Mask"],
    key='Mask_or_image',
    on_change=mask_switcher,
)

# Image display area
col1, col2, col3 = st.columns([0.3, 0.2, 0.2])
with col1:
    st.header("Video")
    # Switcher videofile\webcamera
    if input_source == 'File':
        vid_area = st.image(image_input())
        st.session_state.vid_area = vid_area
    elif input_source == 'USB Device':
        vid_area = webcam_input(app_state)
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
    # if not st.session_state["width_list"]:
    if not app_state.state['width_list']:
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

# # """ Switcher mask/image """
# if mask_radio == "Image":
#     # app_state.update(show_mask=False)
#     st.session_state.show_mask = False
# else:
#     # app_state.update(show_mask=True)
#     st.session_state.show_mask = True

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

if video_file:
    load_video(video_file)

# Play video
if st.session_state.cap and st.session_state.play:
    init_variables()
    print(f"PLAY   {st.session_state['play']}")
    cap = st.session_state.cap
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:

            # Draw info on the frame
            print(app_state.state)
            source, width_pxl, width_mm, = add_info_on_the_frame(
                frame,
                app_state.state['show_mask'],
                app_state.state['width_multiplier'],
            )
            app_state.add_width(width_mm)

            # Process variables
            fps = cap.get(cv2.CAP_PROP_FPS)
            st.session_state.fps = fps
            source = draw_fps(source, cap)

            # Plot
            st.session_state.rolling_1s = round(
                mean_rolling(app_state.width_list, fps), 4
            )
            st.session_state.rolling_10s = round(
                mean_rolling(app_state.width_list, fps, 10), 4
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
width_list = []

# plot_area = st.pyplot(meanplot())

# vid_area = st.image(st.session_state.title_frame)

# else:
#     # End of video reached
#     st.session_state.play = False
