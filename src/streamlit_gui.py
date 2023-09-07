import time
import os.path

import pandas as pd
import streamlit as st
from image_processor import *
from utils import mean_rolling

logger = get_logger("STREAMLIT GUI")
st.set_page_config(layout="wide")

st.title('Filament Measurer')
st.sidebar.header('Control Panel')

# Session variables
st.session_state.play = False
st.session_state.pixel_width_multiplier = 1
if 'title_frame' not in st.session_state:
    st.session_state.title_frame = np.full((480, 640, 3), 255, dtype=np.uint8)

# st.session_state.play_button_disabled = False
# st.session_state.stop_button_disabled = True
# st.session_state.disabled = False

if 'width_list' not in st.session_state:
    st.session_state['width_list'] = []
if 'cap' not in st.session_state:
    st.session_state['cap'] = None
if 'show_mask' not in st.session_state:
    st.session_state['show_mask'] = False
if 'show_every_n_frame' not in st.session_state:
    st.session_state['show_every_n_frame'] = 1
if 'video_path' not in st.session_state:
    st.session_state['video_path'] = 'Empty'
if 'width' not in st.session_state:
    st.session_state['width'] = 1
if 'reference' not in st.session_state:
    st.session_state['reference'] = 1
if 'width_multiplier' not in st.session_state:
    st.session_state['width_multiplier'] = 1
if 'rolling_1s' not in st.session_state:
    st.session_state['rolling_1s'] = 0
if 'rolling_10s' not in st.session_state:
    st.session_state['rolling_10s'] = 0
if 'mean_1' not in st.session_state:
    st.session_state['mean_1'] = []
if 'mean_2' not in st.session_state:
    st.session_state['mean_2'] = []


# Functions
@st.cache_resource
def get_video_filename():
    """ Get file name """
    logger.info('GET FILENAME...')
    ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    filename = (ROOT_DIR + "/data/input/" + st.session_state['filename'])
    # logger.info(f"st.session_state['filename']  {st.session_state['filename']}")
    logger.info(f"filename  {os.path.normpath(filename)}")
    # Return a normalized path.
    return os.path.normpath(filename)


# Function to open video using OpenCV
def open_video():
    """ Open a video, return cap into session state """
    if st.session_state.cap:
        st.session_state.cap.release()
    video_path = st.session_state['video_path']
    st.session_state.cap = cv2.VideoCapture(video_path)
    # _, st.session_state.title_frame = st.session_state.cap.read()


def stop():
    """ Stop the cap """
    logger.info(f"BUTTON Stop")
    st.session_state.play = False
    if 'cap' in st.session_state:
        st.session_state.cap.release()


def change_calibration_multiplier():
    """ The calibration multiplier is used to estimate the current width """
    print(f"width      {st.session_state.width}")
    print(f"reference: {st.session_state.reference}")
    try:
        st.session_state.width_multiplier = st.session_state.reference / st.session_state.width
    except Exception as e:
        logger.info(repr(e))
        st.session_state.width_multiplier = 0.01
    logger.info(f"Calibration multiplier: {st.session_state.width_multiplier}")


def mask_switcher():
    """ Switcher mask/image """
    logger.info(f"BUTTON Mask")
    if mask_radio == 'Image':
        st.session_state.show_mask = False
    else:
        st.session_state.show_mask = True


# Elements
reference = st.sidebar.number_input('Reference width (mm):', value=float(1),
                                    # on_change=change_calibration_multiplier
                                    )
input_source = st.sidebar.selectbox('Input Source', key='input_source',
                                    options=['File', 'USB Device'], index=0)
video_file = st.sidebar.file_uploader('Select a video file',
                                      type=['mp4', 'avi', 'mov'],
                                      # on_change=open_video
                                      )
play_button = st.sidebar.button("Play", key='play_button', on_click=open_video)
stop_button = st.sidebar.button('Stop', key='stop_button', on_click=stop)
# frames_radio = st.sidebar.radio("Show N% of frames", ["100%", "10%"])
# show_10_button = st.sidebar.button('Show 10% of frames', key='show_10_button',
#                                    disabled=st.session_state.disabled)
# show_100_button = st.sidebar.button('Show 100% of frames', key='show_100_button',
#                                     disabled=st.session_state.disabled)
# mask_radio = st.sidebar.radio("Mask/Image", [":rainbow[Image]", "***Mask***"])
mask_radio = st.sidebar.radio("Mask/Image", ["Image", "Mask"],
                              # on_change=mask_switcher
                              )

# Image display area
col1, col2, col3 = st.columns([0.5, 0.2,0.2])
with col1:
    st.header("Video")
    vid_area = st.image(st.session_state.title_frame)
with col2:
    st.header("Results")
    width_pxl = st.text(f'Width, pixels: N/A')
    width_mm = st.text(f'Width, mm:     N/A')
with col3:
    st.header("Mean rolling")
    rolling_1s = st.text(f'1 second:   0')
    rolling_10s = st.text(f'10 seconds: 0')


# st.write(st.session_state)

# Logic

# Change reference multiplier
if reference:
    st.session_state['reference'] = reference
    change_calibration_multiplier()

# """ Switcher mask/image """
logger.info(f"BUTTON Mask")
if mask_radio == 'Image':
    st.session_state.show_mask = False
else:
    st.session_state.show_mask = True

# if frames_radio == '100%':
#     st.session_state['show_every_n_frame'] = 1
# else:
#     st.session_state['show_every_n_frame'] = 10

if stop_button:
    """ Stop the cap """
    logger.info(f"BUTTON Stop")
    st.session_state.play = False
    st.session_state.width_list = []
    if 'cap' in st.session_state:
        st.session_state.cap.release()

if video_file:
    # Get filename, set title frame
    if 'filename' not in st.session_state:
        logger.debug('filename not in session state')
        st.session_state['filename'] = video_file.name
        st.session_state['video_path'] = get_video_filename()
        open_video()
        _, st.session_state.title_frame = st.session_state.cap.read()
    else:
        logger.debug('filename IS in session state')
    vid_area.image(st.session_state.title_frame)

chart_data = pd.DataFrame({
    'col1' : st.session_state.mean_1,
    'col2' : st.session_state.mean_2,
})

st.line_chart(
    chart_data
)


if play_button and video_file:
    logger.info(f"play_button and video_file is TRUE")
    st.session_state['play'] = True
    # Process first frame
    ret, frame = st.session_state.cap.read()
    if ret:
        _, width = process_image(frame=frame, verbose=0)
        st.session_state.width = width
    change_calibration_multiplier()

# Play video
if st.session_state.cap and st.session_state.play:
    print(f"PLAY   {st.session_state['play']}")
    cap = st.session_state.cap
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            # When the video starts
            mask, width = process_image(frame=frame, verbose=0)
            st.session_state.width = width
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
            source, angle = draw_angle_line(source.copy(), mask)
            angle_multiplier = calculate_pixel_multiplier(angle)
            st.session_state.width_list.append(width * angle_multiplier * st.session_state.width_multiplier)
            # width_multiplier_calibrated = change_calibration_multiplier()

            fps = cap.get(cv2.CAP_PROP_FPS)
            st.session_state.rolling_1s = round(mean_rolling(st.session_state.width_list, fps), 4)
            st.session_state.rolling_10s = round(mean_rolling(st.session_state.width_list, fps, 10), 4)
            st.session_state.mean_1.append(st.session_state.rolling_1s)
            st.session_state.mean_2.append(st.session_state.rolling_10s)

            width_pxl.text(f'Width, pixels: {round((width * angle_multiplier), 0)}')
            width_mm.text(f'Width, mm:     {round(width * st.session_state.width_multiplier * angle_multiplier, 3)}')

            rolling_1s.text(f'1 second:   {st.session_state.rolling_1s}')
            rolling_10s.text(f'10 seconds: {st.session_state.rolling_10s}')

            source = draw_fps(source, cap)
            # frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            vid_area.image(source)
            time.sleep(1 / fps)  # keep the fps the same as the original fps

        else:
            # End of video
            logger.info(f"END OF PLAYBACK")
            st.session_state.play = False
            st.session_state.width_list = []
            cap.release()
            width_list = []












# vid_area = st.image(st.session_state.title_frame)

# else:
#     # End of video reached
#     st.session_state.play = False
