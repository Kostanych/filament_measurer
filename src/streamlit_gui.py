import time
import os.path

import streamlit as st
from image_processor import *
from utils import mean_rolling

logger = get_logger("STREAMLIT GUI")

st.title('Filament Measurer')
st.sidebar.header('Control Panel')

# Session variables
st.session_state.play = False
st.session_state.pixel_width_multiplier = 0
if 'title_frame' not in st.session_state:
    st.session_state.title_frame = np.full((480, 640, 3), 255, dtype=np.uint8)
# st.session_state.play_button_disabled = False
# st.session_state.stop_button_disabled = True
# st.session_state.disabled = False

if 'cap' not in st.session_state:
    st.session_state.cap = None
if 'show_mask' not in st.session_state:
    st.session_state.show_mask = False
if 'show_every_n_frame' not in st.session_state:
    st.session_state['show_every_n_frame'] = 1
if 'video_path' not in st.session_state:
    st.session_state['video_path'] = 'Empty'


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


# def mask_switcher():
#     """ Switcher mask/image """
#     logger.info(f"BUTTON Mask")
#     if mask_radio == 'Image':
#         st.session_state.show_mask = False
#     else:
#         st.session_state.show_mask = True


# Elements
calibration_width = st.sidebar.text_input('Change calibration width (mm):', value=0)
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
mask_radio = st.sidebar.radio("Mask/Image", ["Image", "Mask"])

# Image display area
col1, col2 = st.columns([0.7, 0.3])
with col1:
    st.header("Video")
    vid_area = st.image(st.session_state.title_frame)
with col2:
    st.header("Results")
    st.write('Mean width in pixels: ')
    st.write('Mean width in mm: ')

# vid_area = st.image(st.session_state.title_frame)
# st.write('Mean width in pixels: ')
# st.write('Mean width in mm: ')
st.write(st.session_state)

# Set variables
width_list = []

# Logic
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
    if 'cap' in st.session_state:
        st.session_state.cap.release()

if video_file:
    # Get filename, set title frame
    if 'filename' not in st.session_state:
        logger.info('filename not in session state')
        st.session_state['filename'] = video_file.name
        st.session_state['video_path'] = get_video_filename()
        open_video()
        _, st.session_state.title_frame = st.session_state.cap.read()
    else:
        logger.info('filename IS in session state')

if play_button and video_file:
    logger.info(f"play_button and video_file is TRUE")
    st.session_state['play'] = True
    logger.info(f'play = {st.session_state.play}')

if st.session_state.cap and st.session_state.play:
    print(f"PLAY   {st.session_state['play']}")
    cap = st.session_state.cap
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            # When the video starts
            mask, width = process_image(frame=frame, verbose=0)
            # show_mask is missed sometimes. need to fix it
            try:
                if st.session_state.show_mask:
                    source = mask
                else:
                    source = frame
                logger.info(f"SOURCE IS MASK:   {st.session_state.show_mask}")
            except Exception as e:
                print(repr(e))
                source = frame

            # Process frame
            source, angle = draw_angle_line(source.copy(), mask)
            angle_multiplier = calculate_pixel_multiplier(angle)
            width_list.append(width * angle_multiplier)

            fps = cap.get(cv2.CAP_PROP_FPS)
            rolling_1s = mean_rolling(width_list, fps)
            rolling_10s = mean_rolling(width_list, fps, 10)

            source = draw_fps(source, cap)
            # frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            vid_area.image(source)
            time.sleep(1 / fps)  # keep the fps the same as the original fps

        else:
            # End of video
            logger.info(f"END OF PLAYBACK")
            st.session_state.play = False
            cap.release()
            width_list = []

# vid_area = st.image(st.session_state.title_frame)

# else:
#     # End of video reached
#     st.session_state.play = False
