import time

import streamlit as st
from image_processor import *

logger = get_logger("STREAMLIT GUI")


def change_st_state(st_name, state):
    st.session_state[st_name] = state
    logger.info(f"{st_name} - {st.session_state[st_name]}")


def switch_mask_and_image():
    if st.session_state.mask_or_image == 'image':
        return 'mask'
    else:
        return 'image'


@st.cache_resource
def load_video_file():
    filename = ("C://Users/KOS/Downloads/" + st.session_state['filename'])
    # Show the first frame
    logger.info(filename)
    cap = cv2.VideoCapture(filename)
    return cap


st.title('Filament Mesurer')
st.sidebar.header('Control Panel')

# Session variables
st.session_state.play = False
st.session_state.pixel_width_multiplier = 0
if 'title_frame' not in st.session_state:
    st.session_state.title_frame = np.full((480, 640, 3), 255, dtype=np.uint8)
st.session_state.play_button_disabled = False
st.session_state.stop_button_disabled = True
st.session_state.disabled = False

st.session_state.mask_or_image = 'image'
st.session_state['show_every_n_frame'] = 1

# Image display area
vid_area = st.image(st.session_state.title_frame)

calibration_width = st.sidebar.text_input('Change calibration width (mm):', value=0)

input_source = st.sidebar.selectbox('Input Source', key='input_source',
                                    options=['File', 'USB Device'], index=0)
video_file = st.sidebar.file_uploader('Select a video file',
                                      type=['mp4', 'avi', 'mov'])

play_button = st.sidebar.button("Play", key='play_button', )
stop_button = st.sidebar.button('Stop', key='stop_button')
show_10_button = st.sidebar.button('Show 10% of frames', key='show_10_button',
                                   disabled=st.session_state.disabled)
show_100_button = st.sidebar.button('Show 100% of frames', key='show_100_button',
                                    disabled=st.session_state.disabled)
mask_button = st.sidebar.button('Mask/Image', key='mask_button',
                                disabled=st.session_state.disabled)

if video_file:
    st.session_state['filename'] = video_file.name
    logger.info(st.session_state['filename'])
    cap = load_video_file()
    ret, frame = cap.read()
    vid_area.image(frame)

if play_button and video_file:
    cap = load_video_file()
    ret, frame = cap.read()
    vid_area.image(frame)
    logger.info(f'cap  = {cap}')
    logger.info(f'play = {st.session_state.play}')
    # if (not st.session_state['play']) & (video_file != '') & (
    #         input_source == 'File'):
    #     cap = cv2.VideoCapture(st.session_state['filename'])
    # elif input_source == 'USB device':
    #     cap = cv2.VideoCapture(0)
    st.session_state['play'] = True

if stop_button:
    st.session_state.play = False

if show_10_button:
    # Show every 10th frame
    st.session_state['show_every_n_frame'] = 10

if show_100_button:
    # Show all frames
    st.session_state['show_every_n_frame'] = 1

if mask_button:
    switch_mask_and_image()

if st.session_state.play:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.session_state.play = False
        if ret:
            mask, width = process_image(frame=frame, verbose=0)
            frame, angle = draw_angle_line(frame.copy(), mask)
            angle_multiplier = calculate_pixel_multiplier(angle)
            frame = draw_fps(frame, cap)

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            vid_area.image(frame_rgb)
            time.sleep(
                1 / cap.get(cv2.CAP_PROP_FPS))  # Задержка для оригинального FPS
        else:
            # End of video reached
            cap.release()
            cap = None
            # show_play_button = True
            st.session_state.play = False
            break

st.write('Mean width in pixels: ')
st.write('Mean width in mm: ')
