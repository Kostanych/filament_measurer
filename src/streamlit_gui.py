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
st.session_state.play_button_disabled = False
st.session_state.stop_button_disabled = True
st.session_state.disabled = False

if 'mask_or_image' not in st.session_state:
    st.session_state.mask_or_image = st.session_state.get('mask_or_image', 'image')
if 'show_every_n_frame' not in st.session_state:
    logger.info("show_every_n_frame not in a session state!")
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


def change_st_state(st_name, state):
    st.session_state[st_name] = state
    logger.info(f"{st_name} - {st.session_state[st_name]}")


def switch_mask_and_image():
    print(f"DDDDDDDDDDDDDDDDDDDDDDDDDDD    {st.session_state['mask_or_image']}")
    if st.session_state['mask_or_image'] == 'image':
        st.session_state['mask_or_image'] = 'mask'
    else:
        st.session_state['mask_or_image'] = 'image'
    print(f"NEW     DDDDDDDDDDDDDDDDDDD    {st.session_state['mask_or_image']}")


@st.cache_resource
def load_video_file():
    logger.info('GET FILENAME...')
    ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    filename = (ROOT_DIR + "/data/input/" + st.session_state['filename'])
    logger.info(f"st.session_state['filename']  {st.session_state['filename']}")
    logger.info(f"filename  {filename}")

    # Show the first frame
    filename = os.path.normpath(filename)
    cap = cv2.VideoCapture(filename)
    return cap


def stop(cap):
    st.session_state.play = False
    cap.release()
    st.cache_resource.clear()


def main():
    # set variables
    width_list = []
    cap = None

    if video_file:
        st.session_state['filename'] = video_file.name
        cap = load_video_file()
        ret, frame = cap.read()
        vid_area.image(frame)

    if play_button and video_file:
        cap = load_video_file()
        # ret, frame = cap.read()
        # vid_area.image(frame))
        # if (not st.session_state['play']) & (video_file != '') & (
        #         input_source == 'File'):
        #     cap = cv2.VideoCapture(st.session_state['filename'])
        # elif input_source == 'USB device':
        #     cap = cv2.VideoCapture(0)
        st.session_state['play'] = True
        logger.info(f'cap  = {cap}')
        logger.info(f'play = {st.session_state.play}')

    if stop_button:
        stop(cap)

    if show_10_button:
        # Show every 10th frame
        st.session_state['show_every_n_frame'] = 10

    if show_100_button:
        # Show all frames
        st.session_state['show_every_n_frame'] = 1

    if mask_button:
        logger.debug(f"Before mask/image switcher:   {st.session_state['mask_or_image']}")
        switch_mask_and_image()
        logger.debug(f"After mask/image switcher:    {st.session_state['mask_or_image']}")

    if cap and st.session_state.play:
        print(f"PLAY   {st.session_state['play']}")
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                st.session_state.play = False
                cap.release()
                # clear variables
                st.cache_resource.clear()
                width_list = []
            if ret:
                mask, width = process_image(frame=frame, verbose=0)

                if st.session_state['mask_or_image'] == 'image':
                    source = frame
                else:
                    source = mask
                logger.info(f"SOURCE:   {st.session_state['mask_or_image']}")

                # process frame
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
                # End of video reached
                cap.release()
                cap = None
                # show_play_button = True
                st.session_state.play = False
                st.cache_resource.clear()
                break


    st.write('Mean width in pixels: ')
    st.write('Mean width in mm: ')


if __name__ == "__main__":
    main()
