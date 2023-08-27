import time

import streamlit as st
from image_processor import *


def main():
    st.title('Filament Mesurer')
    st.sidebar.header('Control Panel')

    # Session variables
    st.session_state.play = False
    st.session_state.pixel_width_multiplier = 0
    st.session_state.title_frame = np.full((480, 640, 3), 255, dtype=np.uint8)

    # Image display area
    vid_area = st.image(st.session_state.title_frame)


    calibration_width = st.sidebar.text_input('Change calibration width (mm):', value=0)

    input_source = st.sidebar.selectbox('Input Source', ['File', 'USB Device'], index=0)
    video_file = st.sidebar.file_uploader('Select a video file', type=['mp4', 'avi', 'mov'])

    play_button = st.sidebar.button("Play", key='play_button', disabled=not video_file)
    stop_button = st.sidebar.button('Stop', key='stop_button', disabled=True)
    show_10_button = st.sidebar.button('Show 10% of frames', key='show_10_button', disabled=True)
    show_100_button = st.sidebar.button('Show 100% of frames', key='show_100_button', disabled=True)
    mask_button = st.sidebar.button('Mask/Image', key='mask_button', disabled=True)

    if play_button and video_file:
        cap = cv2.VideoCapture("C://Users/KOS/Downloads/" + video_file.name)
        st.session_state.play = True
        # stop_button = False

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                st.session_state.play = False
                break
            mask, width = process_image(frame=frame, verbose=0)
            frame, angle = draw_angle_line(frame.copy(), mask)
            angle_multiplier = calculate_pixel_multiplier(angle)
            frame = draw_fps(frame, cap)

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            vid_area.image(frame_rgb)
            time.sleep(1 / cap.get(cv2.CAP_PROP_FPS))  # Задержка для оригинального FPS


    st.write('Mean width in pixels: ')
    st.write('Mean width in mm: ')

if __name__ == "__main__":
    main()