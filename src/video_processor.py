import logging
import time

import av
import cv2
import streamlit as st
import pandas as pd

from image_processor import add_info_on_the_frame, draw_fps, draw_n_frames, \
    update_title_frame, change_calibration_multiplier, process_image
from plot import update_rolling_plot
from utils import get_logger, make_result_df, mean_rolling, \
    FpsCalculator, init_variables

logger = get_logger()
logging_level = logging.DEBUG

fps_calculator = FpsCalculator()


def play_or_continue_video():
    logger = get_logger("PLAY OR CONTINUE VIDEO", level=logging_level)
    print('play_or_continue_video')
    _, st.session_state.width_pxl = process_image(
        frame=st.session_state.title_frame,
        add_info=False
    )

    if not st.session_state.cap:
        # Create st.session state.cap based on st variables
        logger.info('st.session_state.cap does not exist')
        open_video_source()

    if st.session_state['play']:
        print(f"PLAY   {st.session_state['play']}")
        n_frames = 0
        time_strt = time.time()
        if st.session_state.cap:
            # Change multiplier from the start
            change_calibration_multiplier()
            # init_variables()
            while st.session_state.cap.isOpened():

                ret, frame = st.session_state.cap.read()
                if ret:
                    fps_calculator.tick()  # Update time
                    fps = fps_calculator.get_fps()  # Get mean FPS

                    # time_start = time.time()
                    source, st.session_state.width_pxl, st.session_state.width_mm = add_info_on_the_frame(
                        frame,
                        # app_state
                    )
                    # app_state.add_width(width_mm)
                    n_frames += 1

                    # Process variables
                    st.session_state.fps = fps

                    # Plot the plot
                    plot_means()

                    source = draw_fps(source, fps)
                    source = draw_n_frames(source, n_frames)

                    # Place processed image on the video area
                    st.session_state.vid_area.image(source)
                    st.session_state['last_frame'] = source

                    # time.sleep(1 / fps)  # keep the fps the same as the original fps

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
                        st.session_state.difference_markdown.markdown(
                            f'<span style="font-size: 20px;">Difference(1s mean):{round(st.session_state.reference - st.session_state.rolling_1s, 5)}</span>',
                            unsafe_allow_html=True,
                        )
                    except Exception as e:
                        print(repr(e))
                else:
                    # End of video
                    logger.info("END OF PLAYBACK")
                    st.session_state.play = False
                    st.session_state['last_frame'] = source
                    st.session_state.cap.release()
                    st.session_state.cap = None
                    break  # Close cycle
        update_title_frame(st.session_state['last_frame'])
        real_fps = n_frames / (time.time() - time_strt)
        print(n_frames)
        print(time.time())
        print(time_strt)
        print((time.time() - time_strt))
        logger.info(f"Real fps: {real_fps}")


def open_video_source():
    """Open a video, return cap into session state"""
    logger = get_logger("VIDEO PROCESSOR", level=logging_level)

    if ("video_path" in st.session_state) and (st.session_state["source"] == "File"):
        logger.debug("Video from file")
        video_path = st.session_state["video_path"]
        st.session_state.cap = cv2.VideoCapture(video_path)
    elif st.session_state["source"] == "USB Device":
        logger.debug("Video from USB device")
        st.session_state.cap = cv2.VideoCapture(0)
    else:
        try:
            logger.debug(f"video path {st.session_state['video_path']}")
            logger.debug(f"st.session_state.source : {st.session_state.source}")
        except:
            pass
        logger.info("Select the video first!")
        st.session_state["play"] = False
    # return output_cap


def launch_video_processing():
    logger = get_logger("VIDEO PROCESSOR", level=logging_level)
    if "filename" in st.session_state:
        # check_variables()
        logger.debug('Got the Video file')
        # Get filename, set title frame
        logger.debug('Start to load the video')


def webcam_callback(frame: av.VideoFrame, app_state) -> av.VideoFrame:

    # fps_calculator.tick()  # Update time
    # fps = fps_calculator.get_fps()  # Get mean FPS

    time_start = time.time()
    image = frame.to_ndarray(format="bgr24")
    image, st.session_state.width_pxl, st.session_state.width_mm = add_info_on_the_frame(
        image,
        app_state
    )

    time_end = time.time()
    fps = 1/(time_end - time_start)
    st.session_state.fps = fps

    # Plot the plot
    plot_means(app_state)

    fps_text = f"FPS: { fps:.1f}"
    cv2.putText(
        image,
        fps_text,
        (10, image.shape[0] - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2,
    )

    return av.VideoFrame.from_ndarray(image, format="bgr24")


def plot_means():
    # Plot
    st.session_state.rolling_1s = round(
        mean_rolling(st.session_state.width_list, st.session_state.fps), 4
    )
    st.session_state.rolling_10s = round(
        mean_rolling(st.session_state.width_list, st.session_state.fps, 10), 4
    )
    st.session_state.mean_1.append(st.session_state.rolling_1s)
    st.session_state.mean_2.append(st.session_state.rolling_10s)

    st.session_state.width_pxl_area.markdown(
        f'<span style="font-size: 20px;">Width, pixels: {round(st.session_state.width_pxl, 0)}</span>',
        unsafe_allow_html=True,
    )
    st.session_state.width_mm_area.markdown(
        f'<span style="font-size: 20px;">Width, mm:     {round(st.session_state.width_mm, 3)}</span>',
        unsafe_allow_html=True,
    )

    st.session_state.rolling_1s_markdown.markdown(
        f'<span style="font-size: 20px;">1 second:   {st.session_state.rolling_1s}</span>',
        unsafe_allow_html=True,
    )
    st.session_state.rolling_10s_markdown.markdown(
        f'<span style="font-size: 20px;">10 seconds: {st.session_state.rolling_10s}</span>',
        unsafe_allow_html=True,
    )
