import logging
import time

import av
import cv2
import streamlit as st
import pandas as pd

from image_processor import add_info_on_the_frame, draw_fps, draw_n_frames, \
    update_title_frame
from plot import update_rolling_plot
from files import load_video
from utils import get_logger, make_result_df, mean_rolling, \
    FpsCalculator

logger = get_logger()
logging_level = logging.DEBUG

fps_calculator = FpsCalculator()


def play_video(app_state):
    # init_variables()
    # load_video()
    open_video_source()
    cap = st.session_state.cap
    print(f"PLAY   {st.session_state['play']}")
    n_frames = 0
    time_strt = time.time()
    while cap.isOpened():
        ret, frame = cap.read()

        if ret:
            fps_calculator.tick()  # Update time
            fps = fps_calculator.get_fps()  # Get mean FPS

            # time_start = time.time()
            source, st.session_state.width_pxl, st.session_state.width_mm = add_info_on_the_frame(
                frame,
                app_state
            )
            # app_state.add_width(width_mm)
            n_frames += 1

            # Process variables
            st.session_state.fps = fps

            source = draw_fps(source, fps)
            source = draw_n_frames(source, n_frames)

            # Plot
            st.session_state.rolling_1s = round(
                mean_rolling(app_state.width_list, fps), 4
            )
            st.session_state.rolling_10s = round(
                mean_rolling(app_state.width_list, fps, 10), 4
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
            cap.release()
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

    st.session_state["play"] = True

    logger = get_logger("OPEN VIDEO", level=logging_level)
    logger.info('Try to open the video...')
    # if st.session_state.cap:
    #     st.session_state.cap.release()
    # print(st.session_state["source"])
    # print(st.session_state["video_path"])
    if ("video_path" in st.session_state) and (st.session_state["source"] == "File"):

        logger.info("Video from file")
        video_path = st.session_state["video_path"]
        st.session_state.cap = cv2.VideoCapture(video_path)

        launch_video_processing()
    else:
        logger.info("Select the video first!")
    # return cap


def launch_video_processing():
    logger = get_logger('File processor', level=logging_level)
    if st.session_state.filename:
        # check_variables()
        logger.debug('Got the Video file')
        # Get filename, set title frame
        logger.debug('Start to load the video')


def webcam_callback(frame: av.VideoFrame, app_state) -> av.VideoFrame:
    time_start = time.time()
    image = frame.to_ndarray(format="bgr24")
    image, width_pxl, width_mm = add_info_on_the_frame(
        image,
        app_state
    )
    time_end = time.time()
    fps = 1/(time_end - time_start)
    st.session_state['fps'] = fps

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

