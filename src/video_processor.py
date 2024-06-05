import logging
import os
import sys
import time

import av
import cv2
import streamlit as st
import pandas as pd

from image_processor import (
    add_info_on_the_frame,
    draw_fps,
    draw_n_frames,
    update_title_frame,
    change_calibration_multiplier,
    process_image,
)
from plot import update_rolling_plot
from utils import (
    get_logger,
    make_result_df,
    mean_rolling,
    FpsCalculator,
    init_variables,
)

logger = get_logger()
logging_level = logging.DEBUG

fps_calculator = FpsCalculator()

sys.path.append(os.path.abspath(".."))


def play_or_continue_video():
    logger = get_logger("PLAY OR CONTINUE VIDEO", level=logging_level)
    print("play_or_continue_video")
    _, st.session_state.width_pxl = process_image(
        frame=st.session_state.title_frame, add_info=False
    )

    if not st.session_state.cap:
        open_video_source()

    if st.session_state["play"]:
        n_frames = 0
        time_strt = time.time()
        last_update_time = time.time()
        if st.session_state.cap:
            change_calibration_multiplier()
            while st.session_state.cap.isOpened():
                ret, frame = st.session_state.cap.read()
                if ret:
                    fps_calculator.tick()
                    fps = fps_calculator.get_fps()
                    (
                        source,
                        st.session_state.width_pxl,
                        st.session_state.width_mm,
                    ) = add_info_on_the_frame(frame)
                    n_frames += 1
                    st.session_state.fps = fps
                    plot_means()
                    source = draw_fps(source, fps)
                    source = draw_n_frames(source, n_frames)
                    st.session_state.vid_area.image(source)
                    st.session_state["last_frame"] = source

                    current_time = time.time()
                    update_interval = st.session_state["update_interval"]
                    if (
                        (update_interval == "Every Frame")
                        or (
                            update_interval == "1 Second"
                            and current_time - last_update_time >= 1
                        )
                        or (
                            update_interval == "5 Seconds"
                            and current_time - last_update_time >= 5
                        )
                    ):

                        chart_data = make_result_df()
                        st.session_state.df_points = chart_data
                        update_rolling_plot(st.session_state["plot_area"])
                        st.session_state.difference_markdown.markdown(
                            f'<span style="font-size: 20px;">Difference(1s mean):{round(st.session_state.reference - st.session_state.rolling_1s, 5)}</span>',
                            unsafe_allow_html=True,
                        )
                        last_update_time = current_time
                else:
                    st.session_state.play = False
                    st.session_state["last_frame"] = source
                    st.session_state.cap.release()
                    st.session_state.cap = None
                    break
        update_title_frame(st.session_state["last_frame"])


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
        logger.debug("Got the Video file")
        # Get filename, set title frame
        logger.debug("Start to load the video")


def webcam_callback(frame: av.VideoFrame, app_state) -> av.VideoFrame:

    # fps_calculator.tick()  # Update time
    # fps = fps_calculator.get_fps()  # Get mean FPS

    time_start = time.time()
    image = frame.to_ndarray(format="bgr24")
    (
        image,
        st.session_state.width_pxl,
        st.session_state.width_mm,
    ) = add_info_on_the_frame(image, app_state)

    time_end = time.time()
    fps = 1 / (time_end - time_start)
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
