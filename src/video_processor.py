"""Video processing module for handling video stream and frame processing"""

import logging
import time

import cv2
import streamlit as st

from config import config
from image_processor import (
    add_info_on_the_frame,
    change_calibration_multiplier,
    draw_fps,
    draw_n_frames,
    process_image,
    to_display,
    update_title_frame,
)
from plot import update_rolling_plot
from utils import AppState, FpsCalculator, big_text, make_result_df, mean_rolling

logging_level = logging.DEBUG


class VideoProcessor:
    """Handles video processing, frame-by-frame analysis, and display updates"""

    def __init__(self):
        self.state = AppState()
        self.logger = self.state.get_logger("VIDEO PROCESSOR", level=logging_level)
        self.fps_calculator = FpsCalculator()
        self.last_update_time = time.time()

    def play_or_continue_video(self):
        """Main method to play or continue video processing"""
        self.logger.info("play_or_continue_video")
        _, st.session_state.width_pxl = process_image(
            frame=st.session_state.title_frame, add_info=False
        )

        if not st.session_state.cap:
            self.state.reset_measurements()
            self.open_video_source()

        if st.session_state["play"]:
            n_frames = 0
            source = st.session_state["last_frame"]
            self.last_update_time = time.time()
            if st.session_state.cap:
                change_calibration_multiplier()
                while st.session_state.cap.isOpened():
                    ret, frame = st.session_state.cap.read()
                    if not ret:
                        self.stop_video(source)
                        break
                    # A single broken frame must not stop the whole run:
                    # it is logged and the loop moves on to the next one.
                    try:
                        source = self.process_frame(frame, n_frames)
                    except Exception as e:
                        self.logger.warning(f"Frame {n_frames} skipped: {e!r}")
                    n_frames += 1
                    current_time = time.time()
                    if self.is_time_to_update(current_time):
                        self.update_plot(current_time)
            update_title_frame(st.session_state["last_frame"])

    def is_time_to_update(self, current_time):
        """Check whether the plot should be redrawn on the current frame"""
        interval = config.UPDATE_INTERVALS.get(st.session_state["update_interval"], 0)
        return current_time - self.last_update_time >= interval

    def process_frame(self, frame, n_frames):
        """Process single video frame and update display"""
        self.fps_calculator.tick()
        fps = self.fps_calculator.get_fps()
        (
            source,
            st.session_state.width_pxl,
            st.session_state.width_mm,
        ) = add_info_on_the_frame(frame)
        st.session_state.fps = fps
        plot_means()
        source = draw_fps(source, fps)
        source = draw_n_frames(source, n_frames)
        st.session_state.vid_area.image(to_display(source))
        st.session_state["last_frame"] = source
        return source

    def update_plot(self, current_time):
        """Update plot and difference display"""
        st.session_state.df_points = make_result_df()
        update_rolling_plot(st.session_state["plot_area"])
        difference = round(st.session_state.reference - st.session_state.rolling_1s, 5)
        st.session_state.difference_markdown.markdown(
            big_text(f"Difference(1s mean):{difference}"),
            unsafe_allow_html=True,
        )
        self.last_update_time = current_time

    def open_video_source(self):
        """Open video source (file or USB device)"""
        if ("video_path" in st.session_state) and (
            st.session_state["source"] == "File"
        ):
            self.logger.debug("Video from file")
            cap = cv2.VideoCapture(st.session_state["video_path"])
        elif st.session_state["source"] == "USB Device":
            self.logger.debug("Video from USB device")
            cap = cv2.VideoCapture(0)
        else:
            self.logger.info("Select the video first!")
            st.session_state["play"] = False
            return

        if not cap.isOpened():
            self.logger.warning("Could not open the video source")
            cap.release()
            st.session_state["play"] = False
            st.session_state["status_message"] = "Could not open the video source"
            return

        st.session_state.cap = cap

    def stop_video(self, source):
        """Stop video playback and release resources"""
        st.session_state.play = False
        st.session_state["last_frame"] = source
        if st.session_state.cap:
            st.session_state.cap.release()
            st.session_state.cap = None


def plot_means():
    """Calculate and display rolling means for width measurements"""
    st.session_state.rolling_1s = round(
        mean_rolling(
            st.session_state.width_list,
            st.session_state.fps,
            config.ROLLING_WINDOW_SHORT,
        ),
        4,
    )
    st.session_state.rolling_10s = round(
        mean_rolling(
            st.session_state.width_list,
            st.session_state.fps,
            config.ROLLING_WINDOW_LONG,
        ),
        4,
    )
    st.session_state.mean_1.append(st.session_state.rolling_1s)
    st.session_state.mean_2.append(st.session_state.rolling_10s)
    st.session_state.measurements_total += 1

    st.session_state.width_pxl_area.markdown(
        big_text(f"Width, pixels: {round(st.session_state.width_pxl, 0)}"),
        unsafe_allow_html=True,
    )
    st.session_state.width_mm_area.markdown(
        big_text(f"Width, mm:     {round(st.session_state.width_mm, 3)}"),
        unsafe_allow_html=True,
    )

    st.session_state.rolling_1s_markdown.markdown(
        big_text(f"1 second:   {st.session_state.rolling_1s}"),
        unsafe_allow_html=True,
    )
    st.session_state.rolling_10s_markdown.markdown(
        big_text(f"10 seconds: {st.session_state.rolling_10s}"),
        unsafe_allow_html=True,
    )
