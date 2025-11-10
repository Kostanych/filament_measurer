"""Video processing module for handling video stream and frame processing"""

import logging
import cv2
import streamlit as st
import time
from utils import AppState, FpsCalculator, get_logger, mean_rolling, make_result_df
from plot import update_rolling_plot
from image_processor import (
    add_info_on_the_frame,
    draw_fps,
    draw_n_frames,
    update_title_frame,
    change_calibration_multiplier,
    process_image,
)

logging_level = logging.DEBUG


class VideoProcessor:
    """Handles video processing, frame-by-frame analysis, and display updates"""
    def __init__(self):
        self.state = AppState()
        self.logger = self.state.get_logger("VIDEO PROCESSOR", level=logging.DEBUG)
        self.fps_calculator = FpsCalculator()
        self.last_update_time = time.time()

    def play_or_continue_video(self):
        """Main method to play or continue video processing"""
        self.logger.info("play_or_continue_video")
        _, st.session_state.width_pxl = process_image(
            frame=st.session_state.title_frame, add_info=False
        )

        if not st.session_state.cap:
            self.open_video_source()

        if st.session_state["play"]:
            n_frames = 0
            self.last_update_time = time.time()
            if st.session_state.cap:
                change_calibration_multiplier()
                while st.session_state.cap.isOpened():
                    ret, frame = st.session_state.cap.read()
                    if ret:
                        source = self.process_frame(frame, n_frames)
                        n_frames += 1
                        current_time = time.time()
                        update_interval = st.session_state["update_interval"]
                        if (
                            (update_interval == "Every Frame")
                            or (
                                update_interval == "1 Second"
                                and current_time - self.last_update_time >= 1
                            )
                            or (
                                update_interval == "5 Seconds"
                                and current_time - self.last_update_time >= 5
                            )
                        ):
                            self.update_plot(current_time)
                    else:
                        self.stop_video(source)
                        break
            update_title_frame(st.session_state["last_frame"])

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
        st.session_state.vid_area.image(source)
        st.session_state["last_frame"] = source
        return source

    def update_plot(self, current_time):
        """Update plot and difference display"""
        chart_data = make_result_df()
        st.session_state.df_points = chart_data
        update_rolling_plot(st.session_state["plot_area"])
        st.session_state.difference_markdown.markdown(
            f'<span style="font-size: 20px;">Difference(1s mean):{round(st.session_state.reference - st.session_state.rolling_1s, 5)}</span>',
            unsafe_allow_html=True,
        )
        self.last_update_time = current_time

    def open_video_source(self):
        """Open video source (file or USB device)"""
        if ("video_path" in st.session_state) and (
            st.session_state["source"] == "File"
        ):
            self.logger.debug("Video from file")
            video_path = st.session_state["video_path"]
            st.session_state.cap = cv2.VideoCapture(video_path)
        elif st.session_state["source"] == "USB Device":
            self.logger.debug("Video from USB device")
            st.session_state.cap = cv2.VideoCapture(0)
        else:
            self.logger.info("Select the video first!")
            st.session_state["play"] = False

    def stop_video(self, source):
        """Stop video playback and release resources"""
        st.session_state.play = False
        st.session_state["last_frame"] = source
        st.session_state.cap.release()
        st.session_state.cap = None


def plot_means():
    """Calculate and display rolling means for width measurements"""
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
