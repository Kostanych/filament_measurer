"""Image processing: filament mask, thickness and angle measurement"""

import logging

import cv2
import numpy as np
import streamlit as st
from scipy.optimize import curve_fit

from config import config
from utils import get_logger

logging_level = logging.DEBUG

logger = get_logger("IMAGE PROCESSOR", level=logging_level)


def process_image(frame, add_info=True, verbose=0):
    """
    Take one frame and process it. Return masked frame and mean width of the filament
    Args:
        frame:
            image frame
        add_info:
            add info on the frame
        verbose:
            If True, print more information

    Returns:
        Masked frame and mean width of the filament
    """
    # Example processing: Convert to grayscale and apply thresholding
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, binary_frame = cv2.threshold(
        gray_frame, config.BINARY_THRESHOLD, config.BINARY_MAX_VALUE, cv2.THRESH_BINARY
    )

    if verbose:
        cv2.imshow("image", np.array(frame))
        cv2.waitKey(0)

        cv2.imshow("mask", binary_frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # Measure filament thickness in pixels
    # by averaging the number of pixels per vertical line
    filament_thickness = np.mean(np.sum(binary_frame == 0, axis=0))

    # The Output the measured thickness for the current frame
    logger.debug(
        f"Filament thickness on the current frame: {filament_thickness} pixels"
    )

    if add_info:
        # Display the processed frame with information about the thickness
        thickness_text = f"Filament Thickness: {filament_thickness:.2f} pixels"
        cv2.putText(
            frame,
            thickness_text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2,
        )
        cv2.putText(
            binary_frame,
            thickness_text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2,
        )

    return binary_frame, filament_thickness


def line_func(x, a, b):
    """Just line function"""
    return a * x + b


def fit_filament_line(mask):
    """Fit a straight line into the filament pixels. Return line params or None"""
    y_coords, x_coords = np.where(mask == 0)
    if len(x_coords) < 2:
        return None

    params, _ = curve_fit(line_func, x_coords, y_coords)
    return params


def measure_angle(mask):
    """Compute angle on the filament"""
    params = fit_filament_line(mask)
    if params is None:
        return None

    return np.arctan(params[0]) * 180.0 / np.pi


def draw_angle_line(frame, mask):
    """Draw angle line"""
    params = fit_filament_line(mask)
    if params is None:
        return frame, None

    angle = np.arctan(params[0]) * 180.0 / np.pi
    angle_text = f"Angle: {angle:.2f} degrees"
    cv2.putText(
        frame, angle_text, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2
    )

    y1 = int(line_func(0, *params))
    y2 = int(line_func(frame.shape[1], *params))
    cv2.line(frame, (0, y1), (frame.shape[1], y2), (0, 0, 255), 2)

    return frame, angle


def draw_fps(frame, fps):
    """Draw FPS"""
    fps_text = f"FPS: {fps:.1f}"
    cv2.putText(
        frame,
        fps_text,
        (10, frame.shape[0] - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2,
    )
    return frame


def draw_n_frames(frame, n_frames):
    """Draw processed frames counter"""
    frames_text = f"n_frames: {n_frames:.1f}"
    cv2.putText(
        frame,
        frames_text,
        (10, frame.shape[0] - 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2,
    )
    return frame


def calculate_pixel_multiplier(angle):
    """Calculating the multiplier for a tilted filament"""
    if angle is None:
        return 1
    angle_radians = np.radians(angle)
    cos_angle = np.cos(angle_radians)
    return cos_angle if cos_angle != 0 else 1


def change_calibration_multiplier():
    """The calibration multiplier is used to estimate the current width"""
    if st.session_state.cap:
        _, frame = st.session_state.cap.read()
        _, st.session_state.width_pxl = process_image(frame=frame, verbose=0)
    try:
        # Avoid to divide by zero. 160 is a test mean value.
        if st.session_state.width_pxl == 0:
            st.session_state.width_pxl = config.FALLBACK_WIDTH_PXL
        st.session_state.width_multiplier = (
            st.session_state.reference / st.session_state.width_pxl
        )
    except Exception as e:
        logger.info(repr(e))
        st.session_state.width_multiplier = config.DEFAULT_WIDTH_MULTIPLIER
    logger.info(f"Calibration multiplier:       {st.session_state.width_multiplier}")
    logger.info(f"st.session_state.reference :  {st.session_state.reference}")
    logger.info(f"st.session_state.width_pxl:   {st.session_state.width_pxl}")


def mask_switcher():
    """Switcher mask/image"""
    st.session_state.show_mask = not st.session_state.show_mask
    logger.info(f"Switched! Show mask: {st.session_state.show_mask}!")


def add_info_on_the_frame(frame):
    """Draw text and line info on the frame"""
    # When the video starts
    mask, width_pxl = process_image(frame=frame, add_info=True, verbose=0)
    source = mask if st.session_state.show_mask else frame
    width_mm = width_pxl * st.session_state.width_multiplier

    # Process frame
    try:
        source, angle = draw_angle_line(source, mask)
        angle_multiplier = calculate_pixel_multiplier(angle)
        width_pxl = width_pxl * angle_multiplier
        width_mm = width_pxl * angle_multiplier * st.session_state.width_multiplier
        st.session_state.width_pxl = width_pxl

        st.session_state.width_list.append(width_mm)
    except Exception as e:
        logger.warning(repr(e))

    logger.debug(f"width_multiplier   {st.session_state.width_multiplier}")
    logger.debug(f"width_pxl   {width_pxl}")
    logger.debug(f"width_mm   {width_mm}")
    return source, width_pxl, width_mm


def update_title_frame(frame):
    """
    Update the title frame.
    """
    st.session_state.title_frame = frame
    st.session_state.vid_area.image(st.session_state.title_frame)
