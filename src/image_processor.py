import logging

import av
import cv2
import numpy as np
from scipy.optimize import curve_fit


import pandas as pd
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoHTMLAttributes


from utils import get_logger
import logging

logging_level = logging.INFO
logging_level = logging.DEBUG


def load_image_into_numpy_array(path):
    """Load an image from file into a numpy array.

    Puts image into numpy array to feed into tensorflow graph.
    Note that by convention we put it into a numpy array with shape
    (height, width, channels), where channels=3 for RGB.

    Args:
      path: the file path to the image

    Returns:
      uint8 numpy array with shape (img_height, img_width, 3)
    """
    return np.array(cv2.imread(path))


def process_image(frame, verbose=0):
    """
    Take one frame and process it. Return masked frame and mean width of the filament
    Args:
        frame:
            image frame
        verbose:
            If True, print more information

    Returns:
        Masked frame and mean width of the filament
    """
    logger = get_logger("IMAGE PROCESSOR", level=logging_level)
    # check_variables()
    image_np = np.array(frame)

    # Example processing: Convert to grayscale and apply thresholding
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, binary_frame = cv2.threshold(gray_frame, 127, 255, cv2.THRESH_BINARY)

    if verbose:
        cv2.imshow("image", image_np)
        cv2.waitKey(0)

        cv2.imshow("mask", binary_frame)
        cv2.waitKey(0)

    # Measure filament thickness in pixels
    # by averaging the number of pixels per vertical line
    filament_thickness = np.mean(np.sum(binary_frame == 0, axis=0))

    # The Output the measured thickness for the current frame
    logger.debug(
        "Filament thickness on the current frame: {} pixels".format(filament_thickness)
    )

    # Display the processed frame with information about the thickness
    cv2.putText(
        frame,
        "Filament Thickness: {:.2f} pixels".format(filament_thickness),
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 255),
        2,
    )
    cv2.putText(
        binary_frame,
        "Filament Thickness: {:.2f} pixels".format(filament_thickness),
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 255),
        2,
    )

    # closing all open windows
    cv2.destroyAllWindows()

    return binary_frame, filament_thickness


def line_func(x, a, b):
    """Just line function"""
    return a * x + b


def measure_angle(mask):
    """Compute angle on the filament"""
    y_coords, x_coords = np.where(mask == 0)
    if len(x_coords) < 2:
        return None

    params, _ = curve_fit(line_func, x_coords, y_coords)
    angle = np.arctan(params[0]) * 180.0 / np.pi

    return angle


def draw_angle_line(frame, mask):
    """Draw angle line"""
    print('draw angle')
    angle = measure_angle(mask)
    if angle is not None:
        angle_text = f"Angle: {angle:.2f} degrees"
        cv2.putText(
            frame, angle_text, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2
        )

        y_coords, x_coords = np.where(mask == 0)
        if len(x_coords) >= 2:
            params, _ = curve_fit(line_func, x_coords, y_coords)
            y1 = int(line_func(0, *params))
            y2 = int(line_func(frame.shape[1], *params))
            cv2.line(frame, (0, y1), (frame.shape[1], y2), (0, 0, 255), 2)

    return frame, angle


def draw_fps(frame, cap):
    """Draw FPS"""
    fps = cap.get(cv2.CAP_PROP_FPS)
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


def calculate_pixel_multiplier(angle):
    """Calculating the multiplier for a tilted filament"""
    angle_radians = np.radians(angle)
    cos_angle = np.cos(angle_radians)
    pixel_multiplier = cos_angle if cos_angle != 0 else 1
    return pixel_multiplier


def change_calibration_multiplier():
    """The calibration multiplier is used to estimate the current width"""
    logger = get_logger("CALIBRATION MULTIPLIER", level=logging_level)
    print(f"width      {st.session_state.width_pxl}")
    print(f"reference: {st.session_state.reference}")
    try:
        st.session_state.width_multiplier = (
                st.session_state.reference / st.session_state.width_pxl
        )
    except Exception as e:
        logger.info(repr(e))
        st.session_state.width_multiplier = 0.01
    logger.info(f"Calibration multiplier: {st.session_state.width_multiplier}")


def mask_switcher(mask_radio):
    """Switcher mask/image"""
    logger = get_logger("MASK SWITCHER", level=logging_level)
    logger.info(f"BUTTON Mask")
    if mask_radio == "Image":
        st.session_state.show_mask = False
    else:
        st.session_state.show_mask = True


def connect_camera():
    """Try to open camera using OpenCV"""
    st.session_state.vid_area = webrtc_streamer(key="sample")
    camera_index = None
    for i in range(5):  # Try different indices, like 0 to 4
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            camera_index = i
            break

    if camera_index is not None:
        st.session_state.cap = cv2.VideoCapture(camera_index)
        st.write("Camera connected successfully!")
    else:
        st.write("Couldn't find a connected camera.")


def add_info_on_the_frame(frame, show_mask, width_multiplier):
    """Draw text and line info on the frame"""
    # When the video starts
    mask, width_pxl = process_image(frame=frame, verbose=0)
    width_pxl = width_pxl
    # show_mask is missed sometimes. need to fix it
    try:
        if show_mask:
            source = mask
        else:
            source = frame
    except Exception as e:
        print(repr(e))
        source = frame

    # Process frame
    print('check vars')
    # check_variables()
    source, angle = draw_angle_line(source.copy(), mask)
    angle_multiplier = calculate_pixel_multiplier(angle)
    width_pxl = width_pxl * angle_multiplier
    width_mm = width_pxl * angle_multiplier * width_multiplier

    # width_multiplier_calibrated = change_calibration_multiplier()

    return source, width_pxl, width_mm


def update_title_frame(file_path):
    # Get first frame
    # logger = get_logger('TITLE FRAME', level=logging_level)

    title_cap = cv2.VideoCapture(file_path)
    ret, frame = title_cap.read()
    if ret:
        _, width = process_image(frame=frame, verbose=0)
        st.session_state.width_pxl = width
        st.session_state.title_frame = frame



