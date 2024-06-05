import cv2
import numpy as np
from scipy.optimize import curve_fit

import streamlit as st


from utils import get_logger
import logging

# logging_level = logging.INFO
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

    if add_info:
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


def draw_fps(frame, fps):
    """Draw FPS"""
    # fps = cap.get(cv2.CAP_PROP_FPS)
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
    """Draw FPS"""
    fps_text = f"n_frames: {n_frames:.1f}"
    cv2.putText(
        frame,
        fps_text,
        (10, frame.shape[0] - 50),
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
    if st.session_state.cap:
        _, frame = st.session_state.cap.read()
        _, st.session_state.width_pxl = process_image(frame=frame, verbose=0)
    logger = get_logger("CALIBRATION MULTIPLIER", level=logging_level)
    print(f"width      {st.session_state.width_pxl}")
    print(f"reference: {st.session_state.reference}")
    try:
        # Avoid to divide by zero. 160 is a test mean value.
        if st.session_state.width_pxl == 0:
            st.session_state.width_pxl = 160
        st.session_state.width_multiplier = (
            st.session_state.reference / st.session_state.width_pxl
        )
    except Exception as e:
        logger.info(repr(e))
        st.session_state.width_multiplier = 0.01
    logger.info(f"Calibration multiplier:       {st.session_state.width_multiplier}")
    logger.info(f"st.session_state.reference :  {st.session_state.reference }")
    logger.info(f"st.session_state.width_pxl:   {st.session_state.width_pxl}")


def mask_switcher():
    """Switcher mask/image"""
    logger = get_logger("MASK SWITCHER", level=logging_level)
    logger.info(f"BUTTON Mask")
    if st.session_state.show_mask:
        st.session_state.show_mask = False
    else:
        st.session_state.show_mask = True
    logger.info(f"Switched! Show mask: {st.session_state.show_mask}!")


def add_info_on_the_frame(frame):
    """Draw text and line info on the frame"""
    # When the video starts
    mask, width_pxl = process_image(frame=frame, add_info=True, verbose=0)
    # show_mask is missed sometimes. need to fix it
    try:
        if st.session_state.show_mask:
            source = mask
        else:
            source = frame
    except Exception as e:
        print(repr(e))
        source = frame

    # Process frame
    try:
        source, angle = draw_angle_line(source, mask)
        angle_multiplier = calculate_pixel_multiplier(angle)
        width_pxl = width_pxl * angle_multiplier
        width_mm = width_pxl * angle_multiplier * st.session_state.width_multiplier
        st.session_state.width_pxl = width_pxl
        # width_multiplier_calibrated = change_calibration_multiplier()

        st.session_state.width_list.append(width_mm)
    except Exception as e:
        print(repr(e))

    print(f"angle_multiplier   {angle_multiplier}")
    print(f"width_multiplier   {st.session_state.width_multiplier}")
    print(f"width_pxl   {width_pxl}")
    print(f"width_mm   {width_mm}")
    return source, width_pxl, width_mm


def update_title_frame(frame):
    """
    Update the title frame.
    """
    st.session_state.title_frame = frame
    st.session_state.vid_area.image(st.session_state.title_frame)
