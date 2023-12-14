import logging

import cv2
import numpy as np
from scipy.optimize import curve_fit
import os.path

import altair as alt

import pandas as pd
import streamlit as st

from utils import get_logger, check_variables
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
    check_variables()
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


def get_video_filename():
    """Get file name"""
    logger = get_logger("GET VIDEO FILENAME", level=logging_level)
    check_variables()
    logger.info("GET FILENAME...")
    ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    filename = ROOT_DIR + "/data/input/" + st.session_state["filename"]
    logger.info(f"st.session_state['filename']  {st.session_state['filename']}")
    logger.info(f"filename  {os.path.normpath(filename)}")
    # Return a normalized path.
    return os.path.normpath(filename)


def update_rolling_plot(plot_area):
    """
    Display plot based on data from session state.
    Args:
        plot_area: place to display the plot.
    """
    check_variables()
    try:
        min_value = st.session_state.df_points["values"].min()
        max_value = st.session_state.df_points["values"].max()
        # print(st.session_state.df_points)
        points = (
            alt.Chart(st.session_state.df_points)
            .mark_line()
            .encode(
                x=alt.X("frame"),
                y=alt.Y(
                    "values:Q",
                    scale=alt.Scale(domain=[min_value - 0.2, max_value + 0.2]),
                ),
                color="seconds_count:N",
            )
            .properties(width=1000)
            .configure_axis(labelFontSize=20, titleFontSize=20)
            .configure_legend(titleFontSize=20)
        )
        # Update plot every quarter of a second
        # if st.session_state.df_points["frame"].max() % 6 == 0:
        #     plot_area.altair_chart(points)
        plot_area.altair_chart(points)
    except Exception as e:
        print(repr(e))


def change_calibration_multiplier():
    """The calibration multiplier is used to estimate the current width"""
    logger = get_logger("CALIBRATION MULTIPLIER", level=logging_level)
    check_variables()
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
    check_variables()
    logger.info(f"BUTTON Mask")
    if mask_radio == "Image":
        st.session_state.show_mask = False
    else:
        st.session_state.show_mask = True


def make_result_df(num_seconds=2) -> pd.DataFrame():
    """
    Consumes dataframe and melt it to display on the Altair plot
    Returns:
        melted dataframe.
    """
    check_variables()
    # logger.info(f"MEAN 1: {st.session_state.mean_1}")
    # logger.info(f"MEAN 2: {st.session_state.mean_2}")
    df = pd.DataFrame(
        {
            "Mean 1s": st.session_state.mean_1,
            "Mean 10s": st.session_state.mean_2,
        }
    )
    # logger.info(f"FIRST DF:\n {df}")
    df["frame"] = df.index
    # Cut dataframe to represent X seconds of work.
    max_frame = df.frame.max()
    df = df[df.frame > (max_frame - st.session_state.fps * num_seconds)]
    df = df.melt("frame", var_name="seconds_count", value_name="values")
    # logger.info(f"MELTED DF:\n {df}")
    return df


def update_title_frame(file_path):
    # Get first frame
    # logger = get_logger('TITLE FRAME', level=logging_level)

    title_cap = cv2.VideoCapture(file_path)
    ret, frame = title_cap.read()
    if ret:
        _, width = process_image(frame=frame, verbose=0)
        st.session_state.width_pxl = width
        st.session_state.title_frame = frame


def connect_camera():
    """Try to open not only default camera"""
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


def open_video_source():
    """Open a video, return cap into session state"""
    logger = get_logger("OPEN VIDEO", level=logging_level)

    if st.session_state.cap:
        st.session_state.cap.release()
    if ("video_path" in st.session_state) and (st.session_state["source"] == "File"):
        logger.info("Video from file")
        video_path = st.session_state["video_path"]
        st.session_state.cap = cv2.VideoCapture(video_path)
    elif st.session_state["source"] == "USB Device":
        logger.info("Video from USB")
        # st.session_state.cap = cv2.VideoCapture(0)
        connect_camera()
    else:
        logger.info("Select the video first!")
    # _, st.session_state.title_frame = st.session_state.cap.read()


def stop():
    """Stop the cap"""
    logger = get_logger("STOP VIDEO", level=logging_level)

    check_variables()
    logger.info(f"BUTTON Stop")
    st.session_state.play = False
    logger.info(f"st.session_state.play:   {st.session_state.play}")
    if st.session_state.cap:
        st.session_state.cap.release()
        logger.debug('Cap released')

    # st.session_state.width_list = []

    if st.session_state["width_list"]:
        update_rolling_plot(st.session_state["plot_area"])
