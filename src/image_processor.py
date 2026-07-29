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


def normalize_frame(frame):
    """
    Bring a frame to the 3-channel uint8 BGR form used by the whole pipeline.

    A camera may deliver grayscale, BGRA or nothing at all, especially while
    it is warming up, so anything unusable turns into None instead of a crash.

    Args:
        frame: raw frame from a capture device or a video file.

    Returns:
        BGR frame, or None if the frame cannot be used.
    """
    if frame is None:
        return None

    image = np.asarray(frame)
    if image.size == 0 or image.ndim not in (2, 3):
        return None

    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)

    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    if image.shape[2] == 3:
        return image
    return None


def blank_mask(shape=None):
    """Return an empty mask, meaning no filament was found"""
    height, width = shape or (config.IMAGE_HEIGHT, config.IMAGE_WIDTH)
    return np.full((height, width), config.BINARY_MAX_VALUE, dtype=np.uint8)


def to_display(image):
    """
    Convert a BGR frame to RGB.

    OpenCV keeps images in BGR while Streamlit expects RGB, so without this
    conversion red and blue channels are swapped on the screen.
    """
    if image is None:
        return None
    image = np.asarray(image)
    if image.ndim == 3 and image.shape[2] == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def select_filament_contour(contours, frame_width):
    """
    Pick the contour that belongs to the filament.

    The filament is pulled through the whole field of view, so it always
    reaches both edges of the frame, while dirt sits wherever it landed.
    Size alone is not enough to tell them apart: several specks touching each
    other add up to a blob larger than a thin filament.

    Args:
        contours: contours found on the mask.
        frame_width: width of the frame in pixels.

    Returns:
        The largest contour among those crossing the frame, or simply the
        largest one when nothing crosses it.
    """
    min_span = frame_width * config.MIN_FILAMENT_WIDTH_SPAN
    crossing = [
        contour for contour in contours if cv2.boundingRect(contour)[2] >= min_span
    ]
    return max(crossing or contours, key=cv2.contourArea)


def keep_filament_contour(mask):
    """
    Leave only the filament on the mask, dropping dirt and shadows.

    Dust on the glass and shadows are dark as well, and every dark pixel used
    to be counted as a part of the filament: a dozen specks in view more than
    doubled the measured width.

    Args:
        mask: binary mask where the filament is dark and the background is light.

    Returns:
        Mask of the same shape holding the filament alone. A mask without any
        dark pixels is returned untouched.
    """
    # Contours are searched over the non-zero areas, so the mask is inverted.
    filament = cv2.bitwise_not(mask)
    contours, _ = cv2.findContours(filament, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return mask

    cleaned = np.zeros_like(filament)
    # Filling the outer contour also closes the glare spots inside the filament.
    cv2.drawContours(
        cleaned,
        [select_filament_contour(contours, mask.shape[1])],
        -1,
        color=255,
        thickness=cv2.FILLED,
    )
    return cv2.bitwise_not(cleaned)


def process_image(frame, add_info=True, verbose=0):
    """
    Take one frame and process it. Return masked frame and mean width of the filament

    The filament crosses the frame from edge to edge, so the thickness is the
    number of dark pixels per vertical line averaged over every column.

    Args:
        frame:
            image frame
        add_info:
            add info on the frame
        verbose:
            If True, print more information

    Returns:
        Masked frame and mean width of the filament. An unusable frame gives
        an empty mask and zero thickness.
    """
    normalized = normalize_frame(frame)
    if normalized is None:
        logger.warning("Unusable frame received, thickness is reported as zero")
        return blank_mask(), 0.0

    gray_frame = cv2.cvtColor(normalized, cv2.COLOR_BGR2GRAY)
    _, binary_frame = cv2.threshold(
        gray_frame, config.BINARY_THRESHOLD, config.BINARY_MAX_VALUE, cv2.THRESH_BINARY
    )
    if config.FILTER_FILAMENT_CONTOUR:
        binary_frame = keep_filament_contour(binary_frame)

    if verbose:
        cv2.imshow("image", normalized)
        cv2.waitKey(0)

        cv2.imshow("mask", binary_frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # Measure filament thickness in pixels
    # by averaging the number of pixels per vertical line
    filament_thickness = float(np.mean(np.sum(binary_frame == 0, axis=0)))

    logger.debug(
        f"Filament thickness on the current frame: {filament_thickness} pixels"
    )

    if add_info:
        thickness_text = f"Filament Thickness: {filament_thickness:.2f} pixels"
        cv2.putText(
            normalized,
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
    """
    Fit a straight line into the filament pixels.

    Returns:
        Line params, or None when there is nothing to fit: an empty mask, or
        a mask so dark that it is a broken frame rather than a filament.
    """
    y_coords, x_coords = np.where(mask == 0)
    if len(x_coords) < 2:
        return None

    if len(x_coords) > mask.size * config.MAX_FILAMENT_AREA_FRACTION:
        logger.warning("Frame is too dark to hold a filament, angle is not measured")
        return None

    try:
        params, _ = curve_fit(line_func, x_coords, y_coords)
    except (RuntimeError, ValueError, TypeError) as e:
        logger.warning(f"Line fit failed: {e!r}")
        return None
    return params


def measure_angle(mask):
    """Compute angle on the filament. Returns None when the angle is unknown"""
    params = fit_filament_line(mask)
    if params is None:
        return None

    return np.arctan(params[0]) * 180.0 / np.pi


def draw_angle_line(frame, mask):
    """Draw angle line. Returns the frame untouched when the angle is unknown"""
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
    """
    Calculating the multiplier for a tilted filament.

    A tilted filament is crossed by a vertical scan line along a longer chord
    than its real width, and the ratio between them is the cosine of the tilt.
    """
    if angle is None:
        return 1
    angle_radians = np.radians(angle)
    cos_angle = np.cos(angle_radians)
    return cos_angle if cos_angle != 0 else 1


def measure_filament(frame, width_multiplier, show_mask=False):
    """
    Measure the filament on a single frame.

    This is the pure core of the measurement: it does not touch the session
    state and never raises, so a broken frame costs one zeroed measurement
    instead of the whole video loop.

    Args:
        frame: raw frame.
        width_multiplier: millimetres per pixel, taken from the calibration.
        show_mask: return the mask instead of the frame as the image to show.

    Returns:
        Image to display, width in pixels and width in millimetres.
    """
    try:
        normalized = normalize_frame(frame)
        if normalized is None:
            return blank_mask(), 0.0, 0.0

        mask, width_pxl = process_image(frame=normalized, add_info=True)
        source = mask if show_mask else normalized

        source, angle = draw_angle_line(source, mask)
        width_pxl = width_pxl * calculate_pixel_multiplier(angle)
        width_mm = width_pxl * width_multiplier
        return source, width_pxl, width_mm
    except Exception as e:
        logger.warning(f"Frame measurement failed: {e!r}")
        return blank_mask(), 0.0, 0.0


def change_calibration_multiplier():
    """The calibration multiplier is used to estimate the current width"""
    if st.session_state.cap:
        ret, frame = st.session_state.cap.read()
        if ret:
            _, width_pxl, _ = measure_filament(frame, width_multiplier=1)
            st.session_state.width_pxl = width_pxl
        else:
            logger.warning("Could not read a frame for the calibration")

    # Avoid to divide by zero. 160 is a test mean value.
    if not st.session_state.width_pxl:
        st.session_state.width_pxl = config.FALLBACK_WIDTH_PXL
    st.session_state.width_multiplier = (
        st.session_state.reference / st.session_state.width_pxl
    )
    logger.info(f"Calibration multiplier:       {st.session_state.width_multiplier}")
    logger.info(f"st.session_state.reference :  {st.session_state.reference}")
    logger.info(f"st.session_state.width_pxl:   {st.session_state.width_pxl}")


def mask_switcher():
    """Switcher mask/image"""
    st.session_state.show_mask = not st.session_state.show_mask
    logger.info(f"Switched! Show mask: {st.session_state.show_mask}!")


def add_info_on_the_frame(frame):
    """Measure the frame and store the result in the session state"""
    source, width_pxl, width_mm = measure_filament(
        frame,
        width_multiplier=st.session_state.width_multiplier,
        show_mask=st.session_state.show_mask,
    )
    st.session_state.width_pxl = width_pxl
    st.session_state.width_list.append(width_mm)
    return source, width_pxl, width_mm


def update_title_frame(frame):
    """
    Update the title frame.
    """
    if frame is None:
        return
    st.session_state.title_frame = frame
    if "vid_area" in st.session_state:
        st.session_state.vid_area.image(to_display(frame))
