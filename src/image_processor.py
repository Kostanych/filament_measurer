import cv2
import pandas as pd
import numpy as np

from src.mask_processor import process_contours, measure_length


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
    image_np = np.array(frame)

    # Example processing: Convert to grayscale and apply thresholding
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, binary_frame = cv2.threshold(gray_frame, 127, 255, cv2.THRESH_BINARY)

    if verbose:
        cv2.imshow('image', image_np)
        cv2.waitKey(0)

        cv2.imshow('mask', binary_frame)
        cv2.waitKey(0)

    # Measure filament thickness in pixels
    # by averaging the number of pixels per vertical line
    filament_thickness = np.mean(np.sum(binary_frame == 0, axis=0))

    # Output the measured thickness for the current frame
    print("Filament thickness on the current frame: {} pixels".format(
        filament_thickness))

    # Display the processed frame with information about the thickness
    cv2.putText(frame,
                "Filament Thickness: {:.2f} pixels".format(filament_thickness),
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(binary_frame,
                "Filament Thickness: {:.2f} pixels".format(filament_thickness),
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # closing all open windows
    cv2.destroyAllWindows()

    return binary_frame, filament_thickness


def nothing(args): pass

