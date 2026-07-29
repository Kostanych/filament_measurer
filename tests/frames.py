"""
Synthetic frames standing in for a real microscope.

A camera does not only deliver a clean picture of a filament: while it is
warming up, or when it is unplugged, the frames can be blank, dark, noisy,
grayscale or missing altogether. These builders reproduce such frames so the
pipeline can be tested without any hardware.
"""

import cv2
import numpy as np

HEIGHT = 480
WIDTH = 640


def white_frame():
    """Overexposed frame: no filament is visible at all"""
    return np.full((HEIGHT, WIDTH, 3), 255, dtype=np.uint8)


def black_frame():
    """Closed shutter or unlit scene: every pixel is dark"""
    return np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)


def filament_frame(thickness=14, angle=0.0):
    """
    A dark filament crossing the frame from edge to edge.

    The filament is always drawn across the whole width, the way it looks
    between the extruder and the puller in the real setup.

    Args:
        thickness: filament width in pixels, measured across the filament.
        angle: tilt of the filament in degrees.
    """
    frame = white_frame()
    half_rise = int(np.tan(np.radians(angle)) * WIDTH / 2)
    cv2.line(
        frame,
        (0, HEIGHT // 2 - half_rise),
        (WIDTH, HEIGHT // 2 + half_rise),
        (0, 0, 0),
        thickness,
    )
    return frame


def noisy_frame():
    """Pure sensor noise, as seen on a disconnected input"""
    return np.random.default_rng(0).integers(0, 256, (HEIGHT, WIDTH, 3), dtype=np.uint8)


def grayscale_frame():
    """Single channel frame, delivered by some cameras and codecs"""
    return cv2.cvtColor(filament_frame(), cv2.COLOR_BGR2GRAY)


def bgra_frame():
    """Four channel frame with an alpha layer"""
    return cv2.cvtColor(filament_frame(), cv2.COLOR_BGR2BGRA)


def float_frame():
    """Frame that arrived as floats instead of bytes"""
    return filament_frame().astype(np.float32)


def tiny_frame():
    """Frame far smaller than expected"""
    return np.zeros((2, 2, 3), dtype=np.uint8)


def empty_frame():
    """Frame with no pixels at all"""
    return np.zeros((0, 0, 3), dtype=np.uint8)


# Everything the pipeline may be fed with. None stands for a failed read.
BROKEN_FRAMES = [
    ("none", None),
    ("empty", empty_frame()),
    ("tiny", tiny_frame()),
    ("white", white_frame()),
    ("black", black_frame()),
    ("noise", noisy_frame()),
    ("grayscale", grayscale_frame()),
    ("bgra", bgra_frame()),
    ("float", float_frame()),
]
