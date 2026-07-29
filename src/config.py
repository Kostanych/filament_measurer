"""Configuration holder for Filament Measurer application"""

from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent


class Config:
    """Configuration class with all application constants"""

    # Paths
    DATA_PATH = PROJECT_DIR / "data"
    INPUT_PATH = DATA_PATH / "input"
    OUTPUT_PATH = DATA_PATH / "output"

    # Image processing
    BINARY_THRESHOLD = 127
    BINARY_MAX_VALUE = 255
    IMAGE_WIDTH = 640
    IMAGE_HEIGHT = 480
    IMAGE_SHAPE = (IMAGE_HEIGHT, IMAGE_WIDTH, 3)

    # Measurement defaults
    DEFAULT_REFERENCE_WIDTH_MM = 1.75
    DEFAULT_FPS = 24
    DEFAULT_WIDTH_MULTIPLIER = 0.005
    FALLBACK_WIDTH_PXL = 160

    # A frame with that many dark pixels is not a filament but a garbage frame
    # (closed shutter, camera warming up). Fitting a line into it is pointless
    # and slow, so the angle is not measured at all.
    MAX_FILAMENT_AREA_FRACTION = 0.5

    # Rolling windows (seconds)
    ROLLING_WINDOW_SHORT = 1
    ROLLING_WINDOW_LONG = 10

    # Upper bound of the stored measurements, keeps memory usage constant
    # during a long run. 36000 frames is about 20 minutes at 30 fps.
    MAX_MEASUREMENT_HISTORY = 36000

    # Plot/UI refresh intervals in seconds. Zero means "on every frame"
    UPDATE_INTERVALS = {"Every Frame": 0, "1 Second": 1, "5 Seconds": 5}

    # Plot settings
    PLOT_WIDTH = 1000
    PLOT_Y_MARGIN = 0.2
    PLOT_HISTORY_SECONDS = 2
    FONT_SIZE_LARGE = 20

    # Layout settings
    COL_WIDTHS_MAIN = [0.3, 0.2, 0.2]
    COL_WIDTHS_PLOT = [0.8, 0.2]


config = Config()
