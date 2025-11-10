"""Configuration holder for Filament Measurer application"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, os.getcwd())
PROJECT_DIR = Path(__file__).resolve().parent.parent


class Config:
    """Configuration class with all application constants"""

    # Paths
    DATA_PATH = PROJECT_DIR / 'data'
    INPUT_PATH = DATA_PATH / 'input'
    OUTPUT_PATH = DATA_PATH / 'output'

    # Image processing
    DEFAULT_THRESHOLD = 127
    IMAGE_WIDTH = 640
    IMAGE_HEIGHT = 480
    IMAGE_SIZE = (IMAGE_WIDTH, IMAGE_HEIGHT)

    # Measurement defaults
    DEFAULT_REFERENCE_WIDTH_MM = 1.75
    DEFAULT_FPS = 24

    # Rolling windows (seconds)
    ROLLING_WINDOW_SHORT = 1
    ROLLING_WINDOW_LONG = 10

    # Plot settings
    PLOT_WIDTH = 1000
    PLOT_Y_MARGIN = 0.2
    PLOT_HISTORY_SECONDS = 2
    FONT_SIZE_LARGE = 20

    # Layout settings
    COL_WIDTHS_MAIN = [0.3, 0.2, 0.2]
    COL_WIDTHS_PLOT = [0.8, 0.2]

    # Logging
    LOG_LEVEL_DEBUG = "DEBUG"
    LOG_LEVEL_INFO = "INFO"


config = Config()
