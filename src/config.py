"""Configuration holder """

import os
import sys

from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, os.getcwd())
PROJECT_DIR = Path(__file__).resolve().parent.parent


class Config:
    DATA_PATH = ''
