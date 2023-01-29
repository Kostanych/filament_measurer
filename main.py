import argparse
import sys

import cv2
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from utils import functions

parser = argparse.ArgumentParser()
parser.add_argument("data_path", type=str, default="/home/kos/dev/popeyed_rod_measurer/data/", nargs='?',
                    help="Name of the folder with the data")

try:
    opt = parser.parse_known_args()[0]
except Exception as e:
    print(repr(e))
    parser.print_help()
    sys.exit(0)

data_path = opt.data_path


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


def process_image(image='photo_1.jpg'):
    full_path = opt.data_path + f"input/{image}"
    print('Running inference for {}... '.format(full_path), end='')

    # _image = np.array(full_path).astype('float32')
    # image_new = cv2.resize(_image, interpolation=cv2.INTER_CUBIC)

    image_np = load_image_into_numpy_array(full_path)
    image_color = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
    # image_grayscale = cv2.cvtColor(image_np, cv2.IMREAD_GRAYSCALE)
    # image_np = image_np.astype(np.float32)
    # print(image_np)

    # image_np_with_detections = image_np.copy()

    # #make mask(color)
    min_p = (0, 0, 0)
    max_p = (250, 250, 250)
    mask = cv2.inRange(image_color, min_p, max_p)

    # # make mask(grayscale)
    # white = 255
    # gray = 100
    # # Finds dark pixels
    # mask = cv2.inRange(image_grayscale, gray, white)

    plt.figure()
    cv2.imwrite(opt.data_path + 'output/result.jpeg', image_np)
    # cv2.imshow('Output', image_np)
    cv2.imshow('Output', mask)
    cv2.waitKey(0)

    print('Done')


process_image()

# functions.krutilki(opt.data_path + f"input/photo_1.jpg")



