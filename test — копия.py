import argparse
import sys

import cv2
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

# from utils import functions
from utils.functions import measure_length, process_mask

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
full_path = opt.data_path + f"input/photo_1.jpg"


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


full_path = "C:\\Users\\KOS\\Documents\\dev\\popeyed_rod_measurer\\data\\input\\photo_1.jpg"
color = True
image_np = load_image_into_numpy_array(full_path)

row1 = [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
row2 = [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0]
borders = [row1, row2]


def keep_border_values(row):
    """
    Keep only border values in the list.
    :param row: input list\Series
    :return: processed 2d list
    """
    # to 1d
    # row = [i[0] for i in row]
    row = pd.Series(row)
    row_shift = row.ne(row.shift()).cumsum().astype('uint8')
    borders = [row_shift[0], row_shift[len(row_shift) - 1]]
    print(f"borders: {borders}")
    print(f'row: {np.array(row)}')

    # wr_0 = wr_0.index
    # row_1 = row_1.append(wr_0)
    # row_res = row_res.isin(row_1)


    # print(np.array(row_shift))
    ind = row_shift[row_shift.isin(borders)].index
    print(ind)
    print(ind.unique())
    r = pd.Series(np.zeros(row.shape[0]))
    r.iloc[ind]=255
    # print(r)
    # r = [255 if r.index in ind else 0]
    # r = row.apply(lambda x: 0 if x.index in ind else 255).astype('uint8')
    print(f"row after: {np.array(r)}")

    # return np.stack((r,) * 3, axis=-1)
    return r


# print(keep_border_values(row1))

def prepare_contour(img: np.array):
    """
    Takes first and last columns of image and fill it.
    :param img:
    :return:
    """
    img = img.T
    print(img.shape)
    first_row_n = 0
    last_row_n = len(img) - 1
    first_row = img[0]
    last_row = img[last_row_n]
    for (i, row) in zip([first_row_n, last_row_n], [first_row, last_row]):
        print(f"i   {i}")
        print(f"row {row}")
        # print(img[i])
        img[i] = keep_border_values(row)
    # for n in img[0]:
    #     print(n)
    return img.T

full_path = "C:\\Users\\KOS\\Documents\\dev\\popeyed_rod_measurer\\data\\input\\photo_1.jpg"
# print('Running inference for {}... '.format(full_path), end='')
# _image = np.array(full_path).astype('float32')
# image_new = cv2.resize(_image, interpolation=cv2.INTER_CUBIC)

image_np = load_image_into_numpy_array(full_path)

if color:
    image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
    min_p = (0, 0, 0)
    max_p = (250, 250, 250)
    mask = cv2.inRange(image_np, min_p, max_p)
else:
    image_np = cv2.cvtColor(image_np, cv2.IMREAD_GRAYSCALE)
    min_p = 0
    max_p = 250
    mask = cv2.inRange(image_np, min_p, max_p)


prepare_contour(mask)


