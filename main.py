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


def process_image(image='photo_1.jpg', color=True):
    # while True:
    #     full_path = opt.data_path + f"input/{image}"
    #     # print('Running inference for {}... '.format(full_path), end='')
    #     # _image = np.array(full_path).astype('float32')
    #     # image_new = cv2.resize(_image, interpolation=cv2.INTER_CUBIC)
    #
    #     image_np = load_image_into_numpy_array(full_path)
    #
    #     if color:
    #         image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
    #         min_p = (0, 0, 0)
    #         max_p = (250, 250, 250)
    #         mask = cv2.inRange(image_np, min_p, max_p)
    #     else:
    #         image_np = cv2.cvtColor(image_np, cv2.IMREAD_GRAYSCALE)
    #         min_p = 0
    #         max_p = 250
    #         mask = cv2.inRange(image_np, min_p, max_p)
    #     # image_np = image_np.astype(np.float32)
    #     # print(image_np)
    #
    #     # image_np_with_detections = image_np.copy()
    #
    #     # #make mask(color)
    #
    #     # # make mask(grayscale)
    #     # white = 255
    #     # gray = 100
    #     # # Finds dark pixels
    #     # mask = cv2.inRange(image_grayscale, gray, white)
    #
    #     plt.figure()
    #     cv2.imwrite(opt.data_path + 'output/result.jpeg', image_np)
    #     # cv2.imshow('Output', image_np)
    #     cv2.imshow('Output', mask)
    #     if cv2.waitKey(1) & 0xFF == ord('q'):
    #         # process_image(image_np)
    #         break

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

    cv2.imshow('mask', mask)
    cv2.waitKey(0)

    cv2.imshow('image', image_np)
    cv2.waitKey(0)

    print('Start to prosess mask')
    mask_processed = functions.process_mask(mask)
    print('Complete!')
    cv2.imshow('mask_processed', mask_processed)
    cv2.waitKey(0)

    # closing all open windows
    cv2.destroyAllWindows()

    # print(image_np.shape)
    # print(mask.shape)
    # print(mask_processed.shape)

    print('Done')
    return mask_processed


mask = process_image()
test = mask.T

# print(pd.DataFrame(test))
for col in test:
    #
    row = row.ne(row.shift()).cumsum()
    row = (row != row[0]) & (row != row[row.shape[0] - 1])
    row.astype(int).replace(1, 255)
    # for row in col:
    print(col)

row_orig = pd.Series([0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0])



# img = mask
# ret, thresh = cv2.threshold(img, 127, 255, 0)
# contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#
# cv2.drawContours(img, contours, -1, (0,255,0), 3)


#
# # functions.krutilki(opt.data_path + f"input/photo_1.jpg")
#
# # image_np = load_image_into_numpy_array(full_path)
# img = load_image_into_numpy_array(full_path)
# image_grayscale = cv2.cvtColor(img, cv2.IMREAD_GRAYSCALE)
#
# ret, thresh = cv2.threshold(mask, 0, 250, 0, cv2.THRESH_BINARY)
# # contours, hierarchy = cv2.findContours(thresh, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE) # Use cv2.CCOMP for two level hierarchy
# #cv2.drawContours(img, contours, -1, (0,255,0), 2)
#
# # create an empty mask
# mask = np.zeros(img.shape[:2], dtype=np.uint8)
#
# contours, hierarchy = cv2.findContours(thresh, cv2.RETR_CCOMP,
#                                        cv2.CHAIN_APPROX_SIMPLE)  # Use cv2.CCOMP for two level hierarchy
# # loop through the contours
# for i, cnt in enumerate(contours):
#
#     if hierarchy[0][i][3] != -1:  # basically look for holes
#         # if the size of the contour is less than a threshold (noise)
#         if cv2.contourArea(cnt) < 70:
#             # Fill the holes in the original image
#             cv2.drawContours(img, [cnt], 0, (255), -1)
#
# cv2.imshow("Mask", mask)
#
# cv2.imshow("Img", img)
# image = cv2.bitwise_not(img, img, mask=mask)
# cv2.imshow("Mask", mask)
# cv2.imshow("After", image)
#
# cv2.waitKey()
# cv2.destroyAllWindows()
#
