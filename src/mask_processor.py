import cv2
import numpy as np
import pandas as pd


def measure_length(img: np.array):
    all_pixels = img.shape[0] * img.shape[1]
    non_zero = np.count_nonzero(img)
    avg_pixels = non_zero / img.shape[1]
    print(f"Count of non-zero pixels:  {non_zero}")
    print(f"Count of all pixels:       {all_pixels}")
    print(f"Average width in pixels:   {avg_pixels}")
    return avg_pixels


def process_mask(img: np.array):
    processed = img.copy().T
    for i, row in enumerate(processed):
        # this is awful
        processed[i] = process_column(pd.Series(row), 1)
        processed[i] = process_column(pd.Series(row), 5)
        processed[i] = process_column(pd.Series(row), 100)
        processed[i] = process_column(pd.Series(row), 100)
    return processed.T


def process_column(row_orig: pd.Series, threshhold=10):
    row = row_orig.copy()
    row = row.astype(bool)
    row_res = row.ne(row.shift()).cumsum()
    row_base = (row_res.value_counts(normalize=False).rename('c').to_frame())

    # # find correct and wrong '0'
    # row_0 = row_base[(row_base.index % 2 != 0) & (row_base.c > threshhold)]
    # row_0 = row_0.index
    wr_0 = row_base[(row_base.c <= threshhold) & (row_base.index % 2 != 0)]
    wr_0 = wr_0.index

    # find correct and wrong "1"
    row_1 = row_base[(row_base.index % 2 == 0) & (row_base.c > threshhold)]
    row_1 = row_1.index
    # wr_1 = row_base[(row_base.c <= threshhold) & (row_base.index % 2 == 0)]
    # wr_1 = wr_1.index

    # get true '1' and true '0' pixels
    # row_0 = row_0.append(wr_1)
    row_1 = row_1.append(wr_0)

    # print("row_0")
    # print(row_0)
    # print('row_1')
    # print(row_1)

    # clear row of wrong pixels
    row_res = row_res.isin(row_1)
    # row_res = ~row_res.isin(row_0)
    row_res = row_res.astype(int).replace(1, 255)

    # print(row_res[415:425])

    return row_res


def process_contours(mask):
    # Find the contours of the objects in the image
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    # Find the index of the biggest contour
    max_contour = 0
    for i in range(len(contours)):
        if cv2.contourArea(contours[i]) > cv2.contourArea(contours[max_contour]):
            max_contour = i
            # print(f"NEW MAX CONTOUR    {cv2.contourArea(contours[i])}")
            # print(f"NEW MAX CONTOUR ID {max_contour}")
    # # Create a mask with the biggest contour
    contour_mask = np.zeros_like(mask)
    cv2.fillPoly(contour_mask, pts=[contours[max_contour]], color=(255, 255, 255))
    return contour_mask


