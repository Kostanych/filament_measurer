import cv2
import pandas as pd
import numpy as np


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
    # print(f"borders: {borders}")
    # print(f'row: {np.array(row)}')

    ind = row_shift[row_shift.isin(borders)].index

    r = pd.Series(255, range(row.shape[0]))
    r.iloc[ind] = 0

    return r.astype('uint8')


def prepare_borders(img: np.array):
    """
    Takes first and last columns of image and fill it.
    :param img:
    :return:
    """
    img = img.T
    # print(img.shape)
    first_row_n = 0
    last_row_n = len(img) - 1
    first_row = img[0]
    last_row = img[last_row_n]
    for (i, row) in zip([first_row_n, last_row_n], [first_row, last_row]):
        # print(f"i   {i}")
        # print(f"row {row}")
        # print(img[i])
        img[i] = keep_border_values(row)
    # for n in img[0]:
    #     print(n)
    return img.T


def measure_length(img: np.array):
    all_pixels = img.shape[0] * img.shape[1]
    non_zero = np.count_nonzero(img)
    avg_pixels = non_zero / img.shape[1]
    print(f"Count of non-zero pixels:  {non_zero}")
    print(f"Count of all pixels:       {all_pixels}")
    print(f"Average width in pixels:   {avg_pixels}")
    pass


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


def krutilki(filepath):
    # создаем окно для отображения результата и бегунки
    cv2.namedWindow("setup")
    cv2.createTrackbar("b1", "setup", 0, 255, nothing)
    cv2.createTrackbar("g1", "setup", 0, 255, nothing)
    cv2.createTrackbar("r1", "setup", 0, 255, nothing)
    cv2.createTrackbar("b2", "setup", 255, 255, nothing)
    cv2.createTrackbar("g2", "setup", 255, 255, nothing)
    cv2.createTrackbar("r2", "setup", 255, 255, nothing)

    img = cv2.imread(filepath)  # загрузка изображения

    while True:
        r1 = cv2.getTrackbarPos('r1', 'setup')
        g1 = cv2.getTrackbarPos('g1', 'setup')
        b1 = cv2.getTrackbarPos('b1', 'setup')
        r2 = cv2.getTrackbarPos('r2', 'setup')
        g2 = cv2.getTrackbarPos('g2', 'setup')
        b2 = cv2.getTrackbarPos('b2', 'setup')
        # собираем значения из бегунков в множества
        min_p = (g1, b1, r1)
        max_p = (g2, b2, r2)
        # применяем фильтр, делаем бинаризацию
        img_g = cv2.inRange(img, min_p, max_p)

        cv2.imshow('img', img_g)

        if cv2.waitKey(33) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()


def krutilki_grayscale(filepath):
    cv2.namedWindow("setup")
    cv2.createTrackbar("g1", "setup", 0, 255, nothing)
    cv2.createTrackbar("g2", "setup", 0, 255, nothing)

    img = cv2.imread(filepath)  # загрузка изображения

    while True:
        g1 = cv2.getTrackbarPos('g1', 'setup')
        g2 = cv2.getTrackbarPos('g2', 'setup')

        # собираем значения из бегунков в множества
        min_p = (g1)
        max_p = (g2)
        # применяем фильтр, делаем бинаризацию
        img_g = cv2.inRange(img, min_p, max_p)

        cv2.imshow('img', img_g)

        if cv2.waitKey(33) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()


def nothing(args): pass
