import cv2
import pandas as pd
import numpy as np


def nothing(args): pass


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
