import cv2
import pandas as pd
import numpy as np

from src.mask_processor import process_contours, measure_length


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


def process_image(image, color=True, verbose=0):
    image_np = np.array(image)

    if color:
        image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
        min_p = (10, 10, 10)
        max_p = (240, 240, 240)
        mask = cv2.inRange(image_np, min_p, max_p)
    else:
        image_np = cv2.cvtColor(image_np, cv2.IMREAD_GRAYSCALE)
        min_p = 0
        max_p = 250
        mask = cv2.inRange(image_np, min_p, max_p)



    # full_path = "C:\\Users\\KOS\\Documents\\dev\\popeyed_rod_measurer\\data\\input\\photo_1.jpg"
    # image_np = load_image_into_numpy_array(full_path)
    #
    # image = skimage.color.rgb2gray(image)
    # image_colour = image
    # image[:, 0] = 1
    # image[:, -1] = 1
    #
    # # Find contours at a constant value of 0.8
    # contours = measure.find_contours(image, 0.8)
    #
    # # Find largest contour
    # largest_contour = max(contours, key=lambda contour: len(contour))
    # print('Found max contour')
    #
    # # Make mask
    # mask = polygon2mask(image.shape, largest_contour)
    # mask = invert(convex_hull_image(mask))
    # mask_processed = mask.astype(float)

    if verbose:
        cv2.imshow('image', image_np)
        cv2.waitKey(0)

        cv2.imshow('mask', mask)
        cv2.waitKey(0)

        # print('Start to prosess mask')
        mask_processed = prepare_borders(mask)
        mask_processed = process_contours(mask_processed)
        print('Complete!')
        cv2.imshow('mask_processed', mask_processed)
        cv2.waitKey(0)

        print("Start to compute count of pixels.")
        width = measure_length(mask_processed)
        print('Done')
    else:
        mask_processed = prepare_borders(mask)
        mask_processed = process_contours(mask_processed)
        width = measure_length(mask_processed)

    # closing all open windows
    cv2.destroyAllWindows()

    return mask_processed, width


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

    img[:, 0] = 1
    img[:, -1] = 1
    return img





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

# krutilki("C:\\Users\\KOS\\Documents\\dev\\popeyed_rod_measurer\\data\\input\\photo_1.jpg")
