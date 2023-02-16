import argparse
import io
import sys

import cv2
import numpy as np
import PySimpleGUI as sg
from PySimpleGUI import *
from PIL import Image

# from utils import functions
from utils.functions import measure_length, process_mask, prepare_borders, process_contours

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


def process_image(image='photo_1.jpg', color=True, verbose=0):
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

    if verbose:
        cv2.imshow('image', image_np)
        cv2.waitKey(0)

        cv2.imshow('mask', mask)
        cv2.waitKey(0)

        print('Start to prosess mask')
        mask_processed = prepare_borders(mask)
        mask_processed = process_contours(mask_processed)
        print('Complete!')
        cv2.imshow('mask_processed', mask_processed)
        cv2.waitKey(0)

        print("Start to compute count of pixels.")
        measure_length(mask_processed)
        print('Done')
    else:
        mask_processed = prepare_borders(mask)
        mask_processed = process_contours(mask_processed)
        measure_length(mask_processed)

    # closing all open windows
    cv2.destroyAllWindows()

    return mask_processed


mask = process_image(verbose=0)
a = cv2.imwrite(opt.data_path + "C:\\Users\\KOS\\Documents\\dev\\popeyed_rod_measurer\\data\\output\\photo_1.jpg", mask)
img = mask.copy()




file_types = [("JPEG (*.jpg)", "*.jpg"),
              ("All files (*.*)", "*.*")]
def show_image():
    layout = [
        [sg.Image(key="-IMAGE-")],
        [
            sg.Text("Image File"),
            sg.Input(size=(25, 1), key="-FILE-"),
            sg.FileBrowse(file_types=file_types),
            sg.Button("Load Image"),
        ],
    ]

    window = sg.Window("Image Viewer", layout)

    while True:
        event, values = window.read()
        if event == "Exit" or event == sg.WIN_CLOSED:
            break
        if event == "Load Image":
            filename = values["-FILE-"]
            if os.path.exists(filename):
                image = Image.open(values["-FILE-"])
                image.thumbnail((400, 400))
                bio = io.BytesIO()
                image.save(bio, format="PNG")
                window["-IMAGE-"].update(data=bio.getvalue())

    window.close()


show_image()


#
import PySimpleGUI as sg
import cv2
def Play():
    video_name = sg.popup_get_file('Please enter a video name')
    cap = cv2.VideoCapture(video_name)
    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()
         # Display the resulting frame
        cv2.imshow('frame',frame)
         # Press q to close the video windows before it ends
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
layout = [[sg.Button('Play Video')]]


window = sg.Window('Video Player', layout)
while True:
     event, values = window.read()
     if event == 'Play Video':
        Play()
     elif event == sg.WIN_CLOSED:
        window.close()
        break





