import argparse
import io
import sys

import cv2
import numpy as np
import PySimpleGUI as sg
from PySimpleGUI import *
from PIL import Image

import gui

# from utils import functions
from utils.functions import measure_length, process_mask, prepare_borders, process_contours

parser = argparse.ArgumentParser()
parser.add_argument("data_path", type=str, default="/home/kos/dev/popeyed_rod_measurer/data/", nargs='?',
                    help="Name of the folder with the data")
parser.add_argument("calibration", type=str, default=False, nargs='?',
                    help="If this video for the calibration")
parser.add_argument("calib_width_mm", type=str, default=1.7, nargs='?',
                    help="Value of the calibration object width, in millimeters")

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





# mask, width = process_image(verbose=0)
# a = cv2.imwrite(opt.data_path + "C:\\Users\\KOS\\Documents\\dev\\popeyed_rod_measurer\\data\\output\\photo_1.jpg", mask)
# img = mask.copy()

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


# show_image()

def calibrate_width():
    pass


# def launch_ui():
#     sg.theme('DarkBrown4')
#     calib_multiplier = 0
#     # define the window layout
#     layout = [[sg.Text('Лупоглазый пруткомер', size=(40, 1), justification='center', font='Helvetica 20')],
#               [sg.Image(filename='', key='image')],
#               [sg.Button('Play', size=(10, 1), font='Helvetica 14'),
#                sg.Button('Stop', size=(10, 1), font='Any 14'),
#                sg.Button('Exit', size=(10, 1), font='Helvetica 14'),
#                sg.Button('Calibrate', size=(10, 1), font='Helvetica 14'), ],
#               [sg.Text('Mean width in pixels: '), sg.Text('', size=(15, 1), key='width_value_pxl')],
#               [sg.Text('Mean width in mm:     '), sg.Text('', size=(15, 1), key='width_value_mm')]
#               ]
#
#     # create the window and show it without the plot
#     window = sg.Window('Demo Application - OpenCV Integration',
#                        layout, location=(800, 400))
#
#     # ---===--- Event LOOP Read and display frames, operate the GUI --- #
#     video_name = sg.popup_get_file('Please enter a video name')
#     cap = cv2.VideoCapture(video_name)
#     Play = False
#
#     while True:
#         event, values = window.read(timeout=20)
#
#         # Set blank frame
#         img = np.full((480, 640), 255)
#         # this is faster, shorter and needs less includes
#         imgbytes = cv2.imencode('.png', img)[1].tobytes()
#         window['image'].update(data=imgbytes)
#
#         if event == 'Exit' or event == sg.WIN_CLOSED:
#             return
#
#         elif event == 'Play':
#             Play = True
#
#         elif event == 'Stop':
#             Play = False
#             img = np.full((480, 640), 255)
#             # this is faster, shorter and needs less includes
#             imgbytes = cv2.imencode('.png', img)[1].tobytes()
#             window['image'].update(data=imgbytes)
#
#         elif event == 'Calibrate':
#             ret, frame = cap.read()
#             if not ret:
#                 print("Can't receive frame (stream end?). Exiting ...")
#                 break
#             mask, width = process_image(image=frame, verbose=0)
#             imgbytes = cv2.imencode('.png', mask)[1].tobytes()  # ditto
#             window['image'].update(data=imgbytes)
#             calib_multiplier = opt.calib_width_mm / width
#
#             # print(f"calib_multiplier: {calib_multiplier}")
#             # print(f"opt.calib_width_mm : {opt.calib_width_mm}")
#             # print(f"width: {width}")
#
#
#
#         if Play:
#             ret, frame = cap.read()
#             if not ret:
#                 print("Can't receive frame (stream end?). Exiting ...")
#                 break
#             mask, width = process_image(image=frame, verbose=0)
#             imgbytes = cv2.imencode('.png', mask)[1].tobytes()  # ditto
#             window['image'].update(data=imgbytes)
#
#             window['width_value_pxl'].update(width)
#             window['width_value_mm'].update(width * calib_multiplier)


# launch_ui()


g = gui.Gui(opt)
g.run_gui()
#
# import PySimpleGUI as sg
# import cv2
# def Play():
#     video_name = sg.popup_get_file('Please enter a video name')
#     cap = cv2.VideoCapture(video_name)
#     while(True):
#         # Capture frame-by-frame
#         ret, frame = cap.read()
#          # Display the resulting frame
#         cv2.imshow('frame',frame)
#          # Press q to close the video windows before it ends
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break
#     cap.release()
#     cv2.destroyAllWindows()
# layout = [[sg.Button('Play Video')]]
#
#
# window = sg.Window('Video Player', layout)
# while True:
#      event, values = window.read()
#      if event == 'Play Video':
#         Play()
#      elif event == sg.WIN_CLOSED:
#         window.close()
#         break
