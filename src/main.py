import argparse
import io

import PySimpleGUI as sg
from PySimpleGUI import *
from PIL import Image

import gui

parser = argparse.ArgumentParser()
parser.add_argument("data_path", type=str, default="/home/kos/dev/filament_measurer/data/", nargs='?',
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


g = gui.Gui(opt)
g.run_gui()
