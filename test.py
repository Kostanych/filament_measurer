# import PySimpleGUI as sg
# video_name = sg.popup_get_file('Please enter a video name')
# layout = [[sg.Text('Video'), sg.Video(filename=video_name)],
#             [sg.Button('Start Video')],
#             [sg.Text('Output Parameter 1: '), sg.Text('', size=(15,1), key='parameter1') ],
#             [sg.Text('Output Parameter 2: '), sg.Text('', size=(15,1), key='parameter2')]
#             ]
# window = sg.Window('User Interface', layout)
# while True:
#     event, values = window.read()
#     if event in (None, 'Exit'):
#         break
#     if event == 'Start Video':
#         # update output parameters
#         window['parameter1'].update('Parameter 1 text')
#         window['parameter2'].update('Parameter 2 text')
#     window.close()

import PySimpleGUI as sg
import cv2
import numpy as np

"""
Demo program that displays a webcam using OpenCV
"""


def main():

    sg.theme('DarkBrown4')

    # define the window layout
    layout = [[sg.Text('OpenCV Demo', size=(40, 1), justification='center', font='Helvetica 20')],
              [sg.Image(filename='', key='image')],
              [sg.Button('Play', size=(10, 1), font='Helvetica 14'),
               sg.Button('Stop', size=(10, 1), font='Any 14'),
               sg.Button('Exit', size=(10, 1), font='Helvetica 14'), ],
              [sg.Text('Mean width: '), sg.Text('', size=(15,1), key='parameter2')]
              ]

    # create the window and show it without the plot
    window = sg.Window('Demo Application - OpenCV Integration',
                       layout, location=(800, 400))

    # ---===--- Event LOOP Read and display frames, operate the GUI --- #
    video_name = sg.popup_get_file('Please enter a video name')
    cap = cv2.VideoCapture(video_name)
    Play = False

    while True:
        event, values = window.read(timeout=20)
        if event == 'Exit' or event == sg.WIN_CLOSED:
            return

        elif event == 'Play':
            Play = True
            window['parameter2'].update('Parameter 2 text')

        elif event == 'Stop':
            Play = False
            img = np.full((480, 640), 255)
            # this is faster, shorter and needs less includes
            imgbytes = cv2.imencode('.png', img)[1].tobytes()
            window['image'].update(data=imgbytes)

        if Play:
            ret, frame = cap.read()
            imgbytes = cv2.imencode('.png', frame)[1].tobytes()  # ditto
            window['image'].update(data=imgbytes)


main()
