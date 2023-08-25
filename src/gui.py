import argparse

import cv2
import PySimpleGUI as sg
import numpy as np
import image_processor as fn


class Gui:
    def __init__(self, opt):
        self.opt = opt
        self.width = 0
        self.show_every_n_frame = 1
        self.mask_or_image = 'image'
        self.play = False
        self.filename = ''

        # Set blank start frame
        img = np.full((480, 640), 255)
        self.title_frame = cv2.imencode('.png', img)

    def switch_param(self):
        if self.mask_or_image == 'image':
            return 'mask'
        else:
            return 'image'

    def run_gui(self):
        # sg.theme('DarkBrown4')
        sg.theme('DarkAmber')
        calib_multiplier = 0

        # Define the layout of the UI
        layout = [
            [sg.Text('Лупоглазый пруткомер', size=(40, 1), justification='center', font='Helvetica 20')],
            [sg.Image(filename='', key='image')],
            [sg.Button('Select calibration video')],
            [sg.Button('Change calibration width'),
             sg.InputText(size=(3, 1), key='calibration_input'),
             sg.Text(f'{self.opt.calib_width_mm}', size=(2, 1), key='calibration_value')
             ],
            [sg.Combo(['File', 'USB device'], default_value='File', key='input_source'),
             sg.Button('Load video'),
             sg.Button('Play'),
             sg.Button('Stop'),
             ],
            [sg.Button('Show 10% of frames'), sg.Button('Show 100% of frames'), sg.Button('Mask/Image')],
            [sg.Button('Exit')],
            [sg.Text('Mean width in pixels: '), sg.Text('', size=(15, 1), key='width_value_pxl')],
            [sg.Text('Mean width in mm:     '), sg.Text('', size=(15, 1), key='width_value_mm')]
        ]

        # Set the initial window location (x, y coordinates)
        window_location = (100, 100)  # Adjust these coordinates as needed

        # Create the UI window
        window = sg.Window('Молодец, нашёл', layout, location=window_location)

        # Initialize variables
        cap = None
        show_play_button = True
        fps = 30

        # Main event loop
        while True:
            event, values = window.read(timeout=1000 // fps)  # Update the UI every frame
            # Set main image first frame of current video, or blank image if there was no video
            if not self.play:
                window['image'].update(data=self.title_frame[1].tobytes())
            if event == sg.WINDOW_CLOSED or event == 'Exit':
                break
            elif (event == 'Load video') & (values['input_source'] == 'File'):
                print('load video')
                # Get the filename of the video
                self.filename = sg.popup_get_file('Choose a video file')
                if self.filename:
                    # Load the video
                    cap = cv2.VideoCapture(self.filename)
                    fps = int(cap.get(cv2.CAP_PROP_FPS))
                    # Set the initial frame
                    ret, frame = cap.read()
                    # Set the default title image from the 1st frame of the video
                    self.title_frame = cv2.imencode('.png', frame)
                    window['image'].update(data=self.title_frame[1].tobytes())

            elif event == 'Play':
                print(f'cap  = {cap}')
                print(f'play = {self.play}')
                if (not self.play) & (self.filename != '') & (values['input_source'] == 'File'):
                    cap = cv2.VideoCapture(self.filename)
                elif values['input_source'] == 'USB device':
                    cap = cv2.VideoCapture(0)
                self.play = True

            elif event == 'Stop':
                self.play = False
                img = np.full((480, 640), 255)
                # this is faster, shorter and needs less includes
                imgbytes = cv2.imencode('.png', img)[1].tobytes()
                window['image'].update(data=imgbytes)
                show_play_button = True

            elif event == 'Show 10% of frames':
                # Show every 10th frame
                self.show_every_n_frame = 10

            elif event == 'Show 100% of frames':
                # Show all frames
                self.show_every_n_frame = 1

            elif event == 'Mask/Image':
                self.mask_or_image = self.switch_param()

            elif event == 'Change multiplier':
                calibration_value = values['calibration_input']
                try:
                    calibration_value = float(calibration_value)
                    self.opt.calib_width_mm = calibration_value
                    window['calibration_value'].update(calibration_value)
                except Exception:
                    print("It must be 'float' datatype")

            elif event == 'Select calibration video':
                # Get the filename of the video
                self.filename = sg.popup_get_file('Choose a video file')
                if self.filename:
                    # Load the video
                    cap = cv2.VideoCapture(self.filename)
                    fps = int(cap.get(cv2.CAP_PROP_FPS))
                    # Set the initial frame
                    ret, frame = cap.read()
                    # Set the default title image from the 1st frame of the video
                    self.title_frame = cv2.imencode('.png', frame)
                    window['image'].update(data=self.title_frame[1].tobytes())

                    mask, self.width = fn.process_image(frame=frame, verbose=0)
                    if self.mask_or_image == 'image':
                        source = frame
                    else:
                        source = mask
                    imgbytes = cv2.imencode('.png', source)[1].tobytes()
                    window['image'].update(data=imgbytes)
                    calib_multiplier = self.opt.calib_width_mm / self.width
                    print(f"calib_multiplier: {calib_multiplier}")
                    print(f"opt.calib_width_mm : {self.opt.calib_width_mm}")
                    print(f"width: {self.width}")

            # Start the video
            if cap and self.play:
                # Read the next frame
                ret, frame = cap.read()
                show_play_button = False
                if ret:
                    mask, self.width = fn.process_image(frame=frame, verbose=0)
                    # Show the frame if needed
                    if cap.get(cv2.CAP_PROP_POS_FRAMES) % self.show_every_n_frame == 0:
                        if self.mask_or_image == 'image':
                            source = frame
                        else:
                            source = mask

                        window['image'].update(data=cv2.imencode('.png', source)[1].tobytes())
                        window['width_value_pxl'].update(round(self.width, 0))
                        window['width_value_mm'].update(round(self.width * calib_multiplier, 3))
                else:
                    # End of video reached
                    cap.release()
                    cap = None
                    show_play_button = True
                    self.play = False
                    window['image'].update(filename='')

            # Update button visibility based on conditions
            if show_play_button is True:
                window['Play'].update(visible=True)
                window['Stop'].update(visible=False)
            else:
                window['Play'].update(visible=False)
                window['Stop'].update(visible=True)

        # Clean up
        if cap is not None:
            cap.release()
        # self.eof = True
        window.close()



