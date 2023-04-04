import cv2
import PySimpleGUI as sg
import numpy as np
import utils.functions as fn


class Gui:
    def __init__(self, opt):
        self.opt = opt
        self.width = 0
        self.show_every_n_frame = 1
        self.mask_or_image = 'image'
        self.play = False

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
            [sg.Button('Load video'), sg.Button('Play'), sg.Button('Stop'), sg.Button('Exit'),
             sg.Button('Calibrate'), sg.Button('Show 10% of frames'), sg.Button('Show 100% of frames'),
             sg.Button('Mask/Image')],
            [sg.Text('Mean width in pixels: '), sg.Text('', size=(15, 1), key='width_value_pxl')],
            [sg.Text('Mean width in mm:     '), sg.Text('', size=(15, 1), key='width_value_mm')]
        ]

        # Create the UI window
        window = sg.Window('Video Player', layout)

        # Initialize the video player
        cap = None
        # play = False
        fps = 30
        # show_every_n_frame = 1

        # Main event loop
        while True:
            event, values = window.read(timeout=1000 // fps)  # Update the UI every frame
            # Set main image first frame of current video, or blank image if there was no video
            if not self.play: window['image'].update(data=self.title_frame[1].tobytes())
            # Set blank frame
            # img = np.full((480, 640), 255)
            # this is faster, shorter and needs less includes
            # self.title_frame = cv2.imencode('.png', img)[1].tobytes()
            # window['image'].update(data=self.title_frame)

            if event == sg.WINDOW_CLOSED or event == 'Exit':
                break
            elif event == 'Load video':
                print('load video')
                # Get the filename of the video
                filename = sg.popup_get_file('Choose a video file')
                if filename:
                    # Load the video
                    cap = cv2.VideoCapture(filename)
                    fps = int(cap.get(cv2.CAP_PROP_FPS))
                    # Set the initial frame
                    ret, frame = cap.read()
                    # Set the default title image from the 1st frame of the video
                    self.title_frame = cv2.imencode('.png', frame)
                    window['image'].update(data=self.title_frame[1].tobytes())
            elif event == 'Play':
                self.play = True

            elif event == 'Stop':
                self.play = False
                img = np.full((480, 640), 255)
                # this is faster, shorter and needs less includes
                imgbytes = cv2.imencode('.png', img)[1].tobytes()
                window['image'].update(data=imgbytes)

            elif event == 'Show 10% of frames':
                # Show every 10th frame
                self.show_every_n_frame = 10

            elif event == 'Show 100% of frames':
                # Show all frames
                self.show_every_n_frame = 1

            elif event == 'Mask/Image':
                self.mask_or_image = self.switch_param()

            elif event == 'Calibrate':
                # Get the filename of the video
                filename = sg.popup_get_file('Choose a video file')
                if filename:
                    # Load the video
                    cap = cv2.VideoCapture(filename)
                    fps = int(cap.get(cv2.CAP_PROP_FPS))
                    # Set the initial frame
                    ret, frame = cap.read()
                    # Set the default title image from the 1st frame of the video
                    self.title_frame = cv2.imencode('.png', frame)
                    window['image'].update(data=self.title_frame[1].tobytes())

                ret, frame = cap.read()
                if not ret:
                    print("Can't receive frame (stream end?). Exiting ...")
                    break
                mask, self.width = fn.process_image(image=frame, verbose=0)
                if self.mask_or_image == 'image':
                    source = frame
                else:
                    source = mask
                imgbytes = cv2.imencode('.png', source)[1].tobytes()  # ditto
                window['image'].update(data=imgbytes)
                calib_multiplier = self.opt.calib_width_mm / self.width
                print(f"calib_multiplier: {calib_multiplier}")
                print(f"opt.calib_width_mm : {self.opt.calib_width_mm}")
                print(f"width: {self.width}")

            if cap is not None and self.play:
                # Read the next frame
                ret, frame = cap.read()
                if ret:
                    mask, self.width = fn.process_image(image=frame, verbose=0)
                    # Show the frame if needed
                    if cap.get(cv2.CAP_PROP_POS_FRAMES) % self.show_every_n_frame == 0:
                        if self.mask_or_image == 'image':
                            source = frame
                        else:
                            source = mask

                        window['image'].update(data=cv2.imencode('.png', source)[1].tobytes())
                        window['width_value_pxl'].update(self.width)
                        window['width_value_mm'].update(self.width * calib_multiplier)
                else:
                    # End of video reached
                    cap.release()
                    cap = None
                    self.play = False
                    window['image'].update(filename='')


        # Clean up
        if cap is not None:
            cap.release()
        # self.eof = True
        window.close()
