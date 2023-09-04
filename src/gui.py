
import math

import PySimpleGUI as sg
from image_processor import *
from utils import get_logger, mean_rolling

logger = get_logger("GUI")


class Gui:
    def __init__(self, opt):
        self.opt = opt
        self.width = 0
        self.show_every_n_frame = 1
        self.mask_or_image = 'image'
        self.play = False
        self.filename = ''
        self.rolling_1s = 0

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
        width_multiplier_calibrated = 0

        # Define the layout of the UI
        layout = [
            [sg.Text('Лупоглазый пруткомер', size=(40, 1), justification='center', font='Helvetica 20')],
            [sg.Image(filename='', key='image')],
            # [sg.Button('Select calibration video')],
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
            [sg.Text('Mean width in pixels: '), sg.Text('N/A', size=(10, 1), key='width_value_pxl_angle'),
             sg.Text('Mean rolling 1 second: '), sg.Text('N/A', size=(10, 1), key='rolling_1s'),
             ],
            [sg.Text('Mean width in mm:    '), sg.Text('N/A', size=(10, 1), key='width_value_mm_angle'),
             sg.Text('Mean rolling 10 second: '), sg.Text('N/A', size=(10, 1), key='rolling_10s')
             ],
            [
             sg.Text('angle_multiplier: '), sg.Text('N/A', size=(10, 1), key='angle_multiplier'),
             ]
        ]

        # Set the initial window location (x, y coordinates)
        window_location = (100, 100)  # Adjust these coordinates as needed

        # Create the UI window
        window = sg.Window('Молодец, нашёл', layout, location=window_location)

        # Initialize variables
        cap = None
        show_play_button = True
        fps = 30
        width_list = []
        output_size = (640, 480)

        # Main event loop
        while True:
            event, values = window.read(timeout=1000 // fps)  # Update the UI every frame
            # Set main image first frame of current video, or blank image if there was no video
            if not self.play:
                window['image'].update(data=self.title_frame[1].tobytes())
            if event == sg.WINDOW_CLOSED or event == 'Exit':
                break
            elif (event == 'Load video') & (values['input_source'] == 'File'):
                logger.info('load video')
                # Get the filename of the video
                self.filename = sg.popup_get_file('Choose a video file')
                if self.filename:
                    # Load the video
                    cap = cv2.VideoCapture(self.filename)
                    fps = int(cap.get(cv2.CAP_PROP_FPS))
                    # Set the initial frame
                    ret, frame = cap.read()
                    # Set the default title image from the 1st frame of the video
                    frame = cv2.resize(frame, output_size)
                    self.title_frame = cv2.imencode('.png', frame)
                    # Set width different from the zero
                    _, self.width = process_image(frame=frame, verbose=0)
                    window['image'].update(data=self.title_frame[1].tobytes())
                    width_multiplier_calibrated = self.change_calibration_multiplier()

            elif event == 'Play':
                logger.info(f'cap  = {cap}')
                logger.info(f'play = {self.play}')
                if (not self.play) & (self.filename != '') & (values['input_source'] == 'File'):
                    cap = cv2.VideoCapture(self.filename)
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                elif values['input_source'] == 'USB device':
                    cap = cv2.VideoCapture(0)
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 64)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 48)
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

            elif event == 'Change calibration width':
                calibration_value = values['calibration_input']
                try:
                    calibration_value = float(calibration_value)
                    self.opt.calib_width_mm = calibration_value
                    window['calibration_value'].update(calibration_value)
                    width_multiplier_calibrated = self.change_calibration_multiplier()
                except ZeroDivisionError:
                    logger.info('You should select video first!')
                except Exception:
                    logger.info("It must be 'float' datatype")

            # Start the video
            if cap and self.play:
                # Read the next frame
                ret, frame = cap.read()
                show_play_button = False
                if ret:
                    frame = cv2.resize(frame, output_size)
                    mask, self.width = process_image(frame=frame, verbose=0)

                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    # Show the frame if needed
                    if cap.get(cv2.CAP_PROP_POS_FRAMES) % self.show_every_n_frame == 0:
                        if self.mask_or_image == 'image':
                            source = frame
                        else:
                            source = mask

                        source, angle = draw_angle_line(source.copy(), mask)
                        angle_multiplier = calculate_pixel_multiplier(angle)
                        width_list.append(self.width * angle_multiplier)
                        rolling_1s = mean_rolling(width_list, fps)
                        rolling_10s = mean_rolling(width_list, fps, 10)
                        source = draw_fps(source, cap)

                        window['image'].update(data=cv2.imencode('.png', source)[1].tobytes())

                        # window['width_value_pxl'].update(round(self.width, 0))
                        window['width_value_pxl_angle'].update(
                            round((self.width * angle_multiplier), 0))

                        # window['width_value_mm'].update(round(self.width * width_multiplier_calibrated, 3))
                        window['width_value_mm_angle'].update(
                            round(self.width * width_multiplier_calibrated * angle_multiplier, 3))

                        window['angle_multiplier'].update(angle_multiplier)
                        window['rolling_1s'].update(rolling_1s)
                        window['rolling_10s'].update(rolling_10s)
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

    def change_calibration_multiplier(self):
        """ The calibration multiplier is used to estimate the current width """
        calib_multiplier = self.opt.calib_width_mm / self.width
        logger.info(f"Calibration multiplier: {calib_multiplier}")
        return calib_multiplier

