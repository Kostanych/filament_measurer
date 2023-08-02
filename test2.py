import cv2
import numpy as np


def measure_filament_thickness(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Failed to open the video file.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Add code here to process the frame and measure the filament thickness in pixels

        # Example processing: Convert to grayscale and apply thresholding
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, binary_frame = cv2.threshold(gray_frame, 127, 255, cv2.THRESH_BINARY)

        # Measure filament thickness in pixels by averaging the number of pixels per vertical line
        filament_thickness = np.mean(np.sum(binary_frame == 255, axis=0))

        # Output the measured thickness for the current frame
        print("Filament thickness on the current frame: {} pixels".format(
            filament_thickness))

        # Display the processed frame with information about the thickness
        cv2.putText(frame,
                    "Filament Thickness: {:.2f} pixels".format(filament_thickness),
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Processed Frame", frame)

        # To stop the video processing, press 'q' in the output window
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


# Пример использования функции:
video_path = "C:\\Users\\KOS\\Downloads\\v2_hz.avi"
measure_filament_thickness(video_path)
