import cv2
import time
import numpy as np
from Utilities import max_frame_count

FRAME_LENGTH = 1 / 30.0

cap = None
recording = False
recorded_video = []


def open_camera():
    """Turns on the default camera of the system and returns the camera data stream.
    Throws an exception if no camera is found, or if the camera is already running"""
    global cap
    cap = cv2.VideoCapture(0)
    if cap.isOpened():
        return camera_streamer()
    raise Exception("Failed to init camera")


def camera_streamer():
    """an infinite iterator that yields tuples containing the currrent camera frame,
    and how much time is needed till the next frame is ready"""
    global cap, recording, recorded_video

    next_frame_time = time.time() + FRAME_LENGTH
    _, image = cap.read()
    while True:
        current_time = time.time()

        waiting_time = next_frame_time - current_time
        yield image, waiting_time

        if current_time >= next_frame_time:
            _, image = cap.read()
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGBA)
            image = cv2.flip(image, 1)
            if recording:
                recorded_video.append(image)
                if len(recorded_video) > max_frame_count:
                    recorded_video.pop(0)

            next_frame_time += FRAME_LENGTH


def start_recording():
    """Starts saving the data from the default camera’s stream.
    Throws an exception if no camera is running or if the system is already recording."""
    global cap, recording, recorded_video
    if recording:
        raise Exception("Camera already recording")
    if not cap.isOpened():
        raise Exception("Camera not available")
    recording = True
    recorded_video = []


def stop_recording():
    """Stops saving the data, turns off the default camera, and returns the recorded data.
    Throws an exception if the system is not recording."""
    global cap, recording, recorded_video
    if not recording:
        raise Exception("Camera already recording")
    cap.release()
    recording = False
    return np.array(recorded_video)