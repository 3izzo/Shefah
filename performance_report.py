from datetime import datetime
from preprocess_videos import *
import cv2

start_time = datetime.now()
video_path = ".\\videos\\2.mp4"
frame_index = 0
for frame in get_video_frames(video_path):
    cropped_frame = cv2.cvtColor(
        get_frames_mouth(face_detector, predictor, frame), cv2.COLOR_BGR2RGB
    )
    frame_index += 1

delta_time = datetime.now() - start_time
print(
    "Done in",
    delta_time.microseconds / 1000000 + delta_time.seconds,
    "s",
)

