import numpy as np
from datetime import datetime
from preprocess_videos import get_video_frames,get_frames_mouth, face_detector, predictor
from utilities import max_frame_count, frame_h, frame_w
import cv2

video_path = ".\\videos\\speaker2\\2.mp4"


def preprocessing(video_path):
    frames = []
    i = 0
    for frame in get_video_frames(video_path):
        cropped_frame = cv2.cvtColor(
            get_frames_mouth(face_detector, predictor, frame), cv2.COLOR_BGR2RGB
        )
        frames.append(cropped_frame)
        i += 1
    while i < max_frame_count:
        frames.append(np.zeros((frame_h, frame_w, 3)))
        i += 1
    frames = np.array([frames])
    return frames


start_time = datetime.now()
frames = preprocessing(video_path)
delta_time = datetime.now() - start_time

print(
    "preprocessing done in",
    delta_time.microseconds / 1000000 + delta_time.seconds,
    "s",
)

start_time = datetime.now()
from predict import load_model, decode_predict_ctc
from utilities import translate_label_to_number

delta_time = datetime.now() - start_time

print(
    "tensorflow init in",
    delta_time.microseconds / 1000000 + delta_time.seconds,
    "s",
)

start_time = datetime.now()

shefah_model = load_model()
delta_time = datetime.now() - start_time

print(
    "model loaded in",
    delta_time.microseconds / 1000000 + delta_time.seconds,
    "s",
)

start_time = datetime.now()
y_prediction = shefah_model.predict(frames)
delta_time = datetime.now() - start_time
print(
    "prediction done in",
    delta_time.microseconds / 1000000 + delta_time.seconds,
    "s",
)

start_time = datetime.now()

decoded_prediction = decode_predict_ctc(y_prediction)
predicted = decoded_prediction[0]
predicted_as_numbers = translate_label_to_number(predicted)

delta_time = datetime.now() - start_time
print(
    "postprocessing done in",
    delta_time.microseconds / 1000000 + delta_time.seconds,
    "s",
)