from cv2 import data
import skvideo.io
import fnmatch
import os
import numpy as np
import dlib
import cv2
from datetime import datetime


input_videos_dir = ".\\GRID\\videos"
input_aligns_dir = ".\\GRID\\align"
output_videos_dir = ".\\GRID PRE"


def find_files(directory, pattern):
    for root, dirs, files in os.walk(directory):
        # print(root, dirs, files)
        for basename in files:
            if fnmatch.fnmatch(basename, pattern):
                filename = os.path.join(root, basename)
                yield filename


def get_video_frames(path):
    videogen = skvideo.io.vreader(path)
    frames = np.array([frame for frame in videogen])
    return frames


def get_frames_mouth(detector, predictor, frame):
    MOUTH_WIDTH = 100
    MOUTH_HEIGHT = 50
    HORIZONTAL_PAD = 0.19

    normalize_ratio = None

    dets = detector(frame, 1)
    shape = None
    i = 0
    for k, d in enumerate(dets):
        shape = predictor(frame, d)
        i = -1
    if shape is None:  # Detector doesn't detect face, just return as is
        return frame
    mouth_points = []
    parts = shape.parts()
    for part in parts:
        i += 1
        if i < 48:  # Only take mouth region
            continue
        mouth_points.append((part.x, part.y))
    np_mouth_points = np.array(mouth_points)
    mouth_centroid = np.mean(np_mouth_points[:, -2:], axis=0)

    if normalize_ratio is None:
        mouth_left = np.min(np_mouth_points[:, :-1]) * (1.0 - HORIZONTAL_PAD)
        mouth_right = np.max(np_mouth_points[:, :-1]) * (1.0 + HORIZONTAL_PAD)

        normalize_ratio = MOUTH_WIDTH / float(mouth_right - mouth_left)

    new_img_shape = (
        int(frame.shape[1] * normalize_ratio),
        int(frame.shape[0] * normalize_ratio),
    )
    resized_img = cv2.resize(
        src=frame, dsize=new_img_shape, interpolation=cv2.INTER_CUBIC
    )
    mouth_centroid_norm = mouth_centroid * normalize_ratio

    mouth_l = int(mouth_centroid_norm[0] - MOUTH_WIDTH / 2)
    mouth_r = int(mouth_centroid_norm[0] + MOUTH_WIDTH / 2)
    mouth_t = int(mouth_centroid_norm[1] - MOUTH_HEIGHT / 2)
    mouth_b = int(mouth_centroid_norm[1] + MOUTH_HEIGHT / 2)

    return resized_img[mouth_t:mouth_b, mouth_l:mouth_r]


def make_dir(dir):
    try:
        os.makedirs(dir)
    except FileExistsError as exc:
        pass


def from_file(path):
    with open(path, "r") as f:
        lines = f.readlines()
    align = [
        (int(y[0]) / 1000, int(y[1]) / 1000, y[2])
        for y in [x.strip().split(" ") for x in lines]
    ]
    align = strip(align, ["sp", "sil"])
    return get_sentence(align)


def strip(align, items):
    return [sub for sub in align if sub[2] not in items]


def get_sentence(align):
    return " ".join([y[-1] for y in align if y[-1] not in ["sp", "sil"]])


name_dic = {}
for align_path in find_files(input_aligns_dir, "*.align"):
    actual_name = from_file(align_path)

    align_path = align_path.replace(".align", "")
    id = align_path.split("\\")[-1]
    name_dic[id] = actual_name


def preproc_speaker(speaker_index):
    for video_path in find_files(
        input_videos_dir + "\\s" + str(speaker_index), "*.mpg"
    ):
        start_time = datetime.now()
        original_name = video_path.replace(".mpg", "").split("\\")[-1]
        storage_dir = (
            output_videos_dir
            + "\\s"
            + str(speaker_index)
            + "\\"
            + name_dic[original_name]
        )
        make_dir(storage_dir + "\\unmirrored")
        make_dir(storage_dir + "\\mirrored")
        frame_index = 0
        for frame in get_video_frames(video_path):
            cropped_frame = cv2.cvtColor(
                get_frames_mouth(face_detector, predictor, frame), cv2.COLOR_BGR2RGB
            )
            mirrored = cv2.flip(cropped_frame, 1)
            cv2.imwrite(
                "%s\\unmirrored\\frame%d.png" % (storage_dir, frame_index),
                cropped_frame,
            )
            cv2.imwrite(
                "%s\\mirrored\\frame%d.png" % (storage_dir, frame_index), mirrored
            )
            frame_index += 1
        delta_time = datetime.now() - start_time
        print(
            "Done",
            storage_dir,
            "in",
            delta_time.microseconds / 1000000 + delta_time.seconds,
            "s",
        )


import multiprocessing
from joblib import Parallel, delayed

num_cores = multiprocessing.cpu_count()
print(num_cores)
make_dir(output_videos_dir)
face_detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(".\\shape_predictor_68_face_landmarks.dat")

Parallel(n_jobs=num_cores)(
    delayed(preproc_speaker)(speaker_index) for speaker_index in range(1, 13)
)
