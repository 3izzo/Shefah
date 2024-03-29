import fnmatch
import os
import numpy as np
import dlib
import cv2
from datetime import datetime

from file_manager import get_video_frames
from utilities import input_videos_dir, output_videos_dir

def find_files(directory, pattern):
    for root, dirs, files in os.walk(directory):
        # print(root, dirs, files)
        for basename in files:
            if fnmatch.fnmatch(basename, pattern):
                filename = os.path.join(root, basename)
                yield filename





def get_frames_mouth(detector, predictor, frame, interface=None):
    MOUTH_WIDTH = 100
    MOUTH_HEIGHT = 50
    HORIZONTAL_PAD = 0.6

    normalize_ratio = None

    dets = detector(frame)
    shape = None
    for k, d in enumerate(dets):
        if shape == None:
            shape = predictor(frame, d)
            if not interface == None:
                interface.face_video.append(frame[d.top() : d.bottom(), d.left() : d.right()])
        else:
            shape_size = shape.rect.area()
            shape_temp = predictor(frame, d)
            shape_temp_size = shape_temp.rect.area()
            if shape_temp_size > shape_size:
                shape = shape_temp

    if shape is None:
        raise Exception("No Face Detected")
    mouth_points = []
    parts = shape.parts()
    i = -1
    for part in parts:
        i += 1
        if i < 48:  # Only take mouth region
            continue
        mouth_points.append((part.x, part.y))
    np_mouth_points = np.array(mouth_points)
    mouth_centroid = np.mean(np_mouth_points[:, -2:], axis=0)

    if normalize_ratio is None:
        mouth_left = np.min(np_mouth_points[:, :-1])
        mouth_right = np.max(np_mouth_points[:, :-1])
        mouth_left_padded = mouth_left - (mouth_right - mouth_left) * HORIZONTAL_PAD
        mouth_right_padded = mouth_right + (mouth_right - mouth_left) * HORIZONTAL_PAD

        normalize_ratio = MOUTH_WIDTH / float(mouth_right_padded - mouth_left_padded)

    new_img_shape = (
        int(frame.shape[1] * normalize_ratio),
        int(frame.shape[0] * normalize_ratio),
    )
    resized_img = cv2.resize(src=frame, dsize=new_img_shape, interpolation=cv2.INTER_CUBIC)
    mouth_centroid_norm = mouth_centroid * normalize_ratio

    mouth_l = int(int(mouth_centroid_norm[0]) - MOUTH_WIDTH / 2)
    mouth_r = mouth_l + MOUTH_WIDTH
    if mouth_l < 0:
        mouth_l = 0
        mouth_r = MOUTH_WIDTH
        if mouth_r > resized_img.shape[1]:
            return
    if mouth_r > resized_img.shape[1]:
        mouth_r = resized_img.shape[1]
        mouth_l = mouth_r - MOUTH_WIDTH
        if mouth_l < 0:
            return

    mouth_t = int(int(mouth_centroid_norm[1]) - MOUTH_HEIGHT / 2)
    mouth_b = mouth_t + MOUTH_HEIGHT
    if not interface == None:
        ROI = resized_img.copy()
        i = -1
        for part in parts:
            i += 1
            if i < 48:  # Only take mouth region
                continue
            ROI[int(part.y * normalize_ratio), int(part.x * normalize_ratio)] = [0, 255, 0]

        interface.ROI_video.append(ROI[mouth_t:mouth_b, mouth_l:mouth_r])

    return resized_img[mouth_t:mouth_b, mouth_l:mouth_r]


def make_dir(dir):
    try:
        os.makedirs(dir)
    except FileExistsError as exc:
        pass


def preproc_speaker(speaker_index):

    for video_path in find_files(input_videos_dir + "\\speaker" + str(speaker_index + 1), "[0-9].mp4"):
        start_time = datetime.now()
        storage_dir = video_path.replace(input_videos_dir, output_videos_dir).replace(".mp4", "")
        make_dir(storage_dir)

        frame_index = 0
        for frame in get_video_frames(video_path):
            cropped_frame = cv2.cvtColor(get_frames_mouth(face_detector, predictor, frame), cv2.COLOR_BGR2RGB)

            cv2.imwrite(
                "%s\\frame%d.png" % (storage_dir, frame_index),
                cropped_frame,
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


face_detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(".\\shape_predictor_68_face_landmarks.dat")
if __name__ == "__main__":

    def get_number_of_speakers():
        count = 0
        for root, dirs, files in os.walk(input_videos_dir):
            count = len(dirs)
            break
        return count

    import multiprocessing
    from joblib import Parallel, delayed

    num_cores = multiprocessing.cpu_count()
    num_speakers = get_number_of_speakers()
    print("cores :", num_cores, " speakers : ", num_speakers)
    make_dir(output_videos_dir)

    Parallel(n_jobs=num_cores)(delayed(preproc_speaker)(speaker_index) for speaker_index in range(num_speakers))
