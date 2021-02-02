import numpy as np
import cv2
import os
import fnmatch

letters = ["ا", "ب", "ت", "ة", "ث", "ح", "خ", "د",
           "ر", "س", "ص", "ع", "ف", "ل", "م", "ن", "و", "ي", ]

mapping = {
    "0": "صفر",
    "1": "واحد",
    "2": "اثنين",
    "3": "ثلاثة",
    "4": "اربعة",
    "5": "خمسة",
    "6": "ستة",
    "7": "سبعة",
    "8": "ثمانية",
    "9": "تسعة",
}
max_label_length = 14
max_frame_count = 86
frame_h = 50
frame_w = 100


def load_video_frames(path):
    res = []
    i = 0
    while True:
        name = path + '\\frame%d.png' % i
        frame = cv2.imread(name)
        if(type(frame) == type(None)):
            break
        res.append(frame)
        i += 1
    while(i < max_frame_count):
        res.append(np.zeros((frame_h, frame_w, 3)))
        i += 1
    return np.array(res)


def find_dirs(directory, pattern):
    for root, dirs, files in os.walk(directory):
        # print(root, dirs, files)
        for basename in dirs:
            if fnmatch.fnmatch(basename, pattern):
                dir = os.path.join(root, basename)
                yield dir


def get_video_and_label(path):
    frames = load_video_frames(path)
    split_path = path.split("\\")
    label_as_numbers = split_path[-1].split(".")[0]
    numbers = label_as_numbers.split(" ")
    label = ""
    for n in numbers:
        label += mapping[n] + " "
    label += " " * (max_label_length - len(label))
    # print(label_as_numbers, label, len(label))
    return frames, translate_label_to_array(label)


def translate_label_to_array(label):
    arr = np.empty((max_label_length))
    for i in range(max_label_length):
        if(i >= len(label)):
            arr[i] = len(letters)

        letter = label[i]
        if(letter == ' '):
            arr[i] = len(letters)
        else:
            arr[i] = (letters.index(letter))
    return arr


def get_train_test_data():
    videos = []
    labels = []
    for dir in find_dirs(".\\PreprocessedVideos", "[0-9] [0-9]"):
        frames, label = get_video_and_label(dir)
        videos.append(frames)
        # print(label)
        labels.append(label)

    videos_count = len(videos)
    training_ratio = 0.75

    x_train = []
    y_train = []

    np.random.seed(69)
    for i in range(int(videos_count * training_ratio)):
        random_index = np.random.randint(
            0, int(videos_count * training_ratio) - i)
        x_train.append(videos[random_index])
        y_train.append(labels[random_index])

        videos.pop(random_index)
        labels.pop(random_index)

    x_train = np.array(x_train)
    y_train = np.array(y_train)
    x_test = np.array(videos)
    y_test = np.array(labels)

    return x_train, y_train, x_test, y_test
