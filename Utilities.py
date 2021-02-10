import numpy as np
import cv2
import os
import fnmatch
from fuzzywuzzy import fuzz


letters = [
    "ا",
    "ب",
    "ت",
    "ة",
    "ث",
    "ح",
    "خ",
    "د",
    "ر",
    "س",
    "ص",
    "ع",
    "ف",
    "ل",
    "م",
    "ن",
    "و",
    "ي",
]

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
max_letter_index = len(letters) + 2  # 20
frame_h = 50
frame_w = 100

video_cache = {}


def load_video_frames(path):
    if path in video_cache:
        return video_cache[path]
    frames = []
    i = 0
    while True:
        name = path + "\\frame%d.png" % i
        frame = cv2.imread(name)
        if type(frame) == type(None):
            break
        frames.append(frame)
        i += 1
    # padding
    while i < max_frame_count:
        frames.append(np.zeros((frame_h, frame_w, 3)))
        i += 1
    # normalize the frame
    try:
        res = np.array(frames).astype(np.float32) / 255
        video_cache[path] = res
        return res
    except:
        print("error loading frames from", path, np.array(frames).shape)


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
    label_as_numbers = split_path[-2].split(".")[0]
    numbers = label_as_numbers.split(" ")
    label = ""
    for n in numbers:
        label += mapping[n] + " "
    # label += " " * (max_label_length - len(label))
    # print(label_as_numbers, label, len(label))
    return frames, translate_label_to_array(label)


def translate_label_to_array(label):
    arr = np.empty((max_label_length))
    for i in range(max_label_length):
        if i >= len(label):
            arr[i] = len(letters) + 1
        else:
            letter = label[i]
            if letter == " ":
                arr[i] = len(letters)
            else:
                arr[i] = letters.index(letter)
    return arr.astype(np.int8)


def translate_array_to_label(arr):
    label = ""
    for i in arr:
        if i == -1:
            break
        if i >= len(letters):
            label += " "
        else:
            letter = letters[i]
            label += letter
    return label


def translate_label_to_number(label):
    label = label.strip()
    res = ""
    words = label.split(" ")
    for word in words:
        res += str(translate_word_to_number(word)) + " "
    return res.strip()


def translate_word_to_number(word):
    res = 0
    best_ratio = 0
    for key in mapping.keys():
        value = mapping[key]
        ratio = fuzz.ratio(word, value)
        if ratio > best_ratio:
            res = key
            best_ratio = ratio
    return res


def get_train_validation_test_paths():
    paths = []
    labels = []
    for dir in find_dirs(".\\PreprocessedVideos", "mirrored"):
        paths.append(dir)
        split_path = dir.split("\\")
        numbers = split_path[-2].split(".")[0].split(" ")
        label = ""
        for n in numbers:
            label += mapping[n] + " "
        label = label.strip()
        labels.append(translate_label_to_array(label))

        dir = dir.replace("mirrored", "unmirrored")
        paths.append(dir)
        split_path = dir.split("\\")
        numbers = split_path[-2].split(".")[0].split(" ")
        label = ""
        for n in numbers:
            label += mapping[n] + " "
        label = label.strip()
        labels.append(translate_label_to_array(label))
    videos_count = len(paths)
    training_ratio = 0.70
    validation_ratio = 0.15

    x_train = []
    y_train = []
    x_validation = []
    y_validation = []

    x_test = paths[:]
    y_test = labels[:]
    np.random.seed(69)
    for i in range(int(videos_count * training_ratio)):
        random_index = np.random.randint(0, len(x_test))
        x_train.append(paths[random_index])
        y_train.append(labels[random_index])

        x_test.pop(random_index)
        y_test.pop(random_index)

    for i in range(int(videos_count * validation_ratio)):
        random_index = np.random.randint(0, len(x_test))
        x_validation.append(paths[random_index])
        y_validation.append(labels[random_index])

        x_test.pop(random_index)
        y_test.pop(random_index)

    x_train = np.array(x_train)
    y_train = np.array(y_train)

    x_test = np.array(x_test)
    y_test = np.array(y_test)

    return x_train, y_train, x_validation, y_validation, x_test, y_test
