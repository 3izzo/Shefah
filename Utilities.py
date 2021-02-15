import numpy as np
import cv2
import os
import re
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
max_frame_count = 50
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
            # print(basename,re.match(basename, pattern))
            if re.match(pattern, basename):
                dir = os.path.join(root, basename)
                yield dir


def get_label_from_path(path):
    split_path = path.split("\\")
    label_as_numbers = split_path[-2].split(".")[0]
    numbers = label_as_numbers.split(" ")
    label = ""
    for n in numbers:
        label += mapping[n] + " "
    return translate_label_to_array(label)


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


SPEAKER_TRAIN_COUNT = 11
SPEAKER_VALIDATION_COUNT = 2


def get_train_validation_test_paths():
    speakers_paths = []
    for dir in find_dirs(".\\PreprocessedVideos", "speaker([1-9]|([0-9][0-9]))"):
        speakers_paths.append(dir)

    train_paths = []
    train_labels = []
    validation_paths = []
    validation_labels = []
    test_paths = []
    test_labels = []

    # choose random speakers and all of their videos to the training data
    for i in range(SPEAKER_TRAIN_COUNT):
        # choose random speaker from speakers_paths
        random_index = np.random.randint(0, len(speakers_paths))
        speaker_path = speakers_paths.pop(random_index)

        # go through every video of the speaker
        for dir in find_dirs(speaker_path, "mirrored"):
            train_paths.append(dir)
            train_labels.append(get_label_from_path(dir))
        for dir in find_dirs(speaker_path, "unmirrored"):
            train_paths.append(dir)
            train_labels.append(get_label_from_path(dir))

    # choose random speakers and all of their videos to the validation data
    for i in range(SPEAKER_VALIDATION_COUNT):
        # choose random speaker from speakers_paths
        random_index = np.random.randint(0, len(speakers_paths))
        speaker_path = speakers_paths.pop(random_index)

        # go through every video of the speaker
        for dir in find_dirs(speaker_path, "mirrored"):
            validation_paths.append(dir)
            validation_labels.append(get_label_from_path(dir))
        for dir in find_dirs(speaker_path, "unmirrored"):
            validation_paths.append(dir)
            validation_labels.append(get_label_from_path(dir))

    for speaker_path in speakers_paths:
        for dir in find_dirs(speaker_path, "mirrored"):
            test_paths.append(dir)
            test_labels.append(get_label_from_path(dir))
        for dir in find_dirs(speaker_path, "unmirrored"):
            test_paths.append(dir)
            test_labels.append(get_label_from_path(dir))
    return (
        train_paths,
        train_labels,
        validation_paths,
        validation_labels,
        test_paths,
        test_labels,
    )


get_train_validation_test_paths()