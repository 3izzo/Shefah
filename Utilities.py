import numpy as np
import cv2
from fuzzywuzzy import fuzz
from numpy import random as numpy_random
import random
from file_manager import find_dirs

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
max_frame_count = 75
max_letter_index = len(letters) + 2  # 20
frame_h = 50
frame_w = 100

input_videos_dir = ".\\Videos"
output_videos_dir = ".\\PreprocessedVideos"
checkpoints_dir = ".\\Checkpoints"
checkpoint_pattern = checkpoints_dir + "\\cp-{epoch:04d}.ckpt"

seed = 69

video_cache = {}


def load_frames_for_training(path):
    if path in video_cache:
        return video_cache[path]
    frames = []
    i = 0
    while i < max_frame_count:
        name = path + "\\frame%d.png" % i
        frame = cv2.imread(name)
        if type(frame) == type(None):
            break
        frames.append(frame)
        i += 1
    else:
        print(
            "WARNING VIDEO %s HAS MORE FRAMES THAN THE LIMIT OF %d. Any frame above the limit will be ignored"
            % (name, max_frame_count)
        )
    # padding
    frames = add_padding(frames)
    # normalize the frame
    try:
        res = np.array(frames).astype(np.float32) / 255
        video_cache[path] = res
        return res
    except:
        print("error loading frames from", path, np.array(frames).shape)


def mirror_frames(frames):
    mirrored = []
    for frame in frames:
        # cv2.imshow(frame)
        mirrored.append(cv2.flip(frame, 1))
        # cv2.imshow(mirrored)
    return mirrored


zeros_frame = np.zeros((frame_h, frame_w, 3))


def add_padding(frames):

    if len(frames) >= max_frame_count:
        return frames
    y = np.array([zeros_frame] * (max_frame_count - len(frames)))
    frames = np.concatenate([frames, y])

    return frames


def get_label_from_path(path):
    split_path = path.split("\\")
    label_as_numbers = split_path[-1].split(".")[0]
    numbers = label_as_numbers.split(" ")
    label = ""
    for n in numbers:
        label += mapping[n] + " "
    return label


def translate_label_to_array(label):
    """
    ex: input:  'واحد'
        result: [17,0,5,7]
    """
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
    """
    ex: input: [17,0,5,7]
        result:  'واحد'
    """
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
    res += str(translate_word_to_number(words[0])) + " "
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


def translate_word_to_word(word):
    """ خسه، خمسه"""
    res = 0
    best_ratio = 0
    for key in mapping.keys():
        value = mapping[key]
        ratio = fuzz.ratio(word, value)
        if ratio > best_ratio:
            res = value
            best_ratio = ratio
    return res


def get_train_validation_test_paths(trainCount, valCount):
    numpy_random.seed(seed)
    speakers_paths = []
    for dir in find_dirs(".\\PreprocessedVideos", "speaker([1-9]|([0-9][0-9]))"):
        speakers_paths.append(dir)

    train_paths = []
    train_labels = []
    validation_paths = []
    validation_labels = []
    test_paths = []
    test_labels = []

    print("training speakers:")
    # choose random speakers and all of their videos to the training data
    for i in range(trainCount):

        # choose random speaker from speakers_paths
        random_index = np.random.randint(0, len(speakers_paths))
        speaker_path = speakers_paths.pop(random_index)
        print(speaker_path)
        get_paths_lables(train_paths, train_labels, speaker_path)
    print("validation speakers:")

    # choose random speakers and all of their videos to the validation data
    for i in range(valCount):
        # choose random speaker from speakers_paths
        random_index = np.random.randint(0, len(speakers_paths))
        speaker_path = speakers_paths.pop(random_index)
        print(speaker_path)

        get_paths_lables(validation_paths, validation_labels, speaker_path)
    print("testing speakers:")

    for speaker_path in speakers_paths:
        print(speaker_path)
        get_paths_lables(test_paths, test_labels, speaker_path)

    random.Random(seed).shuffle(train_paths)
    random.Random(seed).shuffle(train_labels)

    random.Random(seed + 1).shuffle(validation_paths)
    random.Random(seed + 1).shuffle(validation_labels)

    random.Random(seed - 1).shuffle(test_paths)
    random.Random(seed - 1).shuffle(test_labels)
    return (
        train_paths,
        train_labels,
        validation_paths,
        validation_labels,
        test_paths,
        test_labels,
    )


def cross_validation(dataCount, numberOfFolds, currnetFold):
    """
    Split the data on the number of folds. \n
    currnet fold should be larger than or equal to 1 \n
    return a one dataset for a model to train but differs each time you use different currentFold .
    """
    numpy_random.seed(seed)
    speakers_paths = []
    for dir in find_dirs(".\\PreprocessedVideos", "speaker([1-9]|([0-9][0-9]))"):
        speakers_paths.append(dir)

    train_paths = []
    train_labels = []
    validation_paths = []
    validation_labels = []

    test_paths = []
    test_labels = []

    print("testing speakers:")
    # choose speakers and all of their videos to the test data
    start = int(((currnetFold - 1) * (dataCount / numberOfFolds)))
    end = int(currnetFold * (dataCount / numberOfFolds))
    for i in range(start, end):
        # choose random speaker from speakers_paths
        speaker_path = speakers_paths.pop(start)
        print(speaker_path)
        get_paths_lables(test_paths, test_labels, speaker_path)

    print("training speakers:", len(speakers_paths))
    for speaker_path in speakers_paths:
        print(speaker_path)
        get_paths_lables(train_paths, train_labels, speaker_path)

    random.Random(seed).shuffle(train_paths)
    random.Random(seed).shuffle(train_labels)

    random.Random(seed).shuffle(test_paths)
    random.Random(seed).shuffle(test_labels)
    return (
        train_paths,
        train_labels,
        validation_paths,
        validation_labels,
        test_paths,
        test_labels,
    )


def get_paths_lables(paths, labels, speaker_path):
    # go through every video of the speaker
    for dir in find_dirs(speaker_path, "[0-9]"):
        label = get_label_from_path(dir)
        paths.append(dir)
        labels.append(translate_label_to_array(label))
