import numpy as np
import os
import re
import skvideo.io
from tkinter import filedialog as fd

video_cache = {}

class OperationCancelled(Exception):
    pass

def get_video():
    """Prompts the user to select a video file through a file selection
    dialog and returns the result.
    Throws an exception if the user cancels or selects a file that is
    invalid or cannot be read."""
    path = fd.askopenfilename(
        filetypes=(
            ("video file", "*.mp4"),
            ("video file", "*.mkv"),
            ("video file", "*.mpg"),
            ("video file", "*.wmv"),
            ("video file", "*.m4p"),
            ("video file", "*.m4v"),
            ("video file", "*.mpg"),
            ("video file", "*.mpeg"),
            ("video file", "*.m2v"),
            ("video file", "*.wav"),
            ("video file", "*.webm"),
            ("all files", "*.*"),
        )
    )
    if path == "":
        raise OperationCancelled

    if is_valid_video_file(path):
        return get_video_frames(path)
    raise Exception("Operation cancled or video incompatible")


def is_valid_video_file(path):
    """Receives a file path and returns whether the given path represents a
    readable path and if the path points to a supported video file"""

    # Check framerate
    try:
        meta = skvideo.io.ffprobe(path)
        framerate = meta["video"]["@avg_frame_rate"].split("/")
        framerate = int(framerate[0]) / float(framerate[1])
        print(framerate)
        if framerate < 24 or framerate > 35:
            return False
        skvideo.io.vreader(path)
    except Exception:
        return False

    return True


def get_video_frames(path):
    """Reads the frames from the given path"""
    videogen = skvideo.io.vreader(path)
    frames = np.array([frame for frame in videogen])
    return frames


def find_dirs(directory, pattern):
    for root, dirs, files in os.walk(directory):
        # print(root, dirs, files)
        for basename in dirs:
            # print(basename,re.match(basename, pattern))
            if re.match(pattern, basename):
                dir = os.path.join(root, basename)
                yield dir

