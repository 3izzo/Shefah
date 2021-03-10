import numpy as np
import keras
from Utilities import *


class DataGenerator(keras.utils.Sequence):
    def __init__(
        self,
        list_videos,
        list_labels,
        batch_size=32,
        input_shape=(32, 32, 32),
        n_classes=max_letter_index,
        shuffle=True,
    ):
        "Initialization"
        self.input_shape = input_shape
        self.batch_size = batch_size
        self.list_videos = list_videos
        self.list_labels = list_labels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        "Denotes the number of batches per epoch"
        return int(np.ceil(len(self.list_videos) * 2 / self.batch_size))

    def __getitem__(self, index):
        "Generate one batch of data"
        # Generate indexes of the batch
        indexes = self.indexes[int(
            index * self.batch_size/2): int((index + 1) * self.batch_size/2)]

        # Find list of IDs
        list_videos_temp = [self.list_videos[k] for k in indexes]
        list_labels_temp = [self.list_labels[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_videos_temp, list_labels_temp)
        item = {
            "the_input": X,
            "the_labels": y,
            "input_length": np.array([max_frame_count] * len(X)),
            "label_length": np.array([max_label_length] * len(y)),
        }
        return item, y

    def on_epoch_end(self):
        "Updates indexes after each epoch"
        self.indexes = np.arange(len(self.list_videos))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_videos_temp, list_labels_temp):
        # X : (n_samples, *input_shape, n_channels)
        "Generates data containing batch_size samples"
        # Initialization
        # print(self.input_shape)
        # print((self.batch_size, *self.input_shape))
        X = np.empty((len(list_videos_temp) * 2, *self.input_shape))
        y = np.empty((len(list_videos_temp) * 2, max_label_length), dtype=int)
        # Generate data
        for i, ID in enumerate(list_videos_temp):
            # Store sample
            X[2*i] = load_video_frames(list_videos_temp[i])
            y[2*i] = list_labels_temp[i]
            X[2*i+1] = mirror_frames(X[2*i])
            y[2*i+1] = list_labels_temp[i]

            # np.load("data/" + ID + ".npy")

            # Store class
            #  = self.labels[ID]
        random.Random(seed).shuffle(X)
        random.Random(seed).shuffle(y)

        return X, y
