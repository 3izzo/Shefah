from keras.layers.convolutional import Conv3D, ZeroPadding3D
from keras.layers.pooling import MaxPooling3D
from keras.layers.core import Dense, Activation, SpatialDropout3D, Flatten
from keras.layers.wrappers import Bidirectional, TimeDistributed
from keras.layers.recurrent import GRU
from keras.layers.normalization import BatchNormalization
from keras.layers import Input
from keras.models import Model
from keras import backend as K
import skvideo.io
import errno
import sys
import fnmatch
import os
import numpy as np
import dlib
import cv2
import tensorflow as tf
from layers import CTC


def get_model(img_c=3, img_w=50, img_h=100, frames_n=86, absolute_max_string_len=14, output_size=20):
    if K.image_data_format() == 'channels_first':
        input_shape = (img_c, frames_n, img_w, img_h)
    else:
        input_shape = (frames_n, img_w, img_h, img_c)

    input_data = Input(name='the_input', shape=input_shape, dtype='float32')

    zero1 = ZeroPadding3D(padding=(1, 2, 2), name='zero1')(input_data)
    conv1 = Conv3D(32, (3, 5, 5), strides=(1, 2, 2),
                   kernel_initializer='he_normal', name='conv1')(zero1)
    batc1 = BatchNormalization(name='batc1')(conv1)
    actv1 = Activation('relu', name='actv1')(batc1)
    drop1 = SpatialDropout3D(0.5)(actv1)
    maxp1 = MaxPooling3D(pool_size=(1, 2, 2),
                         strides=(1, 2, 2), name='max1')(drop1)

    zero2 = ZeroPadding3D(padding=(1, 2, 2), name='zero2')(maxp1)
    conv2 = Conv3D(64, (3, 5, 5), strides=(1, 1, 1),
                   kernel_initializer='he_normal', name='conv2')(zero2)
    batc2 = BatchNormalization(name='batc2')(conv2)
    actv2 = Activation('relu', name='actv2')(batc2)
    drop2 = SpatialDropout3D(0.5)(actv2)
    maxp2 = MaxPooling3D(pool_size=(1, 2, 2),
                         strides=(1, 2, 2), name='max2')(drop2)

    zero3 = ZeroPadding3D(padding=(1, 1, 1), name='zero3')(maxp2)
    conv3 = Conv3D(96, (3, 3, 3), strides=(1, 1, 1),
                   kernel_initializer='he_normal', name='conv3')(zero3)
    batc3 = BatchNormalization(name='batc3')(conv3)
    actv3 = Activation('relu', name='actv3')(batc3)
    drop3 = SpatialDropout3D(0.5)(actv3)
    maxp3 = MaxPooling3D(pool_size=(1, 2, 2),
                         strides=(1, 2, 2), name='max3')(drop3)

    resh1 = TimeDistributed(Flatten())(maxp3)

    gru_1 = Bidirectional(GRU(256, return_sequences=True,
                              kernel_initializer='Orthogonal', name='gru1'), merge_mode='concat')(resh1)
    gru_2 = Bidirectional(GRU(256, return_sequences=True,
                              kernel_initializer='Orthogonal', name='gru2'), merge_mode='concat')(gru_1)

    # transforms RNN output to character activations:
    dense1 = Dense(output_size, kernel_initializer='he_normal',
                   name='dense1')(gru_2)

    y_pred = Activation('softmax', name='softmax')(dense1)

    labels = Input(name='the_labels', shape=[
                   absolute_max_string_len], dtype='float32')
    input_length = Input(name='input_length', shape=[1], dtype='int64')
    label_length = Input(name='label_length', shape=[1], dtype='int64')

    loss_out = CTC('ctc', [y_pred, labels, input_length, label_length])

    return Model(inputs=[input_data, labels, input_length, label_length], outputs=loss_out)
    # return tf.keras.models.Sequential([
    #     tf.keras.layers.Flatten(input_shape=input_shape),
    #     tf.keras.layers.Dense(128, activation='relu'),
    #     tf.keras.layers.Dropout(0.2),
    #     tf.keras.layers.Dense(14)
    # ])


count = 10
test = 2
train = count - test


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
    while(i < 86):
        res.append(np.zeros((50, 100, 3)))
        i += 1
    return np.array(res)


def find_dirs(directory, pattern):
    for root, dirs, files in os.walk(directory):
        # print(root, dirs, files)
        for basename in dirs:
            if fnmatch.fnmatch(basename, pattern):
                dir = os.path.join(root, basename)
                yield dir
# model = get_model()
# loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
# model.compile(optimizer='adam',
#               loss=loss_fn,
#               metrics=['accuracy'])


# model.fit(x_train, y_train, epochs=500)
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

# path = ".\\PreprocessedVideos\\speaker3\\0 1"


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


def translate_label_to_array(label):
    arr = np.empty((14))
    for i in range(14):
        if(i >= len(label)):
            arr[i] = 18

        letter = label[i]
        if(letter == ' '):
            arr[i] = 18
        else:
            arr[i] = (letters.index(letter))
    return arr


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
    random_index = np.random.randint(0, int(videos_count * training_ratio) - i)
    x_train.append(videos[random_index])
    y_train.append(labels[random_index])

    videos.pop(random_index)
    labels.pop(random_index)

x_train = np.array(x_train)
y_train = np.array(y_train)
x_test = videos
y_test = labels

# print(len(y_train), len(x_train))
# print(len(y_test), len(x_test))

model = get_model()

loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=True)

model.compile(optimizer='adam',
              loss=loss_fn,
              metrics=['accuracy'])
inputs = {'the_input': x_train,
          'the_labels': y_train,
          'input_length': np.array([86] * 60),
          'label_length': np.array([14] * 60),
          }
outputs = {'ctc': np.zeros([60])}


model.fit(inputs,outputs, epochs=5)

print(y_train.shape)
# print(x_train)

# list of videos
#   list of frames
#       list of rows
#           list of cols
#               list of chan
#                   data
#                   60, 75, 100, 50, 3
# [ [],[],[],[],[],[],[10] ] 60,
# => [] 60,10
