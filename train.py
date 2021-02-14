from keras.models import Model
from keras import backend as K
import sys
import numpy as np
import tensorflow as tf
from layers import CTC
from Utilities import *
from data_generator import DataGenerator
from jiwer import wer
from keras import optimizers
from model1 import ShefahModel

# Enable GPU Accleration
gpus = tf.config.experimental.list_physical_devices("GPU")
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices("GPU")
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)

tf.compat.v1.disable_eager_execution()

# Get Data
(
    x_train,
    y_train,
    x_validation,
    y_validation,
    x_test,
    y_test,
) = get_train_validation_test_paths()

# create model
shefah_model = ShefahModel()
model = shefah_model.model

# compile model
model.compile(
    optimizer=optimizers.RMSprop(lr=0.01),
    loss={"ctc": lambda y_true, y_pred: y_pred},
    metrics=[
        "accuracy",
    ],
)


checkpoint_pattern = ".\\Checkpoints\\cp-{epoch:04d}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_pattern)

cp_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_pattern, verbose=1, save_weights_only=True, save_freq=50
)

latest = tf.train.latest_checkpoint(checkpoint_dir)

if latest:
    model.load_weights(latest)
    print("-------------------------------------")
    print("loaded weights from %s" % latest)
else:
    # print(paths)
    train_generator = DataGenerator(
        x_train, y_train, input_shape=shefah_model.input_shape, batch_size=25
    )
    validation_generator = DataGenerator(
        x_validation, y_validation, input_shape=shefah_model.input_shape, batch_size=25
    )
    model.fit(
        train_generator,
        validation_data=validation_generator,
        epochs=500,
        callbacks=[cp_callback],
    )
