from keras.models import Model, load_model
from keras import backend as K
import tensorflow as tf
from Utilities import *
from data_generator import DataGenerator
from keras.optimizers import Adam
from model1 import ShefahModel
import sys

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


pretrained_model = ShefahModel(output_size=28)
pretrained_model.model.load_weights(".\\overlapped-weights368.h5")

copy_weights_till_layer = "dense1"
weights = {}
for pretrained_layer in pretrained_model.model.layers:
    if pretrained_layer.name == copy_weights_till_layer:
        break
    weights[pretrained_layer.name] = pretrained_layer.get_weights()

# clear gpu memory
K.clear_session()

# create model
shefah_model = ShefahModel()
model = shefah_model.model

# compile model
model.compile(
    optimizer=Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08),
    loss={"ctc": lambda y_true, y_pred: y_pred},
)


# transfer weights to shefah model
for layer in model.layers:
    if layer.name == copy_weights_till_layer:
        break
    print(layer.name)
    layer.set_weights(weights[layer.name])
    layer.trainable = False



# Get Data
(
    x_train,
    y_train,
    x_validation,
    y_validation,
    x_test,
    y_test,
) = get_train_validation_test_paths(int(sys.argv[1]), int(sys.argv[2]))

start_epoch = 0

train_generator = DataGenerator(
    x_train, y_train, input_shape=shefah_model.input_shape, batch_size=64
)
validation_generator = DataGenerator(
    x_validation, y_validation, input_shape=shefah_model.input_shape, batch_size=16
)
cp_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_pattern, verbose=1, save_weights_only=True, save_freq=200
)
model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=5000,
    initial_epoch=start_epoch,
    callbacks=[cp_callback],
)
