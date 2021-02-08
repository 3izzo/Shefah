from keras.layers.convolutional import Conv3D, ZeroPadding3D
from keras.layers.pooling import MaxPooling3D
from keras.layers.core import Dense, Activation, SpatialDropout3D, Flatten
from keras.layers.wrappers import Bidirectional, TimeDistributed
from keras.layers.recurrent import GRU
from keras.layers.normalization import BatchNormalization
from keras.layers import Input
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


class ShefahModel(object):
    def __init__(
        self,
        img_c=3,
        img_w=frame_w,
        img_h=frame_h,
        frames_n=max_frame_count,
        absolute_max_string_len=max_label_length,
        output_size=max_letter_index,
    ):
        self.img_c = img_c
        self.img_w = img_w
        self.img_h = img_h
        self.frames_n = frames_n
        self.absolute_max_string_len = absolute_max_string_len
        self.output_size = output_size

        if K.image_data_format() == "channels_first":
            self.input_shape = (self.img_c, self.frames_n, self.img_h, self.img_w)
        else:
            self.input_shape = (self.frames_n, self.img_h, self.img_w, self.img_c)

        self.input_data = Input(
            name="the_input", shape=self.input_shape, dtype="float32"
        )

        self.zero1 = ZeroPadding3D(padding=(1, 2, 2), name="zero1")(self.input_data)
        self.conv1 = Conv3D(
            32,
            (3, 5, 5),
            strides=(1, 2, 2),
            kernel_initializer="he_normal",
            name="conv1",
        )(self.zero1)
        self.batc1 = BatchNormalization(name="batc1")(self.conv1)
        self.actv1 = Activation("relu", name="actv1")(self.batc1)
        self.drop1 = SpatialDropout3D(0.5)(self.actv1)
        self.maxp1 = MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2), name="max1")(
            self.drop1
        )

        self.zero2 = ZeroPadding3D(padding=(1, 2, 2), name="zero2")(self.maxp1)
        self.conv2 = Conv3D(
            64,
            (3, 5, 5),
            strides=(1, 1, 1),
            kernel_initializer="he_normal",
            name="conv2",
        )(self.zero2)
        self.batc2 = BatchNormalization(name="batc2")(self.conv2)
        self.actv2 = Activation("relu", name="actv2")(self.batc2)
        self.drop2 = SpatialDropout3D(0.5)(self.actv2)
        self.maxp2 = MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2), name="max2")(
            self.drop2
        )

        self.zero3 = ZeroPadding3D(padding=(1, 1, 1), name="zero3")(self.maxp2)
        self.conv3 = Conv3D(
            96,
            (3, 3, 3),
            strides=(1, 1, 1),
            kernel_initializer="he_normal",
            name="conv3",
        )(self.zero3)
        self.batc3 = BatchNormalization(name="batc3")(self.conv3)
        self.actv3 = Activation("relu", name="actv3")(self.batc3)
        self.drop3 = SpatialDropout3D(0.5)(self.actv3)
        self.maxp3 = MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2), name="max3")(
            self.drop3
        )

        self.resh1 = TimeDistributed(Flatten())(self.maxp3)

        self.gru_1 = Bidirectional(
            GRU(
                256, return_sequences=True, kernel_initializer="Orthogonal", name="gru1"
            ),
            merge_mode="concat",
        )(self.resh1)
        self.gru_2 = Bidirectional(
            GRU(
                256, return_sequences=True, kernel_initializer="Orthogonal", name="gru2"
            ),
            merge_mode="concat",
        )(self.gru_1)

        # transforms RNN output to character activations:
        self.dense1 = Dense(
            self.output_size, kernel_initializer="he_normal", name="dense1"
        )(self.gru_2)

        self.y_pred = Activation("softmax", name="softmax")(self.dense1)

        self.labels = Input(
            name="the_labels", shape=[self.absolute_max_string_len], dtype="float32"
        )
        self.input_length = Input(name="input_length", shape=[1], dtype="int64")
        self.label_length = Input(name="label_length", shape=[1], dtype="int64")

        self.loss_out = CTC(
            "ctc", [self.y_pred, self.labels, self.input_length, self.label_length]
        )

        self.model = Model(
            inputs=[self.input_data, self.labels, self.input_length, self.label_length],
            outputs=self.loss_out,
        )

    def summary(self):
        Model(inputs=self.input_data, outputs=self.y_pred).summary()

    def predict(self, input_batch):
        partial_model = Model(self.input_data, self.y_pred)

        # runs the model in training mode
        output_train = partial_model(input_batch, training=True)
        # runs the model in test mode
        output_test = partial_model(input_batch, training=False)
        # the first 0 indicates test
        # return self.test_function([input_batch, 0])[0]
        return output_train, output_test

    @property
    def test_function(self):
        # captures output of softmax so we can decode the output during visualization
        return K.function(
            [self.input_data, K.learning_phase()], [self.y_pred, K.learning_phase()]
        )


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

(
    x_train,
    y_train,
    x_validation,
    y_validation,
    x_test,
    y_test,
) = get_train_validation_test_data()

shefah_model = ShefahModel()
model = shefah_model.model


model.compile(
    optimizer=optimizers.RMSprop(lr=0.01),
    loss={"ctc": lambda y_true, y_pred: y_pred},
    metrics=["accuracy"],
)


checkpoint_path = ".\\Checkpoints\\cp-{epoch:04d}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)
# from keras.utils import plot_model
# plot_model(model, to_file='model.png', show_shapes=True)
# from IPython.display import Image
# Image(filename='model.png')
# pass
# Create a callback that saves the model's weights every 1 epochs
cp_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path, verbose=1, save_weights_only=True, save_freq=500
)

latest = tf.train.latest_checkpoint(checkpoint_dir)

if latest:
    model.load_weights(latest)
    print("-------------------------------------")
    print("loaded weights from %s" % latest)
else:
    # print(paths)
    train_generator = DataGenerator(
        x_train,
        y_train,
        input_shape=shefah_model.input_shape,
    )
    validation_generator = DataGenerator(
        x_validation,
        y_validation,
        input_shape=shefah_model.input_shape,
    )
    model.fit(
        train_generator,
        validation_data=validation_generator,
        epochs=500,
        callbacks=[cp_callback],
    )

np.set_printoptions(threshold=sys.maxsize)
# print(result1)
# print(result)
print(
    "================================================================================"
)


# print(decoded1)
def decode_predict_ctc(out, top_paths=5):
    results = []
    beam_width = 5
    if beam_width < top_paths:
        beam_width = top_paths
    for i in range(top_paths):
        res = K.get_value(
            K.ctc_decode(
                out,
                input_length=np.ones(out.shape[0]) * out.shape[1],
                greedy=False,
                beam_width=beam_width,
                top_paths=top_paths,
            )[0][i]
        )[0]
        text = translate_array_to_label(res)
        results.append(text)
    return results


def test_model(x, y):
    total_wer = 0
    for i in range(len(y)):
        result1, result2 = shefah_model.predict(np.array([x[i]]))
        # decoded1, decoded2 = K.ctc_decode(y_pred=result2, input_length=np.array([max_frame_count]),
        #                                   greedy=True)
        print(
            "%d==============================================================================="
            % i
        )
        # paths = [path.numpy() for path in decoded1[0]]
        # # print(paths)
        # print(result1)
        top_pred_texts = decode_predict_ctc(result1)
        print(top_pred_texts)
        predicted = top_pred_texts[0]
        actual = translate_array_to_label(y[i])
        predicted_as_numbers = translate_label_to_number(predicted)
        actual_as_numbers = translate_label_to_number(actual)
        wordError = wer(actual_as_numbers, predicted_as_numbers)
        total_wer += wordError
        print(
            "Predicted: ",
            predicted_as_numbers,
            ", Actual:",
            actual_as_numbers,
            predicted,
            actual,
            wordError,
        )
    print(total_wer / len(x))


test_model(x_train, y_train)
print("===============================================================================")
print("===============================================================================")
print("===============================================================================")
test_model(x_test, y_test)
