from keras import backend as K
import numpy as np
import tensorflow as tf
from Utilities import *
from data_generator import DataGenerator
from keras import optimizers
from model1 import ShefahModel
from preprocess_videos import *

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

latest = tf.train.latest_checkpoint(checkpoint_dir)

if latest:
    model.load_weights(latest)
    print("-------------------------------------")
    print("loaded weights from %s" % latest)
else:
    print("No weights found")


def decode_predict_ctc(out, top_paths=5):
    beam_width = 200
    decoded = K.ctc_decode(
        out[0],
        input_length=np.ones(out[0].shape[0]) * out[0].shape[1],
        greedy=False,
        beam_width=beam_width,
        top_paths=top_paths,
    )
    paths = [path.numpy() for path in decoded[0]]

    res = []
    for output in paths[0]:
        res.append(translate_array_to_label(output))
    return res


(
    x_train,
    y_train,
    x_validation,
    y_validation,
    x_test,
    y_test,
) = get_train_validation_test_paths()


def test_model(x, y):
    labels = []
    predictions = []
    for i in range(len(y)):
        frames = load_video_frames(x[i])
        video = np.array([frames])
        result = shefah_model.predict(video)
        # decoded1, decoded2 = K.ctc_decode(y_pred=result2, input_length=np.array([max_frame_count]),
        #                                   greedy=True)
        print(
            "==============================================================================="
        )
        # paths = [path.numpy() for path in decoded1[0]]
        # # print(paths)
        # print(result1)
        top_pred_texts = decode_predict_ctc(result)
        print(top_pred_texts)
        predicted = top_pred_texts[0]
        actual = translate_array_to_label(y[i])
        predicted_as_numbers = int(translate_label_to_number(predicted))
        actual_as_numbers = int(translate_label_to_number(actual))

        labels.append(actual_as_numbers)
        predictions.append(predicted_as_numbers)
        print(
            "Predicted: ",
            predicted_as_numbers,
            predicted,
            ", Actual:",
            actual_as_numbers,
            actual,
        )
    print("Confusion Matrix:")
    print(tf.math.confusion_matrix(labels, np.array(predictions)))


count_labels = np.zeros((10))
for i in range(len(y_train)):
    count_labels[
        int(translate_label_to_number(translate_array_to_label(y_train[i])))
    ] += 1
    print(x_train[i], translate_label_to_number(translate_array_to_label(y_train[i])))

for i in range(len(y_validation)):
    count_labels[
        int(translate_label_to_number(translate_array_to_label(y_validation[i])))
    ] += 1
    print(
        x_validation[i],
        translate_label_to_number(translate_array_to_label(y_validation[i])),
    )

for i in range(len(y_test)):
    count_labels[
        int(translate_label_to_number(translate_array_to_label(y_test[i])))
    ] += 1
    print(x_test[i], translate_label_to_number(translate_array_to_label(y_test[i])))

print(count_labels)
print("Training data =================================================================")
test_model(x_train, y_train)
print("===============================================================================")
print("===============================================================================")
print("Validation data ===============================================================")
test_model(x_validation, y_validation)
print("===============================================================================")
print("===============================================================================")
print("Tesing data ===================================================================")
test_model(x_test, y_test)