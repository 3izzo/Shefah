from keras import backend as K
import numpy as np
import tensorflow as tf
from Utilities import *
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


def test_data(x, y, shefah_model, print_info=False):

    labels = []
    predictions = []
    accuracy = 0
    for i in range(len(x)):
        frames = load_video_frames(x[i])
        video = np.array([frames])
        result = shefah_model.predict(video)
        # decoded1, decoded2 = K.ctc_decode(y_pred=result2, input_length=np.array([max_frame_count]),
        #                                   greedy=True)

        # paths = [path.numpy() for path in decoded1[0]]
        # # print(paths)
        # print(result1)
        top_pred_texts = decode_predict_ctc(result)
        predicted = top_pred_texts[0]
        actual = translate_array_to_label(y[i])
        predicted_as_numbers = int(translate_label_to_number(predicted))
        actual_as_numbers = int(translate_label_to_number(actual))

        labels.append(actual_as_numbers)
        predictions.append(predicted_as_numbers)
        if actual_as_numbers == predicted_as_numbers:
            accuracy += 1
        if print_info:
            print(
                "Predicted: ",
                predicted_as_numbers,
                predicted,
                ", Actual:",
                actual_as_numbers,
                actual,
            )
    confusion_matrix = tf.math.confusion_matrix(labels, np.array(predictions))
    accuracy = accuracy / len(y)
    count_labels = np.zeros((10))
    for label in labels:
        count_labels[label] += 1
    if print_info:
        print("Accuracy: ", accuracy)
        print("Count: ", count_labels)
        print("Confusion Matrix:")
        print(confusion_matrix)

    return accuracy, count_labels, confusion_matrix


def test_model(shefah_model, print_info=False):

    if print_info:
        print("Training data =======================================================")
    train_a, train_c, train_m = test_data(x_train, y_train, shefah_model, print_info)
    validation_a, validation_c, validation_m = (None, None, None)
    if len(x_validation) > 0:
        if print_info:
            print(
                "====================================================================="
            )
            print(
                "====================================================================="
            )
            print(
                "Validation data ====================================================="
            )
        validation_a, validation_c, validation_m = test_data(
            x_validation, y_validation, shefah_model, print_info
        )
    if print_info:
        print("=====================================================================")
        print("=====================================================================")
        print("Tesing data =========================================================")
    test_a, test_c, test_m = test_data(x_test, y_test, shefah_model, print_info)
    return train_a, train_m, validation_a, validation_m, test_a, test_m


if __name__ == "__main__":
    # create model
    shefah_model = ShefahModel()
    model = shefah_model.model

    # compile model
    model.compile()

    checkpoint_dir = os.path.dirname(checkpoint_pattern)

    latest = tf.train.latest_checkpoint(checkpoint_dir)
    if latest:
        model.load_weights(latest)
        print("-------------------------------------")
        print("loaded weights from %s" % latest)
    else:
        print("No weights found")

    test_model(shefah_model, True)
