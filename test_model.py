from keras import backend as K
import numpy as np
import logging
import tensorflow as tf
from Utilities import *
from model1 import ShefahModel
from preprocess_videos import *
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


def decode_predict_ctc(out, top_paths=5):
    beam_width = 200
    decoded = K.ctc_decode(
        out,
        input_length=np.ones(out.shape[0]) * out.shape[1],
        greedy=False,
        beam_width=beam_width,
        top_paths=top_paths,
    )
    paths = [path.numpy() for path in decoded[0]]

    res = []
    for output in paths[0]:
        res.append(translate_array_to_label(output))
    return res


if(sys.argv[1].lower() != "cross"):
    (
        x_train,
        y_train,
        x_validation,
        y_validation,
        x_test,
        y_test,
    ) = get_train_validation_test_paths(int(sys.argv[1]), int(sys.argv[2]))
else:
    (
        x_train,
        y_train,
        x_validation,
        y_validation,
        x_test,
        y_test,
    ) = cross_validation(int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]))
    checkpoints_dir = ".\\Cross_Val_Checkpoints\\Fold"+sys.argv[4]
    checkpoint_pattern = checkpoints_dir + "\\cp-{epoch:04d}.ckpt"


def test_data(x, y, shefah_model, print_info=False):

    labels = []
    predictions = []
    accuracy = 0
    videos = []
    actuals_as_numbers = []
    for i in range(len(x)):
        frames = load_video_frames(x[i])
        actual = translate_array_to_label(y[i])
        actual_as_numbers = int(translate_label_to_number(actual))
        videos.append(frames)
        actuals_as_numbers.append(actual_as_numbers)
    results = shefah_model.predict(np.array(videos))

    for result, actual_as_numbers in zip(results[0], actuals_as_numbers):
        top_pred_texts = decode_predict_ctc(np.array([result]))
        predicted = top_pred_texts[0]

        predicted_as_numbers = int(translate_label_to_number(predicted))

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

    # if print_info:
        # print("Training data =======================================================")
    train_a, train_c, train_m =(None,None,None) # test_data(x_train, y_train, shefah_model, print_info)
    validation_a, validation_c, validation_m = (None, None, None)
    # if len(x_validation) > 0:
    #     if print_info:
    #         print(
    #             "====================================================================="
    #         )
    #         print(
    #             "====================================================================="
    #         )
    #         print(
    #             "Validation data ====================================================="
    #         )
    #     validation_a, validation_c, validation_m = test_data(
    #         x_validation, y_validation, shefah_model, print_info
    #     )
    if print_info:
        print("=====================================================================")
        print("=====================================================================")
        print("Tesing data =========================================================")
    test_a, test_c, test_m = test_data(
        x_test, y_test, shefah_model, print_info)
    return train_a, train_m, validation_a, validation_m, test_a, test_m


if __name__ == "__main__":
    tf.get_logger().setLevel(logging.ERROR)
    if(sys.argv[1].lower() == "cross"):
        testAllCheckpoints = sys.argv[5].lower() == "true"
    else:
        testAllCheckpoints = sys.argv[3].lower() == "true"
    
    print(testAllCheckpoints)
    checkpoints = []
    if testAllCheckpoints:
        for file in find_files(checkpoints_dir, "*.ckpt.data*"):
            checkpoints.append(file.split(".data")[0])
    else:
        checkpoints = [tf.train.latest_checkpoint(
            os.path.dirname(checkpoint_pattern))]

    if len(checkpoints) == 0:
        print("No Weights Found")
    else:
        i = 0
        for checkpoint in checkpoints[-5:-1]:
            # if i < 40:
            #     i += 1
            #     continue
            # create model
            shefah_model = ShefahModel()
            model = shefah_model.model

            # compile model
            model.compile()
            model.load_weights(checkpoint)
            testModel = test_model(shefah_model, not testAllCheckpoints)
            print(checkpoint)

            print("Accuracy on Training = ", testModel[0])
            print("Confusion Matrix: ", testModel[1])
            print("Accuracy on Testing = ", testModel[4])
            print("Confusion Matrix: ", testModel[5])
            print("===============")
            tf.keras.backend.clear_session()
