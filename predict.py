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

# create model
shefah_model = ShefahModel()
model = shefah_model.model

# compile model
model.compile()


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


video_path = ".\\Videos\\speaker1\\7.mp4"

frames = []
i = 0
print("preprocessing video")
for frame in get_video_frames(video_path):
    cropped_frame = cv2.cvtColor(
        get_frames_mouth(face_detector, predictor, frame), cv2.COLOR_BGR2RGB
    )
    frames.append(cropped_frame)
    i += 1
while i < max_frame_count:
    frames.append(np.zeros((frame_h, frame_w, 3)))
    i += 1
frames = np.array([frames])
print("preprocessing done")

y_prediction = shefah_model.predict(frames)
decoded_prediction = decode_predict_ctc(y_prediction)
predicted = decoded_prediction[0]
predicted_as_numbers = translate_label_to_number(predicted)
print("===============================================================================")
print(predicted_as_numbers, predicted)
