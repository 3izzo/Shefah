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

class ShefahModel(object):
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


x_train, y_train, x_test, y_test = get_train_test_data()

model = get_model()

loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=True)

model.compile(optimizer='adam',
              loss=loss_fn,
              metrics=['accuracy'])


checkpoint_path = ".\\Checkpoints\\cp-{epoch:04d}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)


# Create a callback that saves the model's weights every 1 epochs
cp_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path,
    verbose=1,
    save_weights_only=True,
    save_freq=2)

latest = tf.train.latest_checkpoint(checkpoint_dir)

if(latest):
    model.load_weights(latest)
    print("-------------------------------------")
    print("loaded weights from %s" % latest)
else:
    inputs = {'the_input': x_train,
              'the_labels': y_train,
              'input_length': np.array([86] * 60),
              'label_length': np.array([14] * 60),
              }
    outputs = {'ctc': np.zeros([60])}
    model.fit(inputs, outputs, epochs=5, callbacks=[cp_callback])


inputs = {'the_input': x_test,
          'the_labels': y_test,
          'input_length': np.array([86] * 20),
          'label_length': np.array([14] * 20),
          }
result = model.predict(inputs)
np.set_printoptions(threshold=sys.maxsize)
print(result)
decoded = K.ctc_decode(y_pred={'ctc': result}, input_length=np.array([86] * 20),
                       greedy=True)
print(decoded)
