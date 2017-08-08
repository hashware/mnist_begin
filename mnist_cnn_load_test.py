'''Trains a simple convnet on the MNIST dataset.

Gets to 99.25% test accuracy after 12 epochs
(there is still a lot of margin for parameter tuning).
16 seconds per epoch on a GRID K520 GPU.

from https://github.com/fchollet/keras/blob/master/examples/mnist_cnn.py
'''

from __future__ import print_function

import os
import tensorflow as tf
from tensorflow.contrib.keras.python import keras
from tensorflow.contrib.keras.python.keras.models import Sequential
from tensorflow.contrib.keras.python.keras.layers import Dense, Dropout, Flatten
from tensorflow.contrib.keras.python.keras.layers import Conv2D, MaxPooling2D
from tensorflow.contrib.keras.python.keras import backend as K
from dataset import mnist
from tensorflow.contrib import slim
import numpy as np
import pickle
import mnist_data


os.environ["CUDA_VISIBLE_DEVICES"] = "0"
gpu_options = tf.GPUOptions(allow_growth=True)
config = tf.ConfigProto(gpu_options=gpu_options)
session = tf.Session(config=config)
K.set_session(session)


(x_train, y_train), (x_val, y_val), (x_test, y_test) = mnist_data.get_datas()

model = keras.models.load_model('keras_mnist_cnn_model.h5')

score = model.evaluate(x_test, y_test, verbose=1)
print('Test loss:', score[0])
print('Test accuracy:', score[1])


