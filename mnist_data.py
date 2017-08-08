# -*- coding: utf-8 -*-

'''

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



os.environ["CUDA_VISIBLE_DEVICES"] = "0"
gpu_options = tf.GPUOptions(allow_growth=True)
config = tf.ConfigProto(gpu_options=gpu_options)
session = tf.Session(config=config)
K.set_session(session)



batch_size = 128
num_classes = 10
epochs = 10

# input image dimensions
img_rows, img_cols = 28, 28

(x_train, y_train), (x_test, y_test) = ([] , []) , ([], [])
(x_val, y_val) = ([], [])
data_dir = './dataset/data'


with open(data_dir + '/mnist_3.pkl', 'rb') as f:
    p = pickle.load(f)
    x_train = p['train_images']
    y_train = p['train_labels']
    x_val = p['val_images']
    y_val = p['val_labels']
    
    x_test = p['test_images']
    y_test = p['test_labels']




# the data, shuffled and split between train and test sets
# (x_train, y_train), (x_test, y_test) = mnist.load_data()

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_val = x_val.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_val /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_val = keras.utils.to_categorical(y_val, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)


def get_datas():
    return (x_train, y_train), (x_val, y_val), (x_test, y_test)

def get_shape():
    return input_shape





