# -*- coding: utf-8 -*-

'''
convert data to train, val, test(50000, 10000, 10000) pkl
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gzip
import os
import sys

import numpy as np
from six.moves import urllib
import tensorflow as tf
import pickle

os.environ["CUDA_VISIBLE_DEVICES"] = ""


_TRAIN_DATA_FILENAME = 'train-images-idx3-ubyte.gz'
_TRAIN_LABELS_FILENAME = 'train-labels-idx1-ubyte.gz'
_TEST_DATA_FILENAME = 't10k-images-idx3-ubyte.gz'
_TEST_LABELS_FILENAME = 't10k-labels-idx1-ubyte.gz'


_IMAGE_SIZE = 28
_NUM_CHANNELS = 1

# The names of the classes.
_CLASS_NAMES = [
    'zero',
    'one',
    'two',
    'three',
    'four',
    'five',
    'size',
    'seven',
    'eight',
    'nine',
]

def _extract_images(filename, num_images):
    """Extract the images into a numpy array.
    
    Args:
      filename: The path to an MNIST images file.
      num_images: The number of images in the file.
    
    Returns:
      A numpy array of shape [number_of_images, height, width, channels].
    """
    print('Extracting images from: ', filename)
    with gzip.open(filename) as bytestream:
        bytestream.read(16)
        buf = bytestream.read(
            _IMAGE_SIZE * _IMAGE_SIZE * num_images * _NUM_CHANNELS)
        data = np.frombuffer(buf, dtype=np.uint8)
        data = data.reshape(num_images, _IMAGE_SIZE, _IMAGE_SIZE, _NUM_CHANNELS)
    return data

def _extract_labels(filename, num_labels):
    """Extract the labels into a vector of int64 label IDs.
    
    Args:
      filename: The path to an MNIST labels file.
      num_labels: The number of labels in the file.
    
    Returns:
      A numpy array of shape [number_of_labels]
    """
    print('Extracting labels from: ', filename)
    with gzip.open(filename) as bytestream:
        bytestream.read(8)
        buf = bytestream.read(1 * num_labels)
        labels = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)
    return labels

dataset_dir = 'data'

data_filename = os.path.join(dataset_dir, _TRAIN_DATA_FILENAME)
labels_filename = os.path.join(dataset_dir, _TRAIN_LABELS_FILENAME)
train_num_images = 60000
train_images1 = _extract_images(data_filename, train_num_images)
train_labels1 = _extract_labels(labels_filename, train_num_images)

print(train_images1.shape)
print(train_images1.shape)

train_images = train_images1[:50000, :, :, :]
train_labels = train_labels1[:50000]
print(train_images.shape)
print(train_labels.shape)

val_images = train_images1[50000:, :, :, :]
val_labels = train_labels1[50000:]
print(val_images.shape)
print(val_labels.shape)

data_filename = os.path.join(dataset_dir, _TEST_DATA_FILENAME)
labels_filename = os.path.join(dataset_dir, _TEST_LABELS_FILENAME)
test_num_images = 10000
test_images = _extract_images(data_filename, test_num_images)
test_labels = _extract_labels(labels_filename, test_num_images)

pkl = {}
pkl['train_images'] = train_images
pkl['train_labels'] = train_labels

pkl['val_images'] = val_images
pkl['val_labels'] = val_labels

pkl['test_images'] = test_images
pkl['test_labels'] = test_labels


with open('data/mnist_3.pkl', 'wb') as f:
    pickle.dump(pkl, f)

with open('data/mnist_3.pkl', 'rb') as f:
    p = pickle.load(f)
    train_images = p['train_images']
    print(type(train_images))  # <class 'numpy.ndarray'>







