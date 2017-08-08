
from __future__ import print_function

import os
import tensorflow as tf
import mnist
from tensorflow.contrib import slim
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = ""

(x_train, y_train), (x_test, y_test) = ([] ,[]) ,([], [])

data_dir = './data'

with tf.Graph().as_default():
    dataset = mnist.get_split('train', data_dir)
    data_provider = slim.dataset_data_provider.DatasetDataProvider(
            dataset, common_queue_capacity=32, common_queue_min=1)
    image, label = data_provider.get(['image', 'label'])
    with tf.Session() as sess:
        with slim.queues.QueueRunners(sess):
            for i in range(60000):
                x_train1, y_train1 = sess.run([image, label])
                x_train.append(x_train1)
                y_train.append(y_train1)


with tf.Graph().as_default():
    dataset = mnist.get_split('test', data_dir)
    data_provider = slim.dataset_data_provider.DatasetDataProvider(
            dataset, common_queue_capacity=32, common_queue_min=1)
    image, label = data_provider.get(['image', 'label'])
    with tf.Session() as sess:
        with slim.queues.QueueRunners(sess):
            for i in range(10000):
                x_train1, y_train1 = sess.run([image, label])
                x_test.append(x_train1)
                y_test.append(y_train1)


x_train = np.array(x_train)
y_train = np.array(y_train)
x_test = np.array(x_test)
y_test = np.array(y_test)

f = {}
f['x_train'] = x_train
f['y_train'] = y_train
f['x_test'] = x_test
f['y_test'] = y_test

np.save('aaa.npz', f)

print('end')

