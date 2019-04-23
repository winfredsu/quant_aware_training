import tensorflow as tf
import numpy as np
import pickle
import os
from config import *

AUTOTUNE = tf.data.experimental.AUTOTUNE

def bin2array(filename):
    data = pickle.load(open(filename, 'rb'), encoding='bytes')
    images = data[b'data']
    num_images = images.shape[0]
    images = images.reshape((num_images, 3, 32, 32))
    images = np.transpose(images, (0,2,3,1))
    labels = np.array(data[b'labels'])
    return images, labels

def preprocess(image, label):
    # resize and scale to [-1,1)
    image = tf.image.resize(image, IMG_SHAPE[:2])
    image -= 128.0
    image /= 128.0
    return image, label

def get_dataset():
    # here we merge 'data_batch' and 'test_batch'
    images = np.zeros((0,32,32,3))
    labels = np.zeros((0,))
    for i in range(1,6):
        images_batch, labels_batch = bin2array('./dataset/data_batch_'+str(i))
        images = np.concatenate((images, images_batch), axis=0)
        labels = np.concatenate((labels, labels_batch), axis=0) 
    images_batch, labels_batch = bin2array('./dataset/test_batch')
    images = np.concatenate((images, images_batch), axis=0)
    labels = np.concatenate((labels, labels_batch), axis=0) 
    
    # split into train/val/test sets
    ds_train = tf.data.Dataset.from_tensor_slices((images[:TRAIN_SPLIT], labels[:TRAIN_SPLIT])).map(preprocess)
    ds_train = ds_train.cache()
    ds_train = ds_train.apply(tf.data.experimental.shuffle_and_repeat(buffer_size=10000))
    ds_train = ds_train.batch(BATCH_SIZE).prefetch(AUTOTUNE)
    ds_val   = tf.data.Dataset.from_tensor_slices((images[TRAIN_SPLIT:VAL_SPLIT], labels[TRAIN_SPLIT:VAL_SPLIT])).map(preprocess)
    ds_val   = ds_val.cache().apply(tf.data.experimental.shuffle_and_repeat(buffer_size=10000))
    ds_val   = ds_val.batch(BATCH_SIZE).prefetch(AUTOTUNE)
    ds_test  = tf.data.Dataset.from_tensor_slices((images[VAL_SPLIT:TEST_SPLIT], labels[VAL_SPLIT:TEST_SPLIT])).map(preprocess)
    ds_test  = ds_test.cache().apply(tf.data.experimental.shuffle_and_repeat(buffer_size=10000))
    ds_test  = ds_test.batch(BATCH_SIZE).prefetch(AUTOTUNE)

    return ds_train, ds_val, ds_test

