import tensorflow as tf
from sklearn.model_selection import train_test_split
import random
from tensorflow.contrib.data import prefetch_to_device, shuffle_and_repeat, map_and_batch
import numpy as np
import pickle
import os
from config import *


AUTOTUNE =tf.contrib.data.AUTOTUNE


def load_imgfiles(paths):
    data = []
    labels = []
    for path in paths:
        file = os.listdir(path)
        for Labelimgpath in file:
            childerfile = os.listdir(DATA_DIR + Labelimgpath + '/')
            for imgPath in childerfile:
                data.append(DATA_DIR + Labelimgpath + '/' + imgPath)
                labels.append(int(Labelimgpath))
    image_list = data
    label_list = labels
    temp = np.array([image_list, label_list])
    temp = temp.transpose()
    np.random.shuffle(temp)
    np.random.shuffle(temp)
    all_image_list = temp[:, 0]
    all_label_list = temp[:, 1]
    n_sample = len(all_label_list)
    n_train = int(n_sample)  # number of trainning samples
    imgs = all_image_list[0:n_train]
    labels = all_label_list[0:n_train]
    labels = [int(float(i)) for i in labels]
    return imgs, labels

def load_data(path):
    imgs,labels =load_imgfiles(path)
    return imgs,labels


def augmentation(image):
    p = random.randint(0, 10)
    seed = random.randint(0, 2 ** 31 - 1)
    if p > 5:
        image = tf.image.random_flip_left_right(image, seed=seed)
    else:
        image = tf.image.random_flip_up_down(image, seed=seed)
    return image

def image_processing(filename, label):
    x = tf.read_file(filename)
    x_decode = tf.image.decode_jpeg(x, channels=3)
    img = tf.image.resize_images(x_decode, [160,160])
    img = img - 128.0
    img = img / 128.0
    augment_flag = False
    if augment_flag:
        p = random.random()
        if p > 0.5:
            img = augmentation(img)

    return img, label
def prepare_ds():
    imgs, labels = load_data([DATA_DIR])
    SEED = np.random.randint(1024)
    img_paths_train, img_paths_val, labels_train, labels_val = train_test_split(imgs, labels, train_size=0.6,stratify=labels, random_state=SEED)
    ds_train = tf.data.Dataset.from_tensor_slices((img_paths_train, labels_train)).map(image_processing)
    ds_train = ds_train.cache()
    ds_train = ds_train.apply(tf.data.experimental.shuffle_and_repeat(buffer_size=4096))
    ds_train = ds_train.batch(BATCH_SIZE).prefetch(AUTOTUNE)

    ds_val  = tf.data.Dataset.from_tensor_slices((img_paths_val, labels_val)).map(image_processing)
    ds_val  = ds_val.cache()
    ds_val  = ds_val.apply(tf.data.experimental.shuffle_and_repeat(buffer_size=4096))
    ds_val  = ds_val.batch(BATCH_SIZE).prefetch(AUTOTUNE)

    return ds_train, ds_val





