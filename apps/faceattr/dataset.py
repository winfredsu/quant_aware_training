import tensorflow as tf
import numpy as np
import os
from config import *

AUTOTUNE = tf.data.experimental.AUTOTUNE

def load_labels(labels_path):
    f = open(labels_path)
    # line 1: number of images
    num_imgs = int(f.readline())
    # line 2: attribute names, 40 in total
    attr_names = f.readline().split()
    col_idx = [i for i,x in enumerate(attr_names) if x in LABEL_NAMES]
    # line 3 to end: 00xx.jpg -1 1 -1 1 ...
    labels = []
    for i in range(DATASET_SIZE):
        labels.append(list(map(np.float32, f.readline().split()[1:])))
    labels = np.array(labels)
    labels[labels<0] = 0
    labels = labels[:, col_idx]
    return labels[:TRAIN_SPLIT], labels[TRAIN_SPLIT:VAL_SPLIT], labels[VAL_SPLIT:]

def load_imgs(imgs_dir):
    img_paths = os.listdir(imgs_dir)
    img_paths.sort()
    img_paths = img_paths[:DATASET_SIZE]
    for i in range(len(img_paths)):
        img_paths[i] = os.path.join(imgs_dir,img_paths[i])
    return img_paths[:TRAIN_SPLIT], img_paths[TRAIN_SPLIT:VAL_SPLIT], img_paths[VAL_SPLIT:]

def preprocess(img_path, label):
    img = tf.io.read_file(img_path)
    # uint8 range: [0,255]
    img = tf.image.decode_jpeg(img, channels=IMG_SHAPE[2])
    # new range: [-1.0,1.0)
    img = tf.image.resize(img, IMG_SHAPE[:2])
    img -= 128.0
    img /= 128.0
    return img, label

def get_dataset():
    img_paths_train, img_paths_val, img_paths_test = load_imgs('./dataset/img_celeba_face')
    labels_train, labels_val, labels_test = load_labels('./dataset/labels_celeba_face.txt')

    ds_train = tf.data.Dataset.from_tensor_slices((img_paths_train, labels_train)).map(preprocess)
    ds_train = ds_train.apply(tf.data.experimental.shuffle_and_repeat(buffer_size=4096))
    ds_train = ds_train.batch(BATCH_SIZE).prefetch(AUTOTUNE)

    ds_val  = tf.data.Dataset.from_tensor_slices((img_paths_val, labels_val)).map(preprocess)
    ds_val  = ds_val.apply(tf.data.experimental.shuffle_and_repeat(buffer_size=4096))
    ds_val  = ds_val.batch(BATCH_SIZE).prefetch(AUTOTUNE)

    ds_test = tf.data.Dataset.from_tensor_slices((img_paths_test, labels_test)).map(preprocess)
    ds_test = ds_test.apply(tf.data.experimental.shuffle_and_repeat(buffer_size=4096))
    ds_test = ds_test.batch(BATCH_SIZE).prefetch(AUTOTUNE)

    return ds_train, ds_val, ds_test
