#!/usr/bin/env python

import tensorflow as tf
import numpy as np
import PIL
from config import *


tf.app.flags.DEFINE_string('frozen_pb', './frozen.pb', 'frozen pb name')
tf.app.flags.DEFINE_string('output_node', 'MobilenetV1/Squeeze', 'name of output name')
tf.app.flags.DEFINE_string('input_image', 'test.jpg', 'input image')

FLAGS = tf.app.flags.FLAGS

def load_labels(labels_path):
    f = open(labels_path)
    # line 1: number of images
    num_imgs = int(f.readline())
    # line 2: attribute names, 40 in total
    attr_names = f.readline().split()
    col_idx = [i for i,x in enumerate(attr_names) if x in LABEL_NAMES]
    label_names = [attr_names[i] for i in col_idx]
    # line 3 to end: 00xx.jpg -1 1 -1 1 ...
    labels = []
    for i in range(DATASET_SIZE):
        labels.append(list(map(np.float32, f.readline().split()[1:])))
    labels = np.array(labels)
    labels[labels<0] = 0
    labels = labels[:, col_idx]
    return labels, label_names

def test():
    sess = tf.InteractiveSession()

    # get the image to be tested
    # image_test = np.array(PIL.Image.open(FLAGS.input_image).resize(IMG_SHAPE[:2])).astype(np.float32)/128.0-1.0
    image = tf.image.decode_jpeg(tf.read_file(FLAGS.input_image),channels=3)
    image = tf.image.resize_images(image, IMG_SHAPE[:2])
    image = tf.expand_dims(image, 0)
    image = image / 128.0 - 1.0

    # get the ground truth
    labels, label_names = load_labels('./dataset/list_attr_celeba.txt')
    gt_val = labels[int(FLAGS.input_image.split('.')[0].split('/')[1])-1]

    # restore frozen graph
    gd = tf.GraphDef.FromString(open(FLAGS.frozen_pb, 'rb').read())
    images, logits = tf.import_graph_def(gd, return_elements = 
        ['images:0', FLAGS.output_node+':0'])
    pred = tf.round(tf.sigmoid(logits))
    pred_val = pred.eval(feed_dict={
        images: image.eval()
    })    
    print('Label Names: ', label_names)
    print('Class Predicted: ', pred_val)
    print('Class Labled: ', gt_val)

    # print the image
    
    # image_test = image.eval().reshape(160,160,3)
    # image_test = 128 * (image_test + 1)
    # print(image_test)
    # im = PIL.Image.fromarray(image_test.astype(np.uint8))
    # im.show()


def main(unused_arg):
    test()

if __name__ == '__main__':
    tf.app.run(main)
