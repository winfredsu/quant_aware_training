#!/usr/bin/env python

import tensorflow as tf
import numpy as np
import PIL
from config import *

tf.app.flags.DEFINE_string('frozen_pb', './frozen.pb', 'frozen pb name')
tf.app.flags.DEFINE_string('output_node', 'MobilenetV1/Squeeze', 'name of output name')
tf.app.flags.DEFINE_string('input_image', 'test.png', 'input image')

FLAGS = tf.app.flags.FLAGS

def test():
    sess = tf.InteractiveSession()

    # get the image to be tested
    # image_test = np.array(PIL.Image.open(FLAGS.input_image).resize(IMG_SHAPE[:2])).astype(np.float32)/128.0-1.0
    image = tf.image.decode_jpeg(tf.read_file(FLAGS.input_image),channels=3)
    image = tf.image.resize_images(image, IMG_SHAPE[:2])
    image = tf.expand_dims(image, 0)
    image = image / 128.0 - 1.0

    # restore frozen graph
    gd = tf.GraphDef.FromString(open(FLAGS.frozen_pb, 'rb').read())
    images, logits = tf.import_graph_def(gd, return_elements = 
        ['images:0', FLAGS.output_node+':0'])
    pred = tf.argmax(logits,1)
    pred_val = pred.eval(feed_dict={
        images: image.eval()
    })    
    print('Class Predicted: ', pred_val)

    # print the image
    
    image_test = image.eval().reshape(160,160,3)
    image_test = 128 * (image_test + 1)
    print(image_test)
    im = PIL.Image.fromarray(image_test.astype(np.uint8))
    im.show()


def main(unused_arg):
    test()

if __name__ == '__main__':
    tf.app.run(main)
