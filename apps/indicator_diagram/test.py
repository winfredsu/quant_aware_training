#!/usr/bin/env python

import tensorflow as tf
import numpy as np
import dataset
from config import *

tf.app.flags.DEFINE_string('frozen_pb', './frozen.pb', 'frozen pb name')
tf.app.flags.DEFINE_string('output_node', 'MobilenetV1/Squeeze', 'name of output name')

FLAGS = tf.app.flags.FLAGS

def test():
    sess = tf.InteractiveSession()

    # get test data
    _, _, ds_test = dataset.get_dataset()
    ds_test_iterator = ds_test.make_initializable_iterator()
    next_test_images, next_test_labels = ds_test_iterator.get_next()
    ds_test_iterator.initializer.run()

    # restore frozen graph
    gd = tf.GraphDef.FromString(open(FLAGS.frozen_pb, 'rb').read())
    images, logits = tf.import_graph_def(gd, return_elements = 
        ['images:0', FLAGS.output_node+':0'])
    labels = tf.placeholder(tf.int64, [BATCH_SIZE, ], name='labels')

    correct_pred = tf.equal(labels, tf.argmax(logits,1))
    acc_op = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    # run test
    total_test_acc = 0
    for i in range(0, TEST_SIZE, BATCH_SIZE):
        images_batch, labels_batch = sess.run([next_test_images, next_test_labels])
        test_acc = acc_op.eval(feed_dict={
            images: images_batch,
            labels: labels_batch
        })
        total_test_acc += test_acc * BATCH_SIZE / TEST_SIZE

    print('total_test_acc', total_test_acc)

def main(unused_arg):
    test()

if __name__ == '__main__':
    tf.app.run(main)
