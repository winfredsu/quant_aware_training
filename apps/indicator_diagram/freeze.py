#!/usr/bin/env python

import tensorflow as tf
import os
import sys
import model
from config import *

sys.path.append('../..')
import quantize

tf.app.flags.DEFINE_string('train_dir', './train', 'training directory')
tf.app.flags.DEFINE_string('output_dir', '.', 'where to write the frozen pb')
tf.app.flags.DEFINE_string('ckpt', 'model.quant.ckpt', 'ckpt to be frozen')
tf.app.flags.DEFINE_string('output_node', 'MobilenetV1/Squeeze', 'name of output name')
tf.app.flags.DEFINE_string('frozen_pb_name', 'frozen.pb', 'output pb name')
FLAGS = tf.app.flags.FLAGS

def freeze():
    if os.path.exists(FLAGS.train_dir) == False:
        os.mkdir(FLAGS.train_dir)

    sess = tf.InteractiveSession()
    images = tf.placeholder(tf.float32, [None]+IMG_SHAPE, name='images')
    logits = model.mobilenet_v1(images, num_classes=NUM_CLASSES, depth_multiplier=DEPTH_MULTIPLIER, dropout_prob=0.0, is_training=False)

    # write a inference graph (for debug)
    # with tf.io.gfile.GFile(os.path.join(FLAGS.output_dir, 'inference_graph_before_quant.pb'), 'wb') as f:
    #     f.write(sess.graph_def.SerializeToString())

    quantize.create_eval_graph()
    
    # write a inference graph (for debug)
    # with tf.io.gfile.GFile(os.path.join(FLAGS.output_dir, 'inference_graph.pb'), 'wb') as f:
    #     f.write(sess.graph_def.SerializeToString())

    # write frozen graph
    saver = tf.train.Saver(tf.global_variables())
    saver.restore(sess, os.path.join(FLAGS.train_dir, FLAGS.ckpt))

    frozen_gd = tf.graph_util.convert_variables_to_constants(
        sess, sess.graph_def, [FLAGS.output_node])
    tf.train.write_graph(
        frozen_gd,
        FLAGS.output_dir,
        FLAGS.frozen_pb_name,
        as_text=False)

def main(unused_arg):
    freeze()

if __name__ == '__main__':
    tf.app.run(main)
