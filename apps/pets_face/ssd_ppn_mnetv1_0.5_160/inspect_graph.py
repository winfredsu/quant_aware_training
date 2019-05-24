import tensorflow as tf
import os

from tensorflow.python.platform import gfile
model = 'inference/frozen_inference_graph.pb'
graph = tf.get_default_graph()
graph_def = graph.as_graph_def()
graph_def.ParseFromString(gfile.FastGFile(model, 'rb').read())
tf.import_graph_def(graph_def, name='graph')
summaryWriter = tf.summary.FileWriter('log/', graph)

os.system("tensorboard --logdir log")