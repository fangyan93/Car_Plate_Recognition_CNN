import numpy as np
import random
import tensorflow as tf
import time
import pickle
import matplotlib.pyplot as plt


def load_model(session, meta_file, checkpoint_file):
  """ Loads a saved TF model from a file.
  Args:
    session: The tf.Session to use.
    meta_file: The .meta file for the network.
    checkpoint_file: The .ckpt file for the network. """
  print("Loading model from file '%s'..." % (meta_file))

  saver = tf.train.import_meta_graph(meta_file)
  # Modify the checkpoint file path so it can find it.
  search_path = "./%s" % (checkpoint_file.split(".")[0])
  saver.restore(session, search_path)


def recognize(x):
	# x: 20,30
	x = np.reshape(x, [1, 20, 30])
	x = np.expand_dims(x,3)
	with tf.Session() as sess:
		load_model(sess, "my_model.meta", "my_model.data-00000-of-00001")
		all_var = tf.get_collection('validation_nodes')
		label = sess.run(all_var[1], feed_dict={all_var[0]:x})
		char = str(chr(int(label))) 

	return char  

def recog(x, sess, all_var):
  x = np.reshape(x, [1, 20, 30])
  x = np.expand_dims(x,3)
  label = sess.run(all_var[1], feed_dict={all_var[0]:x})
 

  return label  