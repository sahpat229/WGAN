import tensorflow as tf
import numpy as np

slim = tf.contrib.slim

class MNIST_Generator():

	def generator(z, is_training, var_coll, upd_coll):
		result = slim.fully_connected(z, 500, variables_collections=var_coll, activation_fn=tf.nn.relu,
			normalizer_fn=slim.layers.batch_norm, normalizer_params={'is_training':is_training, 'updates_collections':upd_coll})
		result = slim.fully_connected(z, 28*28*1, variables_collections=var_coll, activation_fn=tf.nn.relu,
			normalizer_fn=slim.layers.batch_norm, normalizer_params={'is_training':is_training, 'updates_collections':upd_coll})
		result = tf.reshape(result, [-1, 28, 28, 1])
		return result