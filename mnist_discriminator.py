import tensorflow as tf
import numpy as np

slim = tf.contrib.slim

class MNIST_Discriminator():
	
	def discriminator(x, labels, batch_size, var_coll, fc_space_size):
		result = slim.layers.convolution(
				x,
				32,
				kernel_size=[5, 5],
				stride=1,
				padding='SAME',
				activation_fn=tf.nn.relu,
				weights_initializer=tf.contrib.layers.variance_scaling_initializer(),
				variables_collections=var_coll
			)

		result = slim.layers.max_pool2d(result, [2, 2])

		result = slim.layers.convolution(
				x,
				64,
				kernel_size=[5, 5],
				stride=1,
				padding='SAME',
				activation_fn=tf.nn.relu,
				weights_initializer=tf.contrib.layers.variance_scaling_initializer(),
				variables_collections=var_coll
			)

		result = tf.reshape(result, [batch_size, -1])
		result = slim.layers.fully_connected(result, fc_space_size, activation_fn=tf.nn.relu,
			variables_collections=var_coll)
		labels = slim.fully_connected(labels, fc_space_size, activation_fn=tf.nn.relu,
			variables_collections=var_coll)
		result = tf.concat([result, labels], 1)
		result = slim.layers.fully_connected(result, 2*fc_space_size,
			activation_fn=tf.nn.relu, variables_collections=var_coll)
		result = slim.layers.fully_connected(result, 1, activation_fn=None,
			variables_collections=var_coll)
		return result