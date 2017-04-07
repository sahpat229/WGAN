import h5py
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

slim =  tf.contrib.slim

np.random.seed(1234)

class Generator():

	def generator(z):
		"""
		- Compute G(z) as given in the Wasserstein paper
		- z is of shape [batch_size, num_classes + latent_dim]

		TODO: change to nearest neighbor upsampling instead of deconvolution
		"""

		result = fully_connected(z, 4*4*1024)
		result = tf.reshape(result, [-1, 4, 4, 1024])
		result = tf.layers.conv2d_transpose(
				result,
				512,
				kernel_size=[5, 5],
				strides=(1, 1),
				activation=tf.nn.relu,
				kernel_initializer=tf.contrib.layers.variance_scaling_initializer()
			)
		result = tf.layers.conv2d_transpose(
				result,
				256,
				kernel_size=[5, 5],
				strides=(2, 2),
				activation=tf.nn.relu,
				kernel_initializer=tf.contrib.layers.variance_scaling_initializer()
			)
		result = tf.layers.conv2d_transpose(
				result,
				128,
				kernel_size=[5, 5],
				strides=(2, 2),
				activation=tf.nn.relu,
				kernel_initializer=tf.contrib.layers.variance_scaling_initializer()
			)
		result = tf.layers.conv2d_transpose(
				result,
				3,
				kernel_size=[5, 5],
				strides=(2, 2),
				activation=tf.nn.tanh,
				kernel_initializer=tf.contrib.layers.variance_scaling_initializer()
			)
		return result