import h5py
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

slim =  tf.contrib.slim

np.random.seed(1234)

class Generator():

	def gen_conv(input_map, new_size, filters, k_size, stride, 
		activation, is_training):

		result = tf.layers.batch_normalization(
				input_map,
				training=is_training
			)

		result = tf.image.resize_images(result,
				[new_size, new_size],
				method=ResizeMethod.NEAREST_NEIGHBOR
			)

		result = tf.layers.conv2d(
				result,
				filters,
				kernel_size=[k_size, k_size],
				strides=(stride, stride),
				padding='same',
				activation=activation,
				kernel_initializer=tf.contrib.layers.variance_scaling_initializer()
			)
		return result

	def generator(z, is_training):
		"""
		- Compute G(z) as given in the Wasserstein paper
		- z is of shape [batch_size, num_classes + latent_dim]
		- returns x_hat
		- remember, batch norm in the generator, no batch norm in the critic
		"""
		result = slim.fully_connected(z, 4*4*1024)
		result = tf.reshape(result, [-1, 4, 4, 1024])
		result = Generator.gen_conv(result, 8, 512, 5, 2, tf.nn.relu, is_training)
		result = Generator.gen_conv(result, 16, 256, 5, 2, tf.nn.relu, is_training)
		result = Generator.gen_conv(result, 32, 128, 5, 2, tf.nn.relu, is_training)
		result = Generator.gen_conv(result, 64, 3, 5, 2, tf.nn.tanh, is_training)
		return result
