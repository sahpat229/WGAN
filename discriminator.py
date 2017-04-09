import tensorflow as tf
import ops
import numpy as np

slim = tf.contrib.slim

np.random.seed(1234)

class Discriminator():

	def dis_conv(input_map, filters, k_size, stride, activation, var_coll):
		result = slim.layers.convolution(
				input_map,
				filters,
				kernel_size=[k_size, k_size],
				stride=stride,
				activation_fn=activation,
				weights_initializer=tf.contrib.layers.variance_scaling_initializer(),
				variables_collections=var_coll

			)

		# result = tf.layers.conv2d(
		# 		input_map,
		# 		filters,
		# 		kernel_size=[k_size, k_size],
		# 		strides=(stride, stride),
		# 		padding='valid',
		# 		activation=activation,
		# 		kernel_initializer=tf.contrib.layers.variance_scaling_initializer()
		# 	)
		return result

	def discriminator(x, batch_size, num_chars, var_coll):
		"""
		- Compute D(x) or D(x_hat) as given in the Wasserstein paper
		- x is of shape [batch_size, im_width, im_height, im_channels]
		- returns D(x) of shape [batch_size, num_chars+1] (all fonts + fake)
		"""
		result = Discriminator.dis_conv(x, 128, 2, 2, ops.lrelu, var_coll)
		result = Discriminator.dis_conv(result, 256, 2, 2, ops.lrelu, var_coll)
		result = Discriminator.dis_conv(result, 512, 2, 2, ops.lrelu, var_coll)
		result = Discriminator.dis_conv(result, 1024, 2, 1, ops.lrelu, var_coll)
		result = tf.reshape(result, [batch_size, -1])
		result = slim.fully_connected(result, num_chars+1, activation_fn=ops.lrelu, 
			variables_collections=var_coll)
		return result
