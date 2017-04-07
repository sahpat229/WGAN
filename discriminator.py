import tensorflow as tf
import ops

slim = tf.contrib.slim

np.random.seed(1234)

class Discriminator():

	def dis_conv(input_map, filters, k_size, stride, activation):
		result = tf.layers.conv2d(
				input_map,
				filters,
				kernel_size=[k_size, k_size],
				strides=(stride, stride),
				padding='valid',
				activation=activation,
				kernel_initializer=tf.contrib.layers.variance_scaling_initializer()
			)
		return result

	def discriminator(x, num_chars):
		"""
		- Compute D(x) or D(x_hat) as given in the Wasserstein paper
		- x is of shape [batch_size, im_width, im_height, im_channels]
		- returns D(x) of shape [batch_size, num_chars+1] (all fonts + fake)
		"""
		result = Discriminator.dis_conv(x, 128, 5, 2, ops.lrelu)
		result = Discriminator.dis_conv(result, 256, 5, 2, ops.lrelu)
		result = Discriminator.dis_conv(result, 512, 5, 2, ops.lrelu)
		result = Discriminator.dis_conv(result, 1024, 5, 1, ops.lrelu)
		result = slim.fully_connected(result, num_chars+1, activation_fn=ops.lrelu)
		return result
