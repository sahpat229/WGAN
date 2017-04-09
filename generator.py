import h5py
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

slim =  tf.contrib.slim

np.random.seed(1234)

class Generator():

	def gen_conv(input_map, new_size, filters, k_size, stride, 
		activation, is_training, var_coll, upd_coll):

		result = slim.layers.batch_norm(
				input_map,
				is_training=is_training,
				scale=True,
				variables_collections=var_coll,
				updates_collections=upd_coll
			)

		# result = tf.layers.batch_normalization(
		# 		input_map,
		# 		training=is_training,
		# 		name=name+'_batch'
		# 	)

		result = tf.image.resize_images(result,
				[new_size, new_size],
				method=tf.image.ResizeMethod.NEAREST_NEIGHBOR
			)

		result = slim.layers.convolution(
				result,
				filters,
				kernel_size=[k_size, k_size],
				stride=stride,
				padding='SAME',
				activation_fn=activation,
				weights_initializer=tf.contrib.layers.variance_scaling_initializer(),
				variables_collections=var_coll
			)

		# result = tf.layers.conv2d(
		# 		result,
		# 		filters,
		# 		kernel_size=[k_size, k_size],
		# 		strides=(stride, stride),
		# 		padding='same',
		# 		activation=activation,
		# 		kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
		# 		name=name+'_conv'
		# 	)
		return result

	def generator(z, is_training, var_coll, upd_coll):
		"""
		- Compute G(z) as given in the Wasserstein paper
		- z is of shape [batch_size, num_classes + latent_dim]
		- returns x_hat
		- remember, batch norm in the generator, no batch norm in the critic
		"""
		result = slim.fully_connected(z, 4*4*1024, variables_collections=var_coll)
		result = tf.reshape(result, [-1, 4, 4, 1024])
		result = Generator.gen_conv(result, 8, 512, 5, 1, tf.nn.relu, is_training, var_coll, upd_coll)
		result = Generator.gen_conv(result, 16, 256, 5, 1, tf.nn.relu, is_training, var_coll, upd_coll)
		result = Generator.gen_conv(result, 32, 128, 5, 1, tf.nn.relu, is_training, var_coll, upd_coll)
		result = Generator.gen_conv(result, 64, 3, 5, 1, tf.nn.tanh, is_training, var_coll, upd_coll)
		return result
