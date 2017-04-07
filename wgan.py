import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from data import Fonts, Latent
from discriminator import Discriminator
from generator import Generator

np.random.seed(1234)

class WGAN():
	"""
	Improved Wasserstein GAN Model
	"""

	def __init__(self, sess, path, latent_dim, num_classes, batch_size, learning_rate):
		"""
		- sess : tf.Session
		- path: path to fonts file
		- latent_dim: dimension of latent space for z
		- num_classes: number of different classes the discriminator should
			output over.  Does not include the +1 for fake
		- batch_size: batch size
		- learning_rate: learning rate of both discriminator and generator
		"""
		self.sess = sess
		self.real_data = Fonts(path, batch_size)
		self.latent = Latent(self.real_data.num_chars, latent_dim, batch_size)
		self.num_classes = num_classes
		self.batch_size = batch_size
		self.learning_rate = learning_rate

	def serve_epsilon(self):
		return np.random.uniform(size=self.batch_size)

	def build_model(self):
		self.x = tf.placeholder(tf.float32, shape=[self.batch_size, 64, 64, 3])
		self.labels = tf.placeholder(tf.float32, shape=[self.batch_size, self.num_classes+1])
		self.z = tf.placeholder(tf.float32, 
			shape=[self.batch_size, self.latent.output_size])
		self.epsilon = tf.placeholder(tf.float32, shape=[self.batch_size])
		self.is_training = tf.placeholder(tf.bool, shape=[])

		disc_output_x = Discriminator.discriminator(self.x, self.num_classes)
		generator_output = Generator.generator(self.z, self.is_training)
		disc_output_gz = Discriminator.discriminator(generator_output, self.num_classes)

		x_hat = tf.multiply(self.epsilon, self.x) + 
			tf.multiply(1-self.epsilon, self.generator_output)


		# discriminator(generator_output_inner) will be of size:
		# 	[batch_size, num_classes+1]
		# labels will be of shape:
		#	[batch_size, num_classes+1]

		self.generator_loss = tf.reduce_sum(tf.multiply(disc_output_gz, self.labels), axis=1) + \
			tf.reduce_sum(tf.multiply(disc_output_gz, self.labels-1), axis=1)
		self.generator_loss = tf.reduce_sum(self.generator_loss)


		critic_generator_loss = tf.reduce_sum(tf.multiply(disc_output_x, self.labels), axis=1) + \
			tf.reduce_sum(tf.multiply(disc_output_x, self.labels-1), axis=1)


		self.critic_loss = Discriminator.discriminator(generator_output_inner) - \
			discriminator_output
