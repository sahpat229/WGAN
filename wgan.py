import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from data import Fonts, Latent
from discriminator import Discriminator
from generator import Generator

np.random.seed(1234)

## TODO: FIX GRADIENTS


class WGAN():
	"""
	Improved Wasserstein GAN Model
	"""

	def __init__(self, sess, path, latent_dim, num_classes, batch_size, learning_rate_c,
		learning_rate_g, lambdah):
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
		self.learning_rate_c = learning_rate_c
		self.learning_rate_g = learning_rate_g
		self.lambdah = lambdah
		self.build_model()

	def serve_epsilon(self):
		epsilon = np.random.uniform(size=self.batch_size)
		epsilon_return = np.zeros((self.batch_size, 64, 64, 3))
		for index in range(self.batch_size):
			epsilon_return[index, :] = epsilon[index]
		return epsilon_return

	def build_model(self):
		self.x = tf.placeholder(tf.float32, shape=[self.batch_size, 64, 64, 3])
		self.labels = tf.placeholder(tf.float32, shape=[self.batch_size, self.num_classes+1])
		self.z = tf.placeholder(tf.float32, 
			shape=[self.batch_size, self.latent.output_size])
		self.epsilon = tf.placeholder(tf.float32, shape=[self.batch_size, 64, 64, 3])
		self.is_training = tf.placeholder(tf.bool, shape=[])

		disc_output_x = Discriminator.discriminator(self.x, self.batch_size, self.num_classes)
		print("disc_output_x size: ", disc_output_x.get_shape())
		generator_output = Generator.generator(self.z, self.is_training)
		print("generator_output size: ", generator_output.get_shape())
		disc_output_gz = Discriminator.discriminator(generator_output, self.batch_size, self.num_classes)
		print("disc_output_gz: ", disc_output_gz.get_shape())

		interpolates = tf.multiply(self.epsilon, self.x) + \
			tf.multiply(1-self.epsilon, generator_output)

		disc_interpolates = Discriminator.discriminator(interpolates, self.batch_size, self.num_classes)

		# discriminator(generator_output_inner) will be of size:
		# 	[batch_size, num_classes+1]
		# labels will be of shape:
		#	[batch_size, num_classes+1]

		self.generator_loss = tf.reduce_sum(tf.multiply(disc_output_gz, self.labels), axis=1) + \
			tf.reduce_sum(tf.multiply(disc_output_gz, self.labels-1), axis=1)
		batch_gen_loss = self.generator_loss
		self.generator_loss = tf.reduce_sum(self.generator_loss)


		self.disc_loss = tf.reduce_sum(tf.multiply(disc_output_x, self.labels), axis=1) + \
			tf.reduce_sum(tf.multiply(disc_output_x, self.labels-1), axis=1) - batch_gen_loss

		gradients = tf.gradients(disc_interpolates, [interpolates])[0]
		slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
		gradient_penalty = tf.reduce_mean((slopes-1)**2)

		self.disc_loss += self.lambdah*gradient_penalty

sess = tf.Session()
path = '../fonts.hdf5'
latent_dim = 100
num_classes = 62
batch_size =16
learning_rate_c = 1e-4
learning_rate_g = 1e-4
lambdah = 10

wgan = WGAN(sess, path, latent_dim, num_classes, batch_size, 
	learning_rate_c, learning_rate_g, lambdah)