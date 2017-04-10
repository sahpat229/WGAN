import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from data import Fonts, Latent
from discriminator import Discriminator
from generator import Generator
from time import gmtime, strftime

np.random.seed(1234)

## TODO: FIX GRADIENTS


class WGAN():
	"""
	Improved Wasserstein GAN Model
	"""

	def __init__(self, sess, path, latent_dim, num_classes, batch_size, learning_rate_c,
		learning_rate_g, lambdah, num_critic, iterations):
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
		self.num_critic = num_critic
		self.iterations = iterations
		self.time = strftime("%Y-%m-%d %H:%M:%S", gmtime())
		self.build_model()

	def serve_epsilon(self):
		epsilon = np.random.uniform(size=self.batch_size)
		epsilon_return = np.zeros((self.batch_size, 64, 64, 1))
		for index in range(self.batch_size):
			epsilon_return[index, :] = epsilon[index]
		return epsilon_return

	def build_model(self):
		self.x = tf.placeholder(tf.float32, shape=[self.batch_size, 64, 64, 1])
		self.xlabels = tf.placeholder(tf.float32, shape=[self.batch_size, self.num_classes+1])
		self.z = tf.placeholder(tf.float32, 
			shape=[self.batch_size, self.latent.output_size])
		self.zlabels = tf.placeholder(tf.float32, shape=[self.batch_size, self.num_classes+1])
		self.epsilon = tf.placeholder(tf.float32, shape=[self.batch_size, 64, 64, 1])
		self.is_training = tf.placeholder(tf.bool, shape=[])

		gen_var_coll = ["gen_var_coll"]
		gen_upd_coll = ["gen_upd_coll"]

		disc_var_coll = ["disc_var_coll"]

		with tf.variable_scope("generator") as scope:
			generator_output = Generator.generator(self.z, self.is_training, gen_var_coll, gen_upd_coll)
		
		with tf.variable_scope("discriminator") as scope:
			disc_output_x = Discriminator.discriminator(self.x, self.batch_size, 
				self.num_classes, disc_var_coll)
			scope.reuse_variables()
			disc_output_gz = Discriminator.discriminator(generator_output, self.batch_size, 
				self.num_classes, disc_var_coll)
			interpolates = tf.multiply(self.epsilon, self.x) + \
				tf.multiply(1-self.epsilon, generator_output)
			disc_interpolates = Discriminator.discriminator(interpolates, self.batch_size, 
				self.num_classes, disc_var_coll)

		# discriminator(generator_output_inner) will be of size:
		# 	[batch_size, num_classes+1]
		# labels will be of shape:
		#	[batch_size, num_classes+1]

		self.generator_loss = tf.reduce_sum(tf.multiply(disc_output_gz, self.zlabels), axis=1) + \
			tf.reduce_sum(tf.multiply(disc_output_gz, self.zlabels-1), axis=1)
		batch_gen_loss = self.generator_loss
		self.generator_loss = tf.reduce_mean(self.generator_loss)

		self.disc_loss = tf.reduce_sum(tf.multiply(disc_output_x, self.xlabels), axis=1) + \
			tf.reduce_sum(tf.multiply(disc_output_x, self.xlabels-1), axis=1) - batch_gen_loss
		self.disc_loss = tf.reduce_mean(self.disc_loss)

		gradients_per_dim = [tf.gradients(tf.slice(disc_interpolates, [0, i], [self.batch_size, 1]), [interpolates])
			for i in range(self.num_classes + 1)]
		slopes_per_dim = [tf.sqrt(tf.reduce_sum(tf.square(gradients_per_dim[i]), reduction_indices=[1]))
			for i in range(self.num_classes + 1)]
		gradient_penalty_per_dim = [tf.reduce_mean((slopes_per_dim[i]-1)**2)
			for i in range(self.num_classes + 1)]

		total_grad_penalty = tf.zeros([])
		for grad_penalty in gradient_penalty_per_dim:
			total_grad_penalty += grad_penalty
		self.disc_loss += self.lambdah*total_grad_penalty

		self.disc_loss_sum = tf.summary.scalar('Discriminator Loss', self.disc_loss)
		self.gen_loss_sum = tf.summary.scalar('Generator Loss', self.generator_loss)

		self.disc_writer = tf.summary.FileWriter('./logs/' + self.time, self.sess.graph)
		self.gen_writer = tf.summary.FileWriter('./logs/' + self.time, self.sess.graph)

	def optim_init(self):
		update_ops = tf.get_collection("gen_upd_coll")
		updates = tf.group(*update_ops)
		self.gen_optim = tf.group(updates,
			tf.train.AdamOptimizer(
				learning_rate=self.learning_rate_g,
				beta1=0.5,
				beta2=0.9
				).minimize(self.generator_loss)
			)

		self.disc_optim = tf.train.AdamOptimizer(
			learning_rate=self.learning_rate_c,
			beta1=0.5,
			beta2=0.9
			).minimize(self.disc_loss)
	
		self.sess.run(tf.global_variables_initializer())

	def disc_train_iter(self, iteration, x, xlabels, z, zlabels, epsilon):
		disc_loss, _, summary = self.sess.run(
			[self.disc_loss, self.disc_optim, self.disc_loss_sum],
			feed_dict = {
					self.x: x,
					self.xlabels: xlabels,
					self.z: z,
					self.zlabels: zlabels,
					self.epsilon: epsilon,
					self.is_training: True
				}
			)
		print("DISC LOSS: ", disc_loss)
		self.disc_writer.add_summary(summary, iteration)

	def gen_train_iter(self, iteration, x, xlabels, z, zlabels, epsilon):
		gen_loss, _, summary = self.sess.run(
			[self.gen_loss, self.gen_optim, self.gen_loss_sum],
			feed_dict = {
					self.x: x,
					self.xlabels: xlabels,
					self.z: z,
					self.zlabels: zlabels,
					self.epsilon: epsilon,
					self.is_training: True
				}
			)
		print("GEN LOSS: ", gen_loss)
		self.gen_writer.add_summary(summary, iteration)

	def train(self):
		for iteration in range(self.iterations):
			for disc_iter in range(self.num_critic):
				x, xlabels = self.real_data.serve_real()
				z, zlabels = self.latent.serve_latent()
				epsilon = self.serve_epsilon()
				self.disc_train_iter(iteration*self.num_critic + disc_iter,
					x, xlabels, z, zlabels, epsilon)
			x, xlabels = self.real_data.serve_real()
			z, zlabels = self.latent.serve_latent()
			epsilon = self.serve_epsilon()
			self.gen_train_iter(iteration*self.num_critic + disc_iter,
				x, xlabels, z, zlabels, epsilon)


sess = tf.Session()
path = '/media/sahil/NewVolume/College/fonts.hdf5'
latent_dim = 100
num_classes = 62
batch_size =16
learning_rate_c = 1e-4
learning_rate_g = 1e-4
lambdah = 10
num_critic = 5
iterations = 10

wgan = WGAN(sess, path, latent_dim, num_classes, batch_size, 
	learning_rate_c, learning_rate_g, lambdah, num_critic, iterations)
wgan.optim_init()
wgan.train()