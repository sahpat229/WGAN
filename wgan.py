import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from data import Data
from discriminator import Discriminator
from generator import Generator
from time import gmtime, strftime

np.random.seed(1234)

## TODO: FIX GRADIENTS

class WGAN():
	"""
	Improved Wasserstein GAN Model
	"""

	def __init__(self, version, sess, path, latent_dim, num_classes, batch_size, learning_rate_c,
		learning_rate_g, lambdah, num_critic, iterations):
		"""
		- sess : tf.Session
		- path: path to fonts file
		- latent_dim: dimension of latent space for z
		- num_classes: number of different classes the discriminator should
			output over.  Does not include the +1 for fake
		- batch_size: batch size
		- learning_rate_c: learning rate of both discriminator
		- learning_rate_g: learning_rate of the generator
		- lambdah: gradient penalty scaler
		- num_critic: number of iterations critic should run over generator
		- iterations: number of overall iterations
		"""
		self.version = version
		self.sess = sess
		self.data = Data(path, latent_dim, batch_size)
		self.num_classes = num_classes
		self.batch_size = batch_size
		self.learning_rate_c = learning_rate_c
		self.learning_rate_g = learning_rate_g
		self.lambdah = lambdah
		self.num_critic = num_critic
		self.iterations = iterations
		self.time = strftime("%Y-%m-%d %H:%M:%S", gmtime())
		self.data.test()
		self.build_model()

	def serve_epsilon(self):
		"""
		Serve the random number from 0 to 1 for each dimension to make x_hat
		"""
		epsilon = np.random.uniform(size=self.batch_size)
		epsilon_return = np.zeros((self.batch_size, 64, 64, 1))
		for index in range(self.batch_size):
			epsilon_return[index, :] = epsilon[index]
		return epsilon_return

	def build_v1(self):
		self.z = tf.placeholder(tf.float32, 
			shape=[self.batch_size, self.data.latent_output_size])
		with tf.variable_scope("generator") as scope:
			self.generator_output = Generator.generator(self.z, self.is_training, 
				self.gen_var_coll, self.gen_upd_coll)
		
		with tf.variable_scope("discriminator") as scope:
			disc_output_x = Discriminator.discriminator_v1(self.x, self.batch_size, self.num_classes, self.disc_var_coll)
			scope.reuse_variables()
			disc_output_gz = Discriminator.discriminator_v1(self.generator_output, self.batch_size, self.num_classes, disc_var_coll)
			interpolates = tf.multiply(self.epsilon, self.x) + \
				tf.multiply(1-self.epsilon, self.generator_output)
			disc_interpolates = Discriminator.discriminator_v1(interpolates, self.batch_size, self.num_classes, disc_var_coll)

		# discriminator(self.generator_output_inner) will be of size:
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

	def build_v2(self):
		self.z = tf.placeholder(tf.float32, 
			shape=[self.batch_size, self.data.latent_output_size])
		with tf.variable_scope("generator") as scope:
			self.generator_output = Generator.generator(self.z, self.is_training, 
				self.gen_var_coll, self.gen_upd_coll)

		with tf.variable_scope("discriminator") as scope:
			disc_output_x = Discriminator.discriminator_v2(self.x, self.xlabels, 
				self.batch_size, self.disc_var_coll, 100)
			scope.reuse_variables()
			disc_output_gz = Discriminator.discriminator_v2(self.generator_output, self.zlabels,
				self.batch_size, self.disc_var_coll, 100)
			interpolates = tf.multiply(self.epsilon, self.x) + tf.multiply(1-self.epsilon, self.generator_output)
			disc_interpolates = Discriminator.discriminator_v2(interpolates, self.xlabels, self.batch_size,
				self.disc_var_coll, 100)

		self.generator_loss = tf.reduce_mean(disc_output_gz)
		self.disc_loss = tf.reduce_mean(disc_output_x) - self.generator_loss

		gradients = tf.gradients(disc_interpolates, [interpolates])[0]
		slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
		gradient_penalty = tf.reduce_mean((slopes-1)**2)
		self.disc_loss += self.lambdah*gradient_penalty

	def build_v3(self):
		self.z = tf.placeholder(tf.float32,
			shape=[self.batch_size, self.data.latent_dim])
		with tf.variable_scope("generator") as scope:
			self.generator_output = Generator.generator(self.z, self.is_training,
				self.gen_var_coll, self.gen_upd_coll)

		with tf.variable_scope("discriminator") as scope:
			disc_output_x = Discriminator.discriminator_v3(self.x, self.batch_size,
				self.disc_var_coll)
			scope.reuse_variables()
			disc_output_gz = Discriminator.discriminator_v3(self.generator_output, self.batch_size,
				self.disc_var_coll)
			interpolates = tf.multiply(self.epsilon, self.x) + tf.multiply(1-self.epsilon, self.generator_output)
			disc_interpolates = Discriminator.discriminator_v3(interpolates, self.batch_size,
				self.disc_var_coll)

		self.generator_loss = tf.reduce_mean(disc_output_gz)
		self.disc_loss = tf.reduce_mean(disc_output_x) - self.generator_loss

		gradients = tf.gradients(disc_interpolates, [interpolates])[0]
		slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
		gradient_penalty = tf.reduce_mean((slopes-1)**2)
		self.disc_loss += self.lambdah*gradient_penalty

	def build_model(self):
		self.x = tf.placeholder(tf.float32, shape=[self.batch_size, 64, 64, 1])
		self.xlabels = tf.placeholder(tf.float32, shape=[self.batch_size, self.num_classes+1])
		self.zlabels = tf.placeholder(tf.float32, shape=[self.batch_size, self.num_classes+1])
		self.epsilon = tf.placeholder(tf.float32, shape=[self.batch_size, 64, 64, 1])
		self.is_training = tf.placeholder(tf.bool, shape=[])

		self.gen_var_coll = ["gen_var_coll"]
		self.gen_upd_coll = ["gen_upd_coll"]
		self.disc_var_coll = ["disc_var_coll"]

		if self.version == "v1":
			self.build_v1()
		elif self.version == "v2":
			self.build_v2()
		elif self.version == "v3":
			self.build_v3()
		else:
			raise ValueError("Must use either 'v1' or 'v2' for version argument in WGAN")

		self.disc_loss_sum = tf.summary.scalar('Discriminator Loss', self.disc_loss)
		self.gen_loss_sum = tf.summary.scalar('Generator Loss', self.generator_loss)

		self.disc_writer = tf.summary.FileWriter('./logs/' + self.time, self.sess.graph)
		self.gen_writer = tf.summary.FileWriter('./logs/' + self.time, self.sess.graph)

	def optim_init(self):
		update_ops = tf.get_collection("gen_upd_coll")
		updates = tf.group(*update_ops)
		gen_variables = tf.get_collection("gen_var_coll")
		disc_variables = tf.get_collection("disc_var_coll")
		self.gen_optim = tf.group(updates,
			tf.train.AdamOptimizer(
				learning_rate=self.learning_rate_g,
				beta1=0.5,
				beta2=0.9
				).minimize(self.generator_loss, var_list=gen_variables)
			)

		self.disc_optim = tf.train.AdamOptimizer(
			learning_rate=self.learning_rate_c,
			beta1=0.5,
			beta2=0.9
			).minimize(self.disc_loss, var_list=disc_variables)
	
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
			[self.generator_loss, self.gen_optim, self.gen_loss_sum],
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

	def probe(self):	
		if self.version == "v2" or self.version == "v1":
			x, xlabels = self.data.serve_real()
			z, zlabels = self.data.serve_latent()
		else:
			x, xlabels = self.data.serve_real()
			z, zlabels = self.data.serve_latent_orig()
		epsilon = self.serve_epsilon()
		images = self.sess.run(self.generator_output,
			feed_dict = {
				self.x: x,
				self.xlabels: xlabels,
				self.z: z,
				self.zlabels: zlabels,
				self.epsilon: epsilon,
				self.is_training: False
			});
		print(images[0].shape)
		plt.imshow(np.tile(images[0], (1, 1, 3)))
		plt.show()

	def train(self):
		for iteration in range(self.iterations):
			for disc_iter in range(self.num_critic):
				if self.version == "v2" or self.version == "v1":
					x, xlabels = self.data.serve_real()
					z, zlabels = self.data.serve_latent()
				else:
					x, xlabels = self.data.serve_real()
					z, zlabels = self.data.serve_latent_orig()
				epsilon = self.serve_epsilon()
				print("EPSILON [0]: ", epsilon[0])
				print("EPSILON [1]: ", epsilon[1])
				self.disc_train_iter(iteration*self.num_critic + disc_iter,
					x, xlabels, z, zlabels, epsilon)

			if self.version == "v2" or self.version == "v1":
				x, xlabels = self.data.serve_real()
				z, zlabels = self.data.serve_latent()
			else:
				x, xlabels = self.data.serve_real()
				z, zlabels = self.data.serve_latent_orig()
			epsilon = self.serve_epsilon()
			self.gen_train_iter(iteration*self.num_critic + disc_iter,
				x, xlabels, z, zlabels, epsilon)
			if iteration % 10 == 0:
				self.probe()

version = "v3"
sess = tf.Session()
path_sahil_comp = '/media/sahil/NewVolume/College/fonts.hdf5'
path = '../fonts.hdf5'
latent_dim = 25
num_classes = 62
batch_size =16
learning_rate_c = 1e-4
learning_rate_g = 1e-4
lambdah = 10
num_critic = 5
iterations = 10000

wgan = WGAN(version, sess, path_sahil_comp, latent_dim, num_classes, batch_size, 
	learning_rate_c, learning_rate_g, lambdah, num_critic, iterations)
wgan.optim_init()
wgan.train()
