import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from mnist_data import MNIST_Data
from mnist_discriminator import MNIST_Discriminator
from mnist_generator import MNIST_Generator
from time import gmtime, strftime

np.random.seed(1234)

## TODO: FIX GRADIENTS

class MNIST_WGAN():
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
		self.data = MNIST_Data(path, latent_dim, batch_size)
		self.num_classes = num_classes
		self.batch_size = batch_size
		self.learning_rate_c = learning_rate_c
		self.learning_rate_g = learning_rate_g
		self.lambdah = lambdah
		self.num_critic = num_critic
		self.iterations = iterations
		self.time = strftime("%Y-%m-%d %H:%M:%S", gmtime())
		#self.data.test()
		self.build_model()

	def serve_epsilon(self):
		"""
		Serve the random number from 0 to 1 for each dimension to make x_hat
		"""
		epsilon = np.random.uniform(size=self.batch_size)
		epsilon_return = np.zeros((self.batch_size, 28, 28, 1))
		for index in range(self.batch_size):
			epsilon_return[index, :] = epsilon[index]
		return epsilon_return

	def build_v2(self):
		"""
		Version 2:
		- D(x) has 1 output
		- D(x) takes in the one_hot class vector as an input to compute that 1 output
		- G(z) takes in the one_hot class vector as before in Version 1
		"""
		self.z = tf.placeholder(tf.float32, 
			shape=[self.batch_size, self.data.latent_output_size])
		#with tf.variable_scope("generator") as scope:
		self.generator_output = MNIST_Generator.generator(self.z,
			self.data.latent_output_size)

		#with tf.variable_scope("discriminator") as scope:
		disc_output_x = MNIST_Discriminator.discriminator(self.x, self.xlabels, 
			self.data.labels_size, 50)
		#scope.reuse_variables()
		disc_output_gz = MNIST_Discriminator.discriminator(self.generator_output, self.zlabels,
			self.data.labels_size, 50)
		differences = disc_output_gz - disc_output_x
		interpolates = disc_output_x + (self.epsilon*differences)
		disc_interpolates = MNIST_Discriminator.discriminator(interpolates, self.xlabels,
			self.data.labels_size, 50) 

		self.generator_loss = -tf.reduce_mean(disc_output_gz)
		self.disc_loss = tf.reduce_mean(disc_output_gz) - tf.reduce_mean(disc_output_x)

		gradients = tf.gradients(disc_interpolates, [interpolates])[0]
		slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
		gradient_penalty = tf.reduce_mean((slopes-1)**2)
		self.disc_loss += self.lambdah*gradient_penalty

	def build_model(self):
		self.x = tf.placeholder(tf.float32, shape=[self.batch_size, 28, 28, 1])
		self.xlabels = tf.placeholder(tf.float32, shape=[self.batch_size, self.num_classes+1])
		self.zlabels = tf.placeholder(tf.float32, shape=[self.batch_size, self.num_classes+1])
		self.epsilon = tf.random_uniform(
				shape=[self.batch_size, 1],
				minval=0.,
				maxval=1.
			)
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
 
	def disc_train_iter(self, iteration, x, xlabels, z, zlabels):
		disc_loss, _, summary = self.sess.run(
			[self.disc_loss, self.disc_optim, self.disc_loss_sum],
			feed_dict = {
				self.x: x,
				self.xlabels: xlabels,
				self.z: z,
				self.zlabels: zlabels,
				self.is_training: True
				}
			)
		print("DISC LOSS: ", disc_loss)
		self.disc_writer.add_summary(summary, iteration)

	def gen_train_iter(self, iteration, x, xlabels, z, zlabels):
		gen_loss, _, summary = self.sess.run(
			[self.generator_loss, self.gen_optim, self.gen_loss_sum],
			feed_dict = {
				self.x: x,
				self.xlabels: xlabels,
				self.z: z,
				self.zlabels: zlabels,
				self.is_training: True
				}
			)
		print("GEN LOSS: ", gen_loss)
		self.gen_writer.add_summary(summary, iteration)

	def probe(self):	
		x, xlabels = self.data.serve_real()
		z, zlabels = self.data.serve_latent()
		images = self.sess.run(self.generator_output,
			feed_dict = {
				self.x: x,
				self.xlabels: xlabels,
				self.z: z,
				self.zlabels: zlabels,
				self.is_training: False
			});
		print(images[0].shape)
		plt.imshow(np.tile(x[0], (1, 1, 3)))
		plt.show()
		plt.imshow(np.tile(images[0], (1, 1, 3)))
		plt.show()

	def train(self):
		for iteration in range(self.iterations):
			for disc_iter in range(self.num_critic):
				x, xlabels = self.data.serve_real()
				z, zlabels = self.data.serve_latent()
				#print("EPSILON [0]: ", epsilon[0])
				#print("EPSILON [1]: ", epsilon[1])
				self.disc_train_iter(iteration*self.num_critic + disc_iter,
					x, xlabels, z, zlabels)

			x, xlabels = self.data.serve_real()
			z, zlabels = self.data.serve_latent()
			self.gen_train_iter(iteration*self.num_critic + disc_iter,
				x, xlabels, z, zlabels)
			if iteration % 100 == 0:
				self.probe()

version = "v2"
sess = tf.Session()
path_sahil_comp = '../MNIST_data'
latent_dim = 50
num_classes = 10
batch_size = 32
learning_rate_c = 1e-4
learning_rate_g = 1e-4
lambdah = 10
num_critic = 5
iterations = 10000

wgan = MNIST_WGAN(version, sess, path_sahil_comp, latent_dim, num_classes, batch_size, 
	learning_rate_c, learning_rate_g, lambdah, num_critic, iterations)
wgan.optim_init()
wgan.train()
