import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from data import Fonts, Latent

np.random.seed(1234)

class WGAN():
	"""
	Improved Wasserstein GAN Model
	"""

	def __init__(self, sess, path, latent_dim, batch_size):
		"""
		- sess : tf.Session
		- path: path to fonts file
		"""
		self.sess = sess
		self.real_data = Fonts(path, batch_size)
		self.latent = Latent(self.real_data.num_chars, latent_dim, batch_size)