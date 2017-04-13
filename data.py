import h5py
import numpy as np

np.random.seed(1234)

class Data():
	"""
	- Accessing the image data stored in hdf5 format
	- Serving data to the Wasserstein GAN model
	"""

	def __init__(self, path, latent_dim, batch_size):
		self.load_font_file(path)
		self.batch_size = batch_size
		self.latent_dim = latent_dim

	def load_font_file(self, path):
		"""
		Loads the hdf5 file stored in path
		Returns a file object
		"""
		f = h5py.File(path, 'r')
		self.fonts = f['fonts']
		fonts_shape = f['fonts'].shape
		self.num_fonts, self.num_classes = fonts_shape[0], fonts_shape[1]

	def randomize_labels(self):
		self.font_labels = np.random.randint(low=0, high=self.num_fonts, size=self.batch_size)
		self.char_labels = np.random.randint(low=0, high=self.num_classes, size=self.batch_size)

	def serve_real(self):
		"""
		- Serve self.batch_size amount of real font data to the critic
		- Serve self.batch_size amount of labels associated with the font data
		"""
		self.randomize_labels()
		batch_labels = np.zeros((self.batch_size, self.num_classes+1))
		batch_labels[np.arange(self.batch_size), self.char_labels] = 1
		images = np.array([self.font_labels[self.font_labels[i]][labels[i]] for i in range(self.batch_size)])
		return np.expand_dims(images, axis=3), batch_labels

	def serve_latent(self):
		"""
		- Serve batch_size amount of latent variables to the generator
		- Don't need to have fake as an item in one_hot vector
		"""
		row_picker = np.arange(self.batch_size)
		one_hot = np.zeros((self.batch_size, self.num_classes))
		one_hot[row_picker, self.char_labels] = 1

		latent = np.random.uniform(size=(self.batch_size, self.latent_dim))
		feed_vectors = np.concatenate((one_hot, latent), axis=1)

		labels = np.concatenate((one_hot, np.zeros((self.batch_size, 1))), axis=1)
		return feed_vectors, labels
