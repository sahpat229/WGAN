import h5py
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(1234)

class Fonts():
	"""
	Purposes:
		- Accessing the image data stored in hdf5 format
		- Serving "real" data to the Wasserstein GAN model
	"""

	def __init__(self, path):
		self.load_font_file(path)

	def load_font_file(self, path):
		"""
		Loads the hdf5 file stored in path
		Returns a file object
		"""
		f = h5py.File(path, 'r')
		self.fonts = f['fonts']

	def serve_data(self, batch_size):
		"""
		- Serve batch_size amount of real font data to the critic
		- Serve batch_size amount of labels associated with the font data
		- Shape is []
		"""
		pass

	def test_load(self):
		plt.imshow(self.fonts[0][0])
		plt.show()

class Latent():
	"""
	- Serve the latent noise vector to the generator
	- Shape is [batch_size, num_classes+latent_dim]
	"""

	def serve_latent(num_classes, latent_dim, batch_size):
		"""
		- Serve batch_size amount of latent variables to the generator
		"""
		row_picker = np.arange(batch_size)
		class_picker = np.random.randint(low=0, high=num_classes, size=batch_size)
		one_hot = np.zeros((batch_size, num_classes))
		one_hot[row_picker, class_picker] = 1

		latent = np.random.uniform(size=(batch_size, latent_dim))
		feed_vectors = np.concatenate((one_hot, latent), axis=1)
		return feed_vectors

font = Fonts('../fonts.hdf5')
font.test_load()