import h5py
import numpy as np

np.random.seed(1234)

class Fonts():
	"""
	- Accessing the image data stored in hdf5 format
	- Serving "real" data to the Wasserstein GAN model
	"""

	def __init__(self, path, batch_size):
		self.load_font_file(path)
		self.batch_size = batch_size

	def load_font_file(self, path):
		"""
		Loads the hdf5 file stored in path
		Returns a file object
		"""
		f = h5py.File(path, 'r')
		self.fonts = f['fonts']
		fonts_shape = f['fonts'].shape
		self.num_fonts, self.num_chars = fonts_shape[0], fonts_shape[1]

	def serve_real(self):
		"""
		- Serve self.batch_size amount of real font data to the critic
		- Serve self.batch_size amount of labels associated with the font data
		"""
		labels = np.random.randint(low=0, high=self.num_chars, size=self.batch_size)
		batch_labels = np.zeros((self.batch_size, self.num_chars+1))
		batch_labels[np.arange(self.batch_size), labels] = 1
		fonts = np.random.randint(low=0, high=self.num_fonts, size=self.batch_size)
		images = np.array([self.fonts[fonts[i]][labels[i]] for i in range(self.batch_size)])
		return np.expand_dims(images, axis=3), batch_labels

	# def test_load(self):
	#	plt.imshow(self.fonts[0][0])
	#	plt.show()

class Latent():
	"""
	- Serve the latent noise vector to the generator
	- Shape is [batch_size, num_classes+latent_dim]
	"""

	def __init__(self, num_classes, latent_dim, batch_size):
		self.num_classes = num_classes
		self.latent_dim = latent_dim
		self.batch_size = batch_size
		self.output_size = self.num_classes + self.latent_dim

	def serve_latent(self):
		"""
		- Serve batch_size amount of latent variables to the generator
		- Don't need to have fake as an item in one_hot vector
		"""
		row_picker = np.arange(self.batch_size)
		class_picker = np.random.randint(low=0, high=self.num_classes, size=self.batch_size)
		one_hot = np.zeros((self.batch_size, self.num_classes))
		one_hot[row_picker, class_picker] = 1

		latent = np.random.uniform(size=(self.batch_size, self.latent_dim))
		feed_vectors = np.concatenate((one_hot, latent), axis=1)

		labels = np.concatenate((one_hot, np.zeros((self.batch_size, 1))), axis=1)
		return feed_vectors, labels
