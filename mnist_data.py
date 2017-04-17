import numpy as np
import matplotlib.pyplot as plt
import os
import struct

class MNIST_Data():
    """
    Data access for MNIST dataset
    """

    def __init__(self, path, latent_dim, batch_size):
        self.batch_size = batch_size
        self.latent_dim = latent_dim
        self.test_labels_images = [item for item in self.read("testing", path)]
        self.train_labels_images = [item for item in self.read("training", path)]
        self.num_classes = len(self.train_labels_images[0][0])
        self.latent_output_size = self.latent_dim + self.num_classes
        self.labels_size = self.num_classes + 1
        np.random.shuffle(self.train_labels_images)        

    def read(self, dataset, path = "."):
        """
        Python function for importing the MNIST data set.  It returns an iterator
        of 2-tuples with the first element being the label and the second element
        being a numpy.uint8 2D array of pixel data for the given image.
        """
        if dataset is "training":
            fname_img = os.path.join(path, 'train-images.idx3-ubyte')
            fname_lbl = os.path.join(path, 'train-labels.idx1-ubyte')
        elif dataset == "testing":
            fname_img = os.path.join(path, 't10k-images.idx3-ubyte')
            fname_lbl = os.path.join(path, 't10k-labels.idx1-ubyte')    

        # Load everything in some numpy arrays
        with open(fname_lbl, 'rb') as flbl:
            magic, num = struct.unpack(">II", flbl.read(8))
            lbl = np.fromfile(flbl, dtype=np.int8)

        with open(fname_img, 'rb') as fimg:
            magic, num, rows, cols = struct.unpack(">IIII", fimg.read(16))
            img = np.fromfile(fimg, dtype=np.uint8).reshape(len(lbl), rows, cols)

        get_img = lambda idx: (lbl[idx], img[idx])

        # Create an iterator which returns each image in turn
        for i in range(len(lbl)):
            labels = np.zeros(10);
            labels[lbl[i]] = 1;
            yield (labels, np.array(img[i]).reshape(28, 28, 1))

    def pick_random(self):
        self.pickers = np.random.randint(low=0, high=len(self.train_labels_images), size=self.batch_size)

    def serve_real(self):
        self.pick_random()
        one_hot = [self.train_labels_images[i][0] for i in self.pickers]
        one_hot = np.array(one_hot)
        labels = np.concatenate((one_hot, np.zeros((self.batch_size, 1))), axis=1)
        self.num_labels = np.argmax(labels, axis=1)
        images = [self.train_labels_images[i][1] for i in self.pickers]
        images= np.array(images)
        return images/255, labels

    def serve_latent(self):
        row_picker = np.arange(self.batch_size)
        one_hot = np.zeros((self.batch_size, self.num_classes))
        one_hot[row_picker, self.num_labels] = 1
        latent = np.random.uniform(size=(self.batch_size, self.latent_dim))
        feed_vectors = np.concatenate((one_hot, latent), axis=1)
        labels = np.concatenate((one_hot, np.zeros((self.batch_size, 1))), axis=1)
        return feed_vectors, labels