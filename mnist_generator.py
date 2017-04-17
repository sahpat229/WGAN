import tensorflow as tf
import numpy as np
import tflib as lib
from ops import *

slim = tf.contrib.slim

class MNIST_Generator():

	def generator(z, noise_dim):
		FC_DIM = 512
		output = ReLULayer('Generator.1', noise_dim, FC_DIM, z)
		output = ReLULayer('Generator.2', FC_DIM, FC_DIM, z)
		output = ReLULayer('Generator.3', FC_DIM, FC_DIM, z)
		output = lib.ops.linear.Linear('Generator.Out', FC_DIM, 28*28*1, output)
		output = tf.tanh(output)
		output = tf.reshape(output, [-1, 28, 28, 1])
		return output