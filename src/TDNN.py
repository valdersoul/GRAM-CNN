import tensorflow as tf

from ops import conv2d
from base import Model

class TDNN(Model):
	"""
	Time-delayed Nueral Network (cf. http://arxiv.org/abs/1508.06615v4)
	test
	"""
	def __init__(self, input_, embed_dim = 15,
		#feature_maps=[50, 100, 150, 200, 200, 200, 200],
		#kernels=[1,2,3,4,5,6,7],
		feature_maps=[200, 200,200],
		kernels=[1,2,3], 
		checkpoint_dir="checkpoint",
		forward_only=False):
		"""
		Initialize the parameters for TDNN
		Args:
		  embed_dim: the dimensionality of the inputs
		  feature_maps: list of feature maps (for each kernel width)
		  kernels: list of # of kernels (width)
		"""
		self.embed_dim = embed_dim
		self.feature_maps = feature_maps
		self.kernels = kernels
		
		# [batch_size x seq_length x embed_dim x 1]
		length = self.__length(input_)
		input_ = tf.expand_dims(input_, -1)

		layers = []
		for idx, kernel_dim in enumerate(kernels):

			# [batch_size x seq_length x embed_dim x feature_map_dim]
			conv = conv2d(input_, feature_maps[idx], kernel_dim , self.embed_dim,
			            name="kernel%d" % idx)

			# [batch_size x 1 x 1 x feature_map_dim]
			#pool = tf.nn.max_pool(tf.tanh(conv), [1, reduced_length, 1, 1], [1, 1, 1, 1], 'VALID')
			pool = tf.reduce_max(tf.tanh(conv), axis = 1, keep_dims = True)

			layers.append(tf.reshape(pool, [-1, feature_maps[idx]]))



		if len(kernels) > 1:
			self.output = tf.concat(layers, 1)
		else:
			self.output = layers[0]

	def __length(self, sequence):
		used = tf.sign(tf.reduce_max(tf.abs(sequence), reduction_indices=2))
		length = tf.reduce_sum(used, reduction_indices=1)
		length = tf.cast(length, tf.int32)
		return length
