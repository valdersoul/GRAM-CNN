import tensorflow as tf

from ops import conv2d, conv2d_same
from base import Model

class n_gram(Model):
	"""
	The n_gram convolution network
	"""
	def __init__(self, input_, embed_dim = 1300,
		feature_maps=[50,50,50,50,50,50,50,50,50,50,50,50,50,50,50],
		kernels=[1,2,3,4,5,6,7,8,9,10,11,12],
#	        kernels=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15],
#		kernels = [1,2,3,4,5,6,7,8,9,10],
		# feature_maps=[50,150,150,300,200,150,150],
#		kernels=[1,2,3,4,5,6,7],
#		kernels=[1,2,3,4,5],
#		kernels=[1,2,3],
		checkpoint_dir="checkpoint",
		forward_only=False,
		max_seq_len = 200,
		name = "gram_kernel"):

		self.embed_dim = embed_dim
		self.feature_maps = feature_maps
		self.kernels = kernels
		self.name = name

		# make the input as 1 x word_num x feature_dim x 1
		input_ = tf.expand_dims(input_, -1)

		layers = []
		for idx, kernel_dim in enumerate(kernels):
			reduced_length = max_seq_len - kernel_dim + 1
			# [batch_size x seq_length x embed_dim x feature_map_dim]
			conv = conv2d(input_, feature_maps[idx], kernel_dim , self.embed_dim,
			            name="%s%d" % (name, idx))

			# [batch_size x 1 x 1 x feature_map_dim]
			#pool = tf.nn.max_pool(tf.tanh(conv), [1, reduced_length, 1, 1], [1, 1, 1, 1], 'VALID')
			# 1 x reduced_length x feature_map_dim
			layers.append(tf.reshape(tf.tanh(conv), [-1, feature_maps[idx]]))

		outputs = []
		for i in xrange(max_seq_len):
			grams = []
			# for kernel size is one (1-gram)
			grams.append(tf.reshape(tf.gather(layers[0], i), [1,-1]))
			for j in kernels[1:]:
				if i == 0:
					# the first word only have one related conv result
					grams.append(tf.reshape(tf.gather(layers[j - 1], i), [1, -1]))
				else:
					# the indices related to the word
					indices = list(range(i - j + 1 if i - j + 1 > 0 else 0, i + 1 if i + 1 < max_seq_len -j + 1 else max_seq_len -j + 1))
					gram_feature = tf.gather(layers[j - 1], indices)
					grams.append(tf.reshape(tf.reduce_max(gram_feature, axis = 0, keep_dims = True), [1,-1]))
			outputs.append(tf.concat(grams, 1))
		self.output = tf.concat(outputs, 0)
