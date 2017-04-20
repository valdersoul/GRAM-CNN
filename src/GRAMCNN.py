import tensorflow as tf

tf.GraphKeys.VARIABLES = tf.GraphKeys.GLOBAL_VARIABLES

from ops import conv2d, batch_norm, highway, index_along_every_row, unpack
from base import Model
from TDNN import TDNN
from n_gram import n_gram

class GRAMCNN(Model):
	"""docstring for LSTMTDNN"""
	def __init__(self,
				#input_w, input_c, target,
				#max_word_len,
				vocab_len, character_vocab_len, postag_vocab_len,
				use_word = True,
				use_char = True,
				use_pts = True,
				feature_maps=[50, 100, 150, 200, 200, 200, 200],
				kernels=[1,2,3,4,5,6,7],
				forward_only = False,
				hidden_size = 400,
				hidden_layers = 1,
				word_emb = 200, char_emb = 25, pt_emb = 15,
				drop_out = 0.5, num_classes = 3, word2vec = None, highway = True, crf = True, padding = False,
				max_seq_len = 200):

		self.use_word = use_word
		self.use_char = use_char
		self.use_pts  = use_pts

		self.padding = padding
		self.crf = crf
		self.highway = highway
		self.num_classes = num_classes
		self.feature_maps = feature_maps
		self.kernels = kernels
		self.sess = tf.Session()

		self.max_seq_len = max_seq_len

		if not self.padding:
			#BatchSz x max_seq_len x max_word_len
			self.char_input = tf.placeholder(tf.int32, shape = [None, None])
			#BatchSz x max_seq_len
			self.word_input = tf.placeholder(tf.int32, shape = [None])
			self.pt_input   = tf.placeholder(tf.int32, shape = [None])
			#BatchSz x max_seq_len ? or BatchSz x target_line
			self.target = tf.placeholder(tf.int32, shape = [None])
			#self.seq_len = seq_len

		else:
			self.char_input = tf.placeholder(tf.int32, shape = [self.max_seq_len, None])
			self.word_input = tf.placeholder(tf.int32, shape = [self.max_seq_len])
			self.pt_input   = tf.placeholder(tf.int32, shape = [self.max_seq_len])
			self.target     = tf.placeholder(tf.int32, shape = [max_seq_len])
			self.s_len      = tf.placeholder(tf.int32, shape = [None])


		#hidden size for LSTM
		self.h_size = hidden_size
		self.h_size_2 = hidden_size / 2

		self.lstm_layers = hidden_layers
		self.total_emb_size = 0

		#embedding size
		if self.use_word:
			self.w_emb_size = word_emb
			self.total_emb_size += self.w_emb_size
		else:
			self.w_emb_size = 0
			print 'Not use word embedding'
		if self.use_char:
			self.char_emb_dim = sum(feature_maps)
			self.total_emb_size += self.char_emb_dim
		else:
			self.char_emb_dim = 0
			print 'Not use char embedding'
		self.c_emb_size = char_emb
		if self.use_pts:
			self.pt_emb_size = pt_emb

		#drop out rate
		self.drop_rate = tf.placeholder(tf.float32)

		#is trainning or test
		self.forward_only = forward_only

		# vocab for word and character
		self.word_vocab_size = vocab_len
		self.char_vocab_size = character_vocab_len
		self.postag_vocab_size = postag_vocab_len

		self.max_grad_norm = 5
		self.__build(word2vec)

		init = tf.global_variables_initializer()
		self.sess.run(init)

	def __length(self, sequence):
		used = tf.sign(tf.reduce_max(tf.abs(sequence), reduction_indices=2))
		length = tf.reduce_sum(used, reduction_indices=1)
		length = tf.cast(length, tf.int32)
		return length


	def __build(self, pretrained_word_embedding):
		with tf.variable_scope("LSTMTDNN"):
			with tf.device('/cpu:0'):
				if self.use_char:
					self.char_embedding = tf.get_variable("char_matrix",
											[self.char_vocab_size, self.c_emb_size],
											initializer = tf.uniform_unit_scaling_initializer())

				if self.use_pts:
					self.postag_embedding = tf.get_variable("postag_matrix",
											[self.postag_vocab_size, self.pt_emb_size],
											initializer = tf.uniform_unit_scaling_initializer())

				if self.use_word:
					if pretrained_word_embedding is None:
						self.word_embedding = tf.get_variable("word_matrix",
												[self.word_vocab_size, self.w_emb_size],
												initializer = tf.uniform_unit_scaling_initializer())
					else:
						self.word_embedding = tf.get_variable("word_matrix",
												[self.word_vocab_size, self.w_emb_size],
												initializer = tf.constant_initializer(pretrained_word_embedding), trainable = False)

				#char_vecs sentence_len x max_word_len x embedding_len
				if self.use_char:
					char_vecs = tf.nn.embedding_lookup(self.char_embedding, self.char_input)

				#word_vec  sentence_len x embedding_len
				if self.use_word:
					word_vecs = tf.nn.embedding_lookup(self.word_embedding, self.word_input)

				#postag_vec sentence_len x embedding_len
				if self.use_pts:
					pt_vecs   = tf.nn.embedding_lookup(self.postag_embedding, self.pt_input)

			#char_embedding layer
			if self.use_char:
				char_cnn = TDNN(char_vecs, feature_maps = self.feature_maps, kernels = self.kernels, embed_dim = self.c_emb_size)
				# if self.use_pts:
				# 	combined_emb = tf.concat([pt_vecs, char_cnn.output], 1)
				# else:
				combined_emb = char_cnn.output
			if self.use_word:
				combined_emb = tf.concat([word_vecs, combined_emb], 1)

			# 1x1 convolution
			combined_emb = tf.reshape(combined_emb, [-1,self.total_emb_size])

			if self.highway:
				combined_emb = highway(combined_emb, self.total_emb_size, layer_size = 1)

			combined_emb = tf.reshape(combined_emb, [-1,self.total_emb_size])
			combined_emb = tf.expand_dims(combined_emb, 0)
			combined_emb = tf.nn.dropout(combined_emb, keep_prob = 1- self.drop_rate)

			if not self.padding:
				lstm_cell_fw_1 = tf.contrib.rnn.BasicLSTMCell(self.h_size)
				lstm_cell_bw_1 = tf.contrib.rnn.BasicLSTMCell(self.h_size)
				lstm_cell_fw = tf.contrib.rnn.MultiRNNCell([lstm_cell_fw_1] * self.lstm_layers, state_is_tuple=True)
				lstm_cell_bw = tf.contrib.rnn.MultiRNNCell([lstm_cell_bw_1] * self.lstm_layers, state_is_tuple=True)
				self.outputs, _ = tf.nn.bidirectional_dynamic_rnn(
					cell_fw = lstm_cell_fw,
					cell_bw = lstm_cell_bw,
					inputs = combined_emb,
					dtype = tf.float32,
					sequence_length = self.__length(combined_emb)
				)

				out = tf.concat([self.outputs[0], self.outputs[1]], 2)

				# two layer NN
				w_1 = tf.get_variable("w_1", [self.h_size * 2, self.h_size])
				b_1 = tf.get_variable("b_1", [self.h_size])
				linear1 = tf.matmul(tf.reshape(out, [-1, self.h_size * 2]), w_1) + b_1
				w_3 = tf.get_variable("w_3", [self.h_size, self.num_classes])
				b_3 = tf.get_variable("b_3", [self.num_classes])
				self.logits = tf.matmul(tf.tanh(linear1), w_3) + b_3

			else:
				line_layer = 200
				gram_cnn = n_gram(combined_emb, embed_dim = self.total_emb_size, max_seq_len = self.max_seq_len)
				#gram_cnn = fcn(combined_emb, embed_dim = self.total_emb_size, max_seq_len = self.max_seq_len)
				cnn_output = gram_cnn.output
				
				if self.use_pts:
					cnn_output = tf.concat([pt_vecs, cnn_output], 1)

				w_1 = tf.get_variable("w_1", [cnn_output.get_shape()[1], line_layer])
				b_1 = tf.get_variable("b_1", [line_layer])
				linear1 = tf.matmul(cnn_output, w_1) + b_1
				w_2 = tf.get_variable("w_2", [line_layer, self.num_classes])
				b_2 = tf.get_variable("b_2", [self.num_classes])
				self.logits = tf.matmul(tf.tanh(linear1), w_2) + b_2

			if not self.crf:
				self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(self.logits, self.target))
				soft_max = tf.nn.softmax(self.logits)
				self.y_pred = tf.argmax(soft_max, axis = 1)
			else:
				# use crf to do post processing
				unary_scores = tf.reshape(self.logits,
							  [1, -1, self.num_classes])
				if not self.padding:
					log_likelihood, self.transition_params = tf.contrib.crf.crf_log_likelihood(
									unary_scores, tf.reshape(self.target, [1, -1]), self.__length(combined_emb))
				else:
					log_likelihood, self.transition_params = tf.contrib.crf.crf_log_likelihood(
									unary_scores, tf.reshape(self.target, [1, -1]), self.s_len)

				self.loss = tf.reduce_mean(-log_likelihood)

			self.global_step = tf.Variable(0, name='global_step',
											trainable = False,
											collections=[tf.GraphKeys.GLOBAL_STEP, tf.GraphKeys.GLOBAL_VARIABLES])

			self.learning_rate = tf.train.exponential_decay(
							0.002,                # Base learning rate.
							self.global_step,  # Current index into the dataset.
							20000,          # Decay step.
							0.95,                # Decay rate.
							staircase=True)

			self.opt = tf.train.MomentumOptimizer(self.learning_rate,
											 0.9)

			params = tf.trainable_variables()
			grads = []
			for grad in tf.gradients(self.loss, params):
			  if grad is not None:
				grads.append(tf.clip_by_norm(grad, self.max_grad_norm))
			  else:
				grads.append(grad)
			self.optim = self.opt.apply_gradients(zip(grads, params),
										  global_step=self.global_step)

	def train(self, inputs, word_len = []):
		feed_dict = {self.char_input : inputs['char_for'],
					 self.word_input : inputs['word'],
					 self.target     : inputs['label'],
					 self.drop_rate   : 0.5}
		if self.use_pts:
			feed_dict[self.pt_input] = inputs['pts']
		if not self.padding:
			_, batch_loss = self.sess.run( [self.optim, self.loss],
				feed_dict= feed_dict )
		else:
			feed_dict[self.s_len] = word_len
			_, batch_loss = self.sess.run( [self.optim, self.loss],
				feed_dict= feed_dict )

		return batch_loss

	def test(self, inputs, word_len = []):
		feed_dict = {self.char_input : inputs['char_for'],
					 self.word_input : inputs['word'],
					 self.drop_rate   : 0}
		if self.use_pts:
			feed_dict[self.pt_input] = inputs['pts']

		if not self.crf:
			preds = self.sess.run(self.y_pred,
				feed_dict= feed_dict )
		else:
			if not self.padding:
				logit, transition_params = self.sess.run([self.logits, self.transition_params],
					feed_dict= feed_dict )
				preds, _ = tf.contrib.crf.viterbi_decode(
									logit, transition_params)
			else:
				feed_dict[self.s_len] = word_len
				logit, transition_params = self.sess.run([self.logits, self.transition_params],
					feed_dict= feed_dict )
				preds, _ = tf.contrib.crf.viterbi_decode(
									logit[:word_len[0]], transition_params)

		return preds
