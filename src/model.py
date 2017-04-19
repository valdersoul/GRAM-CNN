from keras.models import Model
from keras.layers import Embedding, Input, Dense, Convolution2D, MaxPooling2D, TimeDistributed, Merge, Reshape, Lambda, Bidirectional, LSTM, Dropout, BatchNormalization, Masking
from keras import backend as K

def CNN(x, kernels, filt_num, length, emd_dim, seq_len):
	concate = []
	for idx, kernel_dim in enumerate(kernels):
		reduced_len = length - kernel_dim + 1
		print kernel_dim
		conv = TimeDistributed(Convolution2D(filt_num, emd_dim, kernel_dim, dim_ordering='tf'))(x)
		maxpool = TimeDistributed(MaxPooling2D((1, reduced_len), dim_ordering='tf'))(conv)
		concate.append(maxpool)

	m = Merge(mode = 'concat')(concate)
	m_reshape = Reshape((seq_len, filt_num * len(kernels)))(m)
	return m_reshape

def build(w_emb_len, c_emb_len, emb_matrix, dic_len, c_dic_len, max_seq_len):

	max_len = 13

	char_emb_size = c_emb_len
	word_emb_size = w_emb_len

	kernels = [1,2,3,4,5,6,7]
	filt_num = 25
	
	word = Input(shape = (max_seq_len,), name = 'word')
	word_mask = Masking(mask_value= -1)(word)
	word_vecs = Embedding(dic_len, word_emb_size)(word_mask)

	chars = Input(shape = (max_seq_len, max_len),  name = 'chars')
	chars_vecs = TimeDistributed(Embedding(c_dic_len, char_emb_size))(chars)
	chars_vecs = Reshape((max_seq_len, char_emb_size, max_len, 1))(chars_vecs)

	cnn = CNN(chars_vecs, kernels, filt_num, max_len, char_emb_size, max_seq_len)

	x = Merge(mode = 'concat')([cnn, word_vecs])
	x = BatchNormalization()(x)

	drop_out_x = Dropout(0.5)(x)

	lstm_h = Bidirectional(LSTM(200, activation = 'tanh', return_sequences=True))(x)

	predicts  = TimeDistributed(Dense(3))(lstm_h)

	model = Model(input = [word, chars], output = [predicts])

	return model