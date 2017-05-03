#!/usr/bin/env python

import os
import numpy as np
import re
import optparse
import itertools
from collections import OrderedDict
from utils import create_input
import loader

from utils import models_path, evaluate, eval_script, eval_temp, save_mappings, reload_mappings
from loader import word_mapping, char_mapping, tag_mapping, pt_mapping
from loader import update_tag_scheme, prepare_dataset
from loader import augment_with_pretrained
from gensim.models import word2vec
from GRAMCNN import GRAMCNN

import tensorflow as tf

def get_comma_separated_args(option, opt, value, parser):
    setattr(parser.values, option.dest, value.split(','))

# Read parameters from command line
optparser = optparse.OptionParser()
optparser.add_option(
    "-T", "--train", default="",
    help="Train set location"
)
optparser.add_option(
    "-d", "--dev", default="",
    help="Dev set location"
)
optparser.add_option(
    "-t", "--test", default="",
    help="Test set location"
)
optparser.add_option(
    "-s", "--tag_scheme", default="iobes",
    help="Tagging scheme (IOB or IOBES)"
)
optparser.add_option(
    "-l", "--lower", default="0",
    type='int', help="Lowercase words (this will not affect character inputs)"
)
optparser.add_option(
    "-z", "--zeros", default="0",
    type='int', help="Replace digits with 0"
)
optparser.add_option(
    "-c", "--char_dim", default="25",
    type='int', help="Char embedding dimension"
)
optparser.add_option(
    "-C", "--char_lstm_dim", default="25",
    type='int', help="Char LSTM hidden layer size"
)
optparser.add_option(
    "-b", "--char_bidirect", default="1",
    type='int', help="Use a bidirectional LSTM for chars"
)
optparser.add_option(
    "-w", "--word_dim", default="200",
    type='int', help="Token embedding dimension"
)
optparser.add_option(
    "-W", "--word_lstm_dim", default="100",
    type='int', help="Token LSTM hidden layer size"
)
optparser.add_option(
    "-B", "--word_bidirect", default="1",
    type='int', help="Use a bidirectional LSTM for words"
)
optparser.add_option(
    "-p", "--pre_emb", default="",
    help="Location of pretrained embeddings"
)
optparser.add_option(
    "-A", "--all_emb", default="1",
    type='int', help="Load all embeddings"
)
optparser.add_option(
    "-a", "--cap_dim", default="0",
    type='int', help="Capitalization feature dimension (0 to disable)"
)
optparser.add_option(
    "-f", "--crf", default="1",
    type='int', help="Use CRF (0 to disable)"
)
optparser.add_option(
    "-D", "--dropout", default="0.5",
    type='float', help="Droupout on the input (0 = no dropout)"
)
optparser.add_option(
    "-L", "--lr_method", default="sgd-lr_.005",
    help="Learning method (SGD, Adadelta, Adam..)"
)
optparser.add_option(
    "-r", "--reload", default="0",
    type='int', help="Reload the last saved model"
)

optparser.add_option(
    "-U", "--use_word", default="1",
    type = 'int', help = "Whether to use word embedding"
)

optparser.add_option(
    "-u", "--use_char", default="1",
    type = 'int', help = "Whether to use char embedding"
)
optparser.add_option(
    "-H", "--hidden_layer", default = "1",
    type = 'int', help = "number of layers used in LSTM"
)
optparser.add_option(
    "-K", "--kernel_size", default = "2,3,4,5",
    type  = 'string', action='callback',
    callback=get_comma_separated_args
)

optparser.add_option(
    "-k", "--kernel_num", default = "100,100,100,100",
    type  = 'string', action='callback',
    callback=get_comma_separated_args
)

optparser.add_option(
    "-P", "--padding", default = "0",
    type = 'int', help = "whether padding the input to use gram-CNN"
)

optparser.add_option(
    "-S", "--pts", default = "0",
    type = 'int', help = "whether use the pts tagging"
)

opts = optparser.parse_args()[0]

# Parse parameters
parameters = OrderedDict()

#IOB OR IOEB
parameters['padding'] = opts.padding == 1
parameters['tag_scheme'] = opts.tag_scheme
parameters['lower'] = opts.lower == 1
parameters['zeros'] = opts.zeros == 1
parameters['char_dim'] = opts.char_dim
parameters['char_lstm_dim'] = opts.char_lstm_dim
parameters['char_bidirect'] = opts.char_bidirect == 1
parameters['word_dim'] = opts.word_dim
parameters['word_lstm_dim'] = opts.word_lstm_dim
parameters['word_bidirect'] = opts.word_bidirect == 1
parameters['pre_emb'] = opts.pre_emb
parameters['all_emb'] = opts.all_emb == 1
parameters['cap_dim'] = opts.cap_dim
parameters['crf'] = opts.crf == 1
parameters['dropout'] = opts.dropout
parameters['lr_method'] = opts.lr_method
parameters['use_word'] = opts.use_word == 1
parameters['use_char'] = opts.use_char == 1
parameters['hidden_layer'] = opts.hidden_layer
parameters['reload'] = opts.reload == 1
parameters['kernels'] = [2,3,4,5] if type(opts.kernel_size) == str else map(lambda x : int(x), opts.kernel_size)
parameters['num_kernels'] = [100,100,100,100] if type(opts.kernel_num) == str else map(lambda x : int(x), opts.kernel_num)
parameters['pts'] = opts.pts == 1

model_name = 'use_word' + str(parameters['use_word']) + \
            ' use_char' + str(parameters['use_char']) + \
            ' drop_out' + str(parameters['dropout']) + \
            ' hidden_size' + str(parameters['word_lstm_dim']) + \
            ' hidden_layer' + str(parameters['hidden_layer']) + \
            ' lower' + str(parameters['lower']) + \
            ' allemb' + str(parameters['all_emb']) + \
            ' kernels' + str(parameters['kernels'])[1:-1] + \
            ' num_kernels' + str(parameters['num_kernels'])[1:-1] + \
            ' padding' + str(parameters['padding']) + \
            ' pts' + str(parameters['pts']) + \
            ' w_emb' + str(parameters['word_dim'])

# Check parameters validity
assert os.path.isfile(opts.train)
assert os.path.isfile(opts.dev)
#assert os.path.isfile(opts.test)
assert parameters['char_dim'] > 0 or parameters['word_dim'] > 0
assert 0. <= parameters['dropout'] < 1.0
assert parameters['tag_scheme'] in ['iob', 'iobes']
assert not parameters['all_emb'] or parameters['pre_emb']
assert not parameters['pre_emb'] or parameters['word_dim'] > 0
assert not parameters['pre_emb'] or os.path.isfile(parameters['pre_emb'])

# Check evaluation script / folders
if not os.path.isfile(eval_script):
    raise Exception('CoNLL evaluation script not found at "%s"' % eval_script)
if not os.path.exists(eval_temp):
    os.makedirs(eval_temp)
if not os.path.exists(models_path):
    os.makedirs(models_path)

if 'bin' in parameters['pre_emb']:
    wordmodel = word2vec.Word2Vec.load_word2vec_format(parameters['pre_emb'], binary=True)
else:
    wordmodel = word2vec.Word2Vec.load_word2vec_format(parameters['pre_emb'], binary=False)


# Data parameters
lower = parameters['lower']
zeros = parameters['zeros']
tag_scheme = parameters['tag_scheme']

# Load sentences
train_sentences = loader.load_sentences(opts.train, lower, zeros)
dev_sentences = loader.load_sentences(opts.dev, lower, zeros)

avg_len = sum([len(i) for i in train_sentences]) / float(len(train_sentences))
print "train avg len: %d" % (avg_len)

if os.path.isfile(opts.test):
    test_sentences = loader.load_sentences(opts.test, lower, zeros)

# Use selected tagging scheme (IOB / IOBES)
update_tag_scheme(train_sentences, tag_scheme)
update_tag_scheme(dev_sentences, tag_scheme)
if os.path.isfile(opts.test):
    update_tag_scheme(test_sentences, tag_scheme)

dt_sentences = []
if os.path.isfile(opts.test):
    dt_sentences = dev_sentences + test_sentences
else:
    dt_sentences = dev_sentences

# Create a dictionary / mapping of words
# If we use pretrained embeddings, we add them to the dictionary.
word_to_id = []
char_to_id = []
pt_to_id = []
tag_to_id = []
if not parameters['reload']:
    if parameters['pre_emb']:
        # mapping of words frenquency decreasing
        dico_words_train = word_mapping(train_sentences, lower)[0]
        dico_words, word_to_id, id_to_word = augment_with_pretrained(
            dico_words_train.copy(),
            wordmodel,
            list(itertools.chain.from_iterable(
                [[w[0] for w in s] for s in dt_sentences])
            ) if not parameters['all_emb'] else None
        )
    else:
        dico_words, word_to_id, id_to_word = word_mapping(train_sentences, lower)
        dico_words_train = dico_words


    # Create a dictionary and a mapping for words / POS tags / tags
    dico_chars, char_to_id, id_to_char = char_mapping(train_sentences)
    dico_tags, tag_to_id, id_to_tag = tag_mapping(train_sentences)
    dico_pts, pt_to_id, id_to_pt = pt_mapping(train_sentences + dev_sentences)
    if not os.path.exists(os.path.join(models_path, model_name)):
            os.makedirs(os.path.join(models_path,model_name))
    save_mappings(os.path.join(models_path, model_name, 'mappings.pkl'), word_to_id, char_to_id, tag_to_id, pt_to_id, dico_words, id_to_tag)
else:
    word_to_id, char_to_id, tag_to_id, pt_to_id, dico_words, id_to_tag = reload_mappings(os.path.join(models_path,model_name, 'mappings.pkl'))
    dico_words_train = dico_words
    id_to_word = {v: k for k, v in word_to_id.items()}
# Index data
m3 = 0
train_data,m1 = prepare_dataset(
    train_sentences, word_to_id, char_to_id, tag_to_id, pt_to_id,lower
)
dev_data,m2 = prepare_dataset(
    dev_sentences, word_to_id, char_to_id, tag_to_id, pt_to_id,lower
)
if os.path.isfile(opts.test):
    test_data,m3 = prepare_dataset(
        test_sentences, word_to_id, char_to_id, tag_to_id, pt_to_id,lower
    )

max_seq_len = max(m1,m2,m3)
print "max length is %i" % (max_seq_len)

print "%i / %i  sentences in train / dev." % (
   len(train_data), len(dev_data))

#
# Train network
#
singletons = set([word_to_id[k] for k, v
                  in dico_words_train.items() if v == 1])

n_epochs = 1000  # number of epochs over the training set
freq_eval = 2000  # evaluate on dev every freq_eval steps
best_dev = -np.inf
best_test = -np.inf
count = 0


#initilaze the embedding matrix
word_emb_weight = np.zeros((len(dico_words), parameters['word_dim']))
c_found = 0
c_lower = 0
c_zeros = 0
n_words = len(dico_words)
for i in xrange(n_words):
        word = id_to_word[i]
        if word in wordmodel:
            word_emb_weight[i] = wordmodel[word]
            c_found += 1
        elif re.sub('\d', '0', word) in wordmodel:
            word_emb_weight[i] = wordmodel[
                re.sub('\d', '0', word)
            ]
            c_zeros += 1

print 'Loaded %i pretrained embeddings.' % len(wordmodel.vocab)
print ('%i / %i (%.4f%%) words have been initialized with '
       'pretrained embeddings.') % (
            c_found + c_lower + c_zeros, n_words,
            100. * (c_found + c_lower + c_zeros) / n_words
      )
print ('%i found directly, %i after lowercasing, '
       '%i after lowercasing + zero.') % (
          c_found, c_lower, c_zeros
      )

gramcnn = GRAMCNN(n_words, len(char_to_id), len(pt_to_id),
                    use_word = parameters['use_word'],
                    use_char = parameters['use_char'],
                    use_pts = parameters['pts'],
                    num_classes = len(id_to_tag),
                    word_emb = parameters['word_dim'],
                    drop_out = parameters['dropout'],
                    word2vec = word_emb_weight,feature_maps=parameters['num_kernels'],#,200,200, 200,200],
                    kernels=parameters['kernels'], hidden_size = parameters['word_lstm_dim'], hidden_layers = parameters['hidden_layer'],
                    padding = parameters['padding'], max_seq_len = max_seq_len, train_size = len(train_data))

if parameters['reload']:
    gramcnn.load(models_path ,model_name)

for epoch in xrange(n_epochs):
    epoch_costs = []
    print "Starting epoch %i..." % epoch
    for i, index in enumerate(np.random.permutation(len(train_data))):
        inputs, word_len = create_input(train_data[index], parameters, True, singletons,
            padding = parameters['padding'], max_seq_len = max_seq_len, use_pts = parameters['pts'] )

        assert inputs['char_for']
        assert inputs['word']
        assert inputs['label']

        # break
        if len(inputs['label']) == 1:
            continue
        train_loss = []
        temp = []
        temp.append(word_len)
        batch_loss = gramcnn.train(inputs, temp)

        train_loss.append(batch_loss)

        if(i % 500 == 0 and i != 0):
            print( "Epoch[%d], "%(epoch) + "Iter " + str(i) + \
                    ", Minibatch Loss= " + "{:.6f}".format(np.mean(train_loss[-500:])))
            train_loss = []

        if i % 2000 == 0 and i != 0:
            dev_score = evaluate(parameters, gramcnn, dev_sentences,
                                 dev_data, id_to_tag, padding = parameters['padding'],
                                 max_seq_len = max_seq_len, use_pts = parameters['pts'])
            print "dev_score_end"
            print "Score on dev: %.5f" % dev_score
            if dev_score > best_dev:
                best_dev = dev_score
                print "New best score on dev."
                print "Saving model to disk..."
                gramcnn.save(models_path ,model_name)
            if os.path.isfile(opts.test):
                if i % 8000 == 0 and i != 0:
                    test_score = evaluate(parameters, gramcnn, test_sentences,
                                          test_data, id_to_tag, padding = parameters['padding'],
                                          max_seq_len = max_seq_len, use_pts = parameters['pts'])
                    print "Score on test: %.5f" % test_score
                    if test_score > best_test:
                        best_test = test_score
                        print "New best score on test."
