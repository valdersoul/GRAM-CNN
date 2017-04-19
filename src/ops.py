import math
import numpy as np
import tensorflow as tf

from tensorflow.python.framework import ops

from utils import *

def index_along_every_row(array, index):
    N, _ = array.shape 
    return array[np.arange(N), index]

def unpack(array):
    N, _ = array.shape
    return np.split(array, N)

def conv2d(input_, output_dim, k_h, k_w,
           stddev=0.02, name="conv2d"):
  with tf.variable_scope(name):
    w = tf.get_variable('w', [k_h, k_w, input_.get_shape()[-1], output_dim],
              initializer=tf.truncated_normal_initializer(stddev=stddev))
    b = tf.get_variable('b', output_dim, initializer=tf.truncated_normal_initializer(stddev=stddev))
    conv = tf.nn.conv2d(input_, w, strides=[1, 1, 1, 1], padding='VALID') + b
    return conv

def conv2d_same(input_, output_dim, k_h, k_w,
           stddev=0.02, name="conv2d"):
  with tf.variable_scope(name):
    w = tf.get_variable('w', [k_h, k_w, input_.get_shape()[-1], output_dim],
              initializer=tf.truncated_normal_initializer(stddev=stddev))
    b = tf.get_variable('b', output_dim, initializer=tf.truncated_normal_initializer(stddev=stddev))
    conv = tf.nn.conv2d(input_, w, strides=[1, 1, 1, 1], padding='SAME') + b
    return conv

def highway(x, size, layer_size=1, bias=-2, f=tf.nn.relu):
  """Highway Network (cf. http://arxiv.org/abs/1505.00387).
  
  t = sigmoid(Wy + b)
  z = t * g(Wy + b) + (1 - t) * y
  where g is nonlinearity, t is transform gate, and (1 - t) is carry gate.
  """
  with tf.variable_scope('highway'):
    for idx in xrange(layer_size):
      W_T = tf.get_variable("weight_transform%d" % idx, [size, size], initializer=tf.truncated_normal_initializer(stddev=0.1))
      b_T = bias

      W = tf.get_variable("weight%d" % idx, [size, size],initializer=tf.truncated_normal_initializer(stddev=0.1))
      b = 0.1

      T = tf.sigmoid(tf.matmul(x, W_T) + b_T)
      H = f(tf.matmul(x, W) + b)
      C = 1. - T

      y = tf.add(tf.multiply(H, T), tf.multiply(x, C), "y")
      x = y
  return y

class batch_norm(object):
  """Code modification of http://stackoverflow.com/a/33950177"""
  def __init__(self, epsilon=1e-5, momentum = 0.1, name="batch_norm"):
    with tf.variable_scope(name) as scope:
      self.epsilon = epsilon
      self.momentum = momentum

      self.ema = tf.train.ExponentialMovingAverage(decay=self.momentum)
      self.name=name

  def __call__(self, x, train=True):
    shape = x.get_shape().as_list()

    with tf.variable_scope(self.name) as scope:
      self.gamma = tf.get_variable("gamma", [shape[-1]],
          initializer=tf.random_normal_initializer(1., 0.02))
      self.beta = tf.get_variable("beta", [shape[-1]],
          initializer=tf.constant_initializer(0.))

      mean, variance = tf.nn.moments(x, [0, 1, 2])

      return tf.nn.batch_norm_with_global_normalization(
        x, mean, variance, self.beta, self.gamma, self.epsilon,
        scale_after_normalization=True)
