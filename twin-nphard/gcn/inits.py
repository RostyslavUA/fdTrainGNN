import tensorflow as tf
import numpy as np
from tensorflow.keras.initializers import GlorotUniform
# tf.compat.v1.disable_eager_execution()

flags = tf.compat.v1.flags
FLAGS = flags.FLAGS


def uniform(shape, scale=0.05, name=None):
    """Uniform init."""
    initial = tf.compat.v1.random_uniform(shape, minval=-scale, maxval=scale, dtype=tf.float32)
    return tf.Variable(initial, name=name)


def glorot(shape, name=None):
    """Glorot & Bengio (AISTATS 2010) init."""
    init_range = np.sqrt(6.0/(shape[0]+shape[1]))
    initial = tf.compat.v1.random_uniform(shape, minval=-init_range, maxval=init_range, dtype=tf.float32)
    return tf.Variable(initial, name=name)


def zeros(shape, name=None):
    """All zeros."""
    initial = tf.zeros(shape, dtype=tf.float32)
    return tf.Variable(initial, name=name)


def ones(shape, name=None):
    """All ones."""
    initial = tf.ones(shape, dtype=tf.float32)
    return tf.Variable(initial, name=name)


class GlorotUniformDistr(GlorotUniform):
    """
    Glorot weight initializer (uniform distribution) for distributed models.
    """
    def __call__(self, shape, dtype=None, **kwargs):
        # Make sure that all nodes have same init weights
        K, num_nodes, input_dim, channels = shape
        weights_per_node = super(GlorotUniformDistr, self).__call__((K, 1, input_dim, channels), dtype)
        weights = tf.tile(weights_per_node, [1, num_nodes, 1, 1])
        return weights
