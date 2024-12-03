from .inits import *
from tensorflow.keras import backend as KB
from tensorflow.keras.layers import Layer as LayerKeras
from spektral.layers import ops
import tensorflow as tf
from spektral.layers import ChebConv

# global unique layer ID dictionary for layer name assignment
_LAYER_UIDS = {}


def get_layer_uid(layer_name=''):
    """Helper function, assigns unique layer IDs."""
    if layer_name not in _LAYER_UIDS:
        _LAYER_UIDS[layer_name] = 1
        return 1
    else:
        _LAYER_UIDS[layer_name] += 1
        return _LAYER_UIDS[layer_name]


def sparse_dropout(x, keep_prob, noise_shape):
    """Dropout for sparse tensors."""
    random_tensor = keep_prob
    random_tensor += tf.compat.v1.random_uniform(noise_shape)
    dropout_mask = tf.cast(tf.floor(random_tensor), dtype=tf.bool)
    pre_out = tf.compat.v1.sparse_retain(x, dropout_mask)
    return pre_out * (1./keep_prob)


def sparse_dropout_no_scale(x, keep_prob, noise_shape):
    """Dropout for sparse tensors."""
    random_tensor = keep_prob
    random_tensor += tf.compat.v1.random_uniform(noise_shape)
    dropout_mask = tf.cast(tf.floor(random_tensor), dtype=tf.float64)
    pre_out = x * dropout_mask
    return pre_out


def dot(x, y, sparse=False):
    """Wrapper for tf.matmul (sparse vs dense)."""
    if sparse:
        res = tf.compat.v1.sparse_tensor_dense_matmul(x, y)
    else:
        res = tf.matmul(x, y)
    return res


def maxpooling(x, y, sparse=False):
    """Wrapper for tf.matmul (sparse vs dense)."""
    featuresize = y.shape[1]
    res_cols = []
    if sparse:
        for i in range(featuresize):
            diag = tf.compat.v1.diag(y[:,i])
            imtx = tf.compat.v1.sparse_tensor_dense_matmul(x, diag)
            icol = tf.reduce_max(imtx, axis=1)
            res_cols.append(icol)
        res = tf.reshape(tf.concat(res_cols, axis=0),(-1,featuresize))
    else:
        for i in range(featuresize):
            diag = tf.compat.v1.diag(y[:,i])
            imtx = tf.matmul(x, diag)
            icol = tf.reduce_max(imtx, axis=1)
            res_cols.append(icol)
        res = tf.reshape(tf.concat(res_cols, axis=0),(-1,featuresize))
    return res


class Layer(object):
    """Base layer class. Defines basic API for all layer objects.
    Implementation inspired by keras (http://keras.io).

    # Properties
        name: String, defines the variable scope of the layer.
        logging: Boolean, switches Tensorflow histogram logging on/off

    # Methods
        _call(inputs): Defines computation graph of layer
            (i.e. takes input, returns output)
        __call__(inputs): Wrapper for _call()
        _log_vars(): Log all variables
    """

    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            layer = self.__class__.__name__.lower()
            name = layer + '_' + str(get_layer_uid(layer))
        self.name = name
        self.vars = {}
        logging = kwargs.get('logging', False)
        self.logging = logging
        self.sparse_inputs = False

    def _call(self, inputs):
        return inputs

    def __call__(self, inputs):
        with tf.name_scope(self.name):
            if self.logging and not self.sparse_inputs:
                tf.summary.histogram(self.name + '/inputs', inputs)
            outputs = self._call(inputs)
            if self.logging:
                tf.summary.histogram(self.name + '/outputs', outputs)
            return outputs

    def _log_vars(self):
        for var in self.vars:
            tf.summary.histogram(self.name + '/vars/' + var, self.vars[var])


class Dense(Layer):
    """Dense layer."""
    def __init__(self, input_dim, output_dim, placeholders, dropout=0., sparse_inputs=False,
                 act=tf.nn.relu, bias=False, featureless=False, **kwargs):
        super(Dense, self).__init__(**kwargs)

        if dropout:
            self.dropout = placeholders['dropout']
        else:
            self.dropout = 0.

        self.act = act
        self.sparse_inputs = sparse_inputs
        self.featureless = featureless
        self.bias = bias

        # helper variable for sparse dropout
        self.num_features_nonzero = placeholders['num_features_nonzero']

        with tf.compat.v1.variable_scope(self.name + '_vars'):
            self.vars['weights'] = glorot([input_dim, output_dim],
                                          name='weights')
            if self.bias:
                self.vars['bias'] = zeros([output_dim], name='bias')

        if self.logging:
            self._log_vars()

    def _call(self, inputs):
        x = inputs

        # dropout
        if self.sparse_inputs:
            x = sparse_dropout(x, 1-self.dropout, self.num_features_nonzero)
        else:
            x = tf.nn.dropout(x, rate=self.dropout)

        # transform
        output = dot(x, self.vars['weights'], sparse=self.sparse_inputs)

        # bias
        if self.bias:
            output += self.vars['bias']

        return self.act(output)


class GraphConvolution(Layer):
    """Graph convolution layer."""
    def __init__(self, input_dim, output_dim, placeholders, dropout=0., channel=0, num_channels=1,
                 sparse_inputs=False, act=tf.nn.relu, bias=False,
                 featureless=False, **kwargs):
        super(GraphConvolution, self).__init__(**kwargs)

        if dropout:
            self.dropout = placeholders['dropout']
        else:
            self.dropout = 0.

        self.act = act
        self.sparse_inputs = sparse_inputs
        self.featureless = featureless
        self.bias = bias
        self.channel = int(channel)
        self.num_channels = int(num_channels)
        self.order = int(len(placeholders['support']) / self.num_channels)
        self.support = placeholders['support'][self.channel * self.order: self.channel*self.order+self.order]

        # helper variable for sparse dropout
        self.num_features_nonzero = placeholders['num_features_nonzero']

        with tf.compat.v1.variable_scope(self.name + '_vars'):
            for i in range(len(self.support)):
                if FLAGS.wts_init == 'random':
                    self.vars['weights_' + str(i)] = glorot([input_dim, output_dim],
                                                            name='weights_' + str(i))
                elif FLAGS.wts_init == 'zeros':
                    self.vars['weights_' + str(i)] = zeros([input_dim, output_dim],
                                                            name='weights_' + str(i))
                else:
                    raise NameError('Unsupported wts_init: {}'.format(FLAGS.wts_init))
            if self.bias:
                self.vars['bias'] = zeros([output_dim], name='bias')

        if self.logging:
            self._log_vars()

    def _call(self, inputs):
        x = inputs

        # dropout
        if self.sparse_inputs:
            x = sparse_dropout(x, 1-self.dropout, self.num_features_nonzero)
        else:
            x = tf.nn.dropout(x, rate=self.dropout)

        # convolve
        supports = list()
        for i in range(len(self.support)):
            if not self.featureless:
                pre_sup = dot(x, self.vars['weights_' + str(i)],
                              sparse=self.sparse_inputs)
            else:
                pre_sup = self.vars['weights_' + str(i)]
            support = dot(self.support[i], pre_sup, sparse=True)
            supports.append(support)
        output = tf.add_n(supports)

        # bias
        if self.bias:
            output += self.vars['bias']

        # concated = tf.concat([output, self.act(output)], axis=1)
        # return tf.layers.dense(concated, output.shape[1])
        return self.act(output)


class DChebConv(ChebConv):
    def build(self, input_shape):
        assert len(input_shape) >= 2
        input_dim = input_shape[0][-1]
        num_nodes = input_shape[1][-1]
        self.kernel = self.add_weight(
            shape=(self.K, num_nodes, input_dim, self.channels),
            initializer=self.kernel_initializer,
            name="kernel",
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
        )
        if self.use_bias:
            self.bias = self.add_weight(
                shape=(num_nodes, self.channels,),  # Same bias for all kernels. Contrary to documentation.
                initializer=self.bias_initializer,
                name="bias",
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
            )
        self.built = True

    def call(self, inputs, mask=None):
        x, a = inputs
        if len(a.shape) < 3:  # No batch dim
            x = x[tf.newaxis]
        # Assuming the dims of x [batch, node, feat_in]

        T_0 = x
        output = T_0[:, :, tf.newaxis] @ self.kernel[0, tf.newaxis]  # Broadcast along batch dim

        if self.K > 1:
            T_1 = ops.modal_dot(a, x)
            output += T_1[:, :, tf.newaxis] @ self.kernel[1, tf.newaxis]

        for k in range(2, self.K):
            T_2 = 2 * ops.modal_dot(a, T_1) - T_0
            output += T_2[:, :, tf.newaxis] @ self.kernel[k, tf.newaxis]
            T_0, T_1 = T_1, T_2

        if self.use_bias:
            output = KB.bias_add(output, self.bias[:, tf.newaxis])
        if mask is not None:
            output *= mask[0]
        output = tf.squeeze(output, (0, 2))
        output = self.activation(output)
        return output


class DropoutNoScaling(LayerKeras):
    def __init__(self, rate, **kwargs):
        super(DropoutNoScaling, self).__init__(**kwargs)
        self.rate = rate

    def call(self, inputs, training=None):
        if training:
            # Apply dropout without scaling
            noise_shape = tf.shape(inputs)
            random_tensor = tf.random.uniform(noise_shape)
            dropout_mask = tf.cast(random_tensor >= self.rate, inputs.dtype)
            return inputs * dropout_mask
        return inputs
