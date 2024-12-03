import pdb
import numpy as np
import tensorflow as tf
import networkx as nx


class GlorotUniformDistr(tf.initializers.GlorotUniform):
    """
    Glorot weight initializer (uniform distribution) for distributed models.
    """
    def __call__(self, shape, dtype=None, **kwargs):
        # Make sure that all nodes have same init weights
        K, num_nodes, input_dim, channels = shape
        weights_per_node = super(GlorotUniformDistr, self).__call__((1, 1, input_dim, channels), dtype)
        weights = tf.tile(weights_per_node, [K, num_nodes, 1, 1])
        return weights


def dropout(inputs, rate=0.0):
    noise_shape = tf.shape(inputs)
    random_tensor = tf.random.uniform(noise_shape)
    dropout_mask = tf.cast(random_tensor >= rate, inputs.dtype)
    return inputs * dropout_mask


# DUWMMSE
class DUWMMSE(object):
        # Initialize
        def __init__( self, nNodes, Pmax=1., var=7e-10, feature_dim=3, batch_size=64, layers=4, learning_rate=1e-3,
                      max_gradient_norm=5.0, exp='duwmmse', optimizer='adam', grad_subsample_p=0.0, dropout_op=0.0,
                      reg_constant=0.0):
            self.nNodes = nNodes
            self.Pmax              = tf.cast( Pmax, tf.float64 )
            self.var               = var
            self.feature_dim       = feature_dim
            self.batch_size        = batch_size
            self.layers            = layers
            self.learning_rate     = learning_rate
            self.max_gradient_norm = max_gradient_norm
            self.exp               = exp
            self.global_step = tf.Variable(0, trainable=False, name='global_step')
            self.optimizer = optimizer
            self.grad_subsample_p = grad_subsample_p
            self.dropout_op = dropout_op
            self.reg_constant = reg_constant
            self.build_model()

        # Build Model
        def build_model(self):
            self.init_placeholders()
            self.build_network()
            self.build_objective()

        def init_placeholders(self):
            # CSI [Batch_size X Nodes X Nodes]
            self.H = tf.compat.v1.placeholder(tf.float64, shape=[None, None, None], name="H")
            self.cmat = tf.compat.v1.placeholder(tf.float64, shape=[None, None, None], name="cmat")
            self.subsample_prob = tf.compat.v1.placeholder(tf.float64, shape=[None, 2], name="subsample_prob")
            self.apply_dropout = tf.compat.v1.placeholder(tf.bool, name="apply_dropout")

            # NSI [Batch_size X Nodes X Features]
            #self.x = tf.compat.v1.placeholder(tf.float64, shape=[None, None, self.feature_dim], name="x")

            # Node Weights [Batch_size X Nodes]
            #self.alpha = tf.compat.v1.placeholder(tf.float64, shape=[None, None], name="alpha")

            # Boolean for Training/Inference
            #self.phase = tf.compat.v1.placeholder_with_default(False, shape=(), name='phase')

        # Building network
        def build_network(self):
            # Squared H
            self.Hsq = tf.math.square(self.H)

            # Diag H
            dH =  tf.linalg.diag_part( self.H )
            #self.dH = tf.matrix_diag( dH )
            self.dH = tf.compat.v1.matrix_diag(dH)

            # Retrieve number of nodes for initializing V
            # self.nNodes = tf.shape( self.H )[-1]  # Passed as a parameter during init

            # Maximum V = sqrt(Pmax)
            Vmax = tf.math.sqrt(self.Pmax)

            # Initial V
            V = Vmax * tf.ones([self.batch_size, self.nNodes], dtype=tf.float64)

            self.pow_alloc = []

            # Iterate over layers l
            for l in range(self.layers):
                #with tf.variable_scope('Layer{}'.format(l+1)):
                with tf.compat.v1.variable_scope('Layer{}'.format(l+1)):
                    # Compute U^l
                    U = self.U_block( V )

                    # Compute W^l
                    W_wmmse = self.W_block( U, V )

                    # Learn a^l
                    a = self.gcn('a')

                    # Learn b^l
                    b = self.gcn('b')

                    # Compute Wcap^l = a^l * W^l + b^l
                    W = tf.math.add( tf.math.multiply( a, W_wmmse ), b )

                    # Learn mu^l
                    #mu = tf.get_variable( name='mu', initializer=tf.constant(0., shape=(), dtype=tf.float64))
#                     mu = tf.compat.v1.get_variable( name='mu', initializer=tf.constant(0., shape=(), dtype=tf.float64))

                    # Compute V^l
                    if self.exp == 'wmmse':
                        V = self.V_block( U, W_wmmse, 0. )
                    else:
                        V = self.V_block( U, W, 0. )

                    # Saturation non-linearity  ->  0 <= V <= Vmax
                    V = tf.math.minimum(V, Vmax) + tf.math.maximum(V, 0) - V

            # Final V
            self.pow_alloc = V

        def U_block(self, V):
            # H_ii * v_i
            #num = tf.math.multiply( tf.matrix_diag_part(self.H), V )
            num = tf.math.multiply( tf.compat.v1.matrix_diag_part(self.H), V )

            # sigma^2 + sum_j( (H_ji)^2 * (v_j)^2 )
            den = tf.reshape( tf.matmul( tf.transpose( self.Hsq, perm=[0,2,1] ), tf.reshape( tf.math.square( V ), [-1, self.nNodes, 1] ) ), [-1, self.nNodes] ) + self.var

            # U = num/den
            return( tf.math.divide( num, den ) )

        # Sum-rate = z
        def W_block(self, U, V):
            # 1 - u_i * H_ii * v_i
            #den = 1. - tf.math.multiply( tf.matrix_diag_part(self.H), tf.math.multiply( U, V ) )
            den = 1. - tf.math.multiply( tf.compat.v1.matrix_diag_part(self.H), tf.math.multiply( U, V ) )

            # W = 1/den
            return( tf.math.reciprocal( den ) )

        # Weighted Sum-rate = a * z
        def W_block1(self, U, V):
            # 1 - u_i * H_ii * v_i
            den = 1. - tf.math.multiply( tf.matrix_diag_part(self.H), tf.math.multiply( U, V ) )

            # W = alpha/den
            return( tf.math.divide( self.alpha, den ) )


        def gcn(self, name):
            # 2 Layers
            L = 2

            # Hidden dim = 5
            input_dim = [self.feature_dim,5]
            output_dim = [5,1]

            ## NSI [Batch_size X Nodes X Features]
            x = tf.ones([self.batch_size, self.nNodes, 1, 1], dtype=tf.float64)
            #x = self.x

            #with tf.variable_scope('gcn_'+name):
            with tf.compat.v1.variable_scope('gcn_'+name):
                for l in range(L):
                    #with tf.variable_scope('gc_l{}'.format(l+1)):
                    with tf.compat.v1.variable_scope('gc_l{}'.format(l+1)):
                        # Weights
                        #w1 = tf.get_variable( name='w1', shape=(input_dim[l], output_dim[l]), initializer=tf.initializers.glorot_uniform(), dtype=tf.float64)
                        w1 = tf.compat.v1.get_variable( name='w1', shape=(self.batch_size, self.nNodes, input_dim[l], output_dim[l]),
                                                        initializer=GlorotUniformDistr(l), dtype=tf.float64)
                        #w0 = tf.get_variable( name='w0', shape=(input_dim[l], output_dim[l]), initializer=tf.initializers.glorot_uniform(), dtype=tf.float64)
                        w0 = tf.compat.v1.get_variable( name='w0', shape=(self.batch_size, self.nNodes, input_dim[l], output_dim[l]),
                                                        initializer=GlorotUniformDistr(l), dtype=tf.float64)
                        ## Biases
                        #b1 = tf.get_variable( name='b1', initializer=tf.constant(0.1, shape=(output_dim[l],), dtype=tf.float64) )
                        b1 = tf.compat.v1.get_variable( name='b1', initializer=tf.constant(0.1, shape=(self.batch_size, self.nNodes, 1, output_dim[l],), dtype=tf.float64) )
                        #b0 = tf.get_variable( name='b0', initializer=tf.constant(0.1, shape=(output_dim[l],), dtype=tf.float64) )
                        b0 = tf.compat.v1.get_variable( name='b0', initializer=tf.constant(0.1, shape=(self.batch_size, self.nNodes, 1, output_dim[l],), dtype=tf.float64) )
                        # XW
                        x1 = tf.matmul(x, w1)
                        x0 = tf.matmul(x, w0)
                        x1 = tf.reshape(x1, shape=(self.batch_size, self.nNodes, output_dim[l]))
                        x0 = tf.reshape(x0, shape=(self.batch_size, self.nNodes, output_dim[l]))

                        # dropout
                        H_drop = tf.cond(self.apply_dropout,
                                         lambda: dropout(self.H, self.dropout_op),
                                         lambda: self.H)
                        dH_drop = tf.cond(self.apply_dropout,
                                         lambda: dropout(self.dH, self.dropout_op),
                                         lambda: self.dH)

                        # diag(A)XW0 + AXW1
                        x1 = tf.matmul(H_drop, x1)
                        x0 = tf.matmul(dH_drop, x0)
                        x1 = tf.reshape(x1, shape=(self.batch_size, self.nNodes, 1, output_dim[l]))
                        x0 = tf.reshape(x0, shape=(self.batch_size, self.nNodes, 1, output_dim[l]))
                        ## AXW + B
                        x1 = tf.add(x1, b1)
                        x0 = tf.add(x0, b0)

                        # Combine
                        x = x1 + x0

                        # activation(AXW + B)
                        if l == 0:
                            x = tf.nn.relu(x)
                        else:
                            x = tf.nn.sigmoid(x)
                # Coefficients (a / b) [Batch_size X Nodes]
                output = tf.squeeze(x)

            return output

        def V_block(self, U, W, mu):
            # H_ii * u_i * w_i
            #num = tf.math.multiply( tf.matrix_diag_part(self.H), tf.math.multiply( U, W ) )
            num = tf.math.multiply( tf.compat.v1.matrix_diag_part(self.H), tf.math.multiply( U, W ) )

            # mu + sum_j( (H_ij)^2 * (u_j)^2 *w_j )
            den = tf.math.add( tf.reshape( tf.matmul( self.Hsq, tf.reshape( tf.math.multiply( tf.math.square( U ), W ), [-1, self.nNodes, 1] ) ), [-1, self.nNodes] ), mu)

            # V = num/den
            return( tf.math.divide( num, den ) )

        def build_objective(self):
            # (H_ii)^2 * (v_i)^2
            #num = tf.math.multiply( tf.matrix_diag_part(self.Hsq), tf.math.square( self.pow_alloc ) )
            num = tf.math.multiply( tf.compat.v1.matrix_diag_part(self.Hsq), tf.math.square( self.pow_alloc ) )

            # sigma^2 + sum_j j ~= i ( (H_ji)^2 * (v_j)^2 )
            den = tf.reshape( tf.matmul( tf.transpose( self.Hsq, perm=[0,2,1] ), tf.reshape( tf.math.square( self.pow_alloc ), [-1, self.nNodes, 1] ) ), [-1, self.nNodes] ) + self.var - num
            
            # rate
            rate = tf.math.log( 1. + tf.math.divide( num, den ) ) / tf.cast( tf.math.log( 2.0 ), tf.float64 )

            # Sum Rate = sum_i ( log(1 + SINR) )
            self.utility = tf.reduce_sum( rate, axis=1 )

            # Weighted Sum Rate
            #rate = tf.math.multiply( self.alpha, rate )
            #self.utility = tf.reduce_sum( rate, axis=1 )

            # Minimization objective
            self.obj = -tf.reduce_mean( self.utility )

            if self.exp == 'duwmmse':
                self.init_optimizer()

        def consensus(self, clip_gradients):
            gradients_cons = []
            for grad in clip_gradients:
                if len(grad.shape) > 0:
                    grad_resh = tf.reshape(grad, (self.batch_size, self.nNodes, -1))
                    for _ in range(1):
                        cmat_drop = dropout(self.cmat, self.dropout_op)
                        grad_resh = cmat_drop @ grad_resh  # TODO: advanced minibatching w/ sparse matrix
                    grad = tf.reshape(grad_resh, grad.shape) * self.nNodes
                gradients_cons.append(grad)
            # Sum over minibatch
            for i, grad in enumerate(gradients_cons):
                if len(grad.shape) > 0:
                    gradients_cons[i] = tf.repeat(tf.reduce_sum(grad, 0)[tf.newaxis], self.batch_size, 0)
            return gradients_cons

        def init_optimizer(self):
            # Gradients and SGD update operation for training the model
            self.trainable_params = tf.compat.v1.trainable_variables()

            #Learning Rate Decay
            #starter_learning_rate = self.learning_rate
            #self.learning_rate_decayed = tf.train.exponential_decay(starter_learning_rate, global_step=self.global_step, decay_steps=5000, decay_rate=0.99, staircase=True)

            # SGD with Momentum
            #self.opt = tf.train.GradientDescentOptimizer( learning_rate=learning_rate )
            #self.opt = tf.train.MomentumOptimizer(learning_rate=self.learning_rate_decayed, momentum=0.9, use_nesterov=True )

            # Optimizer
            if self.optimizer == 'adam':
                self.opt = tf.compat.v1.train.AdamOptimizer(learning_rate=self.learning_rate)
            elif self.optimizer == 'gd':
                self.opt = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=self.learning_rate)

            # Compute gradients of loss w.r.t. all trainable variables
            gradients = tf.gradients(self.obj, self.trainable_params)

            # Clip gradients by a given maximum_gradient_norm
            # clip_gradients, _ = tf.clip_by_global_norm(gradients, self.max_gradient_norm)
            clip_gradients = gradients
            if self.max_gradient_norm is not None:
                clip_gradients = [tf.clip_by_norm(grad, self.max_gradient_norm, axes=[0, 2, 3]) if len(grad.shape) == 4
                                  else grad for grad in clip_gradients]
            if self.grad_subsample_p != 0.0:
                bern_pq = tf.repeat(tf.constant([[self.grad_subsample_p, 1-self.grad_subsample_p]],
                                                            dtype=tf.float64),
                                                self.batch_size, 0)
                self.subsample_prob = tf.random.categorical(tf.math.log(bern_pq), self.nNodes, dtype=tf.int32)
                self.subsample_prob = tf.cast(self.subsample_prob, dtype=tf.float64)
                for i, grad in enumerate(clip_gradients):
                    if len(grad.shape) > 0:
                        clip_gradients[i] = grad * self.subsample_prob[:, :, tf.newaxis, tf.newaxis]
            clip_gradients = self.consensus(clip_gradients)
            # Update the model
            self.updates = self.opt.apply_gradients(
                zip(clip_gradients, self.trainable_params), global_step=self.global_step)

        def save(self, sess, path, var_list=None, global_step=None):
            saver = tf.compat.v1.train.Saver(var_list)
            save_path = saver.save(sess, save_path=path, global_step=global_step)

        def restore(self, sess, path, var_list=None):
            saver = tf.compat.v1.train.Saver(var_list)
            saver.restore(sess, save_path=tf.train.latest_checkpoint(path))

        def train(self, sess, inputs ):
            input_feed = dict()
            input_feed[self.H.name] = inputs[0]
            input_feed[self.cmat.name] = inputs[1]
            input_feed[self.apply_dropout.name] = True
            #input_feed[self.x.name] = features
            #input_feed[self.alpha.name] = alpha

            # Training Phase
            #input_feed[self.phase.name] = True

            output_feed = [self.obj, self.utility, self.pow_alloc, self.updates]

            outputs = sess.run(output_feed, input_feed)

            return outputs[0], outputs[1], outputs[2]

        def eval(self, sess, inputs ):
            input_feed = dict()
            input_feed[self.H.name] = inputs
            input_feed[self.apply_dropout.name] = True
            #input_feed[self.x.name] = features
            #input_feed[self.alpha.name] = alpha

            # Training Phase
            #input_feed[self.phase.name] = False

            output_feed = [self.obj,self.utility, self.pow_alloc]

            outputs = sess.run(output_feed, input_feed)

            return outputs[0], outputs[1], outputs[2]


# UWMMSE
class UWMMSE(object):
        # Initialize
        def __init__( self, Pmax=1., var=7e-10, feature_dim=3, batch_size=64, layers=4, learning_rate=1e-3,
                      max_gradient_norm=5.0, exp='uwmmse', optimizer='adam', dropout_op=0.0, reg_constant=0.0):
            self.Pmax              = tf.cast( Pmax, tf.float64 )
            self.var               = var
            self.feature_dim       = feature_dim
            self.batch_size        = batch_size
            self.layers            = layers
            self.learning_rate     = learning_rate
            self.max_gradient_norm = max_gradient_norm
            self.exp               = exp
            self.global_step = tf.Variable(0, trainable=False, name='global_step')
            self.optimizer = optimizer
            self.dropout_op = dropout_op
            self.reg_constant = reg_constant
            self.build_model()

        # Build Model
        def build_model(self):
            self.init_placeholders()
            self.build_network()
            self.build_objective()
            
        def init_placeholders(self):
            # CSI [Batch_size X Nodes X Nodes]
            self.H = tf.compat.v1.placeholder(tf.float64, shape=[None, None, None], name="H")
            self.apply_dropout = tf.compat.v1.placeholder(tf.bool, name="apply_dropout")

            # NSI [Batch_size X Nodes X Features]
            #self.x = tf.compat.v1.placeholder(tf.float64, shape=[None, None, self.feature_dim], name="x")
            
            # Node Weights [Batch_size X Nodes]
            #self.alpha = tf.compat.v1.placeholder(tf.float64, shape=[None, None], name="alpha")
            
            # Boolean for Training/Inference 
            #self.phase = tf.compat.v1.placeholder_with_default(False, shape=(), name='phase')
        
        
        # Building network
        def build_network(self):
            # Squared H 
            self.Hsq = tf.math.square(self.H)
            
            # Diag H
            dH =  tf.linalg.diag_part( self.H ) 
            #self.dH = tf.matrix_diag( dH )
            self.dH = tf.compat.v1.matrix_diag(dH)
            
            # Retrieve number of nodes for initializing V
            self.nNodes = tf.shape( self.H )[-1]

            # Maximum V = sqrt(Pmax)
            Vmax = tf.math.sqrt(self.Pmax)

            # Initial V
            V = Vmax * tf.ones([self.batch_size, self.nNodes], dtype=tf.float64)
            
            self.pow_alloc = []
            self.W_wmmses = []
            self.Ws = []
            # Iterate over layers l
            for l in range(self.layers):
                #with tf.variable_scope('Layer{}'.format(l+1)):
                with tf.compat.v1.variable_scope('Layer{}'.format(l+1)):
                    # Compute U^l
                    U = self.U_block( V )
                    
                    # Compute W^l
                    W_wmmse = self.W_block( U, V )
                    
                    # Learn a^l
                    a = self.gcn('a')

                    # Learn b^l
                    b = self.gcn('b')

                    # Compute Wcap^l = a^l * W^l + b^l
                    W = tf.math.add( tf.math.multiply( a, W_wmmse ), b )
                    self.W_wmmses.append(W_wmmse)
                    self.Ws.append(W)
                    # Learn mu^l
                    #mu = tf.get_variable( name='mu', initializer=tf.constant(0., shape=(), dtype=tf.float64))
#                     mu = tf.compat.v1.get_variable( name='mu', initializer=tf.constant(0., shape=(), dtype=tf.float64))

                    # Compute V^l
                    if self.exp == 'wmmse':
                        V = self.V_block( U, W_wmmse, 0. )
                    else:
                        V = self.V_block( U, W, 0. )
                    
                    # Saturation non-linearity  ->  0 <= V <= Vmax
                    V = tf.math.minimum(V, Vmax) + tf.math.maximum(V, 0) - V

            # Final V
            self.pow_alloc = V
        
        def U_block(self, V):
            # H_ii * v_i
            #num = tf.math.multiply( tf.matrix_diag_part(self.H), V )
            num = tf.math.multiply( tf.compat.v1.matrix_diag_part(self.H), V )
            
            # sigma^2 + sum_j( (H_ji)^2 * (v_j)^2 )
            den = tf.reshape( tf.matmul( tf.transpose( self.Hsq, perm=[0,2,1] ), tf.reshape( tf.math.square( V ), [-1, self.nNodes, 1] ) ), [-1, self.nNodes] ) + self.var 
            
            # U = num/den
            return( tf.math.divide( num, den ) )

        # Sum-rate = z
        def W_block(self, U, V):
            # 1 - u_i * H_ii * v_i
            #den = 1. - tf.math.multiply( tf.matrix_diag_part(self.H), tf.math.multiply( U, V ) )
            den = 1. - tf.math.multiply( tf.compat.v1.matrix_diag_part(self.H), tf.math.multiply( U, V ) )
            
            # W = 1/den
            return( tf.math.reciprocal( den ) )

        # Weighted Sum-rate = a * z
        def W_block1(self, U, V):
            # 1 - u_i * H_ii * v_i
            den = 1. - tf.math.multiply( tf.matrix_diag_part(self.H), tf.math.multiply( U, V ) )
            
            # W = alpha/den
            return( tf.math.divide( self.alpha, den ) )        


        def gcn(self, name):
            # 2 Layers
            L = 2
            
            # Hidden dim = 5
            input_dim = [self.feature_dim,5]
            output_dim = [5,1]
            
            ## NSI [Batch_size X Nodes X Features]
            x = tf.ones([self.batch_size, self.nNodes, 1], dtype=tf.float64)
            #x = self.x
                        
            #with tf.variable_scope('gcn_'+name):
            with tf.compat.v1.variable_scope('gcn_'+name):
                for l in range(L):
                    #with tf.variable_scope('gc_l{}'.format(l+1)):
                    with tf.compat.v1.variable_scope('gc_l{}'.format(l+1)):
                        # Weights
                        #w1 = tf.get_variable( name='w1', shape=(input_dim[l], output_dim[l]), initializer=tf.initializers.glorot_uniform(), dtype=tf.float64)
                        w1 = tf.compat.v1.get_variable( name='w1', shape=(input_dim[l], output_dim[l]),
                                                        initializer=tf.initializers.glorot_uniform(l), dtype=tf.float64)
                        #w0 = tf.get_variable( name='w0', shape=(input_dim[l], output_dim[l]), initializer=tf.initializers.glorot_uniform(), dtype=tf.float64)
                        w0 = tf.compat.v1.get_variable( name='w0', shape=(input_dim[l], output_dim[l]),
                                                        initializer=tf.initializers.glorot_uniform(l), dtype=tf.float64)
                        
                        ## Biases
                        #b1 = tf.get_variable( name='b1', initializer=tf.constant(0.1, shape=(output_dim[l],), dtype=tf.float64) )
                        b1 = tf.compat.v1.get_variable( name='b1', initializer=tf.constant(0.1, shape=(output_dim[l],), dtype=tf.float64) )
                        #b0 = tf.get_variable( name='b0', initializer=tf.constant(0.1, shape=(output_dim[l],), dtype=tf.float64) )
                        b0 = tf.compat.v1.get_variable( name='b0', initializer=tf.constant(0.1, shape=(output_dim[l],), dtype=tf.float64) )
                        
                        # XW
                        x1 = tf.matmul(x, w1)
                        x0 = tf.matmul(x, w0)

                        # dropout
                        H_drop = tf.cond(self.apply_dropout,
                                         lambda: dropout(self.H, self.dropout_op),
                                         lambda: self.H)
                        dH_drop = tf.cond(self.apply_dropout,
                                         lambda: dropout(self.dH, self.dropout_op),
                                         lambda: self.dH)
                        
                        # diag(A)XW0 + AXW1
                        x1 = tf.matmul(H_drop, x1)
                        x0 = tf.matmul(dH_drop, x0)
                        
                        ## AXW + B
                        x1 = tf.add(x1, b1)
                        x0 = tf.add(x0, b0)
                        
                        # Combine
                        x = x1 + x0
                        
                        # activation(AXW + B)
                        if l == 0:
                            x = tf.nn.relu(x)  
                        else:
                            x = tf.nn.sigmoid(x)

                # Coefficients (a / b) [Batch_size X Nodes]
                output = tf.squeeze(x)
            
            return output
        
        def V_block(self, U, W, mu):
            # H_ii * u_i * w_i
            #num = tf.math.multiply( tf.matrix_diag_part(self.H), tf.math.multiply( U, W ) )
            num = tf.math.multiply( tf.compat.v1.matrix_diag_part(self.H), tf.math.multiply( U, W ) )
            
            # mu + sum_j( (H_ij)^2 * (u_j)^2 *w_j )
            den = tf.math.add( tf.reshape( tf.matmul( self.Hsq, tf.reshape( tf.math.multiply( tf.math.square( U ), W ), [-1, self.nNodes, 1] ) ), [-1, self.nNodes] ), mu)
            
            # V = num/den
            return( tf.math.divide( num, den ) )        
                                                                                
        def build_objective(self):
            # (H_ii)^2 * (v_i)^2
            #num = tf.math.multiply( tf.matrix_diag_part(self.Hsq), tf.math.square( self.pow_alloc ) )
            num = tf.math.multiply( tf.compat.v1.matrix_diag_part(self.Hsq), tf.math.square( self.pow_alloc ) )
            
            # sigma^2 + sum_j j ~= i ( (H_ji)^2 * (v_j)^2 ) 
            den = tf.reshape( tf.matmul( tf.transpose( self.Hsq, perm=[0,2,1] ), tf.reshape( tf.math.square( self.pow_alloc ), [-1, self.nNodes, 1] ) ), [-1, self.nNodes] ) + self.var - num 
            
            self.intf = den - self.var
            self.sinr_num = num
            self.sinr_den = den
            self.sinr = tf.math.divide(num, den)
            
            # rate
            rate = tf.math.log( 1. + self.sinr ) / tf.cast( tf.math.log( 2.0 ), tf.float64 )
            
            # Sum Rate = sum_i ( log(1 + SINR) )
            self.utility = tf.reduce_sum( rate, axis=1 )
            
            # Weighted Sum Rate
            #rate = tf.math.multiply( self.alpha, rate )
            #self.utility = tf.reduce_sum( rate, axis=1 )
            
            # Minimization objective
            self.obj = -tf.reduce_mean( self.utility )
            
            if self.exp == 'uwmmse':
                self.init_optimizer()

        def init_optimizer(self):
            # Gradients and SGD update operation for training the model
            self.trainable_params = tf.compat.v1.trainable_variables()
            
            #Learning Rate Decay
            #starter_learning_rate = self.learning_rate
            #self.learning_rate_decayed = tf.train.exponential_decay(starter_learning_rate, global_step=self.global_step, decay_steps=5000, decay_rate=0.99, staircase=True)
            
            # SGD with Momentum
            #self.opt = tf.train.GradientDescentOptimizer( learning_rate=learning_rate )
            #self.opt = tf.train.MomentumOptimizer(learning_rate=self.learning_rate_decayed, momentum=0.9, use_nesterov=True )

            # Optimizer
            if self.optimizer == 'adam':
                self.opt = tf.compat.v1.train.AdamOptimizer(learning_rate=self.learning_rate)
            elif self.optimizer == 'gd':
                self.opt = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=self.learning_rate)

            # Add regularization for stability
            var_reg = tf.reduce_sum(tf.norm(self.pow_alloc - tf.reduce_mean(self.pow_alloc, 0), axis=0)**2)

            # Compute gradients of loss w.r.t. all trainable variables
            gradients = tf.gradients(self.obj + self.reg_constant*var_reg, self.trainable_params)
            self.gradients = gradients
            # Clip gradients by a given maximum_gradient_norm
            # clip_gradients, _ = tf.clip_by_global_norm(gradients, self.max_gradient_norm)
            clip_gradients = gradients
            if self.max_gradient_norm is not None:
                clip_gradients = [tf.clip_by_norm(grad, self.max_gradient_norm) if len(grad.shape) > 0
                                  else grad for grad in clip_gradients]

            # Update the model
            self.updates = self.opt.apply_gradients(
                zip(clip_gradients, self.trainable_params), global_step=self.global_step)
                
        def save(self, sess, path, var_list=None, global_step=None):
            saver = tf.compat.v1.train.Saver(var_list)
            save_path = saver.save(sess, save_path=path, global_step=global_step)

        def restore(self, sess, path, var_list=None):
            saver = tf.compat.v1.train.Saver(var_list)
            saver.restore(sess, save_path=tf.train.latest_checkpoint(path))

        def train(self, sess, inputs ):
            input_feed = dict()
            input_feed[self.H.name] = inputs
            input_feed[self.apply_dropout.name] = True
            #input_feed[self.x.name] = features
            #input_feed[self.alpha.name] = alpha
            
            # Training Phase
            #input_feed[self.phase.name] = True
 
            output_feed = [self.obj, self.utility, self.pow_alloc, self.updates, self.sinr, self.sinr_num, self.sinr_den, self.intf]
                            
            outputs = sess.run(output_feed, input_feed)
            
            return outputs[0], outputs[1], outputs[2]


        def eval(self, sess, inputs ):
            input_feed = dict()
            input_feed[self.H.name] = inputs
            input_feed[self.apply_dropout.name] = True
            #input_feed[self.x.name] = features
            #input_feed[self.alpha.name] = alpha

            # Training Phase
            #input_feed[self.phase.name] = False

            output_feed = [self.obj,self.utility, self.pow_alloc, self.sinr, self.sinr_num, self.sinr_den, self.intf] 
                           
            outputs = sess.run(output_feed, input_feed)
            
            return outputs[0], outputs[1], outputs[2]
