import tensorflow as tf
import copy
import matplotlib.pyplot as plt

tf.random.set_seed(42)


def plot_hist(grad, bins=100):  # For debugging
    grad_flat = tf.reshape(grad, [-1])
    plt.hist(grad_flat, bins=bins)


class D_SGD:
    def __init__(self, alpha=0.1, beta=0.1):
        self.alpha = alpha
        self.beta = beta

    def dim_check(self, grad, w):
        if len(w.shape) == 1:
            w = w[:, tf.newaxis]
            grad = grad[:, tf.newaxis]
        elif len(w.shape) == 3 and 'bias' not in w.name:
            w = w[:, :, tf.newaxis]
            grad = grad[:, :, tf.newaxis]
        return grad, w

    def get_consensus_term(self, w, adj_c):
        deg = tf.sparse.reduce_sum(adj_c, 0)
        deg_diag = deg*tf.sparse.eye(adj_c.shape[0], dtype=tf.float64)
        adj_neg_deg = tf.sparse.add(adj_c, deg_diag*(-1))
        w_flat = tf.reshape(w, (w.shape[0], -1))
        cons_term = tf.sparse.sparse_dense_matmul(adj_neg_deg, w_flat)
        cons_term = tf.reshape(cons_term, w.shape)
        return cons_term

    def type_check(self, adj):
        if 'csc_matrix' in str(type(adj)):
            adj_c = copy.copy(adj)
            adj_c = tf.cast(adj_c.todense(), tf.float64)
        else:
            adj_c = adj
        return adj_c

    def apply_gradients(self, grads_and_vars, adj):
        """
        Assuming that node is the first dim
        """
        adj_c = self.type_check(adj)
        for grad, w in grads_and_vars:
            grad, w = self.dim_check(grad, w)
            grad_std = tf.math.reduce_std(grad)
            local_sg = (grad+tf.random.normal(grad.shape, mean=0, stddev=0.0*grad_std, dtype=tf.float64))
            consensus_term = self.get_consensus_term(w, adj_c)
            w.assign(w - self.alpha*local_sg + self.beta*consensus_term)


class D_Adam(D_SGD):
    def __init__(self, num_layers, alpha=0.001, beta=0.001, beta_1=0.9, beta_2=0.999):
        self.alpha = alpha
        self.beta = beta
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.t = 0
        self.m_t = [0]*num_layers
        self.v_t = [0]*num_layers
        self.eps = 1e-8

    def get_adam_update(self, g_t, m_t, v_t,):
        m_t = self.beta_1*m_t + (1-self.beta_1)*g_t
        v_t = self.beta_2*v_t + (1-self.beta_2)*g_t**2
        mhat_t = m_t/(1 - self.beta_1**self.t)
        vhat_t = v_t/(1 - self.beta_2**self.t)
        return mhat_t, vhat_t, m_t, v_t

    def apply_gradients(self, grads_and_vars, adj):
        adj_c = self.type_check(adj)
        self.t+=1
        for i, (grad, w) in enumerate(grads_and_vars):
            grad, w = self.dim_check(grad, w)
            mhat_t, vhat_t, self.m_t[i], self.v_t[i] = self.get_adam_update(grad, self.m_t[i], self.v_t[i])
            if mhat_t.ndim == 5:  # minibatch
                mhat_t = tf.reduce_sum(mhat_t, 0)
                vhat_t = tf.reduce_sum(vhat_t, 0)
            consensus_term = self.get_consensus_term(w, adj_c)
            w.assign(w - self.alpha*mhat_t/(tf.sqrt(vhat_t)+self.eps) + self.beta*consensus_term)


class D_AMSGrad:
    def __init__(self, learning_rate=0.001, beta_1=0.9, beta_2=0.999, eps=1e-8):
        self.alpha = learning_rate
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.eps = eps
        self.mvec, self.vvec = [], []
        self.vvec, self.uvec_tilde = [], []

    def init_params(self, grads):
        self.mvec = [tf.zeros((grad.shape[0], tf.reduce_prod(grad.shape[1:])), dtype=tf.float64) for grad in grads]
        self.vvec = [tf.zeros((grad.shape[0], tf.reduce_prod(grad.shape[1:])), dtype=tf.float64) for grad in grads]
        self.vvec_hat = [self.eps*tf.ones((grad.shape[0], tf.reduce_prod(grad.shape[1:])), dtype=tf.float64) for grad in grads]
        self.uvec_tilde = [self.eps*tf.ones((grad.shape[0], tf.reduce_prod(grad.shape[1:])), dtype=tf.float64) for grad in grads]

    def apply_gradients(self, grads_and_vars, cmat):
        if len(self.mvec) == 0:
            grads_and_vars_0 = copy.deepcopy(grads_and_vars)
            grads = list(zip(*grads_and_vars_0))[0]
            self.init_params(grads)
        for i, (grad, wmtx) in enumerate(grads_and_vars):
            grad_resh = tf.reshape(grad, (grad.shape[0], -1))
            wmtx_resh = tf.reshape(wmtx, (wmtx.shape[0], -1))
            self.mvec[i] = self.beta_1*self.mvec[i] + (1-self.beta_1)*grad_resh
            self.vvec[i] = self.beta_2*self.vvec[i] + (1-self.beta_2)*grad_resh**2
            vvec_hat_next = tf.maximum(self.vvec_hat[i], self.vvec[i])
            wmtx_resh = cmat.dot(wmtx_resh)
            self.uvec_tilde[i] = cmat.dot(self.uvec_tilde[i])
            uvec = tf.maximum(self.uvec_tilde[i], self.eps)
            wmtx_resh = wmtx_resh - self.alpha*self.mvec[i]/tf.math.sqrt(uvec)
            self.uvec_tilde[i] = self.uvec_tilde[i] - self.vvec_hat[i] + vvec_hat_next
            wmtx.assign(tf.reshape(wmtx_resh, wmtx.shape))
            self.vvec_hat[i] = vvec_hat_next
