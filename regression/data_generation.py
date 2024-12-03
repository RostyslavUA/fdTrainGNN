import numpy as np
import tensorflow as tf
from gcns import gcn_c_chebnet


def get_data(train_samples, test_samples, num_nodes, input_dim, feattype='mix'):
    if feattype == 'discrete':
        x_train = np.random.randint(0, 6, size=(train_samples, num_nodes, input_dim)).astype(np.float64)
        x_test = np.random.randint(0, 6, size=(test_samples, num_nodes, input_dim)).astype(np.float64)
    elif feattype == 'binary':
        x_train_binary_1 = np.random.binomial(1, 0.1, size=(train_samples, num_nodes, 1))
        x_train_binary_2 = np.random.binomial(1, 0.3, size=(train_samples, num_nodes, 1))
        x_train_binary_3 = np.random.binomial(1, 0.5, size=(train_samples, num_nodes, 1))
        x_train_binary_4 = np.random.binomial(1, 0.7, size=(train_samples, num_nodes, 1))
        x_train_binary_5 = np.random.binomial(1, 0.9, size=(train_samples, num_nodes, 1))
        x_train = np.concatenate((x_train_binary_1, x_train_binary_2, x_train_binary_3, x_train_binary_4, x_train_binary_5), axis=-1, dtype=np.float64)
        x_test_binary_1 = np.random.binomial(1, 0.1, size=(test_samples, num_nodes, 1))
        x_test_binary_2 = np.random.binomial(1, 0.3, size=(test_samples, num_nodes, 1))
        x_test_binary_3 = np.random.binomial(1, 0.5, size=(test_samples, num_nodes, 1))
        x_test_binary_4 = np.random.binomial(1, 0.7, size=(test_samples, num_nodes, 1))
        x_test_binary_5 = np.random.binomial(1, 0.9, size=(test_samples, num_nodes, 1))
        x_test = np.concatenate((x_test_binary_1, x_test_binary_2, x_test_binary_3, x_test_binary_4, x_test_binary_5), axis=-1, dtype=np.float64)
    elif feattype == 'sym_norm':
        x_train = np.random.normal(0, 1, size=(train_samples, num_nodes, input_dim))
        x_test = np.random.normal(0, 1, size=(test_samples, num_nodes, input_dim))
    elif feattype == 'asym_norm':
        x_train = np.random.normal(1, 1, size=(train_samples, num_nodes, input_dim))
        x_test = np.random.normal(1, 1, size=(test_samples, num_nodes, input_dim))
    elif feattype == 'sym_unif':
        x_train = np.random.uniform(-1, 1, (train_samples, num_nodes, input_dim))
        x_test = np.random.uniform(-1, 1, (test_samples, num_nodes, input_dim))
    elif feattype == 'asym_unif':
        x_train = np.random.uniform(0, 2, (train_samples, num_nodes, input_dim))
        x_test = np.random.uniform(0, 2, (test_samples, num_nodes, input_dim))
    elif feattype == 'one_hot':
        x_int = np.random.randint(0, input_dim, size=(train_samples, num_nodes))
        x_train = tf.one_hot(x_int, depth=input_dim, axis=-1, dtype=tf.float64)
        x_int = np.random.randint(0, input_dim, size=(test_samples, num_nodes))
        x_test = tf.one_hot(x_int, depth=input_dim, axis=-1, dtype=tf.float64)
    elif feattype == 'mix':
        x_train_asym_unif = np.random.uniform(0, 1, (train_samples, num_nodes, 1))
        x_train_sym_norm = np.random.normal(0, 1, (train_samples, num_nodes, 1))
        x_train_discrete = np.random.randint(0, 6, size=(train_samples, num_nodes, 1))
        x_train_binary_1 = np.random.binomial(1, 0.25, size=(train_samples, num_nodes, 1))
        x_train_binary_2 = np.random.binomial(1, 0.75, size=(train_samples, num_nodes, 1))
        x_int = np.random.randint(0, input_dim//2, size=(train_samples, num_nodes))
        x_train_onehot = tf.one_hot(x_int, depth=input_dim//2, axis=-1)
        x_train = np.concatenate((x_train_asym_unif, x_train_sym_norm, x_train_discrete,
                                  x_train_binary_1, x_train_binary_2, x_train_onehot), axis=-1)
        x_test_asym_unif = np.random.uniform(0, 1, (test_samples, num_nodes, 1))
        x_test_sym_norm = np.random.normal(0, 1, (test_samples, num_nodes, 1))
        x_test_discrete = np.random.randint(0, 6, size=(test_samples, num_nodes, 1))
        x_test_binary_1 = np.random.binomial(1, 0.25, size=(test_samples, num_nodes, 1))
        x_test_binary_2 = np.random.binomial(1, 0.75, size=(test_samples, num_nodes, 1))
        x_int = np.random.randint(0, input_dim, size=(test_samples, num_nodes))
        x_test_onehot = tf.one_hot(x_int, depth=input_dim//2, axis=-1)
        x_test = np.concatenate((x_test_asym_unif, x_test_sym_norm, x_test_discrete,
                                  x_test_binary_1, x_test_binary_2, x_test_onehot), axis=-1)
    return x_train, x_test


def get_labels(x_train, y_train, L_norm_tuple, weights_gen, act_str):
    train_samples = x_train.shape[0]
    for idx in range(train_samples):
        x_in = tf.convert_to_tensor(x_train[idx, :, :], dtype=tf.float64)
        y_out = gcn_c_chebnet(L_norm_tuple, x_in, weights_gen, act_str)
        y_train[idx, :, :] = y_out.numpy()
    return y_train
