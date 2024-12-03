import tensorflow as tf
from utils import *


def convert_sparse_matrix_to_sparse_tensor(X):
    coo = X.tocoo().astype(np.float64)
    indices = np.mat([coo.row, coo.col]).transpose()
    return tf.SparseTensor(indices, coo.data, coo.shape)


def minibatch_gnn(list_adj, list_x_in, list_y_train, weights, gcn_func, shiftop, act_str):
    adj = sp.block_diag(list_adj, format='csr')
    if shiftop == 'nadj':
        # Normalized adjacency
        L_norm = normalize_adj(adj + sp.eye(adj.shape[0]))
        L_norm_tuple = preprocess_adj(adj)
    else:
        # Normalized Laplacian
        L_norm = sp.eye(adj.shape[0]) - normalize_adj(adj)
        L_norm_tuple = normalize_laplacian(adj)
    x_in = tf.concat(list_x_in, 0)
    y_train = tf.concat(list_y_train, 0)
    if 'gcn_d' in gcn_func.__name__:
        loss, gradients = get_gradients(L_norm_tuple, x_in, y_train, weights, gcn_func, act_str)
        gradients_subgraph = []
        a_sizesum = 0
        for a in list_adj:
            a_size = np.shape(a)[0]
            grad_subgraph = []
            for grad_layer in gradients:
                grad_layer_subgraph = grad_layer[a_sizesum:a_sizesum+a_size]
                grad_subgraph.append(grad_layer_subgraph)
            gradients_subgraph.append(grad_subgraph)
            a_sizesum += a_size
        gradients = gradients_subgraph
        loss_subgraph = []
        a_sizesum = 0
        for a in list_adj:
            a_size = np.shape(a)[0]
            loss_g = loss[a_sizesum:a_sizesum+a_size]
            loss_subgraph.append(loss_g)
            a_sizesum += a_size
        loss = loss_subgraph
    elif 'gcn_c' in gcn_func.__name__:
        loss, gradients = get_gradients_c(L_norm_tuple, x_in, y_train, weights, gcn_func, act_str, scale=len(list_adj))
    return loss, gradients


def minibatch_gnn_test(list_adj, list_x_in, list_y_train, weights, gcn_func, shiftop, act_str):
    adj = sp.block_diag(list_adj, format='csr')
    if shiftop == 'nadj':
        # Normalized adjacency
        L_norm = normalize_adj(adj + sp.eye(adj.shape[0]))
        L_norm_tuple = preprocess_adj(adj)
    else:
        # Normalized Laplacian
        L_norm = sp.eye(adj.shape[0]) - normalize_adj(adj)
        L_norm_tuple = normalize_laplacian(adj)
    x_in = tf.concat(list_x_in, 0)
    y_train = tf.concat(list_y_train, 0)
    y_pred = gcn_func(L_norm_tuple, x_in, weights, act_str)
    loss = tf.reduce_mean((y_pred - y_train)**2, (-1))
    loss_per_graph = []
    a_sizesum = 0
    for a in list_adj:
        a_size = np.shape(a)[0]
        loss_g = loss[a_sizesum:a_sizesum+a_size]
        loss_per_graph.append(loss_g)
        a_sizesum += a_size
    loss = loss_per_graph
    return loss


@tf.function
def get_gradients_c(L_norm_tuple, x_in, y_train, weights, gcn_func, act_str, scale=1.0):
    with tf.GradientTape() as g:
        g.watch(weights)
        y_pred = gcn_func(L_norm_tuple, x_in, weights, act_str)
        loss = tf.reduce_mean((y_pred - y_train)**2)
        loss_scaled = loss*scale
    gradients = g.gradient(loss_scaled, weights)
    return loss, gradients


@tf.function
def get_gradients(L_norm_tuple, x_in, y_train, weights, gcn_d, act_str):
    with tf.GradientTape() as g:
        g.watch(weights)
        y_pred = gcn_d(L_norm_tuple, x_in, weights=weights, act_str=act_str)
        loss = tf.reduce_mean((y_pred - y_train)**2, -1)
    gradients = g.gradient(loss, weights)
    return loss, gradients


def get_weights(input_dim, hidden_dim, output_dim, num_layers=2, w_multiplier=1.0, scale_nbr=2.0):
    weights = []
    for i_layer in range(num_layers):
        if i_layer == 0:
            input_dim_layer = input_dim
            output_dim_layer = hidden_dim
        elif i_layer == num_layers - 1:
            input_dim_layer = hidden_dim
            output_dim_layer = output_dim
        else:
            input_dim_layer = hidden_dim
            output_dim_layer = hidden_dim
        wi1_true = w_multiplier * np.random.uniform(0, 1, (input_dim_layer, output_dim_layer))  # Layer i, coefficient 1
        wi2_true = w_multiplier * np.random.uniform(0, 1, (input_dim_layer, output_dim_layer)) * scale_nbr  # Layer i, coefficient 2
        wi1_gen = tf.Variable(wi1_true)
        wi2_gen = tf.Variable(wi2_true)
        weights.extend((wi1_gen, wi2_gen))
    return weights
