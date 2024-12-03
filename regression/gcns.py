import tensorflow as tf


def gcn_c_chebnet(adj_in, x_in, weights, act_str):
    if act_str == 'lrelu':
        act = tf.nn.leaky_relu
    elif act_str == 'sigmoid':
        act = tf.nn.sigmoid
    elif act_str == 'tanh':
        act = tf.nn.tanh
    coords, values, shape = adj_in
    coords = tf.cast(coords, tf.int64)
    adj_ts = tf.sparse.SparseTensor(indices=coords, values=values, dense_shape=shape)
    Ti1 = x_in
    Ti2 = tf.sparse.sparse_dense_matmul(adj_ts, x_in)
    for i in range(len(weights)//2):
        wi1, wi2 = weights[i*2:(i+1)*2]
        h1 = Ti1 @ wi1 + Ti2 @ wi2
        x1 = act(h1)
        if i < len(weights)/2-1:
            Ti1 = x1
            Ti2 = tf.sparse.sparse_dense_matmul(adj_ts, x1)
    return x1


def gcn_d_chebnet(adj_in, x_in, weights, act_str):
    if act_str == 'lrelu':
        act = tf.nn.leaky_relu
    elif act_str == 'sigmoid':
        act = tf.nn.sigmoid
    elif act_str == 'tanh':
        act = tf.nn.tanh
    coords, values, shape = adj_in
    coords = tf.cast(coords, tf.int64)
    adj_ts = tf.sparse.SparseTensor(indices=coords, values=values, dense_shape=shape)
    Ti1 = x_in
    Ti2 = tf.sparse.sparse_dense_matmul(adj_ts, x_in)
    for i in range(len(weights)//2):
        wi1, wi2 = weights[i*2:(i+1)*2]
        h1 = Ti1[:, tf.newaxis] @ wi1 + Ti2[:, tf.newaxis] @ wi2
        h1 = tf.squeeze(h1, 1)
        x1 = act(h1)
        if i < len(weights)/2-1:
            Ti1 = x1
            Ti2 = tf.sparse.sparse_dense_matmul(adj_ts, x1)
    return x1
