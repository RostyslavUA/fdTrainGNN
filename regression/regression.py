import sys
import time
import numpy as np
import pandas as pd
import tensorflow as tf
from utils import *
from training_utils import minibatch_gnn, minibatch_gnn_test, convert_sparse_matrix_to_sparse_tensor, get_weights
from tensorflow.keras.optimizers import Adam, SGD
from data_generation import get_data, get_labels
from gcns import gcn_c_chebnet, gcn_d_chebnet
from optimizers import D_SGD, D_Adam, D_AMSGrad
np.random.seed(42)


def main(args):
    opt = 'adam'
    shiftop = 'nlap'
    gtype = 'ba'
    feattype = 'mix'
    w_multiplier = 1
    act_str = 'sigmoid'
    batch_size = 100
    preset = True
    num_layers = 2
    architecture = 'distributed'
    if len(args) >= 1:
        architecture = args[0]
    if len(args) >= 2:
        feattype = args[1]
    if len(args) >= 3:
        w_multiplier = float(args[2])
    if len(args) >= 4:
        act_str = args[3]
    if len(args) >= 5:
        batch_size = int(args[4])
    if len(args) >= 6:
        opt = args[5]
    if len(args) >= 7:
        num_layers = args[6]
    # Data generation
    num_nodes = 100
    input_dim = 5 if feattype != 'mix' else 10
    hidden_dim = 10
    output_dim = 1
    noise_std = 0.1
    train_samples = 1000
    test_samples = 20
    epochs = 1000
    lr = 0.001

    # Select a model
    gcn = gcn_c_chebnet if architecture == 'centralized' else gcn_d_chebnet
    print(f"{architecture} training")

    # step 1, create graph
    graph, adj, wts = random_graph(num_nodes, k=8, p=0.25, gtype=gtype, gseed=3)
    if shiftop == 'nadj':
        # Normalized adjacency
        L_norm = normalize_adj(adj)
        L_norm = L_norm.tocsr()
        L_norm_tuple = sparse_to_tuple(L_norm)
    else:
        # Normalized Laplacian
        L_norm = sp.eye(adj.shape[0]) - normalize_adj(adj)
        L_norm_tuple = normalize_laplacian(adj)
    # Consensus matrix
    c_mtx_numpy, c_mtx = consensus_matrix(adj)

    # step 2, create random samples
    x_train, x_test = get_data(train_samples, test_samples, num_nodes, input_dim, feattype=feattype)
    weights_gen = get_weights(input_dim, hidden_dim, output_dim, num_layers=num_layers, w_multiplier=w_multiplier, scale_nbr=2.0)
    y_train = np.zeros((train_samples, num_nodes, output_dim))
    y_test = np.zeros((test_samples, num_nodes, output_dim))
    # create labels
    y_train = get_labels(x_train, y_train, L_norm_tuple, weights_gen, act_str)
    y_test = get_labels(x_test, y_test, L_norm_tuple, weights_gen, act_str)

    noise_var = noise_std**2
    print("Noise std: {:.3f}, noise_variance {:.3f}".format(noise_std, noise_var))

    y_train += np.random.normal(0, noise_std, size=(train_samples, num_nodes, output_dim))
    y_test += np.random.normal(0, noise_std, size=(test_samples, num_nodes, output_dim))

    # step 3, create the weights of the model
    weights = get_weights(input_dim, hidden_dim, output_dim, num_layers=num_layers, w_multiplier=w_multiplier, scale_nbr=1.0)
    if architecture == 'distributed':
        weights = [np.repeat(w[tf.newaxis], num_nodes, axis=0) for w in weights]
    weights = [tf.Variable(w, trainable=True) for w in weights]

    # Select an optimizer
    if opt == 'sgd':
        optimizer = SGD(learning_rate=lr)
    elif opt == 'dsgd':
        optimizer = D_SGD(alpha=lr, beta=0.001)
        adj_sp = convert_sparse_matrix_to_sparse_tensor(adj)
    elif opt == 'adam':
        optimizer = Adam(learning_rate=lr)
    elif opt == 'dadam':
        optimizer = D_Adam(alpha=lr, beta=0.001, num_layers=2*num_layers)  # 2 parameters per layer
        adj_sp = convert_sparse_matrix_to_sparse_tensor(adj)
    elif opt == 'dams':
        optimizer = D_AMSGrad(learning_rate=lr)
    gcn.opt = opt

    start_time = time.time()

    loss_tr = np.zeros(epochs)
    loss_te = np.zeros(epochs)
    res = pd.DataFrame([], columns=['epoch', 'mse_train', 'mse_test'])
    num_batches = train_samples//batch_size
    num_batches_test = int(np.ceil(test_samples/batch_size))
    list_adj = [adj]*batch_size
    list_adj_test = [adj]*np.min((batch_size, test_samples))
    for epoch in range(epochs):
        if not preset and epoch > 0:
            x_train, x_test = get_data(train_samples, test_samples, num_nodes, input_dim, feattype=feattype)
            y_train = np.zeros((train_samples, num_nodes, output_dim))
            y_test = np.zeros((test_samples, num_nodes, output_dim))
            y_train = get_labels(x_train, y_train, L_norm_tuple, weights_gen, act_str)
            y_test = get_labels(x_test, y_test, L_norm_tuple, weights_gen, act_str)
        else:
            idx_permutation = np.random.permutation(range(train_samples))
            x_train = x_train[idx_permutation]
            y_train = y_train[idx_permutation]
        loss = np.zeros(num_batches)
        for idx in range(num_batches):
            batch_begin = idx*batch_size
            batch_end = (idx+1)*batch_size
            x_in = x_train[batch_begin:batch_end, :, :]
            y_in = y_train[batch_begin:batch_end]
            list_x_in = [x for x in x_in]
            list_y_in = [y for y in y_in]
            if architecture == 'centralized':
                weights_train = weights
            elif architecture == 'distributed':
                weights_train = [tf.Variable(tf.tile(w_d, [batch_size, 1, 1])) for w_d in weights]  # weights for minibatch
            loss_b, gradients_b = minibatch_gnn(list_adj, list_x_in, list_y_in, weights_train, gcn, shiftop, act_str)
            loss_b = np.mean(loss_b)
            loss[idx] = loss_b
            if architecture == 'distributed':
                gradients_b = list(zip(*gradients_b))
                gradients_b = [tf.reduce_sum(grad, 0) for grad in gradients_b]  # sum over minibatches
            if opt == 'dsgd' or opt == 'dadam':
                optimizer.apply_gradients(zip(gradients_b, weights), adj_sp)
            elif opt == 'dams':
                optimizer.apply_gradients(zip(gradients_b, weights), c_mtx)
            else:
                optimizer.apply_gradients(zip(gradients_b, weights))

            if architecture == 'distributed' and opt != 'dsgd' and opt != 'dadam' and opt != 'dams':
                for _ in range(1):
                    weights_cons = [c_mtx@np.reshape(w_d, (w_d.shape[0], -1)) for w_d in weights]
                    weights_cons = [np.reshape(w_d, weights[i].shape) for i, w_d in enumerate(weights_cons)]
                    weights = [w_d.assign(w_d_cons) for w_d, w_d_cons in zip(weights, weights_cons)]
        # Test
        loss_test = np.zeros(num_batches_test)
        if architecture == 'centralized':
            weights_test = weights
        elif architecture == 'distributed':
            weights_test = [tf.Variable(tf.tile(w_d, [np.min((batch_size, test_samples)), 1, 1])) for w_d in weights]
        for idx in np.random.permutation(range(num_batches_test)):
            batch_begin = idx*np.min((batch_size, test_samples))
            batch_end = (idx+1)*np.min((batch_size, test_samples))
            x_in = x_test[batch_begin:batch_end, :, :]
            y_in = y_test[batch_begin:batch_end]
            list_x_in = [x for x in x_in]
            list_y_in = [y for y in y_in]
            loss_b = minibatch_gnn_test(list_adj_test, list_x_in, list_y_in, weights_test, gcn, shiftop, act_str)
            loss_test[idx] = np.mean(loss_b)
        loss_tr[epoch] = np.mean(loss)
        loss_te[epoch] = np.mean(loss_test)
        print(
            "Epoch: {},".format(epoch),
            "Train MSE: {:.3f}".format(loss_tr[epoch]),
            "Test MSE: {:.3f}".format(loss_te[epoch])
        )
        epoch_res = pd.DataFrame({
            'epoch': epoch,
            'mse_train': loss_tr[epoch],
            'mse_test' : loss_te[epoch],
        }, index=[epoch])
        res = pd.concat([res, epoch_res], ignore_index=True)
        res.to_csv("./output/regression_chebnet_loss_trajectory_{}_{}_{}_{}_wm{}_{}_preset{}_l{}.csv".format(
            opt, shiftop, gtype, feattype, w_multiplier, act_str, int(preset), num_layers))
    print("Done, total runtime {:.3f}".format(time.time() - start_time))


if __name__ == '__main__':
    main(sys.argv[1:])

