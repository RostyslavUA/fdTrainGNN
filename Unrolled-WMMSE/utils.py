import copy
import numpy as np
import scipy.sparse as sp
import networkx as nx
import matplotlib.pyplot as plt


def consensus_matrix(adj_0):
    # Metropolis-Hastings weights
    # https://web.stanford.edu/~boyd/papers/pdf/lmsc_mtns06.pdf
    adj = copy.deepcopy(adj_0)
    size = adj.shape[0]
    if np.max(adj.diagonal(0)) != 0:
        np.fill_diagonal(adj, 0)
    adj = sp.coo_matrix(adj)
    d_vec = np.array(adj.sum(1)) + 1
    d_mtx = sp.diags(d_vec.flatten())
    d_i_mtx = d_mtx.dot(adj)
    d_j_mtx = d_i_mtx.transpose()
    d_max_mtx = d_i_mtx.maximum(d_j_mtx).toarray()
    d_max_inv = np.divide(1, d_max_mtx, out=np.zeros_like(d_max_mtx), where=d_max_mtx != 0)
    on_diag = np.ones(size) - d_max_inv.sum(1)
    np.fill_diagonal(d_max_inv, on_diag)
    c_mtx = sp.csr_matrix(d_max_inv)
    return d_max_inv, c_mtx


def threshold_csi(samp, thr=0.01):
    """
    Threshold CSI, such that resulting adjacency is connected. Select the threshold through bisection:
    threshold(t+1) = threshold(t) + c threshold(t)/2, where c \in (-1, 1).
    :param samp: a CSI sample of the shape [M, M]
    :param thr: convergence threshold
    :return: thresholded, connected adjacency
    """
    h_frac = samp.max()
    h_frac_prev = h_frac
    attempt = 0
    thr_init = thr
    while True:
        adj_0 = np.ones_like(samp)
        adj_0[samp <= h_frac] = 0.0
        c = 1 if nx.is_connected(nx.Graph(adj_0)) else -1
        h_frac = h_frac + c * h_frac/2
        if abs(h_frac - h_frac_prev) < thr:
            adj = adj_0
            break
        else:
            h_frac_prev = h_frac
            attempt += 1
            if attempt == 100:  # Cannot converge. Increasing threshold
                thr += thr_init
                attempt = 0
    if any(adj.flatten() != adj.T.flatten()):
        adj = adj + adj.T
        adj = np.divide(adj, adj, out=np.zeros_like(adj), where=adj != 0)
    np.fill_diagonal(adj, 1.0)
    return adj


def hist_csi(train_H, batch_idx, hist_num=5):
    for i in range(hist_num):
        csi = np.array(np.array(train_H[batch_idx][i]))
        plt.figure()
        plt.hist(csi.flatten())
        plt.grid()
        plt.title(f"batch_idx: {batch_idx}, sample_idx: {i}")
        plt.show(block=False)
