import numpy as np
import scipy.sparse as sp
import networkx as nx


def get_weights_deviation(weights):
    w_diff_l = []
    for w_l in weights:
        w_l_mean = np.mean(w_l, 0)[np.newaxis]
        diff = w_l - w_l_mean
        diff = np.reshape(diff, [-1])
        norm = np.linalg.norm(diff, ord=2)
        norm = np.sum(norm)
        norm /= w_l.shape[0]
        diff_norm_l = np.sqrt(norm)
        w_l_mean_flat = np.reshape(w_l_mean, [-1])
        norm_mean = np.linalg.norm(w_l_mean_flat, ord=2)
        if norm_mean == 0:
            w_diff = 0.0
        else:
            w_diff = diff_norm_l/norm_mean
        w_diff_l.append(w_diff)
    return w_diff_l


def random_graph(size, k=20, p=0.25, gtype='grp', gseed=None):
    if gtype == 'grp':
        graph = nx.gaussian_random_partition_graph(size, k, min(7, k), p, max(0.1, p/3.0), seed=gseed)
    elif gtype == 'ws':
        graph = nx.connected_watts_strogatz_graph(size, k, p, tries=1000, seed=gseed)
    elif gtype == 'er':
        graph = nx.generators.random_graphs.fast_gnp_random_graph(size, float(k) / float(size), seed=gseed)
    elif gtype == 'ba':
        graph = nx.generators.random_graphs.barabasi_albert_graph(size, int(np.round(k * p)), seed=gseed)
    else:
        raise ValueError('Unsupported graph type')
    wts = 10.0*np.random.uniform(0.01, 1.01, (size,))
    for u in graph:
        graph.nodes[u]['weight'] = wts[u]
    adj = nx.adjacency_matrix(graph, nodelist=list(range(size)), weight=None)
    return graph, adj, wts


def sparse_to_tuple(sparse_mx):
    """Convert sparse matrix to tuple representation."""
    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()
    # return d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt).tocoo()


def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    # adj_normalized = normalize_adj(adj)
    adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))
    # adj_normalized = sp.coo_matrix(adj)
    return sparse_to_tuple(adj_normalized)


def normalize_laplacian(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    # adj_normalized = normalize_adj(adj)
    adj_normalized = sp.eye(adj.shape[0]) - normalize_adj(adj)
    # adj_normalized = sp.coo_matrix(adj)
    return sparse_to_tuple(adj_normalized)


def consensus_matrix(adj_0):
    # Metropolis-Hastings weights
    # https://web.stanford.edu/~boyd/papers/pdf/lmsc_mtns06.pdf
    adj = sp.coo_matrix(adj_0)
    size = adj.shape[0]
    add_factor = 1 if np.max(adj.diagonal(0)) == 0 else 0  # check if self-loops
    d_vec = np.array(adj.sum(1))
    d_vec += add_factor
    d_mtx = sp.diags(d_vec.flatten())
    d_i_mtx = d_mtx.dot(adj)
    d_j_mtx = d_i_mtx.transpose()
    d_max_mtx = d_i_mtx.maximum(d_j_mtx).toarray()
    d_max_inv = 1/(d_max_mtx)
    d_max_inv[d_max_inv == np.inf] = 0
    on_diag = np.ones(size) - d_max_inv.sum(1)
    np.fill_diagonal(d_max_inv, on_diag)
    c_mtx = sp.csr_matrix(d_max_inv)
    return d_max_inv, c_mtx


if __name__ == '__main__':
    # debug code for helper functions, check variables in pycharm debug
    num_nodes = 100
    graph, adj, wts = random_graph(num_nodes, k=8, p=0.25, gtype='ba', gseed=None)
    adj_norm = normalize_adj(adj)
    L_norm = normalize_adj(adj + sp.eye(adj.shape[0]))
    c_mtx_numpy, c_mtx = consensus_matrix(adj)
    a = np.ones(num_nodes)
    b = c_mtx.dot(a)
    print(b)
