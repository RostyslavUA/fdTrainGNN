from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import pickle
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import scipy.sparse as sp
from scipy.spatial import distance_matrix
from utils import consensus_matrix

np.random.seed(0)

# Eperiment
dataID = sys.argv[1]

# Number of nodes
nNodes = 25

layout = 'square'
# circle radius or half of a square's side of a simulated area
xy_lim = 500

# Thresholding
threshold = True

# Fading
fading = True

dataID = 'set'+str(dataID)+f'_UMa_Optional_{layout}_n{nNodes}_lim{xy_lim}_thr{int(threshold)}_fading{int(fading)}_mini'

# Path gain exponent
pl = 2.2

# Rayleigh (NLOS) or Rician(LOS) distribution scale
alpha = 1/np.sqrt(2)

# Batch size
batch_size = 64

# Training iterations
tr_iter = 100

# Testing iterations
te_iter = 100


def gen_location_circle(xy_lim, nNodes, min_dist=0, nAttempts=100):
    # Generate coordinates for transmitters
    tx_r = np.random.uniform(low=0, high=xy_lim, size=nNodes)
    angle = np.random.uniform(low=0, high=2*np.pi, size=nNodes)
    transmitters = np.zeros((nNodes, 2))
    transmitters[:, 0] = tx_r*np.cos(angle)
    transmitters[:, 1] = tx_r*np.sin(angle)
    for i in range(nAttempts):
        # Generate random radius for intended receivers
        r_vec = np.random.uniform(low=10, high=100, size=nNodes)
        a_vec = np.random.uniform(low=0, high=360, size=nNodes)
        # Calculate random delta coordinates for intended receivers
        xy_delta = np.zeros_like(transmitters)
        xy_delta[:, 0] = r_vec * np.sin(a_vec*np.pi/180)
        xy_delta[:, 1] = r_vec * np.cos(a_vec*np.pi/180)
        receivers = transmitters + xy_delta
        if np.all(distance_matrix(transmitters, receivers) >= min_dist):
            return transmitters, receivers
        if i == nAttempts-1:
            raise ValueError(f"Some receivers are closer to transmitters than required min distance {min_dist}m.")

            
def gen_location_square(xy_lim, nNodes):
    tx_x = np.random.uniform(low=-xy_lim, high=xy_lim, size=nNodes)
    tx_y = np.random.uniform(low=-xy_lim, high=xy_lim, size=nNodes)
    transmitters = np.zeros((nNodes, 2))
    transmitters[:, 0] = tx_x
    transmitters[:, 1] = tx_y
    rx_x = np.random.uniform(low=-xy_lim, high=xy_lim, size=nNodes)
    rx_y = np.random.uniform(low=-xy_lim, high=xy_lim, size=nNodes)
    receivers = np.zeros((nNodes, 2))
    receivers[:, 0] = rx_x
    receivers[:, 1] = rx_y
    return transmitters, receivers
            
    
def build_adhoc_network(coord, pars, batch_size):
    fc = pars['fc']
    std = pars['std']
    transmitters, receivers = coord
    # Calculate the distance matrix between all pairs of transmitters and receivers
    d_mtx = distance_matrix(transmitters, receivers)
    pl_no_shadowing = gen_pl_uma_optional(fc, d_mtx)
    nNodes = d_mtx.shape[0]
    pl_mtx_db = np.zeros((batch_size, nNodes, nNodes))
    for b in range(batch_size):
        x_g = np.random.normal(0, std, size=d_mtx.shape)
        x_g = np.clip(x_g, 0-2*std, 0+2*std)
        pl_mtx_db[b, :, :] = pl_no_shadowing + x_g  # 64 instances of path loss with lognormal shadowing for each location realization
    h_mtx_lin = 10.0 ** (-pl_mtx_db/20.0)  # Divide by 20 to get magnitude coefficients
    return( dict(zip(['tx', 'rx'],[transmitters, receivers] )), h_mtx_lin, pl_mtx_db, d_mtx )


def rician_distribution(batch_size, alpha):
    # Zero-mean Rician distribution to simulate fading for LOS channel
    x = np.random.normal(0, alpha, (batch_size, nNodes, nNodes))
    y = np.random.normal(0, alpha, (batch_size, nNodes, nNodes))
    samples = np.sqrt(x**2 + y**2)
    return samples


# Simuate Fading
def sample_graph(batch_size, A, alpha=1):
    samples = np.random.rayleigh(alpha, A.shape)
    #samples = (samples + np.transpose(samples,(0,2,1)))/2
    PP = np.multiply(samples[None,:,:], A)
    return PP[0]


def gen_threshold():
    tx_sig = 10*np.log10(5/1e-3)  # dBm
    k = 1.380649e-23  # Boltzmann constant
    T = 290  # Kelvin
    bw = 5e6
    noise_power_lin = k*T*bw
    noise_power = -106.87  # dBm
    req_snr = 10  # dB
    ctt = 5  # control channel tolerance, dB
    thr_pl = tx_sig - noise_power - req_snr + ctt  # path loss threshold
    thr_gain_lin = 10**(-thr_pl/20)  # Divide by 20 to threshold magnitude coefficients
    return thr_gain_lin


# Training Data
def generate_data(batch_size, layout, xy_lim, alpha, nNodes, threshold=False, fading=False):
    pars = {
        "fc": 0.9,  # GHz
        "std": 7.2
    }
    tr_H, te_H = [], []
    thr_gain_lin = gen_threshold()
    for indx in range(tr_iter):
        # sample training data
        if layout == 'circle':
            transmitters, receivers = gen_location_circle(xy_lim=xy_lim, nNodes=nNodes)
        elif layout == 'square':
            transmitters, receivers = gen_location_square(xy_lim=xy_lim, nNodes=nNodes)
        # Generate dict of coordinates, gain (linear), path loss (dB) and distance matrix
        coord, h_mtx_lin, pl_mtx, d_mtx = build_adhoc_network((transmitters, receivers), pars, batch_size)
        if fading:
            # Apply Rayleigh fading with parameter alpha
            H = sample_graph(batch_size, h_mtx_lin, alpha)
        else:
            H = h_mtx_lin
        if threshold:
            # Threshold receivers
            H[H < thr_gain_lin] = 0.0
        tr_H.append( H )

    for indx in range(te_iter):
        # sample test data
        if layout == 'circle':
            transmitters, receivers = gen_location_circle(xy_lim=xy_lim, nNodes=nNodes)
        elif layout == 'square':
            transmitters, receivers = gen_location_square(xy_lim=xy_lim, nNodes=nNodes)
        # Generate dict of coordinates, gain (linear), path loss (dB) and distance matrix
        coord, h_mtx_lin, pl_mtx, d_mtx = build_adhoc_network((transmitters, receivers), pars, batch_size)
        if fading:
            # Apply Rayleigh fading with parameter alpha
            H = sample_graph(batch_size, h_mtx_lin, alpha)
        else:
            H = h_mtx_lin
        if threshold:
            # Threshold receivers
            H[H < thr_gain_lin] = 0.0
        te_H.append( H )

    return( dict(zip(['train_H', 'test_H'],[tr_H, te_H] ) ) )


def gen_pl_umi_nlos(dist, fc=5.8, c=3e8, hbs=1.7, hut=1.7, gamma=3.0):
    he = 1.0
    dbp = 4*(hbs-he)*(hut-he)*fc*1e9/c
    pl1 = 32.4 + 21*np.log10(dist) + 20*np.log10(fc)
    pl2 = 32.4 + 40*np.log10(dist) + 20*np.log10(fc) - 9.5*np.log10(dbp**2 + (hbs - hut)**2)
    pl_umi_los = np.zeros_like(dist)
    pl_umi_los[dist < dbp] = pl1[dist < dbp]
    pl_umi_los[dist >= dbp] = pl2[dist >= dbp]
    h_umi_los_lin = lognormal_pathloss(dist, pl0=pl_umi_los, d0=10, gamma=gamma, std=4.0)
    pl_umi_los = -10*np.log10(h_umi_los_lin)

    pl_umi_nlos_prime = 35.3*np.log10(dist) + 22.4 + 21.3*np.log10(fc) - 0.3*(hut-1.5)
    pl_umi_nlos = np.maximum(pl_umi_los, pl_umi_nlos_prime)
    return pl_umi_nlos


def gen_pl_uma_optional(fc, d_mtx):
    pl = 32.4 + 20*np.log10(fc) + 30*np.log10(d_mtx)
    return pl


def gen_pl_uma_nlos(fc, d_mtx, c=3e8, hbs=1.7, hut=1.7, d0=10, gamma=3.0):
    he = 1.0
    # Compute breakpoint distance
    dbp = 4*(hbs-he)*(hut-he)*fc*1e9/c
    # Distance-dependent LOS path losses
    pl1 = 28.0 + 22*np.log10(d_mtx) + 20*np.log10(fc)
    pl2 = 28.0 + 40*np.log10(d_mtx) + 20*np.log10(fc) - 9*np.log10(dbp**2 - (hbs-hut)**2)
    pl_uma_los = np.zeros_like(d_mtx)
    pl_uma_los[d_mtx < dbp] = pl1[d_mtx < dbp]
    pl_uma_los[d_mtx >= dbp] = pl2[d_mtx >= dbp]
    # Log normal shadowing
    h_mtx_lin = lognormal_pathloss(d_mtx, pl0=pl_uma_los, d0=d0, gamma=gamma, std=4.0)
    pl_uma_los = -10*np.log10(h_mtx_lin)
    # NLOS path loss
    pl_uma_nlos_prime = 13.54 + 39.08*np.log10(d_mtx) + 20*np.log10(fc) - 0.6*(hut - 1.5)
    pl_uma_nlos = np.maximum(pl_uma_los, pl_uma_nlos_prime)
    return pl_uma_nlos


def gen_cmtx(data_H):
    data_C = dict()
    C = []
    for i, H_m in enumerate(data_H['train_H']):
        adj = np.ones_like(H_m)
        adj[H_m == 0] = 0.0
        adj = adj * np.transpose(adj, (0, 2, 1))
        cmtcs = []
        for b, adj_b in enumerate(adj):
            cmtx = consensus_matrix(adj_b)[1]
            cmtcs.append(cmtx)
        cmtcs = sp.vstack(cmtcs)
        C.append(cmtcs)
    data_C['train_cmat'] = C
    return data_C


def lognormal_pathloss(d_mtx, pl0=40, d0=10, gamma=3.0, std=7.0):
    '''
    PL = PL_0 + 10 \gamma \log_{10}\frac{d}{d_0} + X_g
    https://en.wikipedia.org/wiki/Log-distance_path_loss_model
    Args:
        d_mtx: distance matrix

    Returns:
        pl_mtx: path loss matrix
    '''
    # pl_mtx = np.ones_like(d_mtx)
    x_g = np.random.normal(0, std, size=d_mtx.shape)
    x_g = np.clip(x_g, 0-2*std, 0+2*std)
    pl_db_mtx = pl0 + 10.0 * gamma * np.log10(d_mtx/d0) + x_g
    h_mtx = 10.0 ** (-pl_db_mtx/10.0)
    return h_mtx


def hist_pl(pl_umi_nlos_lin):
    # Check if the rx signal - thermal noise > required snr
    pl_umi_nlos = 10*np.log10(pl_umi_nlos_lin)
    k = 1.380649e-23  # Boltzmann constant
    T = 290  # Kelvin
    bw = 5e6
    noise_power_lin = k*T*bw
    noise_power = 10*np.log10(noise_power_lin) + 30 # dBm
    snr_req = 10  # dB
    tx_sig = 10*np.log10(5/1e-3)  # dBm
    rx_sig = tx_sig - pl_umi_nlos - noise_power  # dB
    poor_channels = np.where(rx_sig < snr_req)  # path loss is very high due to large distance. Many poor channels
    plt.hist(rx_sig.flatten(), bins=100)
    plt.xlabel('RSS, dB')
    plt.ylabel('hits')
    plt.grid()
    plt.show()


def scatter_channel_gain(d_mtcs, tr_H, n_samp):
    samp_idx = np.random.randint(0, tr_H.shape[1], size=n_samp)
    d_mtcs_flat = np.array(d_mtcs).flatten()
    tr_H_samp = np.array(tr_H)[range(tr_iter), samp_idx, :]
    tr_H_flat = tr_H_samp.flatten()
    plt.scatter(d_mtcs_flat, 10*np.log10(tr_H_flat))
    ax = plt.gca()
    ax.set_xscale('log')
    plt.xlabel('distance')
    plt.ylabel(r'$H_{dB}$')
    plt.grid()
    plt.show()



def main():
    # Create data path
    if not os.path.exists('data/'+dataID):
        os.makedirs('data/'+dataID)

    # Training data
    data_H = generate_data(batch_size, layout, xy_lim, alpha, nNodes, threshold, fading)
    f = open('data/'+dataID+'/H.pkl', 'wb')
    pickle.dump(data_H, f)
    f.close()
    if threshold:
        data_C = gen_cmtx(data_H)
        f = open('data/'+dataID+'/cmat.pkl', 'wb')
        pickle.dump(data_C, f)
        f.close()


if __name__ == '__main__':
    main()
