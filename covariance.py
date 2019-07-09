from time import time
import numpy as np
import scipy
import seaborn as sns
from nilearn import plotting
from matplotlib import pyplot as plt
import multiprocessing
from joblib import Parallel, delayed

from globals import mem, coordinates, corpus_tfidf, Ni, Nj, Nk, affine, inv_affine
from builds import encode_pmid, encode_feature, decode_pmid, decode_feature
from tools import print_percent
from activity_map import get_all_maps_associated_to_keyword, plot_activity_map

def plot_matrix_heatmap(M):
    sns.heatmap(M)
    plt.show()

def plot_cov_matrix_brain(M, Ni, Nj, Nk, affine, threshold=None):
    
    n_voxels, _ = M.shape
    # coords = np.arange(n_voxels).reshape((Ni, Nj, Nk), order='F')

    coords = np.zeros((Ni, Nj, Nk, 3)).astype(int)

    for k in range(Ni):
         coords[k, :, :, 0] = k
    for k in range(Nj):
         coords[:, k, :, 1] = k
    for k in range(Nk):
         coords[:, :, k, 2] = k

    coords = coords.reshape((-1, 3), order='F')

    coords_world = np.zeros(coords.shape)

    # print(affine)

    for k in range(coords.shape[0]):
        coords_world[k, :] = np.dot(affine, [coords[k, 0], coords[k, 1], coords[k, 2], 1])[:-1]
        # print(coords_world[k, :])

    threshold = np.max(M)*0.1
    # threshold=0.1
    # print(threshold)
    plotting.plot_connectome(M, coords_world, node_size=5, node_color='black', edge_threshold=threshold)
    plt.show()


if __name__ == '__main__':
    # keyword = 'memory'
    # keyword = 'prosopagnosia'
    keyword = 'schizophrenia'
    sigma = 2.
    reduce = 10

    # cov_matrix, coords, affine_r, avg_n_peaks = build_covariance_matrix_from_keyword(keyword, sigma=sigma, reduce=reduce, gaussian_filter=False)
    # print(cov_matrix)
    # print(cov_matrix.shape)

    # cov_array = cov_matrix.toarray() 
    # # print(np.percentile(cov_array, .9999))
    # # print(len(cov_array[cov_array > 0]))
    # # plot_matrix_heatmap(cov_array)

    # threshold = estimate_threshold_covariance(avg_n_peaks, Ni//reduce, Nj//reduce, Nk//reduce, N_simulations=1000, sigma=sigma, apply_gaussian_filter=False)
    # print('Avg peaks : {}'.format(avg_n_peaks))
    # print('Threshold : {}'.format(threshold))
    # print('Plotting')
    # # threshold = '25%'
    # plot_cov_matrix_brain(cov_array, coords, affine_r, threshold)

    maps, Ni_r, Nj_r, Nk_r, affine_r = get_all_maps_associated_to_keyword(keyword, reduce=1, sigma=1.)

    avg_map = average_from_maps(maps)
    var_map = variance_from_maps(maps)

    avg_data = map_to_data(avg_map, Ni_r, Nj_r, Nk_r)
    avg_img = data_to_img(avg_data, affine_r)

    print(avg_map.shape)

    # plot_activity_map(avg_img)
    plot_activity_map(map_to_img(avg_map, Ni_r, Nj_r, Nk_r, affine_r), threshold=0.0)
    plot_activity_map(map_to_img(var_map, Ni_r, Nj_r, Nk_r, affine_r), threshold=0.0)

    # cov_matrix = covariance_from_maps(maps)
    # print(cov_matrix)

    # plot_cov_matrix_brain(cov_matrix, Ni_r, Nj_r, Nk_r, affine_r)

