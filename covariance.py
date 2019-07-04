from time import time
import numpy as np
import scipy
import seaborn as sns
from nilearn import plotting
from matplotlib import pyplot as plt

from joblib import Parallel, delayed

from globals import mem, coordinates, corpus_tfidf, Ni, Nj, Nk, affine, inv_affine
from builds import encode_pmid, encode_feature, decode_pmid, decode_feature
from tools import print_percent

@mem.cache
def build_covariance_matrix_from_keyword(keyword, gaussian_filter=False, sigma=2., reduce=2):
    '''
        Build empirical covariance matrix of the voxel of the activity map associated to the given keyword
    '''
    time0 = time()

    feature_id = encode_feature[keyword]

    print('Get nonzero pmid')
    nonzero_pmids = np.array([int(decode_pmid[index]) for index in corpus_tfidf[:, feature_id].nonzero()[0]])

    # Change box size according to reduce factor
    Ni_r, Nj_r, Nk_r = np.ceil(np.array((Ni, Nj, Nk))/reduce).astype(int)

    # stat_img_data = np.zeros((Ni_r,Nj_r,Nk_r)) # Building blank stat_img with MNI152's shape
    n_observations = len(nonzero_pmids)
    n_voxels = Ni_r*Nj_r*Nk_r

    coords = np.zeros((Ni_r, Nj_r, Nk_r, 3)).astype(int)

    for k in range(Ni_r):
         coords[k, :, :, 0] = k
    for k in range(Nj_r):
         coords[:, k, :, 1] = k
    for k in range(Nk_r):
         coords[:, :, k, 2] = k

    # print(coords)

    coords = coords.reshape(-1, 3)

    # print(coords)

    observations = np.zeros((n_observations, n_voxels))

    # Change affine to new box size
    affine_r = np.copy(affine)
    for i in range(3):
        affine_r[i, i] = affine[i, i]*reduce

    inv_affine_r = np.linalg.inv(affine_r)

    def compute_map(pmid_enumerate):
        n_tot = len(pmid_enumerate)
        observations_ = np.zeros((n_tot, n_voxels))
        p = 0
        for obs_index, pmid in pmid_enumerate:
            stat_img_data = np.zeros((Ni_r,Nj_r,Nk_r)) # Building blank stat_img with MNI152's shape

            # For each coordinates found in pmid (in mm), compute its corresponding voxels coordinates
            # and note it as activated
            # print('{} out of {}'.format(obs_index, n_observations))
            print_percent(p, n_tot)
            for index, row in coordinates.loc[coordinates['pmid'] == pmid].iterrows():
                x, y, z = row['x'], row['y'], row['z']
                i, j, k = np.minimum(np.floor(np.dot(inv_affine_r, [x, y, z, 1]))[:-1].astype(int), [Ni_r-1, Nj_r-1, Nk_r-1])
                stat_img_data[i, j, k] += 1
            
            # With gaussian blurr, sparse calculation may not be efficient (not enough zeros)
            if gaussian_filter:
                stat_img_data = scipy.ndimage.gaussian_filter(stat_img_data, sigma=sigma)

            reshaped = stat_img_data.reshape(-1)
            observations_[p, :] = reshaped
            # observations[obs_index, :] = reshaped
            p += 1

        return observations_

        # return stat_img_data.reshape(-1)s
    n_jobs = 8

    # observations = compute_map(list(enumerate(nonzero_pmids)))
    # compute_map(list(enumerate(nonzero_pmids)))

    splitted_array = np.array_split(np.array(list(enumerate(nonzero_pmids))), n_jobs)
    # print(np.array_split(np.array(list(enumerate(nonzero_pmids))), n_jobs))
    # print(splitted_array)
    results = []
    # for sub_array in splitted_array:
    #     results.append(compute_map(sub_array))
        # for i, pmid in sub_array:
        #     compute_map(i, pmid)
    # observations = np.concatenate(results, axis=0)

    # Parallel(n_jobs=8, require='sharedmem')(compute_map(i, pmid) for i, pmid in enumerate(nonzero_pmids))
    observations = np.concatenate(Parallel(n_jobs=n_jobs)(delayed(compute_map)(sub_array) for sub_array in splitted_array), axis=0)

    print(observations)
    # print(observations.shape)


    # Sparse computation of covariance matrix
    s_X = scipy.sparse.csr_matrix(observations)
    s_Ones = scipy.sparse.csr_matrix(np.ones(n_observations))

    M1 = s_X.transpose().dot(s_X)
    M2 = (s_Ones.dot(s_X)).transpose()
    M3 = s_Ones.dot(s_X)

    s_cov_matrix = M1/n_observations - M2.dot(M3)/(n_observations**2)

    return s_cov_matrix, coords, affine_r

def plot_matrix_heatmap(M):
    sns.heatmap(M)
    plt.show()

def plot_cov_matrix_brain(M, coords, affine):
    coords_world = np.zeros(coords.shape)

    # print(affine)

    for k in range(coords.shape[0]):
        coords_world[k, :] = np.dot(affine, [coords[k, 0], coords[k, 1], coords[k, 2], 1])[:-1]
        # print(coords_world[k, :])

    edge_threshold = np.max(M)*0.1
    plotting.plot_connectome(M, coords_world, node_size=5, node_color='black', edge_threshold=edge_threshold)
    plt.show()



if __name__ == '__main__':
    keyword = 'memory'
    # keyword = 'prosopagnosia'
    sigma = 2.

    cov_matrix, coords, affine_r = build_covariance_matrix_from_keyword(keyword, sigma=sigma, reduce=5, gaussian_filter=False)
    print(cov_matrix)
    print(cov_matrix.shape)

    cov_array = cov_matrix.toarray() 
    # print(np.percentile(cov_array, .9999))
    # print(len(cov_array[cov_array > 0]))
    # plot_matrix_heatmap(cov_array)

    print('Plotting')
    plot_cov_matrix_brain(cov_array, coords, affine_r)

