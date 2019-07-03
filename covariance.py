from time import time
import numpy as np
import scipy
import seaborn as sns
from matplotlib import pyplot as plt

from globals import mem, coordinates, corpus_tfidf, Ni, Nj, Nk, affine, inv_affine
from builds import encode_pmid, encode_feature, decode_pmid, decode_feature

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

    stat_img_data = np.zeros((Ni_r,Nj_r,Nk_r)) # Building blank stat_img with MNI152's shape
    n_observations = len(nonzero_pmids)
    observations = np.zeros((n_observations, Ni_r*Nj_r*Nk_r))

    # Change affine to new box size
    affine_r = np.copy(affine)
    for i in range(3):
        affine_r[i, i] = affine[i, i]*reduce

    inv_affine_r = np.linalg.inv(affine_r)

    for i, pmid in enumerate(nonzero_pmids):
        # For each coordinates found in pmid (in mm), compute its corresponding voxels coordinates
        # and note it as activated
        print('{} out of {}'.format(i, n_observations))
        for index, row in coordinates.loc[coordinates['pmid'] == pmid].iterrows():
            x, y, z = row['x'], row['y'], row['z']
            i, j, k = np.minimum(np.floor(np.dot(inv_affine_r, [x, y, z, 1]))[:-1].astype(int), [Ni_r-1, Nj_r-1, Nk_r-1])
            stat_img_data[i, j, k] += 1
        
        # With gaussian blurr, sparse calculation may not be efficient (not enough zeros)
        if gaussian_filter:
            stat_img_data = scipy.ndimage.gaussian_filter(stat_img_data, sigma=sigma)

        observations[i, :] = stat_img_data.flatten()

    # Sparse computation of covariance matrix
    s_X = scipy.sparse.csr_matrix(observations)
    s_Ones = scipy.sparse.csr_matrix(np.ones(n_observations))

    M1 = s_X.transpose().dot(s_X)
    M2 = (s_Ones.dot(s_X)).transpose()
    M3 = s_Ones.dot(s_X)

    s_cov_matrix = M1/n_observations - M2.dot(M3)/(n_observations**2)

    return s_cov_matrix, affine_r

def plot_matrix(M):
    sns.heatmap(M)
    plt.show()



if __name__ == '__main__':
    keyword = 'prosopagnosia'
    sigma = 2.

    cov_matrix, affine_r = build_covariance_matrix_from_keyword(keyword, sigma=sigma, reduce=10)
    print(cov_matrix)
    print(cov_matrix.shape)

    cov_array = cov_matrix.toarray() 
    print(np.percentile(cov_array, .9999))
    print(len(cov_array[cov_array > 0]))
    plot_matrix(cov_array)