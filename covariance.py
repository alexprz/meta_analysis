from globals import mem, coordinates, corpus_tfidf, Ni, Nj, Nk, inv_affine
from builds import encode_pmid, encode_feature, decode_pmid, decode_feature
from time import time
import numpy as np
import scipy

@mem.cache
def build_covariance_matrix_from_keyword(keyword, sigma=2.):
    '''
        Build empirical covariance matrix of the voxel of the activity map associated to the given keyword
    '''
    time0 = time()
    # corpus_tfidf = scipy.sparse.load_npz(input_path+'corpus_tfidf.npz')

    # encode_feature, decode_feature = build_index('feature_names.txt')
    # encode_pmid, decode_pmid = build_index('pmids.txt')
    
    feature_id = encode_feature[keyword]

    print('Get nonzero pmid')
    nonzero_pmids = np.array([int(decode_pmid[index]) for index in corpus_tfidf[:, feature_id].nonzero()[0]])

    # coordinates = pd.read_csv(input_path+'coordinates.csv')
    stat_img_data = np.zeros((Ni,Nj,Nk)) # Building blank stat_img with MNI152's shape

    n_observations = len(nonzero_pmids)
    observations = np.zeros((n_observations, Ni*Nj*Nk))

    for i, pmid in enumerate(nonzero_pmids):
        # For each coordinates found in pmid (in mm), compute its corresponding voxels coordinates
        # and note it as activated
        print('{} out of {}'.format(i, n_observations))
        for index, row in coordinates.loc[coordinates['pmid'] == pmid].iterrows():
            x, y, z = row['x'], row['y'], row['z']
            i, j, k = np.minimum(np.floor(np.dot(inv_affine, [x, y, z, 1]))[:-1].astype(int), [Ni-1, Nj-1, Nk-1])
            stat_img_data[i, j, k] += 1
        
        # With gaussian blurr, sparse calculation isn't efficient (not enough zeros)
        # stat_img_data = gaussian_filter(stat_img_data, sigma=sigma, truncate=1.)

        observations[i, :] = stat_img_data.flatten()


    s_X = scipy.sparse.csr_matrix(observations)
    s_Ones = scipy.sparse.csr_matrix(np.ones(n_observations))

    M1 = s_X.transpose().dot(s_X)
    M2 = (s_Ones.dot(s_X)).transpose()
    M3 = s_Ones.dot(s_X)

    s_cov_matrix = M1/n_observations - M2.dot(M3)/(n_observations**2)

    return s_cov_matrix

if __name__ == '__main__':
    print(build_covariance_matrix_from_keyword('prosopagnosia'))