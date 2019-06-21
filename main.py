import numpy as np
import pandas as pd
from nilearn import datasets, plotting
from scipy.ndimage import gaussian_filter
import scipy.sparse
import nibabel as nib
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')
from time import time


input_path = 'minimal/'

# Loading MNI152 background and parameters (shape, affine...)
bg_img = datasets.load_mni152_template()
Ni, Nj, Nk = bg_img.shape
affine = bg_img.affine
inv_affine = np.linalg.inv(affine)


def build_activity_map_from_pmid(pmid, sigma=1):
    '''
        Given a pmid, build its corresponding activity map

        pmid : integer found in pmids.txt
        sigma : std used in gaussian blurr
    '''

    coordinates = pd.read_csv(input_path+'coordinates.csv')
    stat_img_data = np.zeros((Ni,Nj,Nk)) # Building blank stat_img with MNI152's shape

    # For each coordinates found in pmid (in mm), compute its corresponding voxels coordinates
    # and note it as activated
    for index, row in coordinates.loc[coordinates['pmid'] == pmid].iterrows():
        x, y, z = row['x'], row['y'], row['z']
        i, j, k, _ = np.rint(np.dot(inv_affine, [x, y, z, 1])).astype(int)
        stat_img_data[i, j, k] = 1
        
    # Add gaussian blurr
    stat_img_data = gaussian_filter(stat_img_data, sigma=sigma)

    return nib.Nifti1Image(stat_img_data, affine)

def plot_activity_map(stat_img, threshold=0.1, glass_brain=False):
    '''
        Plot stat_img on MNI152 background

        stat_img : Object of Nifti1Image Class
        threshold : min value to display (in percent of maximum)
    '''
    if glass_brain:
        plotting.plot_glass_brain(stat_img, black_bg=True, threshold=threshold*np.max(stat_img.get_data()))
    else:
        plotting.plot_stat_map(stat_img, black_bg=True, threshold=threshold*np.max(stat_img.get_data()))
    plotting.show()

def build_index(file_name):
    decode = dict(enumerate(line.strip() for line in open(input_path+file_name)))
    encode = {v: k for k, v in decode.items()}
    
    return encode, decode

def build_activity_map_from_keyword(keyword, sigma=1):
    time0 = time()
    corpus_tfidf = scipy.sparse.load_npz(input_path+'corpus_tfidf.npz')

    encode_feature, decode_feature = build_index('feature_names.txt')
    encode_pmid, decode_pmid = build_index('pmids.txt')
    
    feature_id = encode_feature[keyword]

    print('Get nonzero pmid')
    nonzero_pmids = np.array([int(decode_pmid[index]) for index in corpus_tfidf[:, feature_id].nonzero()[0]])

    coordinates = pd.read_csv(input_path+'coordinates.csv')
    stat_img_data = np.zeros((Ni,Nj,Nk)) # Building blank stat_img with MNI152's shape
    
    print('Get nonzero coordinates')
    nonzero_coordinates = coordinates.loc[coordinates['pmid'].isin(nonzero_pmids)]
    print('Get pmid')
    pmids = np.array(nonzero_coordinates['pmid'])
    print('Build frequencies')
    frequencies = np.array([corpus_tfidf[encode_pmid[str(pmid)], feature_id] for pmid in nonzero_coordinates['pmid']])
    print(len(nonzero_coordinates))
    n_sample = len(nonzero_coordinates)
    print('Build coords')
    coord = np.zeros((n_sample, 3))
    coord[:, 0] = nonzero_coordinates['x']
    coord[:, 1] = nonzero_coordinates['y']
    coord[:, 2] = nonzero_coordinates['z']

    print('Build voxel coords')
    voxel_coords = np.zeros((n_sample, 3)).astype(int)
    for k in range(n_sample):
        voxel_coords[k, :] = np.minimum(np.floor(np.dot(inv_affine, [coord[k, 0], coord[k, 1], coord[k, 2], 1]))[:-1].astype(int), [Ni-1, Nj-1, Nk-1])

    print('Building map')
    for index, value in enumerate(voxel_coords):
        i, j, k = value
        # stat_img_data[i, j, k] += corpus_tfidf[encode_pmid[str(pmids[index])], feature_id]
        stat_img_data[i, j, k] += frequencies[index]

    print('Building time : {}'.format(time()-time0))
    stat_img_data = gaussian_filter(stat_img_data, sigma=1)
    return nib.Nifti1Image(stat_img_data, affine)

def simulate_max_peaks(n_peaks, Ni, Nj, Nk, sigma):
    brain_map = np.random.binomial(n=n_peaks, p=1./(Ni*Nj*Nk), size=(Ni, Nj, Nk)).astype(float)
    # print(brain_map)
    # peaks = np.random.uniform((Ni, Nj, Nk), size=(n_peaks, 3)).astype(int)
    # # print(peaks)
    # brain_map = np.zeros((Ni, Nj, Nk))
    # # print(brain_map.shape)
    # unique, unique_counts = np.unique(peaks, axis=0, return_counts=True)
    # # print(unique)
    # # print(unique_counts)
    # # brain_map[unique] = unique_counts
    # brain_map[unique[:, 0], unique[:, 1], unique[:, 2]] = unique_counts
    # # print(brain_map)
    # # for coord in peaks:
    # #     # print(tuple(coord))
    # #     brain_map[tuple(coord)] += 1
    # # print(brain_map)
    brain_map = gaussian_filter(brain_map, sigma=1)
    # print(brain_map)
    return np.max(brain_map)
    # peaks = np.random.random_integers(N_voxels-1, size=n_peaks)
    # # print(peaks)
    # unique, unique_counts = np.unique(peaks, return_counts=True)
    # print(unique_counts)
    # print(np.max(unique_counts))
    # return np.max(unique_counts)
    # pass


def estimate_threshold_monte_carlo(n_peaks, Ni=Ni, Nj=Nj, Nk=Nk, N_simulations=5000, sigma=1.):
    max_peaks = np.zeros(N_simulations)

    time0 = time()
    for k in range(N_simulations):
        print(k)
        max_peaks[k] = simulate_max_peaks(n_peaks, Ni, Nj, Nk, sigma=sigma)

    print('Time for MC threshold estimation : {}'.format(time()-time0))
    print('Estimated threshold : {}'.format(np.mean(max_peaks)))
    return np.mean(max_peaks)

if __name__ == '__main__':
    # Step 1 : Plot activity map from a given pmid
    # pmid = 22266924 
    # stat_img = build_activity_map_from_pmid(pmid, sigma=1.5)
    # plot_activity_map(stat_img, glass_brain=True)


    # Step 2
    # keyword = 'amygdala'
    # stat_img = build_activity_map_from_keyword(keyword, sigma=1.5)
    # plot_activity_map(stat_img, glass_brain=True, threshold=0.3)

    # print(simulate_max_peaks(10000, Ni, Nj, Nk, sigma=1.))
    print(estimate_threshold_monte_carlo(1000, Ni, Nj, Nk, N_simulations=5000, sigma=1.))
