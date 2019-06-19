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


if __name__ == '__main__':
    # Step 1 : Plot activity map from a given pmid
    # pmid = 22266924 
    # stat_img = build_activity_map_from_pmid(pmid, sigma=1.5)
    # plot_activity_map(stat_img, glass_brain=True)


    # Step 2
    time0 = time()
    corpus_tfidf = scipy.sparse.load_npz(input_path+'corpus_tfidf.npz')

    # feature_file = open(input_path+'feature_names.txt')
    # feature_index = dict()
    # k = 0
    # for line in feature_file:
    #     # print(line.strip())
    #     feature_index[line.strip()] = k
    #     k += 1

    # pmids_file = open(input_path+'pmids.txt')
    # pmids_index = dict()
    # k = 0
    # for line in pmids_file:
    #     # print(line.strip())
    #     pmids_index[int(line.strip())] = k
    #     k += 1
    
    # print(feature_index)
    # print(pmids_index)
    # print(corpus_tfidf[13880, 6179])
    # print(corpus_tfidf[pmids_index[15522765], feature_index['a1']])

    # def build_activity_map_from_feature(feature_name, sigma=1):
    #     feature_id = feature_index[feature_name]
    #     print(feature_id)
    #     frequencies = corpus_tfidf[corpus_tfidf[:, feature_id].nonzero()[0], feature_id]
    #     print(frequencies)

    # build_activity_map_from_feature('memory')

    encode_feature, decode_feature = build_index('feature_names.txt')
    encode_pmid, decode_pmid = build_index('pmids.txt')

    # nonzero_pmids = np.array([pmid[0, 0] for pmid in corpus_tfidf[:, encode_feature['cognitive']]])

    # print(nonzero_pmids)

    feature_id = encode_feature['amygdala']
    # frequencies = corpus_tfidf[:, feature_id].nonzero()[0]
    print('Get nonzero pmid')
    nonzero_pmids = np.array([int(decode_pmid[index]) for index in corpus_tfidf[:, feature_id].nonzero()[0]])
    # print(nonzero_pmids)
    # print(nonzero_pmids.shape)

    coordinates = pd.read_csv(input_path+'coordinates.csv')
    stat_img_data = np.zeros((Ni,Nj,Nk)) # Building blank stat_img with MNI152's shape
    
    # nb = len(nonzero_pmids)
    # count = 0
    # for pmid in nonzero_pmids:
    #     print('{} out of {}'.format(count, nb))
    #     count += 1
    #     for index, row in coordinates.loc[coordinates['pmid'] == pmid].iterrows():
    #         x, y, z = row['x'], row['y'], row['z']
    #         i, j, k, _ = np.floor(np.dot(inv_affine, [x, y, z, 1])).astype(int)
    #         i, j, k = np.minimum([i, j, k], [Ni-1, Nj-1, Nk-1])
    #         # i, j, k, _ = np.rint(np.dot(inv_affine, [x, y, z, 1])).astype(int)
    #         # print(i, j, k)
    #         stat_img_data[i, j, k] += 1#corpus_tfidf[encode_pmid[str(pmid)], feature_id]
    print('Get nonzero coordinates')
    nonzero_coordinates = coordinates.loc[coordinates['pmid'].isin(nonzero_pmids)]
    print('Get pmid')
    pmids = np.array(nonzero_coordinates['pmid'])
    # print('Build frequencies')
    # frequencies = np.array([corpus_tfidf[encode_pmid[str(pmid)], feature_id] for pmid in nonzero_coordinates['pmid']])
    # print(data)
    print(len(nonzero_coordinates))
    n_sample = len(nonzero_coordinates)
    # nonzero_coordinates['feature_frequency'] = pd.Series(data, index=nonzero_coordinates.index)
    # # nonzero_coordinates['feature_frequency'] = np.array(data)
    # print(nonzero_coordinates)

    print('Build coords')
    coord = np.zeros((n_sample, 3))
    coord[:, 0] = nonzero_coordinates['x']
    coord[:, 1] = nonzero_coordinates['y']
    coord[:, 2] = nonzero_coordinates['z']

    print('Build voxel coords')
    voxel_coords = np.zeros((n_sample, 3)).astype(int)
    for k in range(n_sample):
        voxel_coords[k, :] = np.minimum(np.floor(np.dot(inv_affine, [coord[k, 0], coord[k, 1], coord[k, 2], 1]))[:-1].astype(int), [Ni-1, Nj-1, Nk-1])

    # print(coord)
    # print(voxel_coords)
    # print(frequencies)

    print('Building map')
    for index, value in enumerate(voxel_coords):
        # print(i, j, k)
        i, j, k = value
        # print(encode_pmid[str(pmids[index])])
        # stat_img_data[i, j, k] += corpus_tfidf[encode_pmid[str(pmids[index])], feature_id]
        stat_img_data[i, j, k] += corpus_tfidf[encode_pmid[str(pmids[index])], feature_id]
        # stat_img_data[i, j, k] += frequencies[index]
    # print(len(nonzero_coordinates))

    # for index, row in nonzero_coordinates.iterrows():
    #     pmid = row['pmid']
    #     print(index)
    #     # if corpus_tfidf[encode_pmid[str(pmid)], feature_id] == 0:
    #     #     # print('continue')
    #     #     continue
    #     x, y, z = row['x'], row['y'], row['z']
    #     i, j, k, _ = np.floor(np.dot(inv_affine, [x, y, z, 1])).astype(int)
    #     i, j, k = np.minimum([i, j, k], [Ni-1, Nj-1, Nk-1])
    #     # i, j, k, _ = np.rint(np.dot(inv_affine, [x, y, z, 1])).astype(int)
    #     # print(i, j, k)
    #     stat_img_data[i, j, k] += 1


    print('Building time : {}'.format(time()-time0))
    stat_img_data = gaussian_filter(stat_img_data, sigma=1)
    plot_activity_map(nib.Nifti1Image(stat_img_data, affine), glass_brain=True, threshold=0.3)
