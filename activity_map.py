import numpy as np
import nibabel as nib
from nilearn import plotting
from scipy.ndimage import gaussian_filter
from time import time
import matplotlib
matplotlib.use('TkAgg')

from globals import Ni, Nj, Nk, coordinates, corpus_tfidf, affine, inv_affine, gray_mask
from builds import encode_feature, decode_feature, encode_pmid, decode_pmid

def build_activity_map_from_pmid(pmid, sigma=1):
    '''
        Given a pmid, build its corresponding activity map

        pmid : integer found in pmids.txt
        sigma : std used in gaussian blurr
    '''

    # coordinates = pd.read_csv(input_path+'coordinates.csv')
    stat_img_data = np.zeros((Ni,Nj,Nk)) # Building blank stat_img with MNI152's shape

    # For each coordinates found in pmid (in mm), compute its corresponding voxels coordinates
    # and note it as activated
    for index, row in coordinates.loc[coordinates['pmid'] == pmid].iterrows():
        x, y, z = row['x'], row['y'], row['z']
        # i, j, k, _ = np.rint(np.dot(inv_affine, [x, y, z, 1])).astype(int)
        i, j, k = np.minimum(np.floor(np.dot(inv_affine, [x, y, z, 1]))[:-1].astype(int), [Ni-1, Nj-1, Nk-1])
        stat_img_data[i, j, k] += 1
        
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
        plotting.plot_glass_brain(stat_img, black_bg=True, threshold=threshold)#*np.max(stat_img.get_data()))#, threshold=threshold)#threshold*np.max(stat_img.get_data()))
    else:
        plotting.plot_stat_map(stat_img, black_bg=True, threshold=threshold)#*np.max(stat_img.get_data()))#, threshold=threshold)#threshold*np.max(stat_img.get_data()))
    plotting.show()

def build_activity_map_from_keyword(keyword, sigma=1, gray_matter_mask=True):
    '''
        From the given keyword, build its metaanalysis activity map from all related pmids.

        sigma : std of the gaussian blurr
        gray_matter_mask : specify whether the map is restrained to the gray matter or not

        return (stat_img, hist_img, n_sample)
        stat_img : Nifti1Image object, the map where frequencies are added for each activation
        hist_img : Nifti1Image object, the map where 1 is added for each activation
        n_sample : nb of total peaks (inside and outside gray matter)
    '''
    time0 = time()
    
    feature_id = encode_feature[keyword]

    print('Get nonzero pmid')
    nonzero_pmids = np.array([int(decode_pmid[index]) for index in corpus_tfidf[:, feature_id].nonzero()[0]])

    stat_img_data = np.zeros((Ni,Nj,Nk)) # Building blank img with MNI152's shape
    hist_img_data = np.zeros((Ni,Nj,Nk)) # Building blank img with MNI152's shape
    
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
        hist_img_data[i, j, k] += 1
        stat_img_data[i, j, k] += frequencies[index]
    

    if gray_matter_mask:
        stat_img_data = np.ma.masked_array(stat_img_data, np.logical_not(gray_mask.get_data()))
        hist_img_data = np.ma.masked_array(hist_img_data, np.logical_not(gray_mask.get_data()))

    print('Building time : {}'.format(time()-time0))
    stat_img_data = gaussian_filter(stat_img_data, sigma=sigma)
    hist_img_data = gaussian_filter(hist_img_data, sigma=sigma)

    stat_img = nib.Nifti1Image(stat_img_data, affine)
    hist_img = nib.Nifti1Image(hist_img_data, affine)

    return stat_img, hist_img, n_sample



if __name__ == '__main__':
    pmid = 22266924
    keyword = 'prosopagnosia'
    sigma = 2.

    stat_img = build_activity_map_from_pmid(pmid, sigma=sigma)
    plot_activity_map(stat_img, glass_brain=True, threshold=0.)

    stat_img, hist_img, nb_peaks = build_activity_map_from_keyword(keyword, sigma=sigma, gray_matter_mask=True)
    print('Nb peaks : {}'.format(nb_peaks))

    plot_activity_map(hist_img, glass_brain=False, threshold=0.04)