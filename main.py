import numpy as np
import pandas as pd
from nilearn import datasets, plotting, masking
from scipy.ndimage import gaussian_filter
import scipy.sparse
import nibabel as nib
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')
from time import time
from tools import pool_computing, print_percent
import ray
import multiprocessing


input_path = 'minimal/'

# Loading MNI152 background and parameters (shape, affine...)
bg_img = datasets.load_mni152_template()
gray_mask = masking.compute_gray_matter_mask(bg_img)
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
        plotting.plot_glass_brain(stat_img, black_bg=True, threshold=threshold)#*np.max(stat_img.get_data()))#, threshold=threshold)#threshold*np.max(stat_img.get_data()))
    else:
        plotting.plot_stat_map(stat_img, black_bg=True, threshold=threshold)#*np.max(stat_img.get_data()))#, threshold=threshold)#threshold*np.max(stat_img.get_data()))
    plotting.show()

def build_index(file_name):
    '''
        Build decode & encode dictionnary of the given file_name.

        encode : dict
            key : line number
            value : string at the specified line number
        decode : dict (reverse of encode)
            key : string found in the file
            value : number of the line containing the string

        Used for the files pmids.txt & feature_names.txt
    '''
    decode = dict(enumerate(line.strip() for line in open(input_path+file_name)))
    encode = {v: k for k, v in decode.items()}
    
    return encode, decode

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
    corpus_tfidf = scipy.sparse.load_npz(input_path+'corpus_tfidf.npz')

    encode_feature, decode_feature = build_index('feature_names.txt')
    encode_pmid, decode_pmid = build_index('pmids.txt')
    
    feature_id = encode_feature[keyword]

    print('Get nonzero pmid')
    nonzero_pmids = np.array([int(decode_pmid[index]) for index in corpus_tfidf[:, feature_id].nonzero()[0]])

    coordinates = pd.read_csv(input_path+'coordinates.csv')
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

def simulate_max_peaks(n_peaks, Ni, Nj, Nk, sigma):
    '''
        Simulate a map under the null hypothesis and smooth it with a gaussian blurr (ALE method)
        Return the max encountered.
    '''
    brain_map = np.random.binomial(n=n_peaks, p=1./(Ni*Nj*Nk), size=(Ni, Nj, Nk)).astype(float)
    brain_map = np.ma.masked_array(brain_map, np.logical_not(gray_mask.get_data()))
    brain_map = gaussian_filter(brain_map, sigma=sigma)
    return np.max(brain_map)
    # return np.percentile(brain_map, .95)

def simulate_N_maps(N_sim, process_number, kwargs):
    peaks = np.zeros(N_sim)
    for k in range(N_sim):
        peaks[k] = simulate_max_peaks(kwargs['n_peaks'], kwargs['Ni'], kwargs['Nj'], kwargs['Nk'], kwargs['sigma'])
        # print(k)
        print_percent(k, N_sim, prefix='Simulating map with {} peaks ({}) : '.format(kwargs['n_peaks'], process_number))
    return peaks

@ray.remote
def simulate_N_maps_ray(N_sim, kwargs):
    peaks = np.zeros(N_sim)
    for k in range(N_sim):
        peaks[k] = simulate_max_peaks(kwargs['n_peaks'], kwargs['Ni'], kwargs['Nj'], kwargs['Nk'], kwargs['sigma'])
        # print(k)
        print_percent(k, N_sim, prefix='Simulating map with {} peaks : '.format(kwargs['n_peaks']))
    return peaks

def estimate_threshold_monte_carlo(n_peaks, Ni=Ni, Nj=Nj, Nk=Nk, N_simulations=5000, sigma=1.):
    '''
        Generate N_simulation maps under the null hypothesis.
        Take the 95% percentile of the maxima encoutered.
    '''
    max_peaks = np.zeros(N_simulations)
    time0 = time()

    for k in range(N_simulations):
        print(k)
        max_peaks[k] = simulate_max_peaks(n_peaks, Ni, Nj, Nk, sigma=sigma)

    # estimated_threshold = np.max(max_peaks)
    estimated_threshold = np.percentile(max_peaks, .99)
    # estimated_threshold = np.mean(max_peaks)

    print('Time for MC threshold estimation : {}'.format(time()-time0))
    print('Estimated threshold : {}'.format(estimated_threshold))
    return estimated_threshold

def estimate_threshold_monte_carlo_multiprocessing(n_peaks, Ni=Ni, Nj=Nj, Nk=Nk, N_simulations=5000, sigma=1.):
    time0 = time()
    result = pool_computing(simulate_N_maps, N_simulations, n_peaks=n_peaks, Ni=Ni, Nj=Nj, Nk=Nk, sigma=sigma)
    
    # estimated_threshold = np.mean(result)
    estimated_threshold = np.percentile(result, .99)

    print('Time for MC threshold estimation : {}'.format(time()-time0))
    print('Estimated threshold : {}'.format(estimated_threshold))
    return estimated_threshold

def estimate_threshold_monte_carlo_ray(n_peaks, Ni=Ni, Nj=Nj, Nk=Nk, N_simulations=5000, sigma=1.):
    time0 = time()
    nb_processes=multiprocessing.cpu_count()//2
    ray.init(num_cpus=nb_processes)

    kwargs = {
        'Ni': Ni,
        'Nj': Nj,
        'Nk': Nk,
        'sigma': sigma,
        'n_peaks': n_peaks
    }

    n_list = N_simulations//nb_processes*np.ones(nb_processes).astype(int)
    
    # result = pool_computing(simulate_N_maps, N_simulations, n_peaks=n_peaks, Ni=Ni, Nj=Nj, Nk=Nk, sigma=sigma)
    result = ray.get([simulate_N_maps_ray.remote(n, kwargs) for n in n_list])
    # estimated_threshold = np.mean(result)
    estimated_threshold = np.percentile(result, .99)
    print(np.mean(result))

    print('Time for MC threshold estimation : {}'.format(time()-time0))
    print('Estimated threshold : {}'.format(estimated_threshold))
    return estimated_threshold

# def estimate_threshold2_monte_carlo(n_peaks, Ni=Ni, Nj=Nj, Nk=Nk, N_simulations=5000, sigma=1.):
#     max_peaks = np.zeros(N_simulations)

#     time0 = time()

#     brain_map = np.random.binomial(n=N_simulations*n_peaks, p=1./(Ni*Nj*Nk), size=(Ni, Nj, Nk)).astype(float)
#     brain_map = brain_map/N_simulations
#     # print(brain_map)
#     brain_map = gaussian_filter(brain_map, sigma=sigma)

#     estimated_threshold = np.max(brain_map)

#     # for k in range(N_simulations):
#     #     print(k)
#     #     max_peaks[k] = simulate_max_peaks(n_peaks, Ni, Nj, Nk, sigma=sigma)

#     print('Time for MC threshold estimation : {}'.format(time()-time0))
#     print('Estimated threshold : {}'.format(estimated_threshold))
#     return estimated_threshold

if __name__ == '__main__':
    # Step 1 : Plot activity map from a given pmid
    # pmid = 22266924 
    # stat_img = build_activity_map_from_pmid(pmid, sigma=1.5)
    # plot_activity_map(stat_img, glass_brain=True)


    # Step 2
    keyword = 'prosopagnosia'
    sigma = 2
    stat_img, hist_img, nb_peaks = build_activity_map_from_keyword(keyword, sigma=sigma, gray_matter_mask=True)
    print('Nb peaks : {}'.format(nb_peaks))
    # threshold = estimate_threshold_monte_carlo(nb_peaks, Ni, Nj, Nk, N_simulations=5000, sigma=sigma)
    threshold = estimate_threshold_monte_carlo_ray(nb_peaks, Ni, Nj, Nk, N_simulations=5000, sigma=sigma)
    print(threshold)
    plot_activity_map(hist_img, glass_brain=False, threshold=threshold)
    # print(estimate_threshold_monte_carlo_multiprocessing(1000, Ni, Nj, Nk, N_simulations=5000, sigma=sigma))
