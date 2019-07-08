import multiprocessing
from joblib import Parallel, delayed
from time import time
import numpy as np
from scipy.ndimage import gaussian_filter
import itertools

from globals import mem, Ni, Nj, Nk, gray_mask
from tools import print_percent
from activity_map import Maps


def simulate_max_peaks(n_peaks, Ni, Nj, Nk, sigma):
    '''
        Simulate a map under the null hypothesis and smooth it with a gaussian kernel (ALE method)
        Return the max encountered.
    '''
    brain_map = np.random.binomial(n=n_peaks, p=1./(Ni*Nj*Nk), size=(Ni, Nj, Nk)).astype(float)
    brain_map = np.ma.masked_array(brain_map, np.logical_not(gray_mask.get_data()))
    brain_map = gaussian_filter(brain_map, sigma=sigma)
    return np.max(brain_map)

def simulate_N_maps_joblib(N_sim, kwargs):
    '''
        Equivalent to simulate_max_peaks function called N_sim times.
        (Used for multiprocessing with joblib)
    '''
    peaks = np.zeros(N_sim)
    n_peaks, Ni, Nj, Nk, sigma = kwargs['n_peaks'], kwargs['Ni'], kwargs['Nj'], kwargs['Nk'], kwargs['sigma']
    for k in range(N_sim):
        peaks[k] = simulate_max_peaks(n_peaks, Ni, Nj, Nk, sigma)
        print_percent(k, N_sim, prefix='Simulating map with {} peaks : '.format(n_peaks))
    return peaks

# def estimate_threshold_monte_carlo(n_peaks, Ni=Ni, Nj=Nj, Nk=Nk, N_simulations=5000, sigma=1.):
#     '''
#         Generate N_simulation maps under the null hypothesis.
#         Take the 95% percentile of the maxima encoutered.
#     '''
#     max_peaks = np.zeros(N_simulations)
#     time0 = time()

#     for k in range(N_simulations):
#         print(k)
#         max_peaks[k] = simulate_max_peaks(n_peaks, Ni, Nj, Nk, sigma=sigma)

#     estimated_threshold = np.mean(max_peaks)

#     print('Time for MC threshold estimation : {}'.format(time()-time0))
#     print('Estimated threshold : {}'.format(estimated_threshold))
#     return estimated_threshold

@mem.cache
def estimate_threshold_monte_carlo_joblib(n_peaks, Ni=Ni, Nj=Nj, Nk=Nk, N_simulations=5000, sigma=1.):
    '''
        Estimate threshold with Monte Carlo using multiprocessing thanks to joblib module
    '''
    time0 = time()
    nb_processes=multiprocessing.cpu_count()-1

    kwargs = {
        'Ni': Ni,
        'Nj': Nj,
        'Nk': Nk,
        'sigma': sigma,
        'n_peaks': n_peaks
    }

    n_list = N_simulations//nb_processes*np.ones(nb_processes).astype(int)

    result = Parallel(n_jobs=nb_processes, backend='multiprocessing')(delayed(simulate_N_maps_joblib)(n, kwargs) for n in n_list)

    estimated_threshold = np.mean(result)

    print('Time for MC threshold estimation : {}'.format(time()-time0))
    print('Estimated threshold : {}'.format(estimated_threshold))
    return estimated_threshold

# def compute_random_map(pmid_enumerate, n_voxels, Ni_r, Nj_r, Nk_r, inv_affine_r, gaussian_filter):
#     n_tot = len(pmid_enumerate)
#     observations_ = np.zeros((n_tot, n_voxels))
#     p = 0
#     for obs_index, pmid in pmid_enumerate:
#         stat_img_data = np.zeros((Ni_r,Nj_r,Nk_r)) # Building blank stat_img with MNI152's shape

#         # For each coordinates found in pmid (in mm), compute its corresponding voxels coordinates
#         # and note it as activated
#         print_percent(p, n_tot, prefix='Covariance matrix ')
#         for index, row in coordinates.loc[coordinates['pmid'] == pmid].iterrows():
#             x, y, z = row['x'], row['y'], row['z']
#             i, j, k = np.minimum(np.floor(np.dot(inv_affine_r, [x, y, z, 1]))[:-1].astype(int), [Ni_r-1, Nj_r-1, Nk_r-1])
#             stat_img_data[i, j, k] += 1
        
#         # With gaussian kernel, sparse calculation may not be efficient (not enough zeros)
#         if gaussian_filter:
#             stat_img_data = scipy.ndimage.gaussian_filter(stat_img_data, sigma=sigma)

#         observations_[p, :] = stat_img_data.reshape(-1)
#         p += 1

#     return observations_

@mem.cache
def estimate_threshold_covariance(n_peaks, Ni, Nj, Nk, N_simulations=5000, apply_gaussian_filter=False, sigma=2.):
    n_voxels = Ni*Nj*Nk
    for k in range(N_simulations):
        print(k)
        observations = np.zeros((N_simulations, n_voxels))
        brain_map = np.random.binomial(n=n_peaks, p=1./(Ni*Nj*Nk), size=(Ni, Nj, Nk)).astype(float)
        # brain_map = np.ma.masked_array(brain_map, np.logical_not(gray_mask.get_data()))
        if apply_gaussian_filter:
            brain_map = gaussian_filter(brain_map, sigma=sigma)

        observations[k, :] = brain_map.reshape(-1)

    cov_matrix = empirical_cov_matrix(observations).toarray()

    return np.max(cov_matrix)
    # return np.percentile(cov_matrix, .9999)

def simulate_maps(n_peaks, n_maps, Ni, Nj, Nk, sigma):
    random_maps = Maps(Ni=Ni, Nj=Nj, Nk=Nk).randomize(n_peaks, n_maps)
    avg_map, var_map = random_maps.iterative_smooth_avg_var(sigma)
    return avg_map.max(), var_map.max()

def avg_var_threshold_MC_pool(N_sim, kwargs):
    '''
        Equivalent to simulate_max_peaks function called N_sim times.
        (Used for multiprocessing with joblib)
    '''
    avgs, vars = np.zeros(N_sim), np.zeros(N_sim)
    n_peaks, n_maps, Ni, Nj, Nk, sigma = kwargs['n_peaks'], kwargs['n_maps'], kwargs['Ni'], kwargs['Nj'], kwargs['Nk'], kwargs['sigma']
    for k in range(N_sim):
        avgs[k], vars[k] = simulate_maps(n_peaks, n_maps, Ni, Nj, Nk, sigma)
        print_percent(k, N_sim, prefix='Simulating map with {} peaks : '.format(n_peaks))
    return avgs, vars

@mem.cache
def avg_var_threshold_MC(n_peaks, n_maps, Ni=Ni, Nj=Nj, Nk=Nk, N_simulations=5000, sigma=1.):
    '''
        Estimate threshold with Monte Carlo using multiprocessing thanks to joblib module
    '''
    time0 = time()
    nb_processes=multiprocessing.cpu_count()//2

    kwargs = {
        'Ni': Ni,
        'Nj': Nj,
        'Nk': Nk,
        'sigma': sigma,
        'n_peaks': n_peaks,
        'n_maps': n_maps
    }

    n_list = N_simulations//nb_processes*np.ones(nb_processes).astype(int)

    result = np.concatenate(Parallel(n_jobs=nb_processes, backend='multiprocessing')(delayed(avg_var_threshold_MC_pool)(n, kwargs) for n in n_list), axis=1)
    print(result)
    avgs, vars = result[0], result[1]

    avg_threshold = np.percentile(avgs, .95)
    var_threshold = np.percentile(vars, .95)

    print('Time for MC threshold estimation : {}'.format(time()-time0))
    print('Estimated avg threshold : {}'.format(avg_threshold))
    print('Estimated var threshold : {}'.format(var_threshold))
    return avg_threshold, var_threshold

if __name__ == '__main__':
    nb_peaks = 2798
    sigma = 1.5
    # print(estimate_threshold_monte_carlo_joblib(nb_peaks, Ni, Nj, Nk, N_simulations=100, sigma=sigma))
    # print(estimate_threshold_covariance(nb_peaks, Ni//10, Nj//10, Nk//10, N_simulations=5000, sigma=sigma))

    print(avg_var_threshold_MC(10000, 10, N_simulations=100, sigma=2.))
