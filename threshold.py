import multiprocessing
from joblib import Parallel, delayed
from time import time
import numpy as np
from scipy.ndimage import gaussian_filter

from globals import mem, Ni, Nj, Nk, gray_mask
from tools import print_percent


def simulate_max_peaks(n_peaks, Ni, Nj, Nk, sigma):
    '''
        Simulate a map under the null hypothesis and smooth it with a gaussian blurr (ALE method)
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
    for k in range(N_sim):
        peaks[k] = simulate_max_peaks(kwargs['n_peaks'], kwargs['Ni'], kwargs['Nj'], kwargs['Nk'], kwargs['sigma'])
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

    estimated_threshold = np.mean(max_peaks)

    print('Time for MC threshold estimation : {}'.format(time()-time0))
    print('Estimated threshold : {}'.format(estimated_threshold))
    return estimated_threshold

@mem.cache
def estimate_threshold_monte_carlo_joblib(n_peaks, Ni=Ni, Nj=Nj, Nk=Nk, N_simulations=5000, sigma=1.):
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
        'n_peaks': n_peaks
    }

    n_list = N_simulations//nb_processes*np.ones(nb_processes).astype(int)

    result = Parallel(n_jobs=nb_processes)(delayed(simulate_N_maps_joblib)(n, kwargs) for n in n_list)

    estimated_threshold = np.mean(result)

    print('Time for MC threshold estimation : {}'.format(time()-time0))
    print('Estimated threshold : {}'.format(estimated_threshold))
    return estimated_threshold

if __name__ == '__main__':
    nb_peaks = 2798
    sigma = 2.
    print(estimate_threshold_monte_carlo_joblib(nb_peaks, Ni, Nj, Nk, N_simulations=100, sigma=sigma))
