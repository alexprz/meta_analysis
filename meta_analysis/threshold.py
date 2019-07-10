import multiprocessing
from joblib import Parallel, delayed
from time import time
import numpy as np

from .globals import mem, Ni, Nj, Nk
from .tools import print_percent
from .Maps import Maps


def simulate_maps(n_peaks, n_maps, Ni, Nj, Nk, sigma, verbose):
    random_maps = Maps(Ni=Ni, Nj=Nj, Nk=Nk).randomize(n_peaks, n_maps)
    avg_map, var_map = random_maps.iterative_smooth_avg_var(sigma, verbose=verbose)
    return avg_map.max(), var_map.max()

def avg_var_threshold_MC_pool(N_sim, kwargs):
    '''
        Equivalent to avg_var_threshold_MC function called N_sim times.
        (Used for multiprocessing with joblib)
    '''
    avgs, vars = np.zeros(N_sim), np.zeros(N_sim)
    n_peaks, n_maps, Ni, Nj, Nk, sigma, verbose = kwargs['n_peaks'], kwargs['n_maps'], kwargs['Ni'], kwargs['Nj'], kwargs['Nk'], kwargs['sigma'], kwargs['verbose']
    for k in range(N_sim):
        avgs[k], vars[k] = simulate_maps(n_peaks, n_maps, Ni, Nj, Nk, sigma, verbose)
        print_percent(k, N_sim, prefix='Simulating map with {} peaks : '.format(n_peaks))
    return avgs, vars

@mem.cache
def avg_var_threshold_MC(n_peaks, n_maps, Ni=Ni, Nj=Nj, Nk=Nk, N_simulations=5000, sigma=1., verbose=False):
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
        'n_maps': n_maps,
        'verbose': verbose
    }

    n_list = N_simulations//nb_processes*np.ones(nb_processes).astype(int)

    result = np.concatenate(Parallel(n_jobs=nb_processes, backend='multiprocessing')(delayed(avg_var_threshold_MC_pool)(n, kwargs) for n in n_list), axis=1)
    avgs, vars = result[0], result[1]

    avg_threshold = np.percentile(avgs, .99)
    var_threshold = np.percentile(vars, .99)

    print('Time for MC threshold estimation : {}'.format(time()-time0))
    print('Estimated avg threshold : {}'.format(avg_threshold))
    print('Estimated var threshold : {}'.format(var_threshold))
    return avg_threshold, var_threshold

if __name__ == '__main__':
    n_peaks = 2798
    n_maps = 5
    sigma = 2.

    print(avg_var_threshold_MC(n_peaks, n_maps, N_simulations=20, sigma=2.))
