
import multiprocessing
from joblib import Parallel, delayed
from time import time
import numpy as np

from .globals import mem
from .tools import print_percent
from .Maps import Maps

def simulate_maps(random_maps, n_peaks, n_maps, Ni, Nj, Nk, sigma, verbose, p, mask, var, cov):
    if (var, cov) == (False, False):
        # random_maps = Maps(Ni=Ni, Nj=Nj, Nk=Nk).randomize(n_peaks, 1, p=p)
        random_maps.randomize(n_peaks, 1, p=p, inplace=True)
        map = random_maps.avg()*(1./n_maps)
        map = map.smooth(sigma=sigma, inplace=True)
        return map.max(), None, None

    else:
        random_maps = Maps(Ni=Ni, Nj=Nj, Nk=Nk).randomize(n_peaks, n_maps, p=p)
        avg_map, var_map = random_maps.iterative_smooth_avg_var(compute_var=var, sigma=sigma, verbose=verbose)
        return avg_map.max(), var_map.max(), None

def threshold_MC_pool(N_sim, kwargs):
    '''
        Equivalent to threshold_MC function called N_sim times.
        (Used for multiprocessing with joblib)
    '''
    avgs, vars, covs = np.zeros(N_sim), np.zeros(N_sim), np.zeros(N_sim)
    kwargs['random_maps'] = Maps(Ni=kwargs['Ni'], Nj=kwargs['Nj'], Nk=kwargs['Nk'])
    for k in range(N_sim):
        print_percent(k, N_sim, string=f"Simulating map with {kwargs['n_peaks']} peaks : {{1}} out of {{2}} {{0}}%...", verbose=kwargs['verbose'], end='\r', rate=0)
        avgs[k], vars[k], covs[k] = simulate_maps(**kwargs)
    
    return avgs, vars, covs

@mem.cache
def threshold_MC(n_peaks, n_maps, Ni, Nj, Nk, stats=['avg', 'var'], N_simulations=5000, sigma=1., verbose=False, p=None, mask=None):
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
        'verbose': verbose,
        'p': p,
        'mask': mask,
        'var': True if 'var' in stats else False,
        'cov': True if 'cov' in stats else False,
    }

    n_list = N_simulations//nb_processes*np.ones(nb_processes).astype(int)

    result = np.concatenate(Parallel(n_jobs=nb_processes, backend='multiprocessing')(delayed(threshold_MC_pool)(n, kwargs) for n in n_list), axis=1)
    avgs, vars, covs = result[0], result[1], result[2]

    res = dict()
    if 'avg' in stats:
        res['avg'] = np.percentile(avgs, .95)
    if 'var' in stats:
        res['var'] = np.percentile(vars, .95)
    if 'cov' in stats:
        res['cov'] = np.percentile(covs, .95)

    print('Time for MC threshold estimation : {}'.format(time()-time0))
    print('Estimated thresholds : {}'.format(res))

    return res

if __name__ == '__main__':
    n_peaks = 2798
    n_maps = 5
    sigma = 2.

    print(avg_var_threshold_MC(n_peaks, n_maps, N_simulations=20, sigma=2.))
