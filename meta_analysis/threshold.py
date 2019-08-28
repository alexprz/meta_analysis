"""Implement functions to compute thresholds following a null hypothesis."""
import multiprocessing
from joblib import Parallel, delayed
from time import time
import numpy as np

from .globals import mem
from .tools import print_percent
from .Maps import Maps


def simulate_maps(rand_maps, n_peaks, n_maps, Ni, Nj, Nk, sigma, verbose, p,
                  mask, var, cov):
    """Simulate maps and return max of avg and var encoutered."""
    if (var, cov) == (False, False):
        rand_maps.randomize((n_peaks, 1), p=p, inplace=True)
        map = rand_maps.avg()*(1./n_maps)
        map = map.smooth(sigma=sigma, inplace=True)
        return map.max(), None, None

    else:
        rand_maps.randomize((n_peaks, n_maps), p=p, inplace=True)
        avg_map, var_map = rand_maps.iterative_smooth_avg_var(compute_var=var,
                                                              sigma=sigma,
                                                              verbose=verbose)
        return avg_map.max(), var_map.max(), None


def threshold_MC_pool(N_sim, kwargs):
    """Equivalent to threshold_MC function called N_sim times."""
    avgs, vars, covs = np.zeros(N_sim), np.zeros(N_sim), np.zeros(N_sim)
    Ni, Nj, Nk = kwargs['Ni'], kwargs['Nj'], kwargs['Nk']
    mask = kwargs['mask']
    kwargs['rand_maps'] = Maps(Ni=Ni, Nj=Nj, Nk=Nk, mask=mask)
    for k in range(N_sim):
        s = f"Simulating map with {kwargs['n_peaks']} peaks : "\
            f"{{1}} out of {{2}} {{0}}%..."
        print_percent(k, N_sim, string=s, verbose=kwargs['verbose'],
                      end='\r', rate=0)
        avgs[k], vars[k], covs[k] = simulate_maps(**kwargs)

    return avgs, vars, covs


@mem.cache
def threshold_MC(n_peaks, n_maps, Ni, Nj, Nk, stats=['avg', 'var'], N_sim=5000,
                 sigma=1., verbose=False, p=None, mask=None):
    """
    Estimate threshold with Monte Carlo.

    Args:
        n_peaks(int): Number of peaks to sample accross the n_maps.
        n_maps(int): Number of maps.
        Ni, Nj, Nk(int, int, int): Size of the box.
        stats(list, Optional): List of string giving the wanted stats.
            Available are 'avg' and 'var'.
        N_sim(int): Number of simulations.
        sigma(float): Standard deviation used to smooth the simulated maps.
        verbose(bool): Whether should print log.
        p: Null distribution. See Maps.randomize doc for more information.
        mask: Mask to apply to the null distribution. See Maps.randomize
            for more information.

    Returns:
        (dict) Dictionary which keys are the elements of stats and values
            the corresponding threhsold.

    """
    time0 = time()
    nb_processes = multiprocessing.cpu_count()//2

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

    n_list = N_sim//nb_processes*np.ones(nb_processes).astype(int)

    result = np.concatenate(Parallel(n_jobs=nb_processes,
                                     backend='multiprocessing')(
        delayed(threshold_MC_pool)(n, kwargs) for n in n_list), axis=1)
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
