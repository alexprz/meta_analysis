import numpy as np

from globals import Ni, Nj, Nk

import threshold as thr
import activity_map as am
import plotting
from Maps import Maps


if __name__ == '__main__':
    # Step 1 : Plot activity map from a given pmid
    # pmid = 22266924 
    pmid = 16723214 
    stat_img = am.build_activity_map_from_pmid(pmid, sigma=1.5)
    # plotting.plot_activity_map(stat_img, glass_brain=True, threshold=0.)

    # Step 2
    keyword = 'prosopagnosia'
    # keyword = 'language'
    # keyword = 'schizophrenia'
    sigma = 2.

    maps_HD = Maps(keyword, sigma=None, reduce=1, normalize=False)
    

    avg, var = maps_HD.iterative_smooth_avg_var(sigma=sigma, verbose=True)
    avg_img, var_img = avg.to_img(), var.to_img()

    n_peaks = int(maps_HD.sum())
    n_maps = maps_HD.n_maps

    print('Nb peaks : {}'.format(n_peaks))
    print('Nb maps : {}'.format(n_maps))

    avg_threshold, var_threshold = thr.avg_var_threshold_MC(n_peaks, n_maps, maps_HD.Ni, maps_HD.Nj, maps_HD.Nk, N_simulations=5, sigma=sigma, verbose=True)
    # threshold = 0.0007
    # avg_threshold, var_threshold = 0, 0
    plotting.plot_activity_map(avg_img, glass_brain=False, threshold=avg_threshold)#0.0007)
    plotting.plot_activity_map(var_img, glass_brain=False, threshold=var_threshold)#0.000007)

    # Step 3 : Covariance matrix between voxels
    maps_LD = Maps(keyword, sigma=None, reduce=5, normalize=False)
    cov_matrix = maps_LD.cov()
    print(cov_matrix)
    # cov.plot_cov_matrix_brain(cov_matrix, maps_LD.Ni, maps_LD.Nj, maps_LD.Nk, maps_LD.affine, threshold=50)
