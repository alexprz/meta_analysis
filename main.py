import numpy as np

from globals import Ni, Nj, Nk

import covariance as cov
import threshold as thr
import activity_map as am


if __name__ == '__main__':
    # Step 1 : Plot activity map from a given pmid
    pmid = 22266924 
    stat_img = am.build_activity_map_from_pmid(pmid, sigma=1.5)
    am.plot_activity_map(stat_img, glass_brain=True, threshold=0.)

    # Step 2
    keyword = 'prosopagnosia'
    # keyword = 'schizophrenia'
    sigma = 2.

    maps_HD = am.Maps(keyword, sigma=sigma, reduce=1, normalize=False)
    
    avg_img = maps_HD.avg().to_img()
    var_img = maps_HD.var().to_img()

    n_peaks = np.sum(maps_HD.n_peaks())
    print('Nb peaks : {}'.format(n_peaks))

    # threshold = thr.estimate_threshold_monte_carlo_joblib(n_peaks, Ni, Nj, Nk, N_simulations=5000, sigma=sigma)
    threshold = 0.0007
    am.plot_activity_map(avg_img, glass_brain=False, threshold=0.0007)
    am.plot_activity_map(var_img, glass_brain=False, threshold=0.000007)

    # Step 3 : Covariance matrix between voxels
    maps_LD = am.Maps(keyword, sigma=None, reduce=5, normalize=False)
    cov_matrix = maps_LD.cov()
    print(cov_matrix)
    cov.plot_cov_matrix_brain(cov_matrix, maps_LD.Ni, maps_LD.Nj, maps_LD.Nk, maps_LD.affine, threshold=50)
