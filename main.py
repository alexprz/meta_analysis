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
    sigma = 2.
    stat_img, hist_img, nb_peaks = am.build_activity_map_from_keyword(keyword, sigma=sigma, gray_matter_mask=True)
    print('Nb peaks : {}'.format(nb_peaks))

    threshold = thr.estimate_threshold_monte_carlo_joblib(nb_peaks, Ni, Nj, Nk, N_simulations=100, sigma=sigma)
    am.plot_activity_map(hist_img, glass_brain=False, threshold=threshold)

    # Step 3 : Covariance matrix between voxels
    print(cov.build_covariance_matrix_from_keyword(keyword, sigma=sigma))
