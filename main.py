import numpy as np
import pandas as pd

import meta_analysis
from meta_analysis import threshold as thr
from meta_analysis import plotting, Maps

from globals import coordinates, corpus_tfidf, Ni, Nj, Nk, affine, inv_affine
from tools import build_activity_map_from_pmid, build_df_from_keyword

if __name__ == '__main__':
    # Step 1 : Plot activity map from a given pmid
    pmid = 16723214 
    stat_img = build_activity_map_from_pmid(pmid, sigma=1.5)
    plotting.plot_activity_map(stat_img, glass_brain=True, threshold=0.)

    # Step 2
    keyword = 'prosopagnosia'
    # keyword = 'memory'
    # keyword = 'language'
    # keyword = 'schizophrenia'
    sigma = 2.
    df = build_df_from_keyword(keyword)
    
    maps_HD = Maps(df, Ni=Ni, Nj=Nj, Nk=Nk, affine=affine, reduce=1, groupby_col='pmid')
    

    avg, var = maps_HD.iterative_smooth_avg_var(sigma=sigma, verbose=True)
    avg_img, var_img = avg.to_img(), var.to_img()

    n_peaks = int(maps_HD.sum())
    n_maps = maps_HD.n_maps

    print('Nb peaks : {}'.format(n_peaks))
    print('Nb maps : {}'.format(n_maps))

    avg_threshold, var_threshold = thr.avg_var_threshold_MC(n_peaks, n_maps, maps_HD.Ni, maps_HD.Nj, maps_HD.Nk, N_simulations=5, sigma=sigma, verbose=True)

    plotting.plot_activity_map(avg_img, glass_brain=False, threshold=avg_threshold)
    plotting.plot_activity_map(var_img, glass_brain=False, threshold=var_threshold)

    # # Step 3 : Covariance matrix between voxels
    maps_LD = Maps(df, Ni=Ni, Nj=Nj, Nk=Nk, affine=affine, reduce=5, groupby_col='pmid')
    maps_LD.smooth(sigma=sigma, verbose=True)
    cov_matrix = maps_LD.cov()
    print(cov_matrix)
    plotting.plot_cov_matrix_brain(cov_matrix, maps_LD.Ni, maps_LD.Nj, maps_LD.Nk, maps_LD.affine, threshold=0.2)


