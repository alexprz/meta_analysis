import numpy as np
import pandas as pd
import copy
import pickle
import scipy

import os
here = os.path.dirname(os.path.abspath(__file__))

import meta_analysis
from meta_analysis import threshold as thr
from meta_analysis import plotting, Maps

from globals import coordinates, corpus_tfidf, Ni, Nj, Nk, affine, inv_affine, pickle_path, gray_mask
from tools import build_activity_map_from_pmid, build_df_from_keyword, build_avg_map_corpus

import matplotlib
matplotlib.use('MacOsx')

if __name__ == '__main__':
    # Step 1 : Plot activity map from a given pmid
    pmid = 16723214 
    stat_img = build_activity_map_from_pmid(pmid, sigma=1.5)
    # plotting.plot_activity_map(stat_img, glass_brain=True, threshold=0.)

    # Step 2
    keyword = 'prosopagnosia'
    # keyword = 'memory'
    # keyword = 'language'
    # keyword = 'schizophrenia'
    sigma = 2.
    N_sim = 5
    gray_mask_data = gray_mask.get_data()

    df = build_df_from_keyword(keyword)
    
    maps_HD = Maps(df, Ni=Ni, Nj=Nj, Nk=Nk, affine=affine, reduce=1, groupby_col='pmid', mask=gray_mask_data)
    

    avg, var = maps_HD.iterative_smooth_avg_var(sigma=sigma, verbose=True)
    avg_img, var_img = avg.to_img(), var.to_img()

    n_peaks = int(maps_HD.sum())
    n_maps = maps_HD.n_maps

    # print('Nb peaks : {}'.format(n_peaks))
    # print('Nb maps : {}'.format(n_maps))

    # with open("{}all_maps_avg_sigma_{}.pickle".format(pickle_path, sigma), 'rb') as file:
    #     loaded_avg = pickle.load(file)

    # p = loaded_avg.normalize(inplace=True)

    thresholds = thr.threshold_MC(n_peaks, n_maps, maps_HD.Ni, maps_HD.Nj, maps_HD.Nk, stats=['avg', 'var'], N_simulations=N_sim, sigma=sigma, verbose=True, mask=gray_mask_data)
    # thresholds = thr.threshold_MC(n_peaks, n_maps, maps_HD.Ni, maps_HD.Nj, maps_HD.Nk, stats=['avg', 'var'], N_simulations=N_sim, sigma=sigma, verbose=True, p=p)
    # thresholds = thr.threshold_MC(n_peaks, n_maps, maps_HD.Ni, maps_HD.Nj, maps_HD.Nk, stats=['avg'], N_simulations=N_sim, sigma=sigma, verbose=True)

    avg_threshold = thresholds['avg']
    var_threshold = thresholds['var']

    plotting.plot_activity_map(avg_img, glass_brain=False, threshold=avg_threshold)
    plotting.plot_activity_map(var_img, glass_brain=False, threshold=var_threshold)


    # # Step 3 : Covariance matrix between voxels
    # maps_LD = Maps(df, Ni=Ni, Nj=Nj, Nk=Nk, affine=affine, reduce=5, groupby_col='pmid')
    # maps_LD.smooth(sigma=sigma, verbose=True)
    # cov_matrix = maps_LD.cov()
    # print(cov_matrix)
    # print(np.max(cov_matrix))
    # print(scipy.stats.describe(cov_matrix, axis=None))
    # plotting.plot_cov_matrix_brain(cov_matrix, maps_LD.Ni, maps_LD.Nj, maps_LD.Nk, maps_LD.affine, threshold=0.2)

