import numpy as np
import pandas as pd
import copy
import pickle
import scipy
import nilearn

import meta_analysis
from meta_analysis import threshold as thr
from meta_analysis import plotting, Maps

from globals import coordinates, corpus_tfidf, Ni, Nj, Nk, affine, inv_affine, pickle_path, gray_mask, atlas, template
from tools import build_activity_map_from_pmid, build_df_from_keyword, build_avg_map_corpus

import matplotlib
matplotlib.use('MacOsx')
from matplotlib import pyplot as plt

if __name__ == '__main__':
    keyword = 'prosopagnosia'
    sigma = 2.
    N_sim = 5

    df = build_df_from_keyword(keyword)
    
    maps = Maps(df, template=template, groupby_col='pmid', mask=gray_mask, atlas=atlas)
    
    print(maps)

    avg = maps.avg()
    var = maps.var()

    plotting.plot_activity_map(avg.to_img_atlas(), title='Avg atlas')
    plotting.plot_activity_map(var.to_img_atlas(), title='Var atlas')
    plt.show()

    avg_smoothed, var_smoothed = maps.iterative_smooth_avg_var(sigma=sigma, verbose=True)

    n_peaks = len(df.index)
    threshold = thr.threshold_MC(n_peaks, maps.n_maps, maps._Ni, maps._Nj, maps._Nk, N_simulations=N_sim, sigma=sigma, verbose=True, mask=gray_mask)

    plotting.plot_activity_map(avg_smoothed.to_img(), title='Avg smoothed thresholded', threshold=threshold['avg'])
    plotting.plot_activity_map(var_smoothed.to_img(), title='Var smoothed thresholded', threshold=threshold['var'])
    plt.show()
