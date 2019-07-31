import numpy as np
import pandas as pd
import copy
import pickle
import scipy
import nilearn
from matplotlib import pyplot as plt
import seaborn as sns

import meta_analysis
from meta_analysis import threshold as thr
from meta_analysis import plotting, Maps

from globals import coordinates, corpus_tfidf, Ni, Nj, Nk, affine, inv_affine, pickle_path, gray_mask, atlas, template
from tools import build_activity_map_from_pmid, build_df_from_keyword, build_avg_map_corpus

from benchmark import benchmark, pearson_distance

if __name__ == '__main__':
    keyword = 'language'
    sigma = 2.
    N_sim = 5

    # df = build_df_from_keyword(keyword)
    
    # maps = Maps(df, template=template, groupby_col='pmid', mask=gray_mask, atlas=atlas)
    
    # print(maps)

    # avg = maps.avg()
    # var = maps.var()

    # plotting.plot_activity_map(avg.to_img_atlas(), title='Avg atlas')
    # plotting.plot_activity_map(var.to_img_atlas(), title='Var atlas')
    # plt.show()

    # avg_smoothed, var_smoothed = maps.iterative_smooth_avg_var(sigma=sigma, verbose=True)

    # n_peaks = len(df.index)
    # threshold = thr.threshold_MC(n_peaks, maps.n_maps, maps._Ni, maps._Nj, maps._Nk, N_simulations=N_sim, sigma=sigma, verbose=True, mask=gray_mask)

    # plotting.plot_activity_map(avg_smoothed.to_img(), title='Avg smoothed thresholded', threshold=threshold['avg'])
    # plotting.plot_activity_map(var_smoothed.to_img(), title='Var smoothed thresholded', threshold=threshold['var'])
    # plt.show()

    # cov, labels = maps.cov()
    # nilearn.plotting.plot_matrix(cov, labels=labels)
    # nilearn.plotting.show()

    # df = build_df_from_keyword(keyword)
    
    # maps = Maps(df, template=template, groupby_col='pmid', mask=gray_mask, atlas=atlas)

    # maps.smooth(sigma=2, inplace=True, verbose=True)

    # plotting.plot_activity_map(maps.avg().to_img())
    # plt.show()

    # imgs = maps.to_img(sequence=True, verbose=True)

    # print(imgs)

    # for k in range(5):
    #     plotting.plot_activity_map(imgs[k])

    # plt.show()

    from nilearn import datasets

    adhd_dataset = datasets.fetch_adhd(n_subjects=20)
    func_filenames = adhd_dataset.func

    print(func_filenames)

    print('Loading files')
    maps = Maps(func_filenames)
    from nilearn.decomposition import DictLearning, CanICA

    n_components=3

    # dict_learn = DictLearning(n_components=n_components, smoothing_fwhm=6.,
    #                       memory="nilearn_cache", memory_level=2,
    #                       random_state=0)

    dict_learning = DictLearning(n_components=n_components,
                                 memory="nilearn_cache", memory_level=2,
                                 verbose=1,
                                 random_state=0,
                                 n_epochs=1,
                                 mask_strategy='template',
                                 n_jobs=-2)
    canica = CanICA(n_components=n_components,
                    memory="nilearn_cache", memory_level=2,
                    threshold=3.,
                    n_init=1,
                    verbose=1,
                    mask_strategy='template',
                    n_jobs=-2)

    # # print('converting')
    # # imgs = maps.avg().to_img(sequence=False)
    # # print(imgs)
    # print('training')
    imgs = maps.to_img(sequence=True, verbose=True)
    dict_learning.fit(imgs)
    # canica.fit(imgs)

    components_img = dict_learning.components_img_
    components_img = canica.components_img_

    nilearn.plotting.plot_prob_atlas(components_img, view_type='filled_contours',
                             title='CanICA')
    nilearn.plotting.show()




    # plotting.plot_activity_map(avg_smoothed.to_img(), title='Avg smoothed thresholded', threshold=threshold['avg'])
    # plotting.plot_activity_map(var_smoothed.to_img(), title='Var smoothed thresholded', threshold=threshold['var'])
    # plt.show()

    # cov, labels = maps.cov()
    # nilearn.plotting.plot_matrix(cov, labels=labels)
    # nilearn.plotting.show()

    atlas_dict = {
        'Harvard Oxford': nilearn.datasets.fetch_atlas_harvard_oxford('cort-maxprob-thr25-2mm'),
        'Harvard Oxford 2': nilearn.datasets.fetch_atlas_harvard_oxford('sub-maxprob-thr0-2mm'),
        'Harvard Oxford 3': nilearn.datasets.fetch_atlas_harvard_oxford('sub-maxprob-thr50-2mm'),
        # 'AAL': datasets.fetch_atlas_pauli_2017(version='det')
    }

    criteria = [
        pearson_distance
    ]

    benchmarks = benchmark(maps.avg(), atlas_dict, criteria, verbose=True)

    print(benchmarks)

    sns.catplot(x='Criterion', y='Value', hue='Atlas', data=benchmarks, height=6, kind="bar", palette="muted")
    plt.show()