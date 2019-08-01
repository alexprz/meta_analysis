import numpy as np
import pandas as pd
import copy
import pickle
import scipy
import nilearn
from matplotlib import pyplot as plt
import seaborn as sns
import nibabel as nib

import meta_analysis
from meta_analysis import threshold as thr
from meta_analysis import plotting, Maps

from globals import coordinates, corpus_tfidf, Ni, Nj, Nk, affine, inv_affine, pickle_path, gray_mask, atlas, template
from tools import build_activity_map_from_pmid, build_df_from_keyword, build_avg_map_corpus

from benchmark import benchmark, pearson_distance

if __name__ == '__main__':
    # keyword = 'bilinguals'
    keyword = 'language'
    # keyword = 'prosopagnosia'
    sigma = 2.
    N_sim = 5


    # df = build_df_from_keyword(keyword)
    
    # maps = Maps(df, template=template, groupby_col='pmid', mask=gray_mask, atlas=atlas)
    # maps.smooth(sigma=sigma, inplace=True, verbose=True)

    # null_atlas = {'maps': nib.Nifti1Image(np.zeros((Ni, Nj, Nk)), affine), 'labels': ['r0']}

    # maps_atlas = maps.apply_atlas(null_atlas)
    # img = maps_atlas.avg().to_img_atlas(ignore_bg=False)
    # print(img.get_data().shape)
    # plotting.plot_activity_map(img)
    # plt.show()


    # atlas_dict = {
    #     'Harvard Oxford': nilearn.datasets.fetch_atlas_harvard_oxford('cort-maxprob-thr25-2mm'),
    #     # 'Harvard Oxford 2': nilearn.datasets.fetch_atlas_harvard_oxford('sub-maxprob-thr0-2mm'),
    #     'Harvard Oxford 3': nilearn.datasets.fetch_atlas_harvard_oxford('sub-maxprob-thr50-2mm'),
    #     # 'My atlas !': atlas_homemade,
    #     'Null atlas': null_atlas,
    # }

    # criteria = [
    #     pearson_distance
    # ]



    # benchmarks = benchmark(maps.avg(), atlas_dict, criteria, verbose=True)

    # print(benchmarks)

    # sns.catplot(x='Criterion', y='Value', hue='Atlas', data=benchmarks, height=6, kind="bar", palette="muted")
    # plt.show()

    # exit()

    df = build_df_from_keyword(keyword)
    
    maps = Maps(df, template=template, groupby_col='pmid', mask=gray_mask, atlas=atlas)
    maps.smooth(sigma=sigma, inplace=True, verbose=True)
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

    # adhd_dataset = datasets.fetch_adhd(n_subjects=20)
    # func_filenames = adhd_dataset.func

    # print(func_filenames)

    # print('Loading files')
    # maps = Maps(func_filenames)
    from nilearn.decomposition import DictLearning, CanICA

    n_components=5

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
    # dict_learning.fit(imgs)
    canica.fit(imgs)

    # components_img = dict_learning.components_img_
    components_img = canica.components_img_

    print(components_img.shape)

    # nilearn.plotting.plot_prob_atlas(components_img, view_type='filled_contours',
    #                          title='CanICA')
    # nilearn.plotting.show()

    # print(components_img.get_fdata())
    # print(np.histogram(components_img.get_fdata()))

    maps_atlas_homemade = Maps(components_img, template=template)

    atlas_homemade = maps_atlas_homemade.to_atlas(verbose=True)

    print(atlas_homemade['maps'])

    # plotting.plot_activity_map(atlas_homemade['maps'], title='Generated')
    # plt.show()



    # plotting.plot_activity_map(avg_smoothed.to_img(), title='Avg smoothed thresholded', threshold=threshold['avg'])
    # plotting.plot_activity_map(var_smoothed.to_img(), title='Var smoothed thresholded', threshold=threshold['var'])
    # plt.show()

    # cov, labels = maps.cov()
    # nilearn.plotting.plot_matrix(cov, labels=labels)
    # nilearn.plotting.show()

    null_atlas = {'maps': nib.Nifti1Image(np.zeros((Ni, Nj, Nk)), affine), 'labels': ['r0']}

    atlas_dict = {
        'Harvard Oxford cort-maxprob-thr0-2mm': nilearn.datasets.fetch_atlas_harvard_oxford('cort-maxprob-thr0-2mm'),
        'Harvard Oxford cort-maxprob-thr25-2mm': nilearn.datasets.fetch_atlas_harvard_oxford('cort-maxprob-thr25-2mm'),
        # 'Harvard Oxford 2': nilearn.datasets.fetch_atlas_harvard_oxford('sub-maxprob-thr0-2mm'),
        'Harvard Oxford cort-maxprob-thr50-2mm': nilearn.datasets.fetch_atlas_harvard_oxford('sub-maxprob-thr50-2mm'),
        'Generated atlas (CanICA-{}components)'.format(n_components): atlas_homemade,
        # 'Null atlas': null_atlas,
    }

    for name, atlas in atlas_dict.items():
        plotting.plot_activity_map(atlas['maps'], title=name)

    print(atlas_dict['Harvard Oxford cort-maxprob-thr25-2mm']['labels'])

    criteria = [
        pearson_distance
    ]

    benchmarks = benchmark(maps.avg(), atlas_dict, criteria, verbose=True)

    print(benchmarks)

    # fig, ax = plt.subplots()
    # ax.set(yscale="log")
    sns.catplot(x='Criterion', y='Value', hue='Atlas', data=benchmarks, height=6, kind="bar", palette="muted").set(yscale="log")
    plt.title('Atlas benchmark on \'{}\' keyword'.format(keyword))
    plt.show()