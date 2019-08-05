from nilearn import datasets
import matplotlib
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from meta_analysis import Maps, plotting, print_percent

from tools import build_df_from_keyword
from globals import template, gray_mask

#_________CRITERIA_________#
def pearson_distance(array_ref, array_obs, **kwargs):
    array_ref /= np.sum(array_ref)
    array_obs /= np.sum(array_obs)
    
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.sum(np.nan_to_num(np.true_divide(np.power(array_obs - array_ref, 2), array_ref), posinf=0))

def RMS(array_ref, array_obs, **kwargs):
    return np.sqrt(np.mean(np.power(array_ref-array_obs, 2)))

def Mahalanobis(array_ref, array_obs, **kwargs):
    x = array_obs - array_ref
    inv_S = np.real(np.linalg.inv(kwargs['S']))
    return np.sqrt(np.dot(x.T, np.dot(inv_S, x)))

def zero(arr, arr2, **kwargs):
    return 0

#_________GOODNESS_OF_FIT_________#
def goodness_of_fit_map(img_ref, img_obs, criterion, **kwargs):
    array_ref, array_obs = img_ref.get_fdata(), img_obs.get_fdata()

    if array_ref.shape != array_obs.shape:
        raise ValueError('Images dimensions missmatchs. Img1 : {}, Img2 : {}'.format(array_ref.shape, array_obs.shape))

    return criterion(array_ref, array_obs, **kwargs)

def benchmark_atlas(maps, atlas, criterion, verbose=False, **kwargs):
    maps.apply_atlas(atlas, inplace=True)
    return goodness_of_fit_map(maps.to_img(), maps.to_img_atlas(ignore_bg=True), criterion, **kwargs)

def benchmark(maps, atlas_dict, criteria, verbose=False, **kwargs):
    
    df_list = []
    k, n_tot = 0, len(atlas_dict)*len(criteria)

    for criterion in criteria:
        for name, atlas in atlas_dict.items():
            print_percent(k, n_tot, string='Benchmarking atlas {1} out of {2} : {0:.2f}%...', rate=0, verbose=verbose)
            score = benchmark_atlas(maps, atlas, criterion, verbose=verbose, **kwargs)
            df_list.append([criterion.__name__, name, score])
            k += 1

    return pd.DataFrame(df_list, columns=['Criterion', 'Atlas', 'Value'])



if __name__ == '__main__':
    keyword = 'language'
    sigma = 2.

    df = build_df_from_keyword(keyword)
    maps = Maps(df, template=template, groupby_col='pmid', mask=gray_mask, verbose=True)

    atlas_dict = {
        'Harvard Oxford': datasets.fetch_atlas_harvard_oxford('cort-maxprob-thr25-2mm'),
        'Harvard Oxford 2': datasets.fetch_atlas_harvard_oxford('sub-maxprob-thr0-2mm'),
        'Harvard Oxford 3': datasets.fetch_atlas_harvard_oxford('sub-maxprob-thr50-2mm'),
    }

    criteria = [
        pearson_distance,
        RMS,
        Mahalanobis
    ]

    # print(type(maps._maps.toarray()[0, 0]))

    # array = np.zeros((maps.n_voxels, maps.n_maps), dtype=np.float32)
    # print(array.shape)
    # # print(type(maps._maps.toarray()[0, 0]))
    # maps._maps.toarray(out=array)

    # print(type(array[0, 0]))

    # exit()

    S = maps.cov(atlas=False, shrink='LW', ignore_bg=True, verbose=True)
    benchmarks = benchmark(maps.avg(), atlas_dict, criteria, verbose=True, S=S)
    
    print(benchmarks)
    sns.catplot(x='Criterion', y='Value', hue='Atlas', data=benchmarks, height=6, kind="bar", palette="muted")
    plt.show()
