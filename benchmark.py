from nilearn import datasets
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


from meta_analysis import Maps, plotting, print_percent

from tools import build_df_from_keyword
from globals import template, gray_mask
from clustering import fit_CanICA, fit_DictLearning, fit_Kmeans, fit_Wards, fit_GroupICA, fit_Model

# _________ATLASES_________ #
atlas_HO_0 = datasets.fetch_atlas_harvard_oxford('sub-maxprob-thr0-2mm')
atlas_HO_25 = datasets.fetch_atlas_harvard_oxford('sub-maxprob-thr25-2mm')
atlas_HO_50 = datasets.fetch_atlas_harvard_oxford('sub-maxprob-thr50-2mm')

n_labels = len(atlas_HO_0['labels'])


# _________CRITERIA_________ #
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


# _________GOODNESS_OF_FIT_________ #
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
            print_percent(k, n_tot, string='Benchmarking atlas {1} out of {2} : {0:.2f}%...', rate=0, verbose=verbose, prefix='Benchmark')
            score = benchmark_atlas(maps, atlas, criterion, verbose=verbose, **kwargs)
            df_list.append([criterion.__name__, name, score])
            k += 1

    return pd.DataFrame(df_list, columns=['Criterion', 'Atlas', 'Value'])


if __name__ == '__main__':
    keyword = 'language'
    sigma = 2.

    train_proportion = 0.5
    random_state = 0
    n_components = n_labels-1
    tag = '{}-sigma-{}-{}-components-RS-{}'.format(keyword, sigma, n_components, random_state)
    load = True

    params_CanICA = {
        'n_components':  n_components
    }

    params_DictLearning = {
        'n_components': n_components
    }

    params_Wards = {
        'n_parcels': n_components
    }

    params_Kmeans = {
        'n_parcels': n_components
    }

    df = build_df_from_keyword(keyword)
    maps = Maps(df, template=template, groupby_col='pmid', mask=gray_mask, verbose=True)
    maps.smooth(sigma, inplace=True, verbose=True)
    maps_train, maps_test = maps.split(prop=train_proportion, random_state=random_state)

    maps_avg = maps_test.avg()

    # imgs_3D_list = maps_train.to_img(sequence=True, verbose=True)
    imgs_4D = maps_train.to_img()

    CanICA_imgs = fit_Model(fit_CanICA, imgs_4D, params_CanICA, tag=tag, load=load).components_img_
    # DictLearning_imgs = fit_Model(fit_DictLearning, maps.to_img(), params_DictLearning, tag=tag, load=load).components_img_
    Ward_imgs = fit_Model(fit_Wards, imgs_4D, params_Wards, tag=tag, load=load).labels_img_
    Kmeans_imgs = fit_Model(fit_Kmeans, imgs_4D, params_Kmeans, tag=tag, load=load).labels_img_

    atlas_CanICA = Maps(CanICA_imgs, template=template).to_atlas()
    # atlas_DictLearning = Maps(DictLearning_imgs, template=template).to_atlas()
    atlas_Ward = Maps(Ward_imgs, template=template).to_atlas()
    atlas_Kmeans = Maps(Kmeans_imgs, template=template).to_atlas()

    array_avg = maps_avg.to_array()
    atlas_mean = Maps(array_avg > 0.0003, template=template).to_atlas()

    atlas_dict = {
        'Harvard Oxford 0': atlas_HO_0,
        'Harvard Oxford 25': atlas_HO_25,
        'Harvard Oxford 50': atlas_HO_50,
        'CanICA {} components'.format(n_components): atlas_CanICA,
        # 'Dict Learning {} components'.format(n_components): atlas_DictLearning,
        'Ward {} components'.format(n_components): atlas_Ward,
        'Kmeans {} components'.format(n_components): atlas_Kmeans,
        'Mean thresholded': atlas_mean,
    }

    criteria = [
        pearson_distance
    ]

    for name, atlas in atlas_dict.items():
        plotting.plot_activity_map(atlas['maps'], title=name)

    benchmarks = benchmark(maps_avg, atlas_dict, criteria, verbose=True)

    print(benchmarks)
    sns.catplot(x='Criterion', y='Value', hue='Atlas', data=benchmarks, height=6, kind="bar", palette="muted").set(yscale="log")
    plt.title('Atlas benchmark on \'{}\' keyword'.format(keyword))
    plt.show()
