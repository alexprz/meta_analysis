from nilearn import datasets
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


from meta_analysis import Maps, plotting, print_percent

from tools import build_df_from_keyword, pickle_load, pickle_dump, get_dump_token
from globals import template, gray_mask, pickle_path
from clustering import fit_CanICA, fit_DictLearning, fit_Kmeans, fit_Wards, fit_GroupICA, fit_Model

save_dir = pickle_path+'averaged_maps/'

# _________ATLASES_________ #
atlas_HO_0 = datasets.fetch_atlas_harvard_oxford('sub-maxprob-thr0-2mm')
atlas_HO_25 = datasets.fetch_atlas_harvard_oxford('sub-maxprob-thr25-2mm')
atlas_HO_50 = datasets.fetch_atlas_harvard_oxford('sub-maxprob-thr50-2mm')

n_labels = len(atlas_HO_0['labels'])


# _________CRITERIA_________ #
def pearson_distance(array_ref, array_obs, **kwargs):
    sum_ref = np.sum(array_ref)
    sum_obs = np.sum(array_obs)

    if sum_ref > 0:
        array_ref /= sum_ref

    if sum_obs > 0:
        array_obs /= sum_obs

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
    return goodness_of_fit_map(maps.to_img(), maps.to_img_atlas(ignore_bg=False), criterion, **kwargs)


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


def paiwise_distance(array):
    ni, nj = array.shape

    dist = np.zeros((ni, ni))
    for k in range(ni):
        for l in range(k+1, nj):
            d = np.linalg.norm(array[:, k]-array[:, l])
            dist[k, l] = d
            dist[l, k] = d

    return dist


# def benchmark2(ref_maps, atlas_dict, N_sim, n_peaks, sigma, verbose=False):
#     maps = Maps.copy_header(ref_maps)
#     maps.randomize(n_peaks*np.ones(N_sim), p=ref_maps, inplace=True)
#     maps.smooth(sigma, verbose=True)

#     maps_array_2d = maps.maps.to_array()



#     df_list = []
#     k, n_tot = 0, len(atlas_dict)
#     for name, atlas in atlas_dict.items():
#         print_percent(k, n_tot, string='Benchmarking atlas {1} out of {2}... {0}%', rate=0, prefix='Benchmark 2', verbose=verbose)

#         maps.apply_atlas(atlas, inplace=True)


#         df_list.append(['Correlation distance', name, score])

#     return pd.DataFrame(df_list, columns=['Criterion', 'Atlas', 'Value'])

if __name__ == '__main__':
    keyword = 'language'
    sigma = 2.

    train_proportion = 0.5
    random_state = 0
    n_components = n_labels-1
    alpha = 0.1

    load = True
    tag = '{}-sigma-{}-{}-components-RS-{}'.format(keyword, sigma, n_components, random_state)
    avg_maps_tag = '{}-sigma-{}-normalized'.format(keyword, sigma)

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

    file_path = pickle_path+get_dump_token(tag=avg_maps_tag)
    maps = pickle_load(file_path)
    if maps is None:
        df = build_df_from_keyword(keyword)
        maps = Maps(df, template=template, groupby_col='pmid', mask=gray_mask, verbose=True)
        maps.smooth(sigma, inplace=True, verbose=True)
        maps.normalize(inplace=True)
        pickle_dump(maps, file_path)

    maps_train, maps_test = maps.split(prop=train_proportion, random_state=random_state)

    maps_uniform = Maps(1./maps.n_voxels*np.ones((maps.n_voxels, 1)), Ni=maps.Ni, Nj=maps.Nj, Nk=maps.Nk, affine=maps.affine)
    maps_avg = (1-alpha)*maps_test.avg() + alpha*maps_uniform

    # plotting.plot_activity_map(maps_avg.to_img(), threshold=(1-alpha)*1e-5)
    # plt.show()

    # exit()

    array = maps_avg.to_array()
    print(len(array[array == 0.]))

    # plotting.plot_activity_map(maps_avg.to_img())
    # plt.show()
    # exit()

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
    maps_thr = Maps(array_avg > (1-alpha)*1e-6, template=template)
    atlas_mean = maps_thr.to_atlas()
    atlas_null = Maps(np.zeros(array_avg.shape), template=template).to_atlas()

    atlas_CanICA_thr = Maps(CanICA_imgs, template=template, mask=maps_thr.to_img()).to_atlas()
    atlas_Ward_thr = Maps(Ward_imgs, template=template, mask=maps_thr.to_img()).to_atlas()
    atlas_Kmeans_thr = Maps(Kmeans_imgs, template=template, mask=maps_thr.to_img()).to_atlas()

    # labels, maps = atlas_mean['labels'], atlas_mean['maps']
    # print(labels)
    # print(Maps(maps))
    # exit()

    # maps_null = maps_avg.apply_atlas(atlas_null)
    # print(maps_null._atlas.n_labels)
    # print(maps_null._maps_atlas)
    # print(maps_null.to_array_atlas()[0, 0, 0])
    # print(maps_null.to_array())
    # exit()
    # plotting.plot_activity_map(maps_null.to_img_atlas())
    # plt.show()
    # exit()

    atlas_dict = {
        'Harvard Oxford 0': atlas_HO_0,
        'Harvard Oxford 25': atlas_HO_25,
        'Harvard Oxford 50': atlas_HO_50,
        'CanICA {} components'.format(n_components): atlas_CanICA,
        'CanICA thresholded {} components'.format(n_components): atlas_CanICA_thr,
        # 'Dict Learning {} components'.format(n_components): atlas_DictLearning,
        'Ward {} components'.format(n_components): atlas_Ward,
        'Ward thresholded {} components'.format(n_components): atlas_Ward_thr,
        'Kmeans {} components'.format(n_components): atlas_Kmeans,
        'Kmeans thresholded {} components'.format(n_components): atlas_Kmeans_thr,
        'Mean thresholded': atlas_mean,
        # 'Null': atlas_null,
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
