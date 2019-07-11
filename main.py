import numpy as np
import pandas as pd

import meta_analysis
from meta_analysis import threshold as thr
from meta_analysis import activity_map as am
from meta_analysis import Ni, Nj, Nk, affine, plotting, Maps

from meta_analysis import coordinates, corpus_tfidf, decode_pmid, encode_feature

def build_df_from_keyword(keyword):
    nonzero_pmids = np.array([int(decode_pmid[index]) for index in corpus_tfidf[:, encode_feature[keyword]].nonzero()[0]])
    df = coordinates[coordinates['pmid'].isin(nonzero_pmids)]
    df['weight'] = 1
    return df

if __name__ == '__main__':
    # Step 1 : Plot activity map from a given pmid
    # pmid = 22266924 
    pmid = 16723214 
    stat_img = am.build_activity_map_from_pmid(pmid, sigma=1.5)
    # plotting.plot_activity_map(stat_img, glass_brain=True, threshold=0.)

    # Step 2
    # keyword = 'prosopagnosia'
    keyword = 'memory'
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
    # # threshold = 0.0007
    # # avg_threshold, var_threshold = 0, 0
    plotting.plot_activity_map(avg_img, glass_brain=False, threshold=avg_threshold)#0.0007)
    plotting.plot_activity_map(var_img, glass_brain=False, threshold=var_threshold)#0.000007)

    # # Step 3 : Covariance matrix between voxels
    # maps_LD = Maps(keyword, sigma=None, reduce=5, normalize=False)
    # cov_matrix = maps_LD.cov()
    # print(cov_matrix)
    # cov.plot_cov_matrix_brain(cov_matrix, maps_LD.Ni, maps_LD.Nj, maps_LD.Nk, maps_LD.affine, threshold=50)

    # df = build_df_from_keyword(keyword)
    # print(df)
    # unique = df['pmid'].unique()
    # # print(unique)
    # index_dict = {k:v for v, k in enumerate(unique)}
    # # print(index_dict)

    # # def add_map_id(row, index_dict):
    # #     row['map_id'] = index_dict[row['pmid']]

    # # df.apply(add_map_id, axis=1, index_dict=index_dict)

    # # print(df)
    # df_gb = df.groupby(['pmid'])
    # print(df_gb)
    # # k = 0
    # def my_print(data):
    #     # print(data.index)
    #     # global k
    #     # df['map_id'][data.index] = k
    #     # print(df.loc[data.index])
    #     pmid = data.iloc[0]['pmid']
    #     data['map_id'] = index_dict[pmid]
    #     # k += 1
    #     # print(data[['x', 'y', 'z']])
    #     # print(data)

    #     return data

    # result = df_gb.apply(my_print)
    # print(result)
    # print(df)

    # df_splitted = np.array_split(result, 4, axis=0)

    # def fill_map(data):
    #     print(data[['x', 'y', 'z']])
    # print(df_splitted[0].groupby('pmid').apply(fill_map))

    # df2 = pd.DataFrame(df_gb)
    # print(df2)
    # df = build_df_from_keyword(keyword)
    # maps = Maps(df)
    # print(maps)


