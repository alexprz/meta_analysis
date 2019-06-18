import numpy as np
import pandas as pd
from nilearn import datasets, plotting
from scipy.ndimage import gaussian_filter
import scipy.sparse
import nibabel as nib
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')


input_path = 'minimal/'

# Loading MNI152 background and parameters (shape, affine...)
bg_img = datasets.load_mni152_template()
Ni, Nj, Nk = bg_img.shape
affine = bg_img.affine
inv_affine = np.linalg.inv(affine)


def build_activity_map_from_pmid(pmid, sigma=1):
    '''
        Given a pmid, build its corresponding activity map

        pmid : integer found in pmids.txt
        sigma : std used in gaussian blurr
    '''

    coordinates = pd.read_csv(input_path+'coordinates.csv')
    stat_img_data = np.zeros((Ni,Nj,Nk)) # Building blank stat_img with MNI152's shape

    # For each coordinates found in pmid (in mm), compute its corresponding voxels coordinates
    # and note it as activated
    for index, row in coordinates.loc[coordinates['pmid'] == pmid].iterrows():
        x, y, z = row['x'], row['y'], row['z']
        i, j, k, _ = np.rint(np.dot(inv_affine, [x, y, z, 1])).astype(int)
        stat_img_data[i, j, k] = 1
        
    # Add gaussian blurr
    stat_img_data = gaussian_filter(stat_img_data, sigma=sigma)

    return nib.Nifti1Image(stat_img_data, affine)

def plot_activity_map(stat_img, threshold=0.1):
    '''
        Plot stat_img on MNI152 background

        stat_img : Object of Nifti1Image Class
        threshold : min value to display (in percent of maximum)
    '''
    plotting.plot_glass_brain(stat_img, black_bg=True, threshold=threshold*np.max(stat_img.get_data()))
    plotting.show()

def build_index(file_name):
    decode = dict(enumerate(line.strip() for line in open(input_path+file_name)))
    encode = {v: k for k, v in decode.items()}
    
    return encode, decode


if __name__ == '__main__':
    # Step 1 : Plot activity map from a given pmid
    # pmid = 22266924 
    # stat_img = build_activity_map_from_pmid(pmid, sigma=1.5)
    # plot_activity_map(stat_img)


    # Step 2
    corpus_tfidf = scipy.sparse.load_npz(input_path+'corpus_tfidf.npz')

    # feature_file = open(input_path+'feature_names.txt')
    # feature_index = dict()
    # k = 0
    # for line in feature_file:
    #     # print(line.strip())
    #     feature_index[line.strip()] = k
    #     k += 1

    # pmids_file = open(input_path+'pmids.txt')
    # pmids_index = dict()
    # k = 0
    # for line in pmids_file:
    #     # print(line.strip())
    #     pmids_index[int(line.strip())] = k
    #     k += 1
    
    # print(feature_index)
    # print(pmids_index)
    # print(corpus_tfidf[13880, 6179])
    # print(corpus_tfidf[pmids_index[15522765], feature_index['a1']])

    # def build_activity_map_from_feature(feature_name, sigma=1):
    #     feature_id = feature_index[feature_name]
    #     print(feature_id)
    #     frequencies = corpus_tfidf[corpus_tfidf[:, feature_id].nonzero()[0], feature_id]
    #     print(frequencies)

    # build_activity_map_from_feature('memory')

    encode_feature, decode_feature = build_index('feature_names.txt')
    encode_pmid, decode_pmid = build_index('pmids.txt')


