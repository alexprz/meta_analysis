import numpy as np
import pandas as pd
from nilearn import datasets, plotting
from scipy.ndimage import gaussian_filter
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

    # Building blank stat_img with MNI152's shape
    stat_img_data = np.zeros((Ni,Nj,Nk))

    # For each coordinates found in pmid (in mm), compute its corresponding voxels coordinates
    # and note it as activated
    for index, row in coordinates.loc[coordinates['pmid'] == pmid].iterrows():
        x, y, z = row['x'], row['y'], row['z']
        i, j, k, _ = np.rint(np.dot(inv_affine, np.array([x, y, z, 1])))
        i, j, k = int(i), int(j), int(k)
        stat_img_data[i, j, k] = 1
        
    # Add gaussian blurr and build stat_img
    stat_img_data = gaussian_filter(stat_img_data, sigma=sigma)

    return nib.Nifti1Image(stat_img_data, affine)

def plot_activity_map(stat_img, threshold=0.1):
    '''
        Plot stat_img on MNI152 background

        stat_img : Object of Nifti1Image Class
        threshold : min value to display (in percent of maximum)
    '''
    plotting.plot_stat_map(stat_img, bg_img=bg_img, threshold=threshold*np.max(stat_img.get_data()))
    plotting.show()


if __name__ == '__main__':
    stat_img = build_activity_map_from_pmid(23966927, sigma=1.5)
    plot_activity_map(stat_img)


