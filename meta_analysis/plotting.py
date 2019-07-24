import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from nilearn import plotting
import matplotlib
matplotlib.use('TkAgg')

def plot_activity_map(stat_img, threshold=0., glass_brain=False):
    '''
        Plot stat_img on MNI152 background

        stat_img : Object of Nifti1Image Class
        threshold : min value to display (in percent of maximum)
    '''
    if glass_brain:
        display = plotting.plot_glass_brain(stat_img, black_bg=True, threshold=threshold)#*np.max(stat_img.get_data()))#, threshold=threshold)#threshold*np.max(stat_img.get_data()))
    else:
        display = plotting.plot_stat_map(stat_img, black_bg=True, threshold=threshold)#*np.max(stat_img.get_data()))#, threshold=threshold)#threshold*np.max(stat_img.get_data()))

    return display

def plot_matrix_heatmap(M):
    sns.heatmap(M)
    plt.show()

def plot_cov_matrix_brain(M, Ni, Nj, Nk, affine, threshold=None):
    
    n_voxels, _ = M.shape
    coords = np.zeros((Ni, Nj, Nk, 3)).astype(int)

    for k in range(Ni):
         coords[k, :, :, 0] = k
    for k in range(Nj):
         coords[:, k, :, 1] = k
    for k in range(Nk):
         coords[:, :, k, 2] = k

    coords = coords.reshape((-1, 3), order='F')
    coords_world = np.zeros(coords.shape)

    for k in range(coords.shape[0]):
        coords_world[k, :] = np.dot(affine, [coords[k, 0], coords[k, 1], coords[k, 2], 1])[:-1]

    return plotting.plot_connectome(M, coords_world, node_size=5, node_color='black', edge_threshold=threshold*np.max(M))
