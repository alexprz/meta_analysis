import numpy as np
import nibabel as nib
from scipy.ndimage import gaussian_filter
import scipy
import multiprocessing
from joblib import Parallel, delayed

from .globals import mem, Ni, Nj, Nk, coordinates, corpus_tfidf, affine, inv_affine
from .builds import encode_feature, decode_feature, encode_pmid, decode_pmid
from .tools import print_percent, index_3D_to_1D

def build_activity_map_from_pmid(pmid, sigma=1):
    '''
        Given a pmid, build its corresponding activity map

        pmid : integer found in pmids.txt
        sigma : std used in gaussian blurr
    '''
    stat_img_data = np.zeros((Ni,Nj,Nk)) # Building blank stat_img with MNI152's shape

    # For each coordinates found in pmid (in mm), compute its corresponding voxels coordinates
    # and note it as activated
    for index, row in coordinates.loc[coordinates['pmid'] == pmid].iterrows():
        x, y, z = row['x'], row['y'], row['z']
        i, j, k = np.minimum(np.floor(np.dot(inv_affine, [x, y, z, 1]))[:-1].astype(int), [Ni-1, Nj-1, Nk-1])
        stat_img_data[i, j, k] += 1
        
    # Add gaussian blurr
    stat_img_data = gaussian_filter(stat_img_data, sigma=sigma)

    return nib.Nifti1Image(stat_img_data, affine)

def compute_maps(pmids, Ni, Nj, Nk, inv_affine, normalize, sigma, keyword):
    '''
        Given a list of pmids, builds their activity maps (flattened in 1D) on a LIL sparse matrix format.
        Used for multiprocessing in get_all_maps_associated_to_keyword function.

        pmids : list of pmid
        Ni, Nj, Nk : size of the 3D box (used to flatten 3D to 1D indices)
        inv_affine : the affine inverse used to compute voxels coordinates

        Returns sparse LIL matrix of shape (len(pmids), Ni*Nj*Nk) containing all the maps
    '''

    n_pmids = len(pmids)
    maps = scipy.sparse.lil_matrix((n_pmids, Ni*Nj*Nk))

    for count, pmid in enumerate(pmids):

        print_percent(count, n_pmids, prefix='Building maps associated to {} '.format(keyword))
        coordinates_of_interest = coordinates.loc[coordinates['pmid'] == pmid]
        n_peaks = len(coordinates_of_interest)

        for index, row in coordinates_of_interest.iterrows():
            x, y, z = row['x'], row['y'], row['z']
            i, j, k = np.clip(np.floor(np.dot(inv_affine, [x, y, z, 1]))[:-1].astype(int), [0, 0, 0], [Ni-1, Nj-1, Nk-1])
            p = index_3D_to_1D(i, j, k, Ni, Nj, Nk)

            if normalize:
                maps[count, p] += 1./n_peaks
            else:
                maps[count, p] += 1

        if sigma != None:
            data = maps[count, :].toarray().reshape((Ni, Nj, Nk), order='F')
            data = gaussian_filter(data, sigma=sigma)
            maps[count, :] = data.reshape(-1, order='F')

    return maps

@mem.cache
def get_all_maps_associated_to_keyword(keyword, reduce=1, gray_matter_mask=None, normalize=False, sigma=None):
    '''
        Given a keyword, finds every related studies and builds their activation maps.

        gray_matter_mask : if True, only voxels inside gray_matter_mask are taken into account (NOT IMPLEMENTED YET)
        reduce : integer, reducing scale factor. Ex : if reduce=2, aggregates voxels every 2 voxels in each direction.
                Notice that this affects the affine and box size.

        Returns 
            maps: sparse CSR matrix of shape (n_voxels, n_pmids) containing all the related flattenned maps where
                    n_pmids is the number of pmids related to the keyword
                    n_voxels is the number of voxels in the box (may have changed if reduce != 1)
            Ni_r, Nj_r, Nk_r: new box dimension (changed if reduce != 1)
            affine_r: new affine (changed if reduce != 1)
    '''
    # Retrieve keyword related pmids
    nonzero_pmids = np.array([int(decode_pmid[index]) for index in corpus_tfidf[:, encode_feature[keyword]].nonzero()[0]])

    # Computing new box size according to the reducing factor
    Ni_r, Nj_r, Nk_r = np.ceil(np.array((Ni, Nj, Nk))/reduce).astype(int)

    # LIL format allows faster incremental construction of sparse matrix
    maps = scipy.sparse.lil_matrix((len(nonzero_pmids), Ni_r*Nj_r*Nk_r))

    # Changing affine to the new box size
    affine_r = np.copy(affine)
    for i in range(3):
        affine_r[i, i] = affine[i, i]*reduce
    inv_affine_r = np.linalg.inv(affine_r)

    # Multiprocessing maps computation
    n_jobs = multiprocessing.cpu_count()//2
    splitted_array = np.array_split(np.array(nonzero_pmids), n_jobs)
    
    results = Parallel(n_jobs=n_jobs, backend='multiprocessing')(delayed(compute_maps)(sub_array, Ni_r, Nj_r, Nk_r, inv_affine_r, normalize, sigma, keyword) for sub_array in splitted_array)
    
    print('Stacking...')
    maps = scipy.sparse.vstack(results)
    
    # Converting to CSR format (more efficient for operations)
    print('Converting from LIL to CSR format...')
    maps = scipy.sparse.csr_matrix(maps).transpose()
    return maps, Ni_r, Nj_r, Nk_r, affine_r




