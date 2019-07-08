import matplotlib
print(matplotlib.get_backend())
matplotlib.use('TkAgg')
import numpy as np
import nibabel as nib
from nilearn import plotting
from scipy.ndimage import gaussian_filter
from time import time
from matplotlib import pyplot as plt
matplotlib.use('TkAgg')
import scipy
import multiprocessing
from joblib import Parallel, delayed
import copy
# import ray

from globals import mem, Ni, Nj, Nk, coordinates, corpus_tfidf, affine, inv_affine, gray_mask
from builds import encode_feature, decode_feature, encode_pmid, decode_pmid
from tools import print_percent, index_3D_to_1D

matplotlib.use('TkAgg')
print(matplotlib.get_backend())

def build_activity_map_from_pmid(pmid, sigma=1):
    '''
        Given a pmid, build its corresponding activity map

        pmid : integer found in pmids.txt
        sigma : std used in gaussian blurr
    '''

    # coordinates = pd.read_csv(input_path+'coordinates.csv')
    stat_img_data = np.zeros((Ni,Nj,Nk)) # Building blank stat_img with MNI152's shape

    # For each coordinates found in pmid (in mm), compute its corresponding voxels coordinates
    # and note it as activated
    for index, row in coordinates.loc[coordinates['pmid'] == pmid].iterrows():
        x, y, z = row['x'], row['y'], row['z']
        # i, j, k, _ = np.rint(np.dot(inv_affine, [x, y, z, 1])).astype(int)
        i, j, k = np.minimum(np.floor(np.dot(inv_affine, [x, y, z, 1]))[:-1].astype(int), [Ni-1, Nj-1, Nk-1])
        stat_img_data[i, j, k] += 1
        
    # Add gaussian blurr
    stat_img_data = gaussian_filter(stat_img_data, sigma=sigma)

    return nib.Nifti1Image(stat_img_data, affine)

def plot_activity_map(stat_img, threshold=0., glass_brain=False):
    '''
        Plot stat_img on MNI152 background

        stat_img : Object of Nifti1Image Class
        threshold : min value to display (in percent of maximum)
    '''
    if glass_brain:
        plotting.plot_glass_brain(stat_img, black_bg=True, threshold=threshold)#*np.max(stat_img.get_data()))#, threshold=threshold)#threshold*np.max(stat_img.get_data()))
    else:
        plotting.plot_stat_map(stat_img, black_bg=True, threshold=threshold)#*np.max(stat_img.get_data()))#, threshold=threshold)#threshold*np.max(stat_img.get_data()))
    plotting.show()
    plt.show()

# def build_activity_map_from_keyword(keyword, sigma=1, gray_matter_mask=True):
#     '''
#         From the given keyword, build its metaanalysis activity map from all related pmids.

#         sigma : std of the gaussian blurr
#         gray_matter_mask : specify whether the map is restrained to the gray matter or not

#         return (stat_img, hist_img, n_samples)
#         stat_img : Nifti1Image object, the map where frequencies are added for each activation
#         hist_img : Nifti1Image object, the map where 1 is added for each activation
#         n_samples : nb of pmids related to the keyword
#     '''
#     time0 = time()
    
#     feature_id = encode_feature[keyword]

#     print('Get nonzero pmid')
#     nonzero_pmids = np.array([int(decode_pmid[index]) for index in corpus_tfidf[:, feature_id].nonzero()[0]])
#     n_samples = len(nonzero_pmids)

#     freq_img_data = np.zeros((Ni,Nj,Nk)) # Building blank img with MNI152's shape
#     hist_img_data = np.zeros((Ni,Nj,Nk)) # Building blank img with MNI152's shape
    
#     print('Get nonzero coordinates')
#     nonzero_coordinates = coordinates.loc[coordinates['pmid'].isin(nonzero_pmids)]
#     print('Get pmid')
#     pmids = np.array(nonzero_coordinates['pmid'])
#     print('Build frequencies')
#     frequencies = np.array([corpus_tfidf[encode_pmid[str(pmid)], feature_id] for pmid in nonzero_coordinates['pmid']])
#     print(len(nonzero_coordinates))
#     n_peaks = len(nonzero_coordinates)
#     print('Build coords')
#     coord = np.zeros((n_peaks, 3))
#     coord[:, 0] = nonzero_coordinates['x']
#     coord[:, 1] = nonzero_coordinates['y']
#     coord[:, 2] = nonzero_coordinates['z']

#     print('Build voxel coords')
#     voxel_coords = np.zeros((n_peaks, 3)).astype(int)
#     for k in range(n_peaks):
#         voxel_coords[k, :] = np.minimum(np.floor(np.dot(inv_affine, [coord[k, 0], coord[k, 1], coord[k, 2], 1]))[:-1].astype(int), [Ni-1, Nj-1, Nk-1])

#     print('Building map')
#     for index, value in enumerate(voxel_coords):
#         i, j, k = value
#         hist_img_data[i, j, k] += 1
#         freq_img_data[i, j, k] += frequencies[index]
    

#     if gray_matter_mask:
#         freq_img_data = np.ma.masked_array(freq_img_data, np.logical_not(gray_mask.get_data()))
#         hist_img_data = np.ma.masked_array(hist_img_data, np.logical_not(gray_mask.get_data()))

#     print('Building time : {}'.format(time()-time0))
#     freq_gauss_img_data = gaussian_filter(freq_img_data, sigma=sigma)
#     hist_gauss_img_data = gaussian_filter(hist_img_data, sigma=sigma)

#     freq_gauss_img = nib.Nifti1Image(freq_gauss_img_data, affine)
#     hist_gauss_img = nib.Nifti1Image(hist_gauss_img_data, affine)
#     hist_img = nib.Nifti1Image(hist_img_data, affine)

#     return freq_gauss_img, hist_gauss_img, hist_img, n_samples

# def average_activity_map_by_keyword(keyword, sigma=1, gray_matter_mask=True):
#     freq_gauss_img, hist_gauss_img, hist_img, n_samples = build_activity_map_from_keyword(keyword, sigma=sigma, gray_matter_mask=gray_matter_mask)

#     hist_gauss_data = np.array(hist_gauss_img.get_data())
#     hist_data = np.array(hist_img.get_data())


#     avg_gauss_img = nib.Nifti1Image(hist_gauss_data/n_samples, hist_gauss_img.affine)
#     avg_img = nib.Nifti1Image(hist_data/n_samples, hist_img.affine)

#     print(avg_img.get_data())

#     return avg_gauss_img, avg_img, n_samples

def build_activity_map_from_keyword(keyword, normalize=False, sigma=1., gray_matter_mask=None):
    maps, Ni_r, Nj_r, Nk_r, affine_r = get_all_maps_associated_to_keyword(keyword, normalize=normalize, gray_matter_mask=gray_matter_mask)
    

    sum_map = sum_from_maps(maps)
    print(sum_map)
    sum_data = map_to_data(sum_map, Ni_r, Nj_r, Nk_r)
    sum_data = gaussian_filter(sum_data, sigma=sigma)
    sum_img = data_to_img(sum_data, affine_r)

    n_voxels, _ = maps.shape
    e = scipy.sparse.csr_matrix(np.ones(n_voxels))
    n_peaks = int(e.dot(sum_map)[0, 0])

    return sum_img, n_peaks

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
                # print(normalize)
                maps[count, p] += 1./n_peaks
            else:
                maps[count, p] += 1


        # if normalize:
        #     print(n_peaks)
        #     maps[count, :]/=n_peaks

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
    n_jobs = multiprocessing.cpu_count()-1
    splitted_array = np.array_split(np.array(nonzero_pmids), n_jobs)
    
    results = Parallel(n_jobs=n_jobs, backend='multiprocessing')(delayed(compute_maps)(sub_array, Ni_r, Nj_r, Nk_r, inv_affine_r, normalize, sigma, keyword) for sub_array in splitted_array)
    
    print('Stacking...')
    maps = scipy.sparse.vstack(results)
    
    # Converting to CSR format (more efficient for operations)
    print('Converting from LIL to CSR format...')
    maps = scipy.sparse.csr_matrix(maps).transpose()
    return maps, Ni_r, Nj_r, Nk_r, affine_r


class Maps:
    def __init__(self, keyword=None):
        if keyword != None:
            maps, Ni_r, Nj_r, Nk_r, affine_r = get_all_maps_associated_to_keyword(keyword, normalize=False, reduce=1, sigma=None)
            self.n_voxels, self.n_pmids = maps.shape
        else:
            maps, Ni_r, Nj_r, Nk_r, affine_r = None, None, None, None, None
            self.n_voxels, self.n_pmids = None, None

        self.maps = maps
        self.Ni = Ni_r
        self.Nj = Nj_r
        self.Nk = Nk_r
        self.affine = affine_r

    def copy_header(self, other):
        self.n_voxels, self.n_pmids = other.n_voxels, other.n_pmids
        self.Ni = other.Ni
        self.Nj = other.Nj
        self.Nk = other.Nk
        self.affine = other.affine

    @staticmethod
    def map_to_data(map, Ni, Nj, Nk):
        '''
            Convert a sparse matrix of shape (n_voxels, 1) into a dense 3D numpy array of shape (Ni, Nj, Nk).

            Indexing of map is supposed to have been made Fortran like (first index moving fastest).
        '''
        n_voxels, _ = map.shape

        if n_voxels != Ni*Nj*Nk:
            raise ValueError('Map\'s length ({}) does not match given box ({}, {}, {}) of size {}.'.format(n_voxels, Ni, Nj, Nk, Ni*Nj*Nk))

        return map.toarray().reshape((Ni, Nj, Nk), order='F')

    def to_data(self, map_id=None):
        if map_id!=None:
            return self.map_to_data(self.maps[:, map_id], self.Ni, self.Nj, self.Nk)

        if self.n_pmids > 1:
            raise KeyError('This Maps object contains {} maps, specify which map to convert to data.'.format(self.n_pmids))

        return self.map_to_data(self.maps[0, map_id], self.Ni, self.Nj, self.Nk)

    @staticmethod
    def data_to_img(data, affine):
        '''
            Convert a dense 3D data array into a nibabel Nifti1Image.
        '''
        return nib.Nifti1Image(data, affine)

    @staticmethod
    def map_to_img(map, Ni, Nj, Nk, affine):
        '''
            Convert a sparse matrix of shape (n_voxels, 1) into a nibabel Nifti1Image.

            Ni, Nj, Nk are the size of the box used to index the flattened map matrix.
        '''
        return Maps.data_to_img(Maps.map_to_data(map, Ni, Nj, Nk), affine)

    def to_img(self, map_id=None):
        if map_id!=None:
            return self.map_to_img(self.maps[:, map_id], self.Ni, self.Nj, self.Nk, self.affine)

        if self.n_pmids > 1:
            raise KeyError('This Maps object contains {} maps, specify which map to convert to img.'.format(self.n_pmids))

        return self.map_to_img(self.maps[0, map_id], self.Ni, self.Nj, self.Nk, self.affine)

    def n_peaks(self):
        '''
            Returns a numpy array containing the number of peaks in each maps
        '''
        e = scipy.sparse.csr_matrix(np.ones(self.n_voxels))
        return np.array(e.dot(self.maps).toarray()[0])

    def sum(self):
        '''
            Builds the summed map of the given maps on the second axis.

            maps : sparse CSR matrix of shape (n_voxels, n_pmids) where
                n_voxels is the number of voxels in the box
                n_pmids is the number of pmids

            Returns a sparse CSR matrix of shape (n_voxels, 1) representing the flattened summed up map.
        '''
        e = scipy.sparse.csr_matrix(np.ones(self.n_pmids)).transpose()

        # sum_map = Map()

        return self.maps.dot(e)

    def normalize(self, inplace=False):
        '''
            Normalize each maps separatly so that each maps sums to 1.
        '''
        diag = scipy.sparse.diags(np.power(self.n_peaks(), -1))

        if inplace:
            new_maps = self
        else:
            new_maps = copy.copy(self)

        new_maps.maps = self.maps.dot(diag)

        return new_maps

    # def smooth(self, sigma, inplace=False):
    #     '''

    #     '''



    @staticmethod
    def average(maps):
        '''
            Builds the average map of the given maps on the second axis.

            maps : sparse CSR matrix of shape (n_voxels, n_pmids) where
                n_voxels is the number of voxels in the box
                n_pmids is the number of pmids

            Returns a sparse CSR matrix of shape (n_voxels, 1) representing the flattened average map.
        '''
        _, n_pmids = maps.shape
        e = scipy.sparse.csr_matrix(np.ones(n_pmids)/n_pmids).transpose()

        return maps.dot(e)

    def avg(self):
        return self.average(self.maps)


    @staticmethod
    def variance(maps):
        '''
            Builds the variance map of the given maps on the second axis.

            Returns a sparse CSR matrix of shape (n_voxels, 1) representing the flattened variance map.
        '''
        avg_map = Maps.average(maps)
        maps_squared = maps.multiply(maps) # Squared element wise

        avg_squared_map = Maps.average(maps_squared)
        squared_avg_map = avg_map.multiply(avg_map)

        return avg_squared_map - squared_avg_map

    def var(self):
        return self.variance(self.maps)

    def cov(self):
        '''
            Builds the empirical covariance matrix of the given maps on the second axis.

            Returns a sparse CSR matrix of shape (n_voxels, n_voxels) representing the covariance matrix.
        '''

        # _, n_pmids = maps.shape 

        e = scipy.sparse.csr_matrix(np.ones(self.n_pmids)/self.n_pmids).transpose()

        M1 = self.maps.dot(self.maps.transpose())/self.n_pmids
        M2 = self.maps.dot(e)

        return M1 - M2.dot(M2.transpose())

if __name__ == '__main__':
    pmid = 22266924
    keyword = 'prosopagnosia'
    # keyword = 'schizophrenia'
    sigma = 2.

    # stat_img = build_activity_map_from_pmid(pmid, sigma=sigma)
    # plot_activity_map(stat_img, glass_brain=True, threshold=0.)

    # freq_gauss_img, hist_gauss_img, hist_img, n_samples = build_activity_map_from_keyword(keyword, sigma=sigma, gray_matter_mask=True)
    # print('Nb peaks : {}'.format(nb_peaks))
    # plot_activity_map(hist_gauss_img, glass_brain=False, threshold=0.04)

    # avg_gauss_img, avg_img, n_samples = average_activity_map_by_keyword(keyword, sigma=sigma, gray_matter_mask=True)

    # plot_activity_map(avg_img, glass_brain=False, threshold=0.)

    # maps, Ni_r, Nj_r, Nk_r, affine_r = get_all_maps_associated_to_keyword(keyword, normalize=False, reduce=1, sigma=None)
    # print(maps)
    # # print(maps.shape)
    # # print(Ni_r, Nj_r, Nk_r)

    # maps = normalize_maps(maps)
    # sum_map = sum_from_maps(maps)
    # sum_data = map_to_data(sum_map, Ni_r, Nj_r, Nk_r)
    # sum_data = gaussian_filter(sum_data, sigma=sigma)
    # sum_img = data_to_img(sum_data, affine_r)

    # # sum_img, n_peaks = build_activity_map_from_keyword(keyword, normalize=False, sigma=sigma)

    # # plot_activity_map(sum_img, threshold=0.04)
    # plot_activity_map(sum_img, threshold=0.003)

    # plot_activity_map(map_to_img(sum_from_maps(maps), Ni_r, Nj_r, Nk_r, affine_r), threshold=0.4)

    maps = Maps(keyword)
    maps.normalize(inplace=True)
    # maps = maps.normalize()
    maps.to_data(3)
    sum_map = maps.sum()
    # sum_map = maps.var()
    # sum_map = maps.avg()
    # avg_map = maps.avg()
    # img = Maps.map_to_img(avg_map, maps.Ni, maps.Nj, maps.Nk, maps.affine)
    sum_data = Maps.map_to_data(sum_map, maps.Ni, maps.Nj, maps.Nk)
    sum_data = gaussian_filter(sum_data, sigma=sigma)
    sum_img = Maps.data_to_img(sum_data, maps.affine)

    # plot_activity_map(sum_img, threshold=0.000015)
    # plot_activity_map(sum_img, threshold=0.003)
    # plot_activity_map(img, threshold=0.00)

    empty_maps = Maps()
    empty_maps.copy_header(maps)



