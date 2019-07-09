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


class Maps:
    def __init__(self, keyword_or_shape=None, reduce=1, normalize=False, sigma=None, Ni=None, Nj=None, Nk=None, affine=None):
        if isinstance(keyword_or_shape, str):
            maps, Ni, Nj, Nk, affine = get_all_maps_associated_to_keyword(keyword, normalize=normalize, reduce=reduce, sigma=sigma)
            self.n_voxels, self.n_pmids = maps.shape
        
        elif isinstance(keyword_or_shape, tuple):
            if len(keyword_or_shape) != 2:
                raise ValueError('Given shape of length {} is not admissible (should have length 2).'.format(len(keyword_or_shape)))
            
            maps = scipy.sparse.csr_matrix(keyword_or_shape)
            self.n_voxels, self.n_pmids = maps.shape

        elif isinstance(keyword_or_shape, int):
            maps = scipy.sparse.csr_matrix((keyword_or_shape, 1))
            self.n_voxels, self.n_pmids = maps.shape

        elif keyword_or_shape == None:
            maps = None
            self.n_voxels, self.n_pmids = 0, 0

        else:
            raise ValueError('First argument not understood. Must be str, int or length 2 tuple.')

        self._maps = maps
        self.Ni = Ni
        self.Nj = Nj
        self.Nk = Nk
        self.affine = affine

    @classmethod
    def zeros(cls, shape, **kwargs):
        return cls(keyword_or_shape=shape, **kwargs)

    def copy_header(self, other):
        self.Ni = other.Ni
        self.Nj = other.Nj
        self.Nk = other.Nk
        self.affine = other.affine

        return self

    def has_same_header(self, other):
        if self.n_voxels != other.n_voxels or \
           self.n_pmids != other.n_pmids or \
           self.Ni != other.Ni or \
           self.Nj != other.Nj or \
           self.Nk != other.Nk or \
           (self.affine != other.affine).any():
            return False

        return True

    def __str__(self):
        string = '\nMaps object containing {} maps.\n'
        string += '____________Header_____________\n'
        string += 'N Nonzero : {}\n'
        string += 'N voxels : {}\n'
        string += 'N pmids : {}\n'
        string += 'Box size : ({}, {}, {})\n'
        string += 'Affine :\n{}\n'
        string += 'Map : \n{}\n'
        return string.format(self.n_pmids, self.maps.count_nonzero(), self.n_voxels, self.n_pmids, self.Ni, self.Nj, self.Nk, self.affine, self.maps)

    @property
    def maps(self):
        return self._maps

    @maps.setter
    def maps(self, maps):
        if maps == None:
            self.n_voxels, self.n_pmids = 0, 0
        else:
            self.n_voxels, self.n_pmids = maps.shape
        self._maps = maps

    def randomize(self, n_peaks, n_maps, inplace=False):
        maps = scipy.sparse.csr_matrix(np.random.binomial(n=n_peaks, p=1./(self.Ni*self.Nj*self.Nk*n_maps), size=(self.Ni*self.Nj*self.Nk, n_maps)).astype(float))

        new_maps = self if inplace else copy.copy(self)
        new_maps.maps = maps

        return new_maps


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
            return self.map_to_data(self._maps[:, map_id], self.Ni, self.Nj, self.Nk)

        if self.n_pmids > 1:
            raise KeyError('This Maps object contains {} maps, specify which map to convert to data.'.format(self.n_pmids))

        return self.map_to_data(self._maps[:, 0], self.Ni, self.Nj, self.Nk)

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
            return self.map_to_img(self._maps[:, map_id], self.Ni, self.Nj, self.Nk, self.affine)

        if self.n_pmids > 1:
            raise KeyError('This Maps object contains {} maps, specify which map to convert to img.'.format(self.n_pmids))

        return self.map_to_img(self._maps[:, 0], self.Ni, self.Nj, self.Nk, self.affine)

    def n_peaks(self):
        '''
            Returns a numpy array containing the number of peaks in each maps
        '''
        e = scipy.sparse.csr_matrix(np.ones(self.n_voxels))
        return np.array(e.dot(self._maps).toarray()[0])

    def max(self):
        '''
            Maximum element wise
        '''
        return self.maps.max()

    def sum(self):
        '''
            Builds the summed map of the given maps on the second axis.

            maps : sparse CSR matrix of shape (n_voxels, n_pmids) where
                n_voxels is the number of voxels in the box
                n_pmids is the number of pmids

            Returns a sparse CSR matrix of shape (n_voxels, 1) representing the flattened summed up map.
        '''
        e = scipy.sparse.csr_matrix(np.ones(self.n_pmids)).transpose()

        sum_map = Maps()
        sum_map.copy_header(self)
        sum_map.maps = self._maps.dot(e)

        return sum_map

    def normalize(self, inplace=False):
        '''
            Normalize each maps separatly so that each maps sums to 1.
        '''
        diag = scipy.sparse.diags(np.power(self.n_peaks(), -1))

        new_maps = self if inplace else copy.copy(self)
        new_maps.maps = self._maps.dot(diag)
        return new_maps

    @staticmethod
    def smooth_map(map, sigma, Ni, Nj, Nk):
        data = Maps.map_to_data(map, Ni, Nj, Nk)
        data = gaussian_filter(data, sigma=sigma)
        map = scipy.sparse.csr_matrix(data.reshape((-1, 1), order='F'))
        return map

    def smooth(self, sigma, inplace=False, verbose=False):
        '''
            Convolve each map with Gaussian kernel
        '''

        lil_maps = scipy.sparse.lil_matrix(self.maps)

        for k in range(self.n_pmids):
            if verbose: print('Smoothing {} out of {}.'.format(k, self.n_pmids))
            data = self.to_data(k)
            data = gaussian_filter(data, sigma=sigma)
            lil_maps[:, k] = data.reshape((-1, 1), order='F')

        csr_maps = scipy.sparse.csr_matrix(lil_maps)

        new_maps = self if inplace else copy.copy(self)
        new_maps.maps = csr_maps
        return new_maps

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
        avg_map = Maps()
        avg_map.copy_header(self)
        avg_map.maps = self.average(self.maps)

        return avg_map


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
        var_map = Maps()
        var_map.copy_header(self)
        var_map.maps = self.variance(self._maps)
        return var_map

    def cov(self):
        '''
            Builds the empirical covariance matrix of the given maps on the second axis.

            Returns a sparse CSR matrix of shape (n_voxels, n_voxels) representing the covariance matrix.
        '''

        # _, n_pmids = maps.shape 

        e = scipy.sparse.csr_matrix(np.ones(self.n_pmids)/self.n_pmids).transpose()

        M1 = self._maps.dot(self._maps.transpose())/self.n_pmids
        M2 = self._maps.dot(e)

        return M1 - M2.dot(M2.transpose())

    def iterative_smooth_avg_var(self, sigma=None, verbose=False):
        '''
            Compute average and variance of the maps in self.maps (previously smoothed if sigma!=None) iteratively.
            (Less memory usage).
        '''
        # avg_n = Maps().copy_header(self)
        # avg_n.maps = self[0]
        # var_n = Maps(self.n_voxels).copy_header(self)

        current_map = self[0]
        if sigma != None:
            current_map = self.smooth_map(current_map, sigma, self.Ni, self.Nj, self.Nk)
        avg_map_n = copy.copy(current_map)
        var_map_n = Maps.zeros(self.n_voxels).maps

        for k in range(2, self.n_pmids+1):
            if verbose:
                print('Iterative smooth avg var {} out of {}'.format(k, self.n_pmids))
            avg_map_p, var_map_p = copy.copy(avg_map_n), copy.copy(var_map_n)
            current_map = self[k-1]

            if sigma != None:
                current_map = self.smooth_map(current_map, sigma, self.Ni, self.Nj, self.Nk)

            avg_map_n = 1./k*((k-1)*avg_map_p + current_map)
            # var_map_n = (k-2)/(k-1)*var_map_p + (avg_map_p - avg_map_n).power(2) + 1./(k-1)*(current_map-avg_map_n).power(2)
            var_map_n = (k-1)/(k)*var_map_p + (k-1)/(k)*(avg_map_p - avg_map_n).power(2) + 1./(k)*(current_map-avg_map_n).power(2)

        avg = Maps().copy_header(self)
        var = Maps().copy_header(self)
        avg.maps = avg_map_n
        var.maps = var_map_n

        # self.smooth(sigma=sigma, inplace=True)
        # return self.avg(), self.var()
        return avg, var

    def __iadd__(self, val):
        # if not self.has_same_header(val):
        #     raise ValueError('Maps must have same header to be added. Given :\n{}\n{}'.format(self, val))

        self.maps += val.maps

        return self

    def __add__(self, other):
        result = copy.copy(self)
        result += other
        return result

    def __imul__(self, val):
        self.maps *= val
        return self

    def __mul__(self, val):
        result = copy.copy(self)
        result *= val
        return result

    def __getitem__(self, key):
        return self.maps[:, key]


def metaanalysis_img_from_keyword(keyword, sigma=None, reduce=1, normalize=False):
    maps = Maps(keyword, sigma=sigma, reduce=reduce, normalize=normalize)
    maps = maps.avg()
    return maps.to_img()


if __name__ == '__main__':
    pmid = 22266924
    keyword = 'prosopagnosia'
    # keyword = 'schizophrenia'
    # keyword = 'memory'
    sigma = 2.

    # img = metaanalysis_img_from_keyword(keyword, sigma=sigma)
    # plot_activity_map(img, threshold=0.00065)

    # rand_maps = Maps(Ni=Ni, Nj=Nj, Nk=Nk, affine=affine).randomize(100000, 10)
    # print(rand_maps)
    # M1 = Maps(Ni=Ni, Nj=Nj, Nk=Nk).randomize(100000, 10)
    # M2 = Maps(Ni=Ni, Nj=Nj, Nk=Nk).randomize(100000, 10)
    # print(M1)
    # print(M1*2)

    # print(M1+M2)

    # rand_maps.smooth(sigma=sigma, verbose=True, inplace=True)

    # avg, var = rand_maps.iterative_smooth_avg_var(sigma=sigma, verbose=True)

    # print(avg)

    # maps = Maps(keyword)
    # maps_smoothed = Maps(keyword, sigma=2.)
    # avg_img = maps_smoothed.avg().to_img()
    # var_map = maps_smoothed.var()
    # var_img = var_map.to_img()
    
    # avg_map_2, var_map_2 = maps.iterative_smooth_avg_var(sigma=2.)
    # plot_activity_map(avg_img, threshold=0.00065)
    # plot_activity_map(avg_map_2.to_img(), threshold=0.00065)    
    # plot_activity_map(var_img, threshold=0.00003)
    # plot_activity_map(var_map_2.to_img(), threshold=0.00003)

    # print(var_map)    
    # print(var_map_2)   

    maps = Maps(keyword, sigma=None, normalize=False)
    maps_smoothed = Maps(keyword, sigma=sigma, normalize=False)

    avg, var = maps.iterative_smooth_avg_var(sigma=sigma, verbose=True)
    avg2 = maps_smoothed.avg()
    var2 = maps_smoothed.var()

    print(avg)
    print(avg2)
    print(var)
    print(var2)

    plot_activity_map(avg2.to_img(), threshold=0.0003)
    plot_activity_map(avg.to_img(), threshold=0.0003)

    plot_activity_map(var2.to_img(), threshold=0.000005) 
    plot_activity_map(var.to_img(), threshold=0.000005) 


