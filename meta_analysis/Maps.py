import scipy, copy
import numpy as np
import nibabel as nib
from scipy.ndimage import gaussian_filter

from .activity_map import get_all_maps_associated_to_keyword

class Maps:
    def __init__(self, keyword_or_shape=None, reduce=1, normalize=False, sigma=None, Ni=None, Nj=None, Nk=None, affine=None):
        if isinstance(keyword_or_shape, str):
            maps, Ni, Nj, Nk, affine = get_all_maps_associated_to_keyword(keyword_or_shape, normalize=normalize, reduce=reduce, sigma=sigma)
            self.n_voxels, self.n_maps = maps.shape
        
        elif isinstance(keyword_or_shape, tuple):
            if len(keyword_or_shape) != 2:
                raise ValueError('Given shape of length {} is not admissible (should have length 2).'.format(len(keyword_or_shape)))
            
            maps = scipy.sparse.csr_matrix(keyword_or_shape)
            self.n_voxels, self.n_maps = maps.shape

        elif isinstance(keyword_or_shape, int):
            maps = scipy.sparse.csr_matrix((keyword_or_shape, 1))
            self.n_voxels, self.n_maps = maps.shape

        elif keyword_or_shape == None:
            maps = None
            self.n_voxels, self.n_maps = 0, 0

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
           self.n_maps != other.n_maps or \
           self.Ni != other.Ni or \
           self.Nj != other.Nj or \
           self.Nk != other.Nk or \
           np.array_equal(self.affine, other.affine):
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
        return string.format(self.n_maps, self.maps.count_nonzero(), self.n_voxels, self.n_maps, self.Ni, self.Nj, self.Nk, self.affine, self.maps)

    @property
    def maps(self):
        return self._maps

    @maps.setter
    def maps(self, maps):
        if maps == None:
            self.n_voxels, self.n_maps = 0, 0
        else:
            self.n_voxels, self.n_maps = maps.shape
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

        if self.n_maps > 1:
            raise KeyError('This Maps object contains {} maps, specify which map to convert to data.'.format(self.n_maps))

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

        if self.n_maps > 1:
            raise KeyError('This Maps object contains {} maps, specify which map to convert to img.'.format(self.n_maps))

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

    def summed_map(self):
        '''
            Builds the summed map of the given maps on the second axis.

            maps : sparse CSR matrix of shape (n_voxels, n_maps) where
                n_voxels is the number of voxels in the box
                n_maps is the number of pmids

            Returns a sparse CSR matrix of shape (n_voxels, 1) representing the flattened summed up map.
        '''
        sum_map = Maps()
        sum_map.copy_header(self)
        sum_map.maps = self.sum(axis=1)

        return sum_map

    def sum(self, **kwargs):
        return self.maps.sum(**kwargs)

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

        for k in range(self.n_maps):
            if verbose: print('Smoothing {} out of {}.'.format(k, self.n_maps))
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

            maps : sparse CSR matrix of shape (n_voxels, n_maps) where
                n_voxels is the number of voxels in the box
                n_maps is the number of pmids

            Returns a sparse CSR matrix of shape (n_voxels, 1) representing the flattened average map.
        '''
        _, n_maps = maps.shape
        e = scipy.sparse.csr_matrix(np.ones(n_maps)/n_maps).transpose()

        return maps.dot(e)

    def avg(self):
        avg_map = Maps()
        avg_map.copy_header(self)
        avg_map.maps = self.average(self.maps)

        return avg_map


    @staticmethod
    def variance(maps, bias=False):
        '''
            Builds the variance map of the given maps on the second axis.

            Returns a sparse CSR matrix of shape (n_voxels, 1) representing the flattened variance map.
        '''
        _, n_maps = maps.shape

        avg_map = Maps.average(maps)
        maps_squared = maps.multiply(maps) # Squared element wise

        avg_squared_map = Maps.average(maps_squared)
        squared_avg_map = avg_map.multiply(avg_map)

        var = avg_squared_map - squared_avg_map

        if not bias:
            var *= (n_maps/(n_maps-1))

        return var

    def var(self, bias=True):
        var_map = Maps()
        var_map.copy_header(self)
        var_map.maps = self.variance(self._maps, bias=bias)
        return var_map

    def cov(self, bias=False):
        '''
            Builds the empirical unbiased covariance matrix of the given maps on the second axis.

            Returns a sparse CSR matrix of shape (n_voxels, n_voxels) representing the covariance matrix.
        '''
        if not bias and self.n_maps <= 1:
            raise ValueError('Unbiased covariance computation requires at least 2 maps ({} given).'.format(self.n_maps))

        ddof = 0 if bias else 1

        e1 = scipy.sparse.csr_matrix(np.ones(self.n_maps)/(self.n_maps-ddof)).transpose()
        e2 = scipy.sparse.csr_matrix(np.ones(self.n_maps)/(self.n_maps)).transpose()

        M1 = self._maps.dot(e1)
        M2 = self._maps.dot(e2)
        M3 = self._maps.dot(self._maps.transpose())/((self.n_maps-ddof))

        return M3 - M1.dot(M2.transpose())

    def iterative_smooth_avg_var(self, sigma=None, bias=False, verbose=False):
        '''
            Compute average and variance of the maps in self.maps (previously smoothed if sigma!=None) iteratively.
            (Less memory usage).
        '''

        current_map = self[0]
        if sigma != None:
            current_map = self.smooth_map(current_map, sigma, self.Ni, self.Nj, self.Nk)
        avg_map_n = copy.copy(current_map)
        var_map_n = Maps.zeros(self.n_voxels).maps

        for k in range(2, self.n_maps+1):
            if verbose:
                print('Iterative smooth avg var {} out of {}'.format(k, self.n_maps))
            avg_map_p, var_map_p = copy.copy(avg_map_n), copy.copy(var_map_n)
            current_map = self[k-1]

            if sigma != None:
                current_map = self.smooth_map(current_map, sigma, self.Ni, self.Nj, self.Nk)

            avg_map_n = 1./k*((k-1)*avg_map_p + current_map)

            if bias:
                var_map_n = (k-1)/(k)*var_map_p + (k-1)/(k)*(avg_map_p - avg_map_n).power(2) + 1./(k)*(current_map-avg_map_n).power(2)
            else:
                var_map_n = (k-2)/(k-1)*var_map_p + (avg_map_p - avg_map_n).power(2) + 1./(k-1)*(current_map-avg_map_n).power(2)

        avg = Maps().copy_header(self)
        var = Maps().copy_header(self)
        avg.maps = avg_map_n
        var.maps = var_map_n

        return avg, var

    def __iadd__(self, val):
        if not self.has_same_header(val):
            warnings.warn('Added maps don\'t have same header.', UserWarning)

        self.maps += val.maps

        return self

    def __add__(self, other):
        result = copy.copy(self)
        result += other
        return result

    def __imul__(self, val):
        if not self.has_same_header(val):
            warnings.warn('Multiplied maps don\'t have same header.', UserWarning)

        self.maps *= val
        return self

    def __mul__(self, val):
        result = copy.copy(self)
        result *= val
        return result

    def __getitem__(self, key):
        return self.maps[:, key]
