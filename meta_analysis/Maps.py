'''
    TEST
'''
import scipy, copy
import numpy as np
import nibabel as nib
import pandas as pd
from scipy.ndimage import gaussian_filter
from sklearn.covariance import LedoitWolf

from .globals import mem

from .tools import print_percent, index_3D_to_1D

import multiprocessing
from joblib import Parallel, delayed

def compute_maps(df, **kwargs):
    '''
        Given a list of pmids, builds their activity maps (flattened in 1D) on a LIL sparse matrix format.
        Used for multiprocessing in get_all_maps_associated_to_keyword function.

        pmids : list of pmid
        Ni, Nj, Nk : size of the 3D box (used to flatten 3D to 1D indices)
        inv_affine : the affine inverse used to compute voxels coordinates

        Returns sparse LIL matrix of shape (len(pmids), Ni*Nj*Nk) containing all the maps
    '''
    Ni = kwargs['Ni']
    Nj = kwargs['Nj']
    Nk = kwargs['Nk']
    Ni_r = kwargs['Ni_r']
    Nj_r = kwargs['Nj_r']
    Nk_r = kwargs['Nk_r']
    inv_affine = kwargs['inv_affine']
    inv_affine_r = kwargs['inv_affine_r']
    index_dict = kwargs['index_dict']
    n_pmids = kwargs['n_pmids']
    col_names = kwargs['col_names']
    mask = kwargs['mask']

    groupby_col = col_names['groupby']
    x_col = col_names['x']
    y_col = col_names['y']
    z_col = col_names['z']
    weight_col = col_names['weight']

    maps = scipy.sparse.lil_matrix((n_pmids, Ni_r*Nj_r*Nk_r))


    i_row, n_tot = 0, df.shape[0]
    for index, row in df.iterrows():
        print_percent(i_row, n_tot)
        i_row += 1

        map_id, weight = index_dict[row[groupby_col]], row[weight_col]
        x, y, z = row[x_col], row[y_col], row[z_col]
        i_r, j_r, k_r = np.clip(np.floor(np.dot(inv_affine_r, [x, y, z, 1]))[:-1].astype(int), [0, 0, 0], [Ni_r-1, Nj_r-1, Nk_r-1])
        i, j, k = np.clip(np.floor(np.dot(inv_affine, [x, y, z, 1]))[:-1].astype(int), [0, 0, 0], [Ni-1, Nj-1, Nk-1])
        
        if mask is None or mask[i, j, k] == 1:
            p = index_3D_to_1D(i_r, j_r, k_r, Ni_r, Nj_r, Nk_r)
            maps[map_id, p] += weight

    return scipy.sparse.csr_matrix(maps)

@mem.cache
def build_maps_from_df(df, col_names, Ni, Nj, Nk, affine, reduce=1, mask=None):
    '''
        Given a keyword, finds every related studies and builds their activation maps.

        reduce : integer, reducing scale factor. Ex : if reduce=2, aggregates voxels every 2 voxels in each direction.
                Notice that this affects the affine and box size.

        Returns 
            maps: sparse CSR matrix of shape (n_voxels, n_pmids) containing all the related flattenned maps where
                    n_pmids is the number of pmids related to the keyword
                    n_voxels is the number of voxels in the box (may have changed if reduce != 1)
            Ni_r, Nj_r, Nk_r: new box dimension (changed if reduce != 1)
            affine_r: new affine (changed if reduce != 1)
    '''

    # Creating map index
    unique_pmid = df['pmid'].unique()
    n_pmids = len(unique_pmid)
    index_dict = {k:v for v, k in enumerate(unique_pmid)}

    # Computing new box size according to the reducing factor
    Ni_r, Nj_r, Nk_r = np.ceil(np.array((Ni, Nj, Nk))/reduce).astype(int)

    # LIL format allows faster incremental construction of sparse matrix
    maps = scipy.sparse.csr_matrix((n_pmids, Ni_r*Nj_r*Nk_r))

    # Changing affine to the new box size
    affine_r = np.copy(affine)
    for i in range(3):
        affine_r[i, i] = affine[i, i]*reduce
    inv_affine_r = np.linalg.inv(affine_r)
    inv_affine = np.linalg.inv(affine)

    # Multiprocessing maps computation
    n_jobs = multiprocessing.cpu_count()//2
    splitted_df= np.array_split(df, n_jobs, axis=0)

    kwargs = {
        'Ni_r': Ni_r,
        'Nj_r': Nj_r,
        'Nk_r': Ni_r,
        'Ni': Ni,
        'Nj': Nj,
        'Nk': Nk,
        'inv_affine_r': inv_affine_r,
        'inv_affine': inv_affine,
        'index_dict': index_dict,
        'n_pmids': n_pmids,
        'col_names': col_names,
        'mask': mask
    }
    
    results = Parallel(n_jobs=n_jobs, backend='multiprocessing')(delayed(compute_maps)(sub_df, **kwargs) for sub_df in splitted_df)
    
    print('Merging...')
    for m in results:
        maps += m

    maps = maps.transpose()
    
    return maps, Ni_r, Nj_r, Nk_r, affine_r

class Maps:
    def __init__(self, df=None,
                       reduce=1,
                       Ni=None, Nj=None, Nk=None,
                       affine=None,
                       mask=None,
                       nifti_mask=None,
                       groupby_col=None,
                       x_col='x',
                       y_col='y',
                       z_col='z',
                       weight_col='weight',
                       save_memory=True
                       ):
        '''


        Args:
            df (pandas.DataFrame): Pandas DataFrame containing the (x,y,z) coordinates, the weights and the map id. The names of the columns can be specified.
            reduce (int): Reducing factor of the map resolution (e.g. if reduce=2, voxels are 2 times larger in every directions).
            Ni (int): X size of the bounding box.
            Nj (int): Y size of the bounding box.
            Nk (int): Z size of the bounding box.
            affine (numpy.ndarray): Array with shape (4, 4) storing the affine used to compute brain voxels coordinates from world cooridnates.
            mask (numpy.ndarray): Boolean array with shape (Ni, Nj, Nk). If mask[i, j, k] = False, ignore all (x,y,z) coordinates which have their voxels coordinates equal to (i, j, k).
            groupby_col (str): Name of the column on which the groupby operation is operated. Or in an equivalent way, the name of the column storing the ids of the maps.
            x_col (str): Name of the column storing the x coordinates.
            y_col (str): Name of the column storing the y coordinates.
            z_col (str): Name of the column storing the z coordinates.
            weight_col (str): Name of the column storing the weights. 
        '''

        self.Ni = Ni
        self.Nj = Nj
        self.Nk = Nk
        self.affine = affine
        self.mask = mask
        self._save_memory = save_memory
        self._maps_dense =  None


        self.nifti_mask = nifti_mask
        self.id_to_coord_index = None
        self.coord_to_id_index = None


        if isinstance(df, pd.DataFrame):
            if groupby_col == None:
                raise ValueError('Must specify column name to group by maps.')

            if Ni is None or Nj is None or Nk is None or affine is None:
                raise ValueError('Must specify box size (Ni, Nj, Nk) and affine when initializing with pandas dataframe.')

            col_names = {
                'groupby': groupby_col,
                'x': x_col,
                'y': y_col,
                'z': z_col,
                'weight': weight_col
            }

            self.maps, self.Ni, self.Nj, self.Nk, self.affine = build_maps_from_df(df, col_names, Ni, Nj, Nk, affine, mask=mask, reduce=reduce)
            self.n_voxels, self.n_maps = self._maps.shape

        elif isinstance(df, tuple):
            self.maps = scipy.sparse.csr_matrix(df)
            self.n_voxels, self.n_maps = self._maps.shape

        elif isinstance(df, int):
            self.maps = scipy.sparse.csr_matrix((df, 1))
            self.n_voxels, self.n_maps = self._maps.shape

        elif df is None:
            self._maps = None
            self.n_voxels, self.n_maps = 0, 0

        else:
            raise ValueError('First argument not understood. Must be pandas df, int or length 2 tuple.')


    @classmethod
    def zeros(cls, shape, **kwargs):
        return cls(df=shape, **kwargs)

    def copy_header(self, other):
        self.Ni = other.Ni
        self.Nj = other.Nj
        self.Nk = other.Nk
        self.affine = other.affine
        self._save_memory = other._save_memory

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

    def coord_to_id(self, i, j, k):
        return np.ravel_multi_index((i, j, k), (self.Ni, self.Nj, self.Nk), order='F')

    def id_to_coord(self, id):
        return np.unravel_index(id, (self.Ni, self.Nj, self.Nk), order='F')

    def flatten_array(self, array):
        return array.reshape(-1, order='F')

    def unflatten_array(self, array):
        return array.reshape((self.Ni, self.Nj, self.Nk), order='F')

    def apply_mask(self, nifti_mask):

        mask_data = self.flatten_array(nifti_mask.get_fdata())

        filter_matrix = scipy.sparse.diags(mask_data, format='csr')

        self.maps = filter_matrix.dot(self.maps)
        self.nifti_mask = nifti_mask
        self.id_to_coord, self.coord_to_id = self.build_index(nifti_mask)

    def build_index(self, nifti_mask):
        mask_data = self.flatten_array(nifti_mask.get_fdata())

        nonzero_id = np.nonzero(mask_data)[0]

        id_to_coord_index = {id: self.id_to_coord(id) for id in nonzero_id}
        coord_to_id_index = {v: k for k, v in id_to_coord_index.items()}

        return id_to_coord_index, coord_to_id_index


    @property
    def save_memory(self):
        return self._save_memory

    @save_memory.setter
    def save_memory(self, save_memory):
        self._save_memory = save_memory

        if save_memory:
            if hasattr(self, '_maps_dense'): del self._maps_dense
        else:
            self._set_dense_maps()

    @property
    def maps(self):
        return self._maps

    @maps.setter
    def maps(self, maps):
        if maps is None:
            self.n_voxels, self.n_maps = 0, 0

        else:
            self.n_voxels, self.n_maps = maps.shape

        self._maps = maps

        if hasattr(self, '_save_memory') and not self._save_memory:
            self._set_dense_maps()

    def _set_dense_maps(self):
        if self._maps is None:
            self._maps_dense = None
        else:
            self._maps_dense = self._maps.toarray().reshape((self.Ni, self.Nj, self.Nk, self.n_maps), order='F')

    @property
    def mask(self):
        if self._maps is None:
            return None

        return self._maps.reshape((Ni, Nj, Nk), order='F')

    @mask.setter
    def mask(self, mask):
        if mask is None:
            self._mask = None
        else:
            self._mask = mask.reshape(-1, order='F')

    def randomize(self, n_peaks, n_maps, inplace=False, p=None, mask=None):
        if self.Ni is None or self.Nj is None or self.Nk is None:
            raise ValueError('Invalid box size ({}, {}, {}).'.format(Ni, Nj, Nk))

        n_voxels = self.Ni*self.Nj*self.Nk

        if p is None:
            p = np.ones(n_voxels)/n_voxels
        
        elif isinstance(p, Maps):
            if p.n_maps != 1:
                raise ValueError('Maps object should contain exactly one map to serve as distribution. Given has {} maps.'.format(p.n_maps))
            p = p.maps.transpose().toarray()[0]

        elif isinstance(p, np.ndarray):
            if p.shape != (self.Ni, self.Nj, self.Nk):
                raise ValueError('Invalid numpy array to serve as a distribution. Should be of shape ({}, {}, {}).'.format(self.Ni, self.Nj, self.Nk))
            p = p.reshape(-1, order='F')

        # else:
        #     raise ValueError('Invalid distribution p. Must be either Maps object or numpy.ndarray.')

        if mask is not None:
            mask = mask.reshape(-1, order='Fortran')
            p = np.ma.masked_array(p, np.logical_not(mask))
            p /= np.sum(p)

        maps = scipy.sparse.dok_matrix((n_voxels, n_maps))
        voxels_samples = np.random.choice(n_voxels, size=n_peaks, p=p)
        voxels_samples_unique, counts = np.unique(voxels_samples, return_counts=True)

        for i in range(len(voxels_samples_unique)):
            map_id = np.random.randint(n_maps)
            maps[voxels_samples_unique[i], map_id] = counts[i]

        maps = scipy.sparse.csr_matrix(maps)


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
    def data_to_map(data):
        return scipy.sparse.csr_matrix(data.reshape((-1, 1), order='F'))

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
        sum_map.maps = scipy.sparse.csr_matrix(self.sum(axis=1))

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
        data_reshaped = data.reshape((-1, 1), order='F')
        map = scipy.sparse.csr_matrix(data_reshaped)
        return map

    @staticmethod
    def smooth_data(data, sigma):
        return gaussian_filter(data, sigma=sigma)

    def smooth(self, sigma, inplace=False, verbose=False):
        '''
            Convolve each map with Gaussian kernel
        '''

        lil_maps = scipy.sparse.lil_matrix(self.maps)

        for k in range(self.n_maps):
            if verbose: print('Smoothing {} out of {}.'.format(k+1, self.n_maps))
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

    def cov(self, bias=False, shrink=None):
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

        # Empirical covariance matrix
        S =  M3 - M1.dot(M2.transpose())

        if shrink is None:
            return S

        elif shrink == 'LW':
            return LedoitWolf().fit(S.toarray()).covariance_

    def iterative_smooth_avg_var(self, compute_var=True, sigma=None, bias=False, verbose=False):
        '''
            Compute average and variance of the maps in self.maps (previously smoothed if sigma!=None) iteratively.
            (Less memory usage).
        '''

        if not compute_var:
            return self.avg().smooth(sigma=sigma), None

        current_map = self[0] if self.save_memory else self._maps_dense[:, :, :, 0]
        if sigma != None:
                if self.save_memory:
                    current_map = self.smooth_map(current_map, sigma, self.Ni, self.Nj, self.Nk)
                else:
                    current_map = self.smooth_data(current_map, sigma)

        avg_map_n = copy.copy(current_map)
        var_map_n = Maps.zeros(self.n_voxels).maps if self.save_memory else np.zeros((self.Ni, self.Nj, self.Nk))

        for k in range(2, self.n_maps+1):
            if verbose:
                print('Iterative smooth avg var {} out of {}...'.format(k, self.n_maps), end='\r', flush=True)
            avg_map_p, var_map_p = copy.copy(avg_map_n), copy.copy(var_map_n)
            current_map = self[k-1] if self.save_memory else self._maps_dense[:, :, :, k-1]

            if sigma != None:
                if self.save_memory:
                    current_map = self.smooth_map(current_map, sigma, self.Ni, self.Nj, self.Nk)
                else:
                    current_map = self.smooth_data(current_map, sigma)

            avg_map_n = 1./k*((k-1)*avg_map_p + current_map)

            if bias:
                if self.save_memory:
                    var_map_n = (k-1)/(k)*var_map_p + (k-1)/(k)*(avg_map_p - avg_map_n).power(2) + 1./(k)*(current_map-avg_map_n).power(2)
                else:
                    var_map_n = (k-1)/(k)*var_map_p + (k-1)/(k)*np.power(avg_map_p - avg_map_n, 2) + 1./(k)*np.power(current_map-avg_map_n, 2)

            else:
                if self.save_memory:
                    var_map_n = (k-2)/(k-1)*var_map_p + (avg_map_p - avg_map_n).power(2) + 1./(k-1)*(current_map-avg_map_n).power(2)
                else:
                    var_map_n = (k-2)/(k-1)*var_map_p + np.power(avg_map_p - avg_map_n, 2) + 1./(k-1)*np.power(current_map-avg_map_n, 2)

        avg = Maps().copy_header(self)
        var = Maps().copy_header(self)

        avg.maps = avg_map_n if self.save_memory else self.data_to_map(avg_map_n)
        var.maps = var_map_n if self.save_memory else self.data_to_map(var_map_n)

        if verbose:
            print('Iterative smooth avg var {} out of {}... Done'.format(self.n_maps, self.n_maps))

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
        if isinstance(val, Maps) and not self.has_same_header(val):
            warnings.warn('Multiplied maps don\'t have same header.', UserWarning)

        self.maps *= val
        return self

    def __mul__(self, val):
        result = copy.copy(self)
        result *= val
        return result

    def __getitem__(self, key):
        return self.maps[:, key]

