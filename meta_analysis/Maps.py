'''
    TEST
'''
import scipy, copy, nilearn
import numpy as np
import nibabel as nib
import pandas as pd
import nilearn
from scipy.ndimage import gaussian_filter
from sklearn.covariance import LedoitWolf
from nilearn import datasets

from .globals import mem

from .tools import print_percent, index_3D_to_1D

import multiprocessing
from joblib import Parallel, delayed

# @profile
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
    inv_affine = kwargs['inv_affine']
    index_dict = kwargs['index_dict']
    n_maps = kwargs['n_maps']
    mask = kwargs['mask']
    verbose = kwargs['verbose']

    col_names = kwargs['col_names']
    groupby_col = col_names['groupby']
    x_col = col_names['x']
    y_col = col_names['y']
    z_col = col_names['z']
    weight_col = col_names['weight']

    maps = scipy.sparse.lil_matrix((n_maps, Ni*Nj*Nk))

    n_tot = df.shape[0]
    for i_row, row in enumerate(zip(df[groupby_col], df[x_col], df[y_col], df[z_col], df[weight_col])):
        print_percent(i_row, n_tot, string='Loading dataframe {0:.1f}%...', verbose=verbose)

        groupby_id, x, y, z, weight = row
        map_id = index_dict[groupby_id]

        i, j, k = np.clip(np.floor(np.dot(inv_affine, [x, y, z, 1]))[:-1].astype(int), [0, 0, 0], [Ni-1, Nj-1, Nk-1])
        
        if mask is None or mask[i, j, k] == 1:
            p = Maps.coord_to_id(i, j, k, Ni, Nj, Nk)
            maps[map_id, p] += weight

    return scipy.sparse.csr_matrix(maps)

# @mem.cache
def build_maps_from_df(df, col_names, Ni, Nj, Nk, affine, mask=None, verbose=False):
    '''
        Given a keyword, finds every related studies and builds their activation maps.

        reduce : integer, reducing scale factor. Ex : if reduce=2, aggregates voxels every 2 voxels in each direction.
                Notice that this affects the affine and box size.

        Returns 
            maps: sparse CSR matrix of shape (n_voxels, n_maps) containing all the related flattenned maps where
                    n_maps is the number of pmids related to the keyword
                    n_voxels is the number of voxels in the box (may have changed if reduce != 1)
            Ni_r, Nj_r, Nk_r: new box dimension (changed if reduce != 1)
            affine_r: new affine (changed if reduce != 1)
    '''

    df = df.astype({col_names['x']: 'float64',
               col_names['y']: 'float64',
               col_names['z']: 'float64',
               col_names['weight']: 'float64',
               })

    # Creating map index
    unique_pmid = df[col_names['groupby']].unique()
    n_maps = len(unique_pmid)
    index_dict = {k:v for v, k in enumerate(unique_pmid)}

    # LIL format allows faster incremental construction of sparse matrix
    maps = scipy.sparse.csr_matrix((n_maps, Ni*Nj*Nk))

    # Multiprocessing maps computation
    n_jobs = multiprocessing.cpu_count()//2
    splitted_df= np.array_split(df, n_jobs, axis=0)

    kwargs = {
        'Ni': Ni,
        'Nj': Nj,
        'Nk': Nk,
        'inv_affine': np.linalg.inv(affine),
        'index_dict': index_dict,
        'n_maps': n_maps,
        'col_names': col_names,
        'mask': None if mask is None else mask.get_fdata(),
        'verbose': verbose
    }
    
    results = Parallel(n_jobs=n_jobs, backend='multiprocessing')(delayed(compute_maps)(sub_df, **kwargs) for sub_df in splitted_df)
    
    if verbose: print('Merging...')
    for m in results:
        maps += m

    maps = maps.transpose()
    
    return maps

def build_maps_from_img(img):
    img = nilearn.image.load_img(img)
    n_dims = len(img.get_data().shape)

    if n_dims == 4:
        Ni, Nj, Nk, n_maps = img.shape

    elif n_dims == 3:
        Ni, Nj, Nk = img.shape
        n_maps = 1

    else:
        raise ValueError('Image not supported. Must be a 3D or 4D image.')

    data = Maps.flatten_array(img.get_data(), _2D=n_maps)
    maps = scipy.sparse.csr_matrix(data)

    return maps, Ni, Nj, Nk, img.affine

class Atlas:
    def __init__(self, atlas):
        self.atlas = None
        self.atlas_img = None
        self.data = None
        self.labels = None
        self.n_labels = None

        if atlas is None:
            return

        self.atlas = atlas
        self.atlas_img = nilearn.image.load_img(atlas['maps'])
        self.data = self.atlas_img.get_fdata()
        self.labels = atlas['labels']
        self.n_labels = len(self.labels)


class Maps:
    def __init__(self, df=None,
                       template=None,
                       Ni=None, Nj=None, Nk=None,
                       affine=None,
                       mask=None,
                       atlas=None,
                       groupby_col=None,
                       x_col='x',
                       y_col='y',
                       z_col='z',
                       weight_col='weight',
                       save_memory=True,
                       verbose=False
                       ):
        '''


        Args:
            df (pandas.DataFrame): Pandas DataFrame containing the (x,y,z) coordinates, the weights and the map id. The names of the columns can be specified.
            template (nibabel.Nifti1Image): Template storing the box size and affine. If not None, Will overwrite parameters Ni, Nj, Nk and affine.
            Ni (int): X size of the bounding box.
            Nj (int): Y size of the bounding box.
            Nk (int): Z size of the bounding box.
            affine (numpy.ndarray): Array with shape (4, 4) storing the affine used to compute brain voxels coordinates from world cooridnates.
            mask (nibabel.Nifti1Image): Nifti1Image with 0 or 1 data.  0: outside the mask, 1: inside.
            atlas (Object): Object containing a nibabel.Nifti1Image or a path to it in atlas['maps'] and a list of the labels in atlas['labels']
            groupby_col (str): Name of the column on which the groupby operation is operated. Or in an equivalent way, the name of the column storing the ids of the maps.
            x_col (str): Name of the column storing the x coordinates.
            y_col (str): Name of the column storing the y coordinates.
            z_col (str): Name of the column storing the z coordinates.
            weight_col (str): Name of the column storing the weights. 
        '''

        if template is not None and (isinstance(template, nib.Nifti1Image) or isinstance(template, str)):
            template = nilearn.image.load_img(template)
            Ni, Nj, Nk = template.shape
            affine = template.affine

        elif template is not None:
            raise ValueError('Template not understood. Must be a nibabel.Nifti1Image or a path to it.')

        elif isinstance(df, np.ndarray) and len(df.shape) == 3:
            Ni, Nj, Nk = df.shape

        elif isinstance(df, np.ndarray) and len(df.shape) == 4:
            Ni, Nj, Nk, _ = df.shape

        elif isinstance(df, nib.Nifti1Image) or isinstance(df, str):
            pass

        if mask is not None and not isinstance(mask, nib.Nifti1Image):
            raise ValueError('Mask must be an instance of nibabel.Nifti1Image')

        self._save_memory = save_memory
        self._mask = mask
        self._maps = None
        self._atlas = Atlas(atlas)
        self._maps_dense =  None
        self._maps_atlas = None
        self._atlas_filter_matrix = None


        if isinstance(df, pd.DataFrame):
            if groupby_col is None:
                raise ValueError('Must specify column name to group by maps.')

            if Ni is None or Nj is None or Nk is None or affine is None:
                raise ValueError('Must specify Ni, Nj, Nk and affine to initialize with dataframe.')

            col_names = {
                'groupby': groupby_col,
                'x': x_col,
                'y': y_col,
                'z': z_col,
                'weight': weight_col
            }

            self._maps = build_maps_from_df(df, col_names, Ni, Nj, Nk, affine, mask, verbose)

        elif isinstance(df, nib.Nifti1Image) or isinstance(df, str) or isinstance(df, list):
            self._maps, Ni, Nj, Nk, affine = build_maps_from_img(df)

        elif isinstance(df, np.ndarray) and len(df.shape) == 2:
            self._maps = scipy.sparse.csr_matrix(df)

        elif isinstance(df, np.ndarray) and len(df.shape) == 3:
            self._maps = scipy.sparse.csr_matrix(self._flatten_array(df, _2D=1))

        elif isinstance(df, np.ndarray) and len(df.shape) == 4:
            self._maps = scipy.sparse.csr_matrix(df.reshape((-1, df.shape[-1]), order='F'))

        elif isinstance(df, tuple):
            self._maps = scipy.sparse.csr_matrix(df)

        elif isinstance(df, int):
            self._maps = scipy.sparse.csr_matrix((df, 1))

        elif df is None:
            self._maps = None

        elif not isinstance(df, Maps):
            raise ValueError('First argument not understood : {}'.format(type(df)))


        if Ni is None or Nj is None or Nk is None:
            raise ValueError('Must either specify Ni, Nj, Nk or template.')

        self._Ni = Ni
        self._Nj = Nj
        self._Nk = Nk
        self._affine = affine


        if self._mask_dimensions_missmatch():
            raise ValueError('Mask dimensions missmatch. Given box size ({}, {}, {}) whereas mask size is {}. Consider resampling either input data or mask.'.format(self._Ni, self._Nj, self._Nk, self._mask.get_fdata().shape))

        if self._atlas_dimensions_missmatch():
            raise ValueError('Atlas dimensions missmatch. Given box size ({}, {}, {}) whereas atlas size is {}. Consider resampling either input data or atlas.'.format(self._Ni, self._Nj, self._Nk, self._atlas.data.shape))

        if self._box_dimensions_missmatch():
            raise ValueError('Box dimension missmatch. Given box size is ({}, {}, {}) for {} voxels whereas maps contains {} voxels.'.format(self._Ni, self._Nj, self._Nk, self._Ni*self._Nj*self._Nk, self._maps.shape))


        if self._has_mask():
            self.apply_mask(mask)

        self._build_atlas_filter_matrix()
        self._refresh_atlas_maps()

        if not save_memory:
            self._set_dense_maps()


    #_____________PROPERTIES_____________#
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
        self._maps = maps
        self._refresh_atlas_maps()

        if hasattr(self, '_save_memory') and not self._save_memory:
            self._set_dense_maps()

    def _set_maps(self, maps, refresh_atlas_maps=True, refresh_dense_maps=True):
        self._maps = maps

        if refresh_atlas_maps:
            self._refresh_atlas_maps()

        if refresh_dense_maps and hasattr(self, '_save_memory') and not self._save_memory:
            self._set_dense_maps()

    @property
    def n_voxels(self):
        return 0 if self._maps is None else self._maps.shape[0]

    @property
    def n_maps(self):
        return 0 if self._maps is None else self._maps.shape[1]

    #_____________CLASS_METHODS_____________#
    @classmethod
    def zeros(cls, n_voxels, n_maps=1, **kwargs):
        '''
            Create empty maps of the given shape.
            See the Maps.__init__ doc for **kwargs parameters.
        '''
        return cls(df=(n_voxels, n_maps), **kwargs)

    @classmethod
    def random(cls, n_peaks, n_maps, Ni=None, Nj=None, Nk=None, affine=None, template=None, mask=None, atlas=None, p=None):
        '''
            Create the given number of maps and sample peaks on them. 

            See the Maps.__init__ doc for Ni, Nj, Nk, mask, atlas parameters.
            See the Maps.randomize doc for n_peaks, n_maps, p parameters.
        '''
        maps = cls(df=None, Ni=Ni, Nj=Nj, Nk=Nk, affine=affine, template=template, mask=mask, atlas=atlas)

        return maps.randomize(n_peaks, n_maps, p=p, use_mask=(mask is not None), inplace=True)

    @classmethod
    def copy_header(cls, other):
        '''
            Create a Maps instance with same header as the passed Maps object.

            Args:
                other (Maps): Maps instance wanted informations.

            Returns:
                (Maps)
        '''
        maps = cls(Ni=other._Ni, Nj=other._Nj, Nk=other._Nk)
        maps._copy_header(other)
        return maps

    #_____________PRIVATE_TOOLS_____________#
    def _copy_header(self, other):
        self._Ni = other._Ni
        self._Nj = other._Nj
        self._Nk = other._Nk
        self._affine = other._affine
        self._mask = other._mask
        self._save_memory = other._save_memory
        self._atlas = other._atlas

        return self

    def __str__(self):
        string = '\nMaps object containing {} maps.\n'
        string += '____________Header_____________\n'
        string += 'N Nonzero : {}\n'
        string += 'N voxels : {}\n'
        string += 'N pmids : {}\n'
        string += 'Box size : ({}, {}, {})\n'
        string += 'Affine :\n{}\n'
        string += 'Has atlas : {}\n'
        string += 'Map : \n{}\n'
        string += 'Atlas Map : \n{}\n'
        return string.format(self.n_maps, self.maps.count_nonzero(), self.n_voxels, self.n_maps, self._Ni, self._Nj, self._Nk, self._affine, self._has_atlas(), self.maps, self._maps_atlas)

    @staticmethod
    def coord_to_id(i, j, k, Ni, Nj, Nk):
        return np.ravel_multi_index((i, j, k), (Ni, Nj, Nk), order='F')

    @staticmethod
    def id_to_coord(id, Ni, Nj, Nk):
        return np.unravel_index(id, (Ni, Nj, Nk), order='F')
   
    @staticmethod
    def flatten_array(array, _2D=None):
        shape = -1 if _2D is None else (-1, _2D)
        return array.reshape(shape, order='F')

    @staticmethod
    def unflatten_array(array, Ni, Nj, Nk, _4D=None):
        shape = (Ni, Nj, Nk) if _4D is None or _4D == 1 else (Ni, Nj, Nk, _4D)
        return array.reshape(shape, order='F')

    def _coord_to_id(self, i, j, k):
        return self.coord_to_id(i, j, k, self._Ni, self._Nj, self._Nk)

    def _id_to_coord(self, id):
        return self.id_to_coord(id, self._Ni, self._Nj, self._Nk)

    def _flatten_array(self, array, _2D=None):
        return self.flatten_array(array, _2D=_2D)

    def _unflatten_array(self, array, _4D=None):
        return self.unflatten_array(array, self._Ni, self._Nj, self._Nk, _4D=_4D)

    def _build_atlas_filter_matrix(self):
        if not self._has_atlas():
            return

        atlas_data = self._flatten_array(self._atlas.data)
        filter_matrix = scipy.sparse.lil_matrix((self._atlas.n_labels, self._Ni*self._Nj*self._Nk))

        for k in range(self._atlas.n_labels):
            row = atlas_data == k
            filter_matrix[k, row] = 1

        return scipy.sparse.csr_matrix(filter_matrix)

    def _refresh_atlas_maps(self):
        if not self._has_atlas() or self._maps is None:
            return

        if self._atlas_filter_matrix is None:
            self._atlas_filter_matrix = self._build_atlas_filter_matrix()

        self._maps_atlas = self._atlas_filter_matrix.dot(self.maps)

    def _has_atlas(self):
        return self._atlas is not None and self._atlas.atlas is not None

    def _has_mask(self):
        return self._mask is not None and isinstance(self._mask, nib.Nifti1Image)

    def _get_maps(self, map_id=None, atlas=False, dense=False):

        if atlas and dense:
            raise ValueError('No dense maps for atlas.')

        if map_id is None:
            if atlas:
                return self._maps_atlas
            
            elif dense:
                return self._maps_dense

            else:
                return self._maps

        else:
            if atlas:
                return self._maps_atlas[:, map_id]
            
            elif dense:
                return self._maps_dense[:, :, :, map_id]

            else:
                return self._maps[:, map_id]


    def _set_dense_maps(self):
        if self._maps is None:
            self._maps_dense = None
        else:
            self._maps_dense = Maps.unflatten_array(self._maps.toarray(), self._Ni, self._Nj, self._Nk, _4D=self.n_maps)

    def _box_dimensions_missmatch(self):
        Ni, Nj, Nk = self._Ni, self._Nj, self._Nk

        if self._maps is None:
            return False

        if Ni is not None and Nj is not None and Nk is not None and self._maps is not None and Ni*Nj*Nk == self._maps.shape[0]:
            return False

        return True

    def _mask_dimensions_missmatch(self):
        if not self._has_mask():
            return False

        if (self._Ni, self._Nj, self._Nk) == self._mask.shape:
            return False

        return True

    def _atlas_dimensions_missmatch(self):
        if not self._has_atlas():
            return False

        if (self._Ni, self._Nj, self._Nk) == self._atlas.data.shape:
            return False

        return True

    #_____________OPERATORS_____________#

    def __iadd__(self, val):
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

    #_____________DATA_TRANSFORMERS_____________#
    @staticmethod
    def map_to_array(map, Ni, Nj, Nk):
        '''
            Convert a sparse matrix of shape (n_voxels, 1) into a dense 3D numpy array of shape (Ni, Nj, Nk).

            Indexing of map is supposed to have been made Fortran like (first index moving fastest).
        '''
        n_voxels, n_maps = map.shape

        if n_voxels != Ni*Nj*Nk:
            raise ValueError('Map\'s length ({}) does not match given box ({}, {}, {}) of size {}.'.format(n_voxels, Ni, Nj, Nk, Ni*Nj*Nk))

        return Maps.unflatten_array(map.toarray(), Ni, Nj, Nk, _4D=n_maps)

    @staticmethod
    def array_to_map(array):
        return scipy.sparse.csr_matrix(Maps.flatten_array(array, _2D=1))

    @staticmethod
    def array_to_img(array, affine):
        '''
            Convert a dense 3D array into a nibabel Nifti1Image.
        '''
        return nib.Nifti1Image(array, affine)

    @staticmethod
    def map_to_img(map, Ni, Nj, Nk, affine):
        '''
            Convert a sparse matrix of shape (n_voxels, 1) into a nibabel Nifti1Image.

            Ni, Nj, Nk are the size of the box used to index the flattened map matrix.
        '''
        return Maps.array_to_img(Maps.map_to_array(map, Ni, Nj, Nk), affine)

    def to_array(self, map_id=None):
        '''
            Convert one map into a 3D numpy.ndarray.

            Args:
                map_id (int, optional): If int : id of the map to convert (3D output). 
                    If None, converts all the maps (4D output). Defaults to None. 
        
            Returns:
                (numpy.ndarray) 3D array containing the chosen map information.
        '''
        maps = self._maps

        if map_id is not None:
            maps = self._maps[:, map_id]

        return self.map_to_array(maps, self._Ni, self._Nj, self._Nk)

    def to_img(self, map_id=None):
        '''
            Convert one map into a nibabel.Nifti1Image.

            Args:
                map_id (int, optional): If int : id of the map to convert (3D output). 
                    If None, converts all the maps (4D output). Defaults to None. 
        
            Returns:
                (nibabel.Nifti1Image) Nifti1Image containing the chosen map information.
        '''
        if self._affine is None:
            raise ValueError('Must specify affine to convert maps to img.')

        maps = self._maps
        if map_id is not None:
            maps = self._maps[:, map_id]

        return self.map_to_img(maps, self._Ni, self._Nj, self._Nk, self._affine)

    @staticmethod
    def _one_map_to_array_atlas(map, Ni, Nj, Nk, atlas_data, label_range):
        array = np.zeros((Ni, Nj, Nk))

        for k in label_range:
            array[atlas_data == k] = map[k, 0]

        return array

    def to_array_atlas(self, map_id=None, ignore_bg=True):
        '''
            Convert one atlas map into a 3D numpy.array.

            Args:
                map_id (int, optional): If int : id of the map to convert (3D output). 
                    If None, converts all the maps (4D output). Defaults to None.
                ignore_bg (bool, optional): If True: ignore the first label of the atlas (background) which is set to 0 in the returned array.

            Returns:
                (numpy.ndarray) 3D array containing the chosen atlas map information.

            Raises:
                AttributeError: If no atlas has been given to this instance.
        '''
        if not self._has_atlas():
            raise AttributeError('No atlas were given.')

        start = 1 if ignore_bg else 0
        label_range = range(start, self._atlas.n_labels)

        if map_id is None:
            array = np.zeros((self._Ni, self._Nj, self._Nk, self.n_maps))
            
            for k in range(self.n_maps):
                array[:, :, :, k] = self._one_map_to_array_atlas(self._maps_atlas[:, k], self._Ni, self._Nj, self._Nk, self._atlas.data, label_range)
        else:
            array = np.zeros((self._Ni, self._Nj, self._Nk))
            array[:, :, :] = self._one_map_to_array_atlas(self._maps_atlas[:, map_id], self._Ni, self._Nj, self._Nk, self._atlas.data, label_range)

        return array

    def to_img_atlas(self, map_id=None, ignore_bg=True):
        '''
            Convert one atlas map into a nibabel.Nifti1Image.

            Args:
                map_id (int, optional): If int : id of the map to convert (3D output). 
                    If None, converts all the maps (4D output). Defaults to None.
                ignore_bg (bool, optional): If True: ignore the first label of the atlas (background) which is set to 0 in the returned array.

            Returns:
                (nibabel.Nifti1Image) Nifti1Image containing the chosen atlas map information.

            Raises:
                AttributeError: If no atlas as been given to this instance.
        '''
        return self.array_to_img(self.to_array_atlas(map_id=map_id, ignore_bg=ignore_bg), self._affine)

    def _to_map_atlas(self, data):
        if isinstance(data, np.ndarray):
            data = scipy.sparse.csr_matrix(self.flatten_array(data, _2D=1))

        return self._atlas_filter_matrix.dot(data)

    #_____________PUBLIC_TOOLS_____________#
    def apply_mask(self, mask):
        '''
            Set the contribution of every voxels outside the mask to zero.

            Args:
                mask (nibabel.Nifti1Image): Nifti1Image with 0 or 1 array. 0: outside the mask, 1: inside.
        '''
        if not isinstance(mask, nib.Nifti1Image):
            raise ValueError('Mask must be an instance of nibabel.Nifti1Image')

        if self.maps is not None:
            mask_array = self._flatten_array(mask.get_fdata())
            filter_matrix = scipy.sparse.diags(mask_array, format='csr')
            self.maps = filter_matrix.dot(self.maps)

        self._mask = mask

    def randomize(self, n_peaks, n_maps, p=None, override_mask=False, inplace=False):
        '''
            Randomize the maps by sampling n_peaks of weight 1 (may overlap) over n_maps maps.

            Args:
                n_peaks (int): Number of peaks to sample maps-wise. Each peak has a weight 1 and the weights of the peaks sampled on the same voxel of the same map are added.
                n_maps (int): Number of maps on which to perform the sample.
                p (Maps instance or np.ndarray): (Optional) Distribution of probability of the peaks over the voxels.
                    The distribution may be given either by a Maps instance containing 1 map or a np.ndarray of same shape as the box of the current Maps instance (Ni, Nj, Nk).
                    If None, sample uniformly accros the box. Default: None
                override_mask (bool): (Optional) If False, use the mask given when initializing the Maps object.
                    Important : the given distribution p is then shrinked and re-normalized.
                    If True p is unchanged. Default : False.
                inplace (bool): (Optional) Performs the sampling inplace (True) or creates a new instance (False). Default False.

            Returns:
                (Maps instance) Self or a copy depending on inplace.
        '''
        if self._Ni is None or self._Nj is None or self._Nk is None:
            raise ValueError('Invalid box size ({}, {}, {}).'.format(Ni, Nj, Nk))

        n_voxels = self._Ni*self._Nj*self._Nk

        if p is None:
            p = np.ones(n_voxels)/n_voxels
        
        elif isinstance(p, Maps):
            if p.n_maps != 1:
                raise ValueError('Maps object should contain exactly one map to serve as distribution. Given has {} maps.'.format(p.n_maps))
            p = p.maps.transpose().toarray()[0]

        elif isinstance(p, np.ndarray):
            if p.shape != (self._Ni, self._Nj, self._Nk):
                raise ValueError('Invalid numpy array to serve as a distribution. Should be of shape ({}, {}, {}).'.format(self._Ni, self._Nj, self._Nk))
            p = self._flatten_array(p)

        else:
            raise ValueError('Invalid distribution p. Must be either None, Maps object or numpy.ndarray.')

        if not override_mask and self._has_mask():
            mask = self._flatten_array(self._mask.get_fdata())
            p = np.ma.masked_array(p, np.logical_not(mask)).filled(0)
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

    def normalize(self, inplace=False):
        '''
            Normalize each maps separatly so that each maps sums to 1.

            Args:
                inplace (bool, optional): If True performs the normalization inplace else create a new instance.

            Returns:
                (Maps) Self or a copy depending on inplace.
        '''
        diag = scipy.sparse.diags(np.power(self.n_peaks(atlas=False), -1))

        new_maps = self if inplace else copy.copy(self)
        new_maps.maps = self._maps.dot(diag)

        if self._has_atlas():
            diag_atlas = scipy.sparse.diags(np.power(self.n_peaks(atlas=True), -1))
            new_maps._maps_atlas = self._maps_atlas.dot(diag_atlas)

        return new_maps

    @staticmethod
    def _smooth_array(array, sigma):
        return gaussian_filter(array, sigma=sigma)

    @staticmethod
    def _smooth_map(map, sigma, Ni, Nj, Nk):
        array = Maps.map_to_array(map, Ni, Nj, Nk)
        array = Maps._smooth_array(array, sigma=sigma)
        map = Maps.array_to_map(array)
        return map

    @staticmethod
    def _smooth(data, sigma, Ni=None, Nj=None, Nk=None):
        if sigma is None:
            return data

        if isinstance(data, np.ndarray):
            return Maps._smooth_array(data, sigma)

        elif scipy.sparse.issparse(data):
            return Maps._smooth_map(data, sigma, Ni, Nj, Nk)

    @profile
    def smooth(self, sigma, map_id=None, inplace=False, verbose=False):
        '''
            Convolve chosen maps with gaussian kernel.

            Args:
                sigma (float): Standard deviation of the gaussian kernel.
                map_id (int, optional): If None: convolves each maps. If int: convolves only the chosen map. Defaults to None.
                inplace (bool, optional): If True performs the normalization inplace else create a new instance. Defaults to False
                verbose (bool, optional): If True print logs.

        '''
        if map_id is None:
            map_ids = range(self.n_maps)
        else:
            map_ids = [map_id]

        csc_matrices = []

        for k in map_ids:
            print_percent(k, self.n_maps, 'Smoothing {1} out of {2}... {0:.1f}%', rate=0, verbose=verbose)

            if not self.save_memory:
                array = self._get_maps(map_id=k, dense=True)
            else:
                array = self.to_array(k)

            array_smoothed = gaussian_filter(array, sigma=sigma)
            array_smoothed = self._flatten_array(array_smoothed, _2D=1)
            matrix = scipy.sparse.csc_matrix(array_smoothed)
            csc_matrices.append(matrix)

        csr_maps = scipy.sparse.hstack(csc_matrices)
        csr_maps = scipy.sparse.csr_matrix(csr_maps)

        new_maps = self if inplace else copy.copy(self)
        new_maps.maps = csr_maps

        return new_maps

    #_____________STATISTICS_____________#
    def n_peaks(self, atlas=False):
        '''
            Compute the sum of weights in each maps (equivalent to number of peaks if unit weights are 1).

            Args:
                atlas (bool, optional): If True, the atlas maps are considered.
        '''
        return self.sum(atlas=atlas, axis=0, keepdims=True).reshape(-1)

    def max(self, atlas=False, **kwargs):
        '''
            Compute the maximum. axis=None: element-wise, axis=0: maps-wise, axis=1: voxels/labels-wise.

            Args:
                atlas (bool, optional): If True, the atlas maps are considered.
                **kwargs: Kwargs are passed to scipy.sparse.csr_matrix.max() function.
            Returns:
                (numpy.ndarray) 2D numpy array
        '''
        maps = self._get_maps(atlas=atlas)

        max = copy.copy(maps).max(**kwargs)
        if isinstance(max, scipy.sparse.coo.coo_matrix):
            max = max.toarray()
        return max

    def sum(self, atlas=False, axis=None, keepdims=False):
        '''
            Compute the sum. axis=None: element-wise, axis=0: maps-wise, axis=1: voxels/labels-wise.

            Args:
                atlas (bool, optional): If True, the atlas maps are considered.
                **kwargs: Kwargs are passed to scipy.sparse.csr_matrix.sum() function.
            Returns:
                (numpy.ndarray) 2D numpy array
        '''
        maps = self._get_maps(atlas=atlas)

        e1 = scipy.sparse.csr_matrix(np.ones((1, maps.shape[0])))
        e2 = scipy.sparse.csr_matrix(np.ones((maps.shape[1], 1)))

        if axis is None or axis == 0:
            maps = e1.dot(maps)

        if axis is None or axis == 1:
            maps = maps.dot(e2)

        if axis not in [None, 0, 1]:
            raise ValueError('Axis must be None, 0 or 1.')

        return np.array(maps.toarray()) if keepdims else np.squeeze(np.array(maps.toarray()))

    def summed_map(self):
        '''
            Sums all maps.

            Returns:
                (Maps) New Maps instance containing the summed map.
        '''
        sum_map = Maps.copy_header(self)
        sum_map.maps = scipy.sparse.csr_matrix(self.sum(axis=1, keepdims=True))

        return sum_map

    @staticmethod
    def _average(maps):
        '''
            Computes the average map of the given maps on the second axis.

            maps : sparse CSR matrix of shape (n_voxels, n_maps) where
                n_voxels is the number of voxels in the box
                n_maps is the number of pmids

            Returns a sparse CSR matrix of shape (n_voxels, 1) representing the flattened average map.
        '''
        _, n_maps = maps.shape
        e = scipy.sparse.csr_matrix(np.ones(n_maps)/n_maps).transpose()

        return maps.dot(e)


    @staticmethod
    def _variance(maps, bias=False):
        '''
            Computes the variance map of the given maps on the second axis.

            Returns a sparse CSR matrix of shape (n_voxels, 1) representing the flattened variance map.
        '''
        _, n_maps = maps.shape

        avg_map = Maps._average(maps)
        maps_squared = maps.multiply(maps) # Squared element wise

        avg_squared_map = Maps._average(maps_squared)
        squared_avg_map = avg_map.multiply(avg_map)

        var = avg_squared_map - squared_avg_map

        if not bias:
            var *= (n_maps/(n_maps-1))

        return var

    def avg(self):
        '''
            Computes the average map.

            Returns:
                (Maps) New Maps instance containing the average map.
        '''
        avg_map = Maps.copy_header(self)
        avg_map.maps = self._average(self.maps)
        if self._has_atlas():
            avg_map._maps_atlas = self._average(self._maps_atlas)

        return avg_map

    def var(self, bias=True):
        '''
            Computes the variance map.

            Args:
                bias (bool, optional): If True, computes the biased variance (1/n_maps factor), else compute the unbiased variance (1/(n_maps-1) factor).

            Returns:
                (Maps) New Maps instance containing the variance map.
        '''
        var_map = Maps.copy_header(self)
        var_map.maps = self._variance(self._maps, bias=bias)
        if self._has_atlas():
            var_map._maps_atlas = self._variance(self._maps_atlas, bias=bias)

        return var_map

    def cov(self, atlas=True, bias=False, shrink=None, sparse=False, ignore_bg=True):
        '''
            Computes the empirical covariance matrix of the voxels, the observations being the different maps.
            Important : Considering covariance between atlas' labels (atlas=True) instead of voxels is highly recommended
            since the number of voxels is often huge (~1 million), the covariance matrix would be of big shape (1 million, ~1 million) and the computation will probably not finish.

            Args:
                atlas (bool, optional): If True, consider covariance between atlas' labels. If False, covariance between voxels. Default is True (recommended).
                bias (bool, optional): If True, computes the biased covariance, else unbiased. Defaults to False.
                shrink(str, optional): Shrink the covariance matrix. If 'LW', the LedoitWolf method is applied. Default is None.
                sparse(bool, optional): If False, converts the sparse covariance matrix to a dense array. Else, let it sparse. Default is False
                ignore_bg(bool, optional): If True, ignore the first label of the atlas (background).

            Returns:
                (numpy.ndarray or scipy.sparse.csr_matrix) A 2D matrix (sparse or dense depending on sparse parameter) of shape (n_voxels, n_voxels) representing the covariance matrix.
        '''
        if atlas and not self._has_atlas():
            raise ValueError('No atlas. Must specify an atlas when initializing Maps or specify atlas=False in cov() function.')

        if not bias and self.n_maps <= 1:
            raise ValueError('Unbiased covariance computation requires at least 2 maps ({} given).'.format(self.n_maps))

        maps = self._get_maps(atlas=atlas)
        ddof = 0 if bias else 1

        if atlas:
            labels = self._atlas.labels
            if ignore_bg:
                maps[0, :] = 0
                labels = labels[1:]


        e1 = scipy.sparse.csr_matrix(np.ones(self.n_maps)/(self.n_maps-ddof)).transpose()
        e2 = scipy.sparse.csr_matrix(np.ones(self.n_maps)/(self.n_maps)).transpose()

        M1 = maps.dot(e1)
        M2 = maps.dot(e2)
        M3 = maps.dot(maps.transpose())/((self.n_maps-ddof))

        # Empirical covariance matrix
        S =  M3 - M1.dot(M2.transpose())

        if not sparse:
            S = S.toarray()

        if shrink == 'LW':
            S = LedoitWolf().fit(S.toarray()).covariance_

        return S, labels if atlas else S

    @staticmethod
    def _power(map, n):
        if scipy.sparse.issparse(map):
            return map.power(n)

        elif isinstance(map, np.ndarray):
            return np.power(map, n)

        else:
            raise ValueError('Given map type not supported for power : {}'.format(type(map)))

    @staticmethod
    def _iterative_avg(k, previous_avg, new_value):
        if k == 1:
            return new_value
        return 1./k*((k-1)*previous_avg + new_value)

    @staticmethod
    def _iterative_var(k, previous_var, new_avg, new_value, bias=False):
        if k == 1:
            return 0*new_value

        if bias:
            return (k-1)/(k)*previous_var + 1./(k*(k-1))*Maps._power(new_avg - new_value, 2) + 1./(k)*Maps._power(new_value - new_avg, 2)
        else:
            return (k-2)/(k-1)*previous_var + 1./((k-1)**2)*Maps._power(new_avg - new_value, 2) + 1./(k-1)*Maps._power(new_value - new_avg, 2)

    # @profile
    def iterative_smooth_avg_var(self, compute_var=True, sigma=None, bias=False, verbose=False):
        '''
            Compute average and variance of the maps in self.maps (previously smoothed if sigma!=None) iteratively.
            (Less memory usage).
        '''

        if not compute_var:
            return self.avg().smooth(sigma=sigma), None

        avg_map = None
        var_map = None
        avg_map_atlas = None
        var_map_atlas = None

        for k in range(self.n_maps):
            print_percent(k, self.n_maps, 'Iterative smooth avg var {1} out of {2}... {0:.1f}%', rate=0, verbose=verbose)

            current_map = self._get_maps(map_id=k, atlas=False, dense=not self.save_memory)
            current_map = self._smooth(current_map, sigma, self._Ni, self._Nj, self._Nk)

            avg_map = self._iterative_avg(k+1, avg_map, current_map)
            var_map = self._iterative_var(k+1, var_map, avg_map, current_map, bias=bias)

            if self._has_atlas():
                current_map_atlas = self._to_map_atlas(current_map)
                avg_map_atlas = self._iterative_avg(k+1, avg_map_atlas, current_map_atlas)
                var_map_atlas = self._iterative_var(k+1, var_map_atlas, avg_map_atlas, current_map_atlas, bias=bias)

        avg = Maps.copy_header(self)
        var = Maps.copy_header(self)

        if not self.save_memory: avg_map = self.array_to_map(avg_map)
        if not self.save_memory: var_map = self.array_to_map(var_map)
        
        avg._set_maps(avg_map, refresh_atlas_maps=False)
        var._set_maps(var_map, refresh_atlas_maps=False)

        if self._has_atlas():
            avg._maps_atlas = avg_map_atlas
            var._maps_atlas = var_map_atlas

        return avg, var

