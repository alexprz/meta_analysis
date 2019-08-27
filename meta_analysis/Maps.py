"""Maps class."""
import scipy
import copy
import nilearn
import numpy as np
import nibabel as nib
import pandas as pd
from scipy.ndimage import gaussian_filter
from sklearn.covariance import LedoitWolf
from scipy.sparse import csr_matrix

from .globals import mem
from .tools import print_percent

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
    inv_affine = kwargs['inv_affine']
    index_dict = kwargs['index_dict']
    n_maps = kwargs['n_maps']
    mask = kwargs['mask']
    verbose = kwargs['verbose']
    dtype = kwargs['dtype']

    col_names = kwargs['col_names']
    groupby_col = col_names['groupby']
    x_col = col_names['x']
    y_col = col_names['y']
    z_col = col_names['z']
    weight_col = col_names['weight']

    maps = scipy.sparse.lil_matrix((n_maps, Ni*Nj*Nk), dtype=dtype)

    n_tot = df.shape[0]
    for i_row, row in enumerate(zip(df[groupby_col], df[x_col], df[y_col], df[z_col], df[weight_col])):
        print_percent(i_row, n_tot, string='Loading dataframe {0:.1f}%...', verbose=verbose, prefix='Maps')

        groupby_id, x, y, z, weight = row
        map_id = index_dict[groupby_id]

        i, j, k = np.clip(np.floor(np.dot(inv_affine, [x, y, z, 1]))[:-1].astype(int), [0, 0, 0], [Ni-1, Nj-1, Nk-1])

        if mask is None or mask[i, j, k] == 1:
            p = Maps.coord_to_id(i, j, k, Ni, Nj, Nk)
            maps[map_id, p] += weight

    return scipy.sparse.csr_matrix(maps)


@mem.cache
def build_maps_from_df(df, col_names, Ni, Nj, Nk, affine, mask=None, verbose=False, dtype=np.float32):
    '''
        Given a keyword, finds every related studies and builds their activation maps.

        reduce : integer, reducing scale factor. Ex : if reduce=2, aggregates voxels every 2 voxels in each direction.
                Notice that this affects the affine and box size.

        Returns:
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
    index_dict = {k: v for v, k in enumerate(unique_pmid)}

    # LIL format allows faster incremental construction of sparse matrix
    maps = scipy.sparse.csr_matrix((n_maps, Ni*Nj*Nk), dtype=dtype)

    # Multiprocessing maps computation
    n_jobs = multiprocessing.cpu_count()//2
    splitted_df = np.array_split(df, n_jobs, axis=0)

    kwargs = {
        'Ni': Ni,
        'Nj': Nj,
        'Nk': Nk,
        'inv_affine': np.linalg.inv(affine),
        'index_dict': index_dict,
        'n_maps': n_maps,
        'col_names': col_names,
        'mask': None if mask is None else mask.get_fdata(),
        'verbose': verbose,
        'dtype': dtype
    }

    results = Parallel(n_jobs=n_jobs, backend='multiprocessing')(delayed(compute_maps)(sub_df, **kwargs) for sub_df in splitted_df)

    for k, m in enumerate(results):
        if verbose:
            print_percent(k, n_jobs, string='Merging... {0:.2f}%', prefix='Maps')
        maps += m

    maps = maps.transpose()

    return maps


def build_maps_from_img(img, dtype=np.float32):
    img = nilearn.image.load_img(img)
    n_dims = len(img.get_data().shape)

    if n_dims == 4:
        Ni, Nj, Nk, n_maps = img.shape

    elif n_dims == 3:
        Ni, Nj, Nk = img.shape
        n_maps = 1

    else:
        raise ValueError('Image not supported. Must be a 3D or 4D image.')

    data = Maps.flatten_array(img.get_data(), _2D=n_maps).astype(dtype)
    maps = scipy.sparse.csr_matrix(data)

    return maps, Ni, Nj, Nk, img.affine


class Atlas:
    def __init__(self, atlas=None, bg_label='Background'):
        self.atlas = None
        self.atlas_img = None
        self.data = None
        self.labels = None
        self.n_labels = None
        self.bg_label = bg_label
        self.bg_index = None
        self.labels_without_bg = None
        self.labels_range_without_bg = None

        if atlas is None:
            return

        self.atlas = atlas
        self.atlas_img = nilearn.image.load_img(atlas['maps'])
        self.data = self.atlas_img.get_fdata()
        self.labels = atlas['labels']
        self.labels_range = list(range(len(self.labels)))
        self.n_labels = len(self.labels)

        self.set_bg_labels(bg_label)

    def has_background(self):
        return self.bg_index is not None

    def get_labels(self, ignore_bg=False, bg_label=None):
        labels, labels_without_bg = self.labels, self.labels_without_bg

        if bg_label is not None:
            _, labels_without_bg, _ = self.get_bg_labels(bg_label)

        if not ignore_bg:
            return labels

        if self.has_background():
            return labels_without_bg

        print(f'No background label matching \'{bg_label}\' in given atlas. '
              'Consider specifying background label. ignore_bg ignored')

        return labels

    def get_labels_range(self, ignore_bg=False, bg_label=None):
        labels_range, labels_range_without_bg = self.labels_range, self.labels_range_without_bg

        if bg_label is not None:
            _, _, labels_range_without_bg = self.get_bg_labels(bg_label)

        if not ignore_bg:
            return labels_range

        if self.has_background():
            return labels_range_without_bg

        print(f'No background label matching \'{bg_label}\' in given atlas. '
              'Consider specifying background label. ignore_bg ignored')

        return labels_range

    def set_bg_labels(self, bg_label):
        self.bg_index, self.labels_without_bg, self.labels_range_without_bg = self.get_bg_labels(bg_label)

    def get_bg_labels(self, bg_label):

        try:
            bg_index = self.labels.index(bg_label)

            labels_without_bg = copy.copy(self.labels)
            labels_range_without_bg = copy.copy(self.labels_range)

            del labels_without_bg[bg_index]
            del labels_range_without_bg[bg_index]

        except ValueError:
            bg_index, labels_without_bg, labels_range_without_bg = None, None, None

        return bg_index, labels_without_bg, labels_range_without_bg


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
                 verbose=False,
                 dtype=np.float64
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
            raise ValueError('Template not understood.'
                             'Must be a nibabel.Nifti1Image or a path to it.')

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
        self._maps_dense = None
        self._maps_atlas = None
        self._atlas_filter_matrix = None
        self._dtype = dtype

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

            self._maps = build_maps_from_df(df, col_names, Ni, Nj, Nk, affine, mask, verbose, self._dtype)

        elif isinstance(df, nib.Nifti1Image) or isinstance(df, str) or isinstance(df, list):
            self._maps, Ni, Nj, Nk, affine = build_maps_from_img(df, dtype=self._dtype)

        elif isinstance(df, np.ndarray) and len(df.shape) == 2:
            self._maps = scipy.sparse.csr_matrix(df, dtype=self._dtype)

        elif isinstance(df, np.ndarray) and len(df.shape) == 3:
            df = self._flatten_array(df, _2D=1)
            self._maps = scipy.sparse.csr_matrix(df, dtype=self._dtype)

        elif isinstance(df, np.ndarray) and len(df.shape) == 4:
            df = df.reshape((-1, df.shape[-1]), order='F')
            self._maps = scipy.sparse.csr_matrix(df, dtype=self._dtype)

        elif isinstance(df, tuple):
            self._maps = scipy.sparse.csr_matrix(df, dtype=self._dtype)

        elif isinstance(df, int):
            self._maps = scipy.sparse.csr_matrix((df, 1), dtype=self._dtype)

        elif df is None:
            self._maps = None

        elif not isinstance(df, Maps):
            raise TypeError(f'First argument not understood : {type(df)}')

        if Ni is None or Nj is None or Nk is None:
            raise TypeError('Must either specify Ni, Nj, Nk or template.')

        self._Ni = Ni
        self._Nj = Nj
        self._Nk = Nk
        self._affine = affine

        if self._mask_dimensions_missmatch():
            raise ValueError(f'Mask dimensions missmatch. Given box size is '
                             f'({self._Ni}, {self.Nj}, {self.Nk}) whereas '
                             f'mask size is {self._mask.get_fdata().shape}. '
                             f'Consider resampling either input data or mask.')

        if self._atlas_dimensions_missmatch():
            raise ValueError(f'Atlas dimensions missmatch. Given box size is '
                             f'({self.Ni}, {self.Nj}, {self.Nk}) whereas '
                             f'atlas size is {self._atlas.data.shape}. '
                             f'Consider resampling input data or atlas.')

        if self._box_dimensions_missmatch():
            raise ValueError(f'Box dimension missmatch. Given box size is '
                             f'({self.Ni}, {self.Nj}, {self.Nk}) for '
                             f'{self.prod_N()} voxels whereas maps '
                             f'has shape {self._maps.shape}.')

        if self._has_mask():
            self.apply_mask(mask)

        self._refresh_atlas_maps()

        if not save_memory:
            self._set_dense_maps()

    # _____________PROPERTIES_____________ #
    @property
    def save_memory(self):
        return self._save_memory

    @save_memory.setter
    def save_memory(self, save_memory):
        self._save_memory = save_memory

        if save_memory:
            if hasattr(self, '_maps_dense'):
                del self._maps_dense
        else:
            self._set_dense_maps()

    @property
    def maps(self):
        return self._maps

    @maps.setter
    def maps(self, maps):
        if not scipy.sparse.issparse(maps):
            maps = scipy.sparse.csr_matrix(maps)

        self._maps = maps.astype(self._dtype)
        self._refresh_atlas_maps()

        if hasattr(self, '_save_memory') and not self._save_memory:
            self._set_dense_maps()

    def _set_maps(self, maps, refresh_atlas_maps=True, refresh_dense_maps=True):
        self._maps = maps.astype(self._dtype)

        if refresh_atlas_maps:
            self._refresh_atlas_maps()

        if refresh_dense_maps and hasattr(self, '_save_memory') and not self._save_memory:
            self._set_dense_maps()

    @property
    def n_voxels(self):  # Deprecated
        return 0 if self._maps is None else self._maps.shape[0]

    @property
    def n_v(self):
        return 0 if self._maps is None else self._maps.shape[0]

    @property
    def n_maps(self):  # Deprecated
        return 0 if self._maps is None else self._maps.shape[1]

    @property
    def n_m(self):
        return 0 if self._maps is None else self._maps.shape[1]

    @property
    def Ni(self):
        return self._Ni

    @property
    def Nj(self):
        return self._Nj

    @property
    def Nk(self):
        return self._Nk

    @property
    def prod_N(self):
        return self.Ni*self.Nj*self.Nk

    @property
    def affine(self):
        return self._affine

    @property
    def shape(self):
        return (self.Ni, self.Nj, self.Nk, self.n_m)

    @property
    def shape_f(self):
        return (self.n_v, self.n_m)

    def set_coord(self, id, x, y, z, val):
        """Set value to a given x y z coord"""

        if id < 0 or id >= self.n_m:
            raise ValueError(f'Map id must be in [0, {self.n_m}].')

        i, j, k = self.xyz_to_ijk(x, y, z)
        p = self._coord_to_id(i, j, k)
        self.maps[p, id] = val

    def xyz_to_ijk(self, x, y, z):
        if self.affine is None:
            raise ValueError('Maps object should have affine to convert xyz coords.')

        inv_affine = np.linalg.inv(self.affine)
        return np.clip(np.floor(np.dot(inv_affine, [x, y, z, 1]))[:-1].astype(int), [0, 0, 0], [self.Ni-1, self.Nj-1, self.Nk-1])

    # _____________CLASS_METHODS_____________ #
    @classmethod
    def zeros(cls, n_maps=1, **kwargs):
        """
        Create zero-valued maps of the given shape.

        See the Maps.__init__ doc for **kwargs parameters.

        Args:
            n_maps (int, Optional): Number of maps.

        Returns:
            (Maps) Instance of Maps object.

        """
        maps = cls(df=None, **kwargs)  # Build empty maps
        maps.maps = csr_matrix((maps.prod_N, n_maps), dtype=maps._dtype)
        return maps

    @classmethod
    def random(cls, size, p=None, **kwargs):
        """
        Create random maps from given size.

        Must give appropriates kwargs to initialize an empty map.
        See the Maps.__init__ doc.

        Args:
            size: See the Maps.randomize doc.
            p: See the Maps.randomize doc.

        Returns:
            (Maps) Instance of Maps object.

        """
        maps = cls(**kwargs)
        return maps.randomize(size, p=p, override_mask=False, inplace=True)

    @classmethod
    def copy_header(cls, other):
        """
        Create a Maps instance with same header as given Maps object.

        Args:
            other (Maps): Maps instance wanted informations.

        Returns:
            (Maps) Instance of Maps object.

        """
        maps = cls(Ni=other._Ni, Nj=other._Nj, Nk=other._Nk)
        maps._copy_header(other)
        return maps

    # _____________PRIVATE_TOOLS_____________ #
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
        return (
            f'\nMaps object containing {self.n_m} maps.\n'
            f'____________Header_____________\n'
            f'N Nonzero : {self.maps.count_nonzero()}\n'
            f'N voxels : {self.n_v}\n'
            f'N maps : {self.n_m}\n'
            f'Box size : ({self.Ni}, {self.Nj}, {self.Nk})\n'
            f'Affine :\n{self.affine}\n'
            f'Has atlas : {self._has_atlas()}\n'
            f'Map : \n{self.maps}\n'
            f'Atlas Map : \n{self._maps_atlas}\n'
        )

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
        filter_matrix = scipy.sparse.lil_matrix((self._atlas.n_labels, self.prod_N))

        for k in range(self._atlas.n_labels):
            row = atlas_data == k
            filter_matrix[k, row] = 1/np.sum(row)

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

    # _____________OPERATORS_____________ #

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

    def __rmul__(self, val):
        return self.__mul__(val)

    def __getitem__(self, key):
        return self.maps[:, key]

    # _____________DATA_TRANSFORMERS_____________ #
    @staticmethod
    def map_to_array(map, Ni, Nj, Nk):
        '''
            Convert a sparse matrix of shape (n_voxels, 1) into a dense 3D numpy array of shape (Ni, Nj, Nk).

            Indexing of map is supposed to have been made Fortran like (first index moving fastest).
        '''
        n_v, n_maps = map.shape

        if n_v != Ni*Nj*Nk:
            raise ValueError(f'Map\'s length ({n_v}) does not match given box '
                             f'({Ni}, {Nj}, {Nk}) of size {Ni*Nj*Nk}.')

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

    def to_img(self, map_id=None, sequence=False, verbose=False):
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

        if sequence:

            n_jobs = multiprocessing.cpu_count()//2
            splitted_range = np.array_split(range(maps.shape[1]), n_jobs)

            def to_img_pool(maps_range):
                res = []
                n_tot = len(maps_range)
                for i, k in enumerate(maps_range):
                    print_percent(i, n_tot, string='Converting {1} out of {2}... {0:.2f}%', verbose=verbose, rate=0, prefix='Maps')
                    res.append(self.map_to_img(maps[:, k], self._Ni, self._Nj, self._Nk, self._affine))
                return res

            return np.concatenate(Parallel(n_jobs=n_jobs, backend='threading')(delayed(to_img_pool)(sub_array) for sub_array in splitted_range))

        return self.map_to_img(maps, self._Ni, self._Nj, self._Nk, self._affine)

    @staticmethod
    def _one_map_to_array_atlas(map, Ni, Nj, Nk, atlas_data, label_range):
        array = np.zeros((Ni, Nj, Nk))

        for k in label_range:
            array[atlas_data == k] = map[k, 0]

        return array

    def to_array_atlas(self, map_id=None, ignore_bg=True, bg_label=None):
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

        # start = 1 if ignore_bg else 0
        # label_range = range(start, self._atlas.n_labels)

        # label_range = self._atlas.labels_range_without_bg if ignore_bg and self._atlas.has_background() else self._atlas.label_range

        label_range = self._atlas.get_labels_range(ignore_bg=ignore_bg, bg_label=bg_label)
        # label_range = list(range(self._atlas.n_labels))

        # Delete background if any
        # try:
        #     background_index = self._atlas.labels.index(background_label)
        #     del label_range[background_index]
        # except ValueError: # no Background in labels
        #     pass

        if map_id is None and self.n_maps == 1:
            map_id = 0

        if map_id is None:
            array = np.zeros((self._Ni, self._Nj, self._Nk, self.n_maps))

            for k in range(self.n_maps):
                array[:, :, :, k] = self._one_map_to_array_atlas(self._maps_atlas[:, k], self._Ni, self._Nj, self._Nk, self._atlas.data, label_range)
        else:
            array = np.zeros((self._Ni, self._Nj, self._Nk))
            array[:, :, :] = self._one_map_to_array_atlas(self._maps_atlas[:, map_id], self._Ni, self._Nj, self._Nk, self._atlas.data, label_range)

        return array

    def to_img_atlas(self, map_id=None, ignore_bg=False):
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

    def to_atlas(self, bg_label=None):
        '''
            Converts the maps into an atlas by creating a label for each different values.

            Returns:
                (nibabel.Nifti1Image) Nifti1Image containing the atlas
                () Labels of the regions
        '''

        array = self.to_array()

        # print(np.histogram(array))
        # print(np.unique(array))
        # uniques = np.unique(array)
        # n_tot = len(uniques)
        if len(array.shape) == 4:
            array = np.concatenate((np.zeros(array.shape[:-1]+(1,)), array), axis=3)
            array = np.argmax(array, axis=3)
        # array_atlas = np.zeros(array.shape[:-1])

        # # print(np.histogram(np.argmax(array, axis=3)))

        # for k in range(self.n_maps):
        #     print_percent(k, self.n_maps, string='Converting to atlas label {1} out of {2} : {0:.2f}%...', rate=0, verbose=verbose)
        #     array_atlas[array[:, :, :, k] > 0] = k+1

        if self.n_maps == 1:  # Atlas stored on one map
            # n_labels = len(np.unique(self.to_array(0)))
            n_labels = int(np.max(self.to_array(0)))+1

        else:  # Atlas stored on several maps, one label on each
            n_labels = self.n_maps

        if bg_label is not None:
            if not isinstance(bg_label, tuple) or not len(bg_label) == 2:
                raise ValueError('Background label must be a length 2 tuple of shape (bg_label_id, bg_label_name).')

            bg_label_id, bg_label_name = bg_label
            if bg_label_id < 0 or bg_label_id >= n_labels:
                raise ValueError('Given background index out of range. {0} labels detected.'.format(n_labels))

            L1 = ['r{}'.format(k) for k in range(bg_label_id)]
            L2 = [bg_label_name]
            L3 = ['r{}'.format(k) for k in range(bg_label_id+1, n_labels)]

            L = L1+L2+L3

        else:
            L = ['r{}'.format(k) for k in range(n_labels)]

        return {'maps': nib.Nifti1Image(array, self._affine), 'labels': L}

    # _____________PUBLIC_TOOLS_____________ #
    def apply_mask(self, mask):
        '''
            Set the contribution of every voxels outside the mask to zero.

            Args:
                mask (nibabel.Nifti1Image): Nifti1Image with 0 or 1 array. 0: outside the mask, 1: inside.
        '''
        if not isinstance(mask, nib.Nifti1Image):
            raise ValueError('Mask must be an instance of nibabel.Nifti1Image')

        if self.maps is not None:
            mask_array = self._flatten_array(mask.get_fdata()).astype(self._dtype)
            filter_matrix = scipy.sparse.diags(mask_array, format='csr').astype(self._dtype)
            self.maps = filter_matrix.dot(self.maps)

        self._mask = mask

    def apply_atlas(self, atlas, inplace=False):
        new_maps = self if inplace else copy.copy(self)
        new_maps._atlas = Atlas(atlas)
        new_maps._atlas_filter_matrix = new_maps._build_atlas_filter_matrix()
        new_maps._refresh_atlas_maps()

        return new_maps

    def randomize(self, size, p=None, override_mask=False, inplace=False):
        '''
        Randomize the maps by sampling n_peaks of weight 1 (may overlap) over n_maps maps.

        Args:
            size: int or size 2 tuple or 1D numpy.ndarray.
                If 1D numpy.ndarray, creates as many maps as the array length
                and sample the given number of peaks in each maps.
                Each peak has a weight 1 and the weights of the peaks sampled
                on the same voxel of the same map are added.
                If tuple (n_peaks, n_maps) given, creates n_maps, samples
                n_peaks and assign each peak to a map uniformly.
                If int given, equivalent as tuple with n_maps=1.
            p (Maps instance or np.ndarray): (Optional) Distribution of probability of the peaks over the voxels.
                The distribution may be given either by a Maps instance containing 1 map or a np.ndarray of same shape as the box of the current Maps instance (Ni, Nj, Nk).
                If None, sample uniformly accros the box. Default: None
            override_mask (bool): (Optional) If False, use the mask given when initializing the Maps object.
                Important : the given distribution p is then shrinked and re-normalized.
                If True, no mask is used and p is unchanged. Default : False.
            inplace (bool): (Optional) Performs the sampling inplace (True) or creates a new instance (False). Default False.

        Returns:
            (Maps instance) Self or a copy depending on inplace.
        '''
        if self._Ni is None or self._Nj is None or self._Nk is None:
            raise ValueError('Invalid box size ({}, {}, {}).'.format(self._Ni, self._Nj, self._Nk))

        if isinstance(size, int):
            n_peaks, n_maps = size, 1

        elif isinstance(size, tuple):
            if len(size) != 2:
                raise ValueError('If given size is a tuple, must be of size 2 : (n_peaks, n_maps).')
            n_peaks, n_maps = size

        elif isinstance(size, np.ndarray):
            if len(size.shape) != 1:
                raise ValueError('Given size array not understood. Must be a 1D numpy array.')
            n_peaks, n_maps = np.sum(size), size.shape[0]

        else:
            raise ValueError('Given size not understood.')

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

        if isinstance(size, np.ndarray):
            maps_samples = np.repeat(np.arange(n_maps), repeats=size)
        else:
            maps_samples = np.random.choice(n_maps, size=n_peaks)

        for i in range(n_peaks):
            map_id = maps_samples[i]
            maps[voxels_samples[i], map_id] += 1

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

    def threshold(self, threshold, inplace=False):
        '''
            Threshold each map according to the given threshold.

            Args:
                threshold: All value greater or equal to threshold remains
                    unchanged, all the others are set to zero.
                inplace (bool, optional): If True performs the normalization
                    inplace else create a new instance.

            Returns:
                (Maps) Self or a copy depending on inplace.
        '''
        new_maps = self if inplace else copy.copy(self)
        new_maps.maps[new_maps.maps < threshold] = 0.
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

        def smooth_pool(map_ids, self, sigma):

            csc_matrices = []
            n_tot = len(map_ids)
            count = 0
            for k in map_ids:
                print_percent(count, n_tot, 'Smoothing {1} out of {2}... {0:.1f}%', rate=0, verbose=verbose, prefix='Maps')
                count += 1

                if not self.save_memory:
                    array = self._get_maps(map_id=k, dense=True)
                else:
                    array = self.to_array(k)

                array_smoothed = gaussian_filter(array, sigma=sigma)
                array_smoothed = self._flatten_array(array_smoothed, _2D=1)
                matrix = scipy.sparse.csc_matrix(array_smoothed)
                csc_matrices.append(matrix)

            return csc_matrices

        nb_jobs = multiprocessing.cpu_count()//2
        splitted_range = np.array_split(map_ids, nb_jobs)
        csc_matrices = np.concatenate(Parallel(n_jobs=nb_jobs, backend='threading')(delayed(smooth_pool)(sub_array, self, sigma) for sub_array in splitted_range))

        csr_maps = scipy.sparse.hstack(csc_matrices)
        csr_maps = scipy.sparse.csr_matrix(csr_maps)

        new_maps = self if inplace else copy.copy(self)
        new_maps.maps = csr_maps

        return new_maps

    def split(self, prop=0.5, random_state=None):
        np.random.seed(random_state)

        maps_A = Maps.copy_header(self)
        maps_B = Maps.copy_header(self)

        id_sub_maps_A = np.sort(np.random.choice(np.arange(self.n_maps), np.ceil(prop*self.n_maps).astype(int), replace=False))
        id_sub_maps_B = np.sort(np.delete(np.arange(self.n_maps), id_sub_maps_A))

        def filter_matrix(array):
            M = scipy.sparse.lil_matrix((self.n_maps, array.shape[0]))
            for k in range(array.shape[0]):
                M[array[k], k] = 1
            return scipy.sparse.csr_matrix(M)

        filter_matrix_A = filter_matrix(id_sub_maps_A)
        filter_matrix_B = filter_matrix(id_sub_maps_B)

        maps_A.maps = self.maps.dot(filter_matrix_A)
        maps_B.maps = self.maps.dot(filter_matrix_B)

        return maps_A, maps_B

    def shuffle(self, random_state=None, inplace=False):
        """Shuffle the maps index.

        Shuffle the maps index. For example, if 3 maps are stored in this
        order (1, 2, 3), shuffling them may lead to a new order (2, 1, 3).

        Arguments:
            random_state {int} -- Used to initialize the numpy random seed.
        """
        np.random.seed(random_state)

        new_maps = self if inplace else copy.copy(self)
        permutation = np.random.permutation(new_maps.n_m)
        M = scipy.sparse.lil_matrix((new_maps.n_m, new_maps.n_m))

        for k in range(new_maps.n_m):
            M[k, permutation[k]] = 1.
        M = scipy.sparse.csr_matrix(M)

        new_maps.maps = new_maps.maps.dot(M)

        return new_maps

    # _____________STATISTICS_____________ #
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
        maps_squared = maps.multiply(maps)  # Squared element wise

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

    def cov(self, atlas=True, bias=False, shrink=None, sparse=False, ignore_bg=True, verbose=False):
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
            labels = self._atlas.get_labels(ignore_bg=ignore_bg)
            if ignore_bg and self._atlas.has_background():
                maps[self._atlas.bg_index, :] = 0

        if verbose:
            print('Computing cov matrix')
        e1 = scipy.sparse.csr_matrix(np.ones(self.n_maps)/(self.n_maps-ddof)).transpose().astype(self._dtype)
        e2 = scipy.sparse.csr_matrix(np.ones(self.n_maps)/(self.n_maps)).transpose().astype(self._dtype)

        M1 = maps.dot(e1)
        M2 = maps.dot(e2)
        M3 = maps.dot(maps.transpose())/((self.n_maps-ddof))

        # Empirical covariance matrix
        S = M3 - M1.dot(M2.transpose())

        del M1, M2, M3

        print('To dense...') if verbose else None
        if not sparse:
            S = S.toarray()

        if shrink == 'LW':
            print('Shrink') if verbose else None
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
            print_percent(k, self.n_maps, 'Iterative smooth avg var {1} out of {2}... {0:.1f}%', rate=0, verbose=verbose, prefix='Maps')

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

        if not self.save_memory:
            avg_map = self.array_to_map(avg_map)
            var_map = self.array_to_map(var_map)

        avg._set_maps(avg_map, refresh_atlas_maps=False)
        var._set_maps(var_map, refresh_atlas_maps=False)

        if self._has_atlas():
            avg._maps_atlas = avg_map_atlas
            var._maps_atlas = var_map_atlas

        return avg, var
