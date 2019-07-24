import unittest
import numpy as np
import nibabel as nib

from meta_analysis import Maps
from globals_test import affine

class ApplyMaskTestCase(unittest.TestCase):
    def setUp(self):
        self.array = np.array([[[1, 2, 3],
                                [4, 5, 6],
                                [7, 8, 9]]])

        self.array2 = np.array([[[[1, 2, 3],
                                 [4, 5, 6],
                                 [7, 8, 9]]],
                                 [[[9, 8, 7],
                                 [6, 5, 4],
                                 [3, 2, 1]]]])

        self.mask_data = np.array([[[0, 0, 1],
                                    [0, 1, 0],
                                    [0, 0, 0]]])

        self.mask = nib.Nifti1Image(self.mask_data, affine)
        
        self.expected = np.array([[[0, 0, 3],
                                   [0, 5, 0],
                                   [0, 0, 0]]])

        self.expected2 = np.array([[[[0, 0, 3],
                                   [0, 5, 0],
                                   [0, 0, 0]]],
                                   [[[0, 0, 7],
                                   [0, 5, 0],
                                   [0, 0, 0]]]])
        
        self.Ni, self.Nj, self.Nk = self.mask_data.shape

    def test_mask_init_one_map(self):
        maps = Maps(self.array, Ni=self.Ni, Nj=self.Nj, Nk=self.Nk, mask=self.mask)
        self.assertTrue(np.array_equal(maps.to_array(), self.expected))

    def test_mask_init_two_maps(self):
        maps2 = Maps(self.array2, Ni=self.Ni, Nj=self.Nj, Nk=self.Nk, mask=self.mask)
        self.assertTrue(np.array_equal(maps2.to_array(0), self.expected2[0]))
        self.assertTrue(np.array_equal(maps2.to_array(1), self.expected2[1]))
    
    def test_mask_apply_one_map(self):
        maps = Maps(self.array, Ni=self.Ni, Nj=self.Nj, Nk=self.Nk)
        maps.apply_mask(self.mask)

        self.assertTrue(np.array_equal(maps.to_array(), self.expected))

    def test_mask_apply_two_maps(self):
        maps2 = Maps(self.array2, Ni=self.Ni, Nj=self.Nj, Nk=self.Nk)
        maps2.apply_mask(self.mask)

        self.assertTrue(np.array_equal(maps2.to_array(0), self.expected2[0]))
        self.assertTrue(np.array_equal(maps2.to_array(1), self.expected2[1]))

class NormalizeTestCase(unittest.TestCase):
    def setUp(self):
        self.array = np.array([[[1, 0, 0],
                                [0, 1, 0],
                                [0, 0, 2]]])

        self.array2 = np.array([[[[1, 0, 0],
                                 [0, 1, 0],
                                 [0, 0, 2]]],
                                 [[[0, 0, 0],
                                 [0, 1, 0],
                                 [0, 0, 0]]]])
        
        self.expected = np.array([[[0.25, 0, 0],
                                   [0, 0.25, 0],
                                   [0, 0, 0.5]]])

        self.expected2 = np.array([[[[0.25, 0, 0],
                                   [0, 0.25, 0],
                                   [0, 0, 0.5]]],
                                   [[[0, 0, 0],
                                   [0, 1, 0],
                                   [0, 0, 0]]]])

        self.Ni, self.Nj, self.Nk = self.array.shape

    def test_normalize_one_map(self):
        maps = Maps(self.array, Ni=self.Ni, Nj=self.Nj, Nk=self.Nk)
        maps_ = maps.normalize(inplace=False)
        maps.normalize(inplace=True)

        self.assertTrue(np.array_equal(maps.to_array(), self.expected))
        self.assertTrue(np.array_equal(maps_.to_array(), self.expected))

    def test_normalize_two_maps(self):
        maps2 = Maps(self.array2, Ni=self.Ni, Nj=self.Nj, Nk=self.Nk)
        maps2_ = maps2.normalize(inplace=False)
        maps2.normalize(inplace=True)

        self.assertTrue(np.array_equal(maps2.to_array(0), self.expected2[0]))
        self.assertTrue(np.array_equal(maps2.to_array(1), self.expected2[1]))
        self.assertTrue(np.array_equal(maps2_.to_array(0), self.expected2[0]))
        self.assertTrue(np.array_equal(maps2_.to_array(1), self.expected2[1]))




