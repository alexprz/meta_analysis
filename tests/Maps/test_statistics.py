import unittest
import numpy as np
import nibabel as nib

from meta_analysis import Maps
from globals_test import affine

class NPeaksTestCase(unittest.TestCase):
    def setUp(self):
        self.array = np.array([[[1, 2, 3],
                                [4, 5, 6],
                                [7, 8, 9]]])

        self.array2 = np.array([[[[1, 9], [2, 8], [3, 7]],
                                 [[4, 6], [5, 5], [6, 4]],
                                 [[7, 3], [8, 2], [9, 1]]]])

        self.mask_data = np.array([[[0, 0, 1],
                                    [0, 1, 0],
                                    [0, 0, 0]]])

        self.mask = nib.Nifti1Image(self.mask_data, affine)
        
        self.expected = np.array([45])
        self.expected2 = np.array([45, 45])

        self.expected_masked = np.array([8])
        self.expected_masked2 = np.array([8, 12])
        
        self.Ni, self.Nj, self.Nk = self.mask_data.shape

    def test_one_map(self):
        maps = Maps(self.array, Ni=self.Ni, Nj=self.Nj, Nk=self.Nk)
        self.assertTrue(np.array_equal(maps.n_peaks(), self.expected))

    def test_two_maps(self):
        maps2 = Maps(self.array2, Ni=self.Ni, Nj=self.Nj, Nk=self.Nk)
        self.assertTrue(np.array_equal(maps2.n_peaks(), self.expected2))

    def test_one_map_masked(self):
        maps = Maps(self.array, Ni=self.Ni, Nj=self.Nj, Nk=self.Nk, mask=self.mask)
        self.assertTrue(np.array_equal(maps.n_peaks(), self.expected_masked))

    def test_two_maps_masked(self):
        maps2 = Maps(self.array2, Ni=self.Ni, Nj=self.Nj, Nk=self.Nk, mask=self.mask)
        self.assertTrue(np.array_equal(maps2.n_peaks(), self.expected_masked2))

class MaxTestCase(unittest.TestCase):
    def setUp(self):
        self.array = np.array([[[1, 2, 3],
                                [4, 5, 6],
                                [7, 8, 9]]])

        self.array2 = np.array([[[[1, 9], [2, 8], [3, 7]],
                                 [[4, 6], [5, 5], [6, 4]],
                                 [[7, 3], [8, 2], [9, 1]]]])

        self.mask_data = np.array([[[0, 0, 1],
                                    [0, 1, 0],
                                    [0, 0, 0]]])

        self.mask = nib.Nifti1Image(self.mask_data, affine)
        
        self.expected0 = 9.
        self.expected1 = np.array([[9]])
        self.expected2 = np.array([[9, 9]])

        self.expected_masked1 = np.array([[5]])
        self.expected_masked2 = np.array([[5, 7]])
        
        self.Ni, self.Nj, self.Nk = self.mask_data.shape

    def test_one_map(self):
        maps = Maps(self.array, Ni=self.Ni, Nj=self.Nj, Nk=self.Nk)
        self.assertTrue(np.array_equal(maps.max(axis=0), self.expected1))
        self.assertTrue(np.array_equal(maps.max(), self.expected0))

    def test_two_maps(self):
        maps2 = Maps(self.array2, Ni=self.Ni, Nj=self.Nj, Nk=self.Nk)
        self.assertTrue(np.array_equal(maps2.max(axis=0), self.expected2))
        self.assertTrue(np.array_equal(maps2.max(), self.expected0))

    def test_one_map_masked(self):
        maps = Maps(self.array, Ni=self.Ni, Nj=self.Nj, Nk=self.Nk, mask=self.mask)
        self.assertTrue(np.array_equal(maps.max(axis=0), self.expected_masked1))
        self.assertTrue(np.array_equal(maps.max(), 5.))

    def test_two_maps_masked(self):
        maps2 = Maps(self.array2, Ni=self.Ni, Nj=self.Nj, Nk=self.Nk, mask=self.mask)
        self.assertTrue(np.array_equal(maps2.max(axis=0), self.expected_masked2))
        self.assertTrue(np.array_equal(maps2.max(), 7.))

class SumTestCase(unittest.TestCase):
    def setUp(self):
        self.array = np.array([[[1, 2, 3],
                                [4, 5, 6],
                                [7, 8, 9]]])

        self.array2 = np.array([[[[1, 9], [2, 8], [3, 7]],
                                 [[4, 6], [5, 5], [6, 4]],
                                 [[7, 3], [8, 2], [9, 1]]]])

        self.mask_data = np.array([[[0, 0, 1],
                                    [0, 1, 0],
                                    [0, 0, 0]]])

        self.mask = nib.Nifti1Image(self.mask_data, affine)
        
        self.expected0 = 45.
        self.expected1 = np.array([[45]])
        self.expected2 = np.array([[45, 45]])

        self.expected_masked1 = np.array([[8]])
        self.expected_masked2 = np.array([[8, 12]])
        
        self.Ni, self.Nj, self.Nk = self.mask_data.shape

    def test_one_map(self):
        maps = Maps(self.array, Ni=self.Ni, Nj=self.Nj, Nk=self.Nk)
        self.assertTrue(np.array_equal(maps.sum(axis=0), self.expected1))
        self.assertTrue(np.array_equal(maps.sum(), self.expected0))

    def test_two_maps(self):
        maps2 = Maps(self.array2, Ni=self.Ni, Nj=self.Nj, Nk=self.Nk)
        self.assertTrue(np.array_equal(maps2.sum(axis=0), self.expected2))
        self.assertTrue(np.array_equal(maps2.sum(), 2*self.expected0))

    def test_one_map_masked(self):
        maps = Maps(self.array, Ni=self.Ni, Nj=self.Nj, Nk=self.Nk, mask=self.mask)
        self.assertTrue(np.array_equal(maps.sum(axis=0), self.expected_masked1))
        self.assertTrue(np.array_equal(maps.sum(), 8.))

    def test_two_maps_masked(self):
        maps2 = Maps(self.array2, Ni=self.Ni, Nj=self.Nj, Nk=self.Nk, mask=self.mask)
        self.assertTrue(np.array_equal(maps2.sum(axis=0), self.expected_masked2))
        self.assertTrue(np.array_equal(maps2.sum(), 20.))
