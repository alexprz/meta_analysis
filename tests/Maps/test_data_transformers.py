import unittest
from hypothesis import given
import numpy as np
import scipy

from meta_analysis import Maps
from globals_test import random_permitted_case_3D, random_permitted_case_1D, empty_maps, random_maps, gray_mask, template, atlas, affine

class ToArrayTestCase(unittest.TestCase):
    def setUp(self):
        self.array2D = np.random.rand(2, 2)
        self.array4D = np.random.rand(3, 3, 3, 1)
        self.array4D_2 = np.random.rand(3, 3, 3, 2)
        self.array3D = np.random.rand(3, 3, 3)

    def test_array_2D(self):
        maps = Maps(self.array2D, Ni=2, Nj=1, Nk=1)
        self.assertTrue(np.array_equal(maps.to_array(), self.array2D.reshape((2, 1, 1, 2))))
    
    def test_array_2D_one(self):
        maps = Maps(self.array2D, Ni=2, Nj=1, Nk=1)
        self.assertTrue(np.array_equal(maps.to_array(0), self.array2D.reshape((2, 1, 1, 2))[:, :, :, 0]))
        self.assertTrue(np.array_equal(maps.to_array(1), self.array2D.reshape((2, 1, 1, 2))[:, :, :, 1]))
    
    def test_array_3D(self):
        maps = Maps(self.array3D)
        self.assertTrue(np.array_equal(maps.to_array(), self.array3D))

    def test_array_4D(self):
        maps = Maps(self.array4D)
        self.assertTrue(np.array_equal(maps.to_array(), self.array4D[:, :, :, 0]))

    def test_array_4D_one(self):
        maps = Maps(self.array4D)
        self.assertTrue(np.array_equal(maps.to_array(0), self.array4D[:, :, :, 0]))

    def test_array_4D_2_all(self):
        maps = Maps(self.array4D_2)
        self.assertTrue(np.array_equal(maps.to_array(), self.array4D_2))
        
    def test_array_4D_2_one(self):
        maps = Maps(self.array4D_2)
        self.assertTrue(np.array_equal(maps.to_array(0), self.array4D_2[:, :, :, 0]))
        
