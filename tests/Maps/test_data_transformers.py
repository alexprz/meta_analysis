import unittest
from hypothesis import given
import numpy as np
import scipy

from meta_analysis import Maps
from globals_test import random_permitted_case_3D, random_permitted_case_1D, empty_maps, random_maps, gray_mask, template, atlas, affine, fmri_img

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
        
class ToImgTestCase(unittest.TestCase):
    def setUp(self):
        self.array2D = np.random.rand(2, 2)
        self.array4D = np.random.rand(3, 3, 3, 1)
        self.array4D_2 = np.random.rand(3, 3, 3, 2)
        self.array3D = np.random.rand(3, 3, 3)
        self.affine = np.eye(4)

    def test_array_2D(self):
        maps = Maps(self.array2D, Ni=2, Nj=1, Nk=1, affine=self.affine)
        img = maps.to_img()
        self.assertTrue(np.array_equal(img.get_fdata(), self.array2D.reshape((2, 1, 1, 2))))
        self.assertTrue(np.array_equal(img.affine, self.affine))
    
    def test_array_2D_one(self):
        maps = Maps(self.array2D, Ni=2, Nj=1, Nk=1, affine=self.affine)
        img0 = maps.to_img(0)
        img1 = maps.to_img(1)
        self.assertTrue(np.array_equal(img0.get_fdata(), self.array2D.reshape((2, 1, 1, 2))[:, :, :, 0]))
        self.assertTrue(np.array_equal(img1.get_fdata(), self.array2D.reshape((2, 1, 1, 2))[:, :, :, 1]))
        self.assertTrue(np.array_equal(img0.affine, self.affine))    
        self.assertTrue(np.array_equal(img1.affine, self.affine))

    def test_array_3D(self):
        maps = Maps(self.array3D, affine=self.affine)
        img = maps.to_img()
        self.assertTrue(np.array_equal(img.get_fdata(), self.array3D))
        self.assertTrue(np.array_equal(img.affine, self.affine))

    def test_array_4D(self):
        maps = Maps(self.array4D, affine=self.affine)
        img = maps.to_img()
        self.assertTrue(np.array_equal(img.get_fdata(), self.array4D[:, :, :, 0]))
        self.assertTrue(np.array_equal(img.affine, self.affine))

    def test_array_4D_one(self):
        maps = Maps(self.array4D, affine=self.affine)
        img = maps.to_img(0)
        self.assertTrue(np.array_equal(img.get_fdata(), self.array4D[:, :, :, 0]))
        self.assertTrue(np.array_equal(img.affine, self.affine))

    def test_array_4D_2_all(self):
        maps = Maps(self.array4D_2, affine=self.affine)
        img = maps.to_img()
        self.assertTrue(np.array_equal(img.get_fdata(), self.array4D_2))
        self.assertTrue(np.array_equal(img.affine, self.affine))

    def test_array_4D_2_one(self):
        maps = Maps(self.array4D_2, affine=self.affine)
        img = maps.to_img(0)
        self.assertTrue(np.array_equal(img.get_fdata(), self.array4D_2[:, :, :, 0]))
        self.assertTrue(np.array_equal(img.affine, self.affine))

    def test_loaded_3D_img(self):
        maps = Maps(template)
        self.assertTrue(np.array_equal(maps.to_img().get_fdata(), template.get_fdata()))
    
    def test_loaded_4D_img(self):
        maps = Maps(fmri_img)
        self.assertTrue(np.array_equal(maps.to_img().get_fdata(), fmri_img.get_fdata()))

    def test_forbidden_array_2D(self):
        maps = Maps(self.array2D, Ni=2, Nj=1, Nk=1)
        with self.assertRaises(ValueError):
            maps.to_img()    
    def test_forbidden_array_2D_one(self):
        maps = Maps(self.array2D, Ni=2, Nj=1, Nk=1)
        with self.assertRaises(ValueError):
            maps.to_img()    
    def test_forbidden_array_3D(self):
        maps = Maps(self.array3D)
        with self.assertRaises(ValueError):
            maps.to_img()
    def test_forbidden_array_4D(self):
        maps = Maps(self.array4D)
        with self.assertRaises(ValueError):
            maps.to_img()
    def test_forbidden_array_4D_one(self):
        maps = Maps(self.array4D)
        with self.assertRaises(ValueError):
            maps.to_img()
    def test_forbidden_array_4D_2_all(self):
        maps = Maps(self.array4D_2)
        with self.assertRaises(ValueError):
            maps.to_img()        
    def test_forbidden_array_4D_2_one(self):
        maps = Maps(self.array4D_2)
        with self.assertRaises(ValueError):
            maps.to_img()
