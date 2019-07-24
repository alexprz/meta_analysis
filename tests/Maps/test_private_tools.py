import unittest
from hypothesis import given
import numpy as np
import scipy

from meta_analysis import Maps
from .globals import random_permitted_case_3D, random_permitted_case_1D, empty_maps, random_maps, gray_mask, template, atlas

class CoordinatesTestCase(unittest.TestCase):
    @given(args=random_permitted_case_3D())
    def test_coord_id(self, args):
        i, j, k, Ni, Nj, Nk = args
        p = Maps.coord_to_id(i, j, k, Ni, Nj, Nk)
        self.assertEqual(Maps.id_to_coord(p, Ni, Nj, Nk), (i, j, k))

    @given(args=random_permitted_case_1D())
    def test_id_coord(self, args):
        p, Ni, Nj, Nk = args
        i, j, k = Maps.id_to_coord(p, Ni, Nj, Nk)
        self.assertEqual(Maps.coord_to_id(i, j, k, Ni, Nj, Nk), p)

class MaskTestCase(unittest.TestCase):
    def test_no_mask(self):
        maps = Maps(np.array([[0.]]), Ni=1, Nj=1, Nk=1)
        self.assertFalse(maps._has_mask())

    def test_mask(self):
        maps = Maps(template=template, mask=gray_mask)
        self.assertTrue(maps._has_mask())

class AtlasTestCase(unittest.TestCase):
    def test_no_atlas(self):
        maps = Maps(np.array([[0.]]), Ni=1, Nj=1, Nk=1)
        self.assertFalse(maps._has_atlas())

    def test_atlas(self):
        maps = Maps(template=template, atlas=atlas)
        self.assertTrue(maps._has_atlas())


