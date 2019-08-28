import unittest
from hypothesis import given, settings
import hypothesis.strategies as strats
import numpy as np
import scipy

from meta_analysis import Maps
from globals_test import gray_mask, template, atlas, df_ex, Ni, Nj, Nk, \
 groupby_col, affine, array2D, array3D, array4D_1, array4D_2, \
 example_maps, array2D_missmatch, array3D_missmatch, array4D_1_missmatch, \
 array4D_2_missmatch, gray_mask_missmatch, fmri_img, gray_mask_2, atlas_2

class DataFrameInitTestCase(unittest.TestCase):
    def test_allowed_template(self):
        maps = Maps(df_ex, template=template, groupby_col=groupby_col)
        self.assertEqual(maps.n_maps, 2)
        self.assertFalse(maps._has_mask())
        self.assertFalse(maps._has_atlas())

    def test_allowed_template_mask(self):
        maps = Maps(df_ex, template=template, groupby_col=groupby_col, mask=gray_mask)
        self.assertTrue(maps._has_mask())

    def test_allowed_template_atlas(self):
        maps = Maps(df_ex, template=template, groupby_col=groupby_col, atlas=atlas)
        self.assertTrue(maps._has_atlas())

    def test_allowed_template_mask_atlas(self):
        maps = Maps(df_ex, template=template, groupby_col=groupby_col, mask=gray_mask, atlas=atlas)
        self.assertTrue(maps._has_mask())
        self.assertTrue(maps._has_atlas())

    def test_allowed_manual(self):
        maps = Maps(df_ex, Ni=Ni, Nj=Nj, Nk=Nk, affine=affine, groupby_col=groupby_col)
        self.assertEqual(maps.n_maps, 2)
        self.assertFalse(maps._has_mask())
        self.assertFalse(maps._has_atlas())

    def test_allowed_manual_mask(self):
        maps = Maps(df_ex, Ni=Ni, Nj=Nj, Nk=Nk, affine=affine, groupby_col=groupby_col, mask=gray_mask)
        self.assertTrue(maps._has_mask())

    def test_allowed_manual_atlas(self):
        maps = Maps(df_ex, Ni=Ni, Nj=Nj, Nk=Nk, affine=affine, groupby_col=groupby_col, atlas=atlas)
        self.assertTrue(maps._has_atlas())

    def test_allowed_manual_mask_atlas(self):
        maps = Maps(df_ex, Ni=Ni, Nj=Nj, Nk=Nk, affine=affine, groupby_col=groupby_col, mask=gray_mask, atlas=atlas)
        self.assertTrue(maps._has_mask())
        self.assertTrue(maps._has_atlas())

    def test_forbidden(self):
        with self.assertRaises(TypeError):
            maps = Maps(df_ex)

        with self.assertRaises(TypeError):
            maps = Maps(df_ex, groupby_col=groupby_col)

        with self.assertRaises(TypeError):
            maps = Maps(df_ex, Ni=Ni, Nj=Nj, affine=affine, groupby_col=groupby_col)

        with self.assertRaises(TypeError):
            maps = Maps(df_ex, Ni=Ni, Nj=Nj, Nk=Nk, groupby_col=groupby_col)

        with self.assertRaises(TypeError):
            maps = Maps(df_ex, Ni=Ni, Nj=Nj, Nk=Nk, affine=affine)

        with self.assertRaises(TypeError):
            maps = Maps(df_ex, Ni=Ni, Nj=Nj, Nk=Nk)

        with self.assertRaises(TypeError):
            maps = Maps(df_ex, template=template)

    def test_mask_missmatch_manual(self):
        with self.assertRaises(ValueError):
            maps = Maps(df_ex, Ni=Ni, Nj=Nj, Nk=Nk, affine=affine, groupby_col=groupby_col, mask=gray_mask_missmatch)
            maps = Maps(df_ex, Ni=Ni-1, Nj=Nj, Nk=Nk, affine=affine, groupby_col=groupby_col, mask=gray_mask)

    def test_atlas_missmatch_manual(self):
        with self.assertRaises(ValueError):
            maps = Maps(df_ex, template, groupby_col=groupby_col, mask=gray_mask_missmatch)

class ArrayInitTestCase(unittest.TestCase):
    def test_allowed_2D_template(self):
        maps = Maps(array2D, template=template)
    def test_allowed_3D_template(self):
        maps = Maps(array3D, template=template)
    def test_allowed_4D_1_template(self):
        maps = Maps(array4D_1, template=template)
    def test_allowed_4D_2_template(self):
        maps = Maps(array4D_2, template=template)

    def test_allowed_2D_manual(self):
        maps = Maps(array2D, Ni=Ni, Nj=Nj, Nk=Nk, affine=affine)
    def test_allowed_3D_manual(self):
        maps = Maps(array3D, Ni=Ni, Nj=Nj, Nk=Nk, affine=affine)
    def test_allowed_4D_1_manual(self):
        maps = Maps(array4D_1, Ni=Ni, Nj=Nj, Nk=Nk, affine=affine)
    def test_allowed_4D_2_manual(self):
        maps = Maps(array4D_2, Ni=Ni, Nj=Nj, Nk=Nk, affine=affine)

    def test_allowed_2D_manual_no_affine(self):
        maps = Maps(array2D, Ni=Ni, Nj=Nj, Nk=Nk)
    def test_allowed_3D_manual_no_affine(self):
        maps = Maps(array3D, Ni=Ni, Nj=Nj, Nk=Nk)
    def test_allowed_4D_1_manual_no_affine(self):
        maps = Maps(array4D_1, Ni=Ni, Nj=Nj, Nk=Nk)
    def test_allowed_4D_2_manual_no_affine(self):
        maps = Maps(array4D_2, Ni=Ni, Nj=Nj, Nk=Nk)

    def test_allowed_3D_manual_no_data(self):
        maps = Maps(array3D)
        self.assertEqual(array3D.shape, (maps._Ni, maps._Nj, maps._Nk))
    def test_allowed_4D_1_manual_no_data(self):
        maps = Maps(array4D_1)
        self.assertEqual(array4D_1.shape[:-1], (maps._Ni, maps._Nj, maps._Nk))
    def test_allowed_4D_2_manual_no_data(self):
        maps = Maps(array4D_2)
        self.assertEqual(array4D_2.shape[:-1], (maps._Ni, maps._Nj, maps._Nk))

    def test_forbidden(self):
        with self.assertRaises(TypeError):
            maps = Maps(array2D)

    def test_forbidden_manual(self):
        with self.assertRaises(TypeError):
            maps = Maps(array2D, Ni=Ni, Nj=Nj, affine=affine)
        with self.assertRaises(TypeError):
            maps = Maps(array2D, Ni=Ni, Nj=Nj)

    @given(di=strats.integers(min_value=-Ni+1, max_value=Ni-1),
           dj=strats.integers(min_value=-Ni+1, max_value=Ni-1),
           dk=strats.integers(min_value=-Ni+1, max_value=Ni-1))
    @settings(max_examples=2)
    def test_box_missmatch_manual(self, di, dj, dk):
        if (di, dj, dk) == (0, 0, 0):
            return
        with self.assertRaises(ValueError):
            maps = Maps(array2D, Ni=Ni+di, Nj=Nj+dj, Nk=Nk+dk, affine=affine)
        # with self.assertRaises(TypeError):
        #     maps = Maps(array3D, Ni=Ni+di, Nj=Nj+dj, Nk=Nk+dk, affine=affine)
        # with self.assertRaises(TypeError):
        #     maps = Maps(array4D_1, Ni=Ni+di, Nj=Nj+dj, Nk=Nk+dk, affine=affine)
        # with self.assertRaises(TypeError):
        #     maps = Maps(array4D_2, Ni=Ni+di, Nj=Nj+dj, Nk=Nk+dk, affine=affine)

    def test_box_missmatch_template(self):
        with self.assertRaises(ValueError):
            maps = Maps(array2D_missmatch, template=template)
        with self.assertRaises(ValueError):
            maps = Maps(array3D_missmatch, template=template)
        with self.assertRaises(ValueError):
            maps = Maps(array4D_1_missmatch, template=template)
        with self.assertRaises(ValueError):
            maps = Maps(array4D_2_missmatch, template=template)

    def test_mask_missmatch_manual(self):
        with self.assertRaises(ValueError):
            maps = Maps(array2D, Ni=Ni, Nj=Nj, Nk=Nk, affine=affine, mask=gray_mask_missmatch)
        with self.assertRaises(ValueError):
            maps = Maps(array3D, Ni=Ni, Nj=Nj, Nk=Nk, affine=affine, mask=gray_mask_missmatch)
        with self.assertRaises(ValueError):
            maps = Maps(array4D_1, Ni=Ni, Nj=Nj, Nk=Nk, affine=affine, mask=gray_mask_missmatch)
        with self.assertRaises(ValueError):
            maps = Maps(array4D_2, Ni=Ni, Nj=Nj, Nk=Nk, affine=affine, mask=gray_mask_missmatch)

    def test_mask_missmatch_template(self):
        with self.assertRaises(ValueError):
            maps = Maps(array2D, template=template, mask=gray_mask_missmatch)
        with self.assertRaises(ValueError):
            maps = Maps(array3D, template=template, mask=gray_mask_missmatch)
        with self.assertRaises(ValueError):
            maps = Maps(array4D_1, template=template, mask=gray_mask_missmatch)
        with self.assertRaises(ValueError):
            maps = Maps(array4D_2, template=template, mask=gray_mask_missmatch)

class ShapeInitTestCase(unittest.TestCase):
    def setUp(self):
        self.shape1D = Ni*Nj*Nk
        self.shape2D = (Ni*Nj*Nk, 2)
        self.shape1D_missmatch = Ni*Nj*Nk+1
        self.shape2D_missmatch = (Ni*Nj*Nk+1, 2)

    def test_allowed_1D_template(self):
        maps = Maps(self.shape1D, template=template)
    def test_allowed_2D_template(self):
        maps = Maps(self.shape2D, template=template)

    def test_allowed_1D_manual(self):
        maps = Maps(self.shape1D, Ni=Ni, Nj=Nj, Nk=Nk, affine=affine)
    def test_allowed_2D_manual(self):
        maps = Maps(self.shape2D, Ni=Ni, Nj=Nj, Nk=Nk, affine=affine)

    def test_allowed_1D_manual_no_affine(self):
        maps = Maps(self.shape1D, Ni=Ni, Nj=Nj, Nk=Nk)
    def test_allowed_2D_manual_no_affine(self):
        maps = Maps(self.shape2D, Ni=Ni, Nj=Nj, Nk=Nk)

    def test_forbidden(self):
        with self.assertRaises(TypeError):
            maps = Maps(self.shape1D)
        with self.assertRaises(TypeError):
            maps = Maps(self.shape2D)

    def test_forbidden_manual(self):
        with self.assertRaises(TypeError):
            maps = Maps(self.shape1D, Ni=Ni, Nj=Nj, affine=affine)
        with self.assertRaises(TypeError):
            maps = Maps(self.shape2D, Ni=Ni, Nj=Nj, affine=affine)

    def test_box_missmatch_manual(self):
        with self.assertRaises(ValueError):
            maps = Maps(self.shape1D_missmatch, Ni=Ni, Nj=Nj, Nk=Nk, affine=affine)
        with self.assertRaises(ValueError):
            maps = Maps(self.shape2D_missmatch, Ni=Ni, Nj=Nj, Nk=Nk, affine=affine)

    def test_box_missmatch_template(self):
        with self.assertRaises(ValueError):
            maps = Maps(self.shape1D_missmatch, template=template)
        with self.assertRaises(ValueError):
            maps = Maps(self.shape2D_missmatch, template=template)


class NoneInitTestCase(unittest.TestCase):
    def test_allowed_template(self):
        maps = Maps(template=template)

    def test_allowed_manual(self):
        maps = Maps(Ni=Ni, Nj=Nj, Nk=Nk, affine=affine, mask=gray_mask)

    def test_allowed_template_mask(self):
        maps = Maps(template=template)

    def test_allowed_manual_mask(self):
        maps = Maps(Ni=Ni, Nj=Nj, Nk=Nk, affine=affine, mask=gray_mask)

    def test_allowed_template_atlas(self):
        maps = Maps(template=template, atlas=atlas)

    def test_allowed_manual_atlas(self):
        maps = Maps(Ni=Ni, Nj=Nj, Nk=Nk, affine=affine, atlas=atlas)

    def test_allowed_manual_no_affine(self):
        maps = Maps(Ni=Ni, Nj=Nj, Nk=Nk)

    def test_forbidden(self):
        with self.assertRaises(TypeError):
            maps = Maps()

        with self.assertRaises(TypeError):
            maps = Maps(Ni=Ni, Nj=Nj, affine=affine)

class ImgInitTestCase(unittest.TestCase):
    def test_allowed(self):
        maps = Maps(fmri_img)

    def test_allowed_atlas(self):
        maps = Maps(fmri_img, atlas=atlas_2)

    def test_allowed_mask(self):
        maps = Maps(fmri_img, mask=gray_mask_2)

    def test_atlas_missmatch(self):
        with self.assertRaises(ValueError):
            maps = Maps(fmri_img, atlas=atlas)

    def test_mask_missmatch(self):
        with self.assertRaises(ValueError):
            maps = Maps(fmri_img, mask=gray_mask)
