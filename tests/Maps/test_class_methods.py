"""Test the classmethods of Maps."""
import unittest
import numpy as np

from meta_analysis import Maps
from globals_test import template, Ni, Nj, Nk, affine


class EmptyTestCase(unittest.TestCase):
    """Test Maps.empty classmethod."""

    def test_no_kwargs(self):
        """Test without kerwords."""
        with self.assertRaises(TypeError):
            Maps.empty()

    def test_template(self):
        """Test from template."""
        maps = Maps.empty(template=template)
        self.assertEqual(maps.Ni, Ni)
        self.assertEqual(maps.Nj, Nj)
        self.assertEqual(maps.Nk, Nk)
        self.assertTrue(np.array_equal(maps.affine, affine))
        self.assertTrue(maps.maps is None)


class ZerosTestCase(unittest.TestCase):
    """Test Maps.zeros classmethod."""

    def test_no_kwargs(self):
        """Test without kerwords."""
        with self.assertRaises(TypeError):
            Maps.zeros()

    def test_one_map_template(self):
        """Test creating one map from template."""
        maps = Maps.zeros(template=template)
        arr = maps.to_array()
        self.assertTrue(arr.all() == 0)
        self.assertEqual(maps.n_m, 1)
        self.assertEqual(maps.n_v, Ni*Nj*Nk)

    def test_one_map_manual(self):
        """Test creating one map from Ni Nj Nk."""
        maps = Maps.zeros(Ni=Ni, Nj=Nj, Nk=Nk)
        arr = maps.to_array()
        self.assertTrue(arr.all() == 0)
        self.assertEqual(maps.n_m, 1)
        self.assertEqual(maps.n_v, Ni*Nj*Nk)

    def test_two_maps_template(self):
        """Test creating several maps from template."""
        maps = Maps.zeros(n_maps=2, template=template)
        arr = maps.to_array()
        self.assertTrue(arr.all() == 0)
        self.assertEqual(maps.n_m, 2)
        self.assertEqual(maps.n_v, Ni*Nj*Nk)

    def test_two_maps_manual(self):
        """Test creating several maps from Ni Nj Nk."""
        maps = Maps.zeros(n_maps=2, Ni=Ni, Nj=Nj, Nk=Nk)
        arr = maps.to_array()
        self.assertTrue(arr.all() == 0)
        self.assertEqual(maps.n_m, 2)
        self.assertEqual(maps.n_v, Ni*Nj*Nk)


class RandomTestCase(unittest.TestCase):
    """Test Maps.random classmethod."""

    def setUp(self):
        """Set up sizes."""
        self.size1 = 10
        self.size1_bis = 10
        self.size2 = (10, 2)
        self.size3 = np.array([10, 5])

    def test_no_kwargs(self):
        """Test without keywords."""
        with self.assertRaises(TypeError):
            Maps.random(size=None)

    def test_one_map_template(self):
        """Test creating one map from template."""
        maps = Maps.random(self.size1, template=template)
        arr = maps.to_array()
        self.assertEqual(np.sum(arr), self.size1)
        self.assertEqual(maps.n_m, 1)
        self.assertEqual(maps.n_v, Ni*Nj*Nk)

    def test_one_map_manual(self):
        """Test creating one map from Ni Nj Nk."""
        maps = Maps.random(self.size1, Ni=Ni, Nj=Nj, Nk=Nk)
        arr = maps.to_array()
        self.assertEqual(np.sum(arr), self.size1)
        self.assertEqual(maps.n_m, 1)
        self.assertEqual(maps.n_v, Ni*Nj*Nk)

    def test_one_map_bis_template(self):
        """Test creating one map from template."""
        maps = Maps.random(self.size1_bis, template=template)
        arr = maps.to_array()
        self.assertEqual(np.sum(arr), self.size1)
        self.assertEqual(maps.n_m, 1)
        self.assertEqual(maps.n_v, Ni*Nj*Nk)

    def test_one_map_bis_manual(self):
        """Test creating one map from Ni Nj Nk."""
        maps = Maps.random(self.size1_bis, Ni=Ni, Nj=Nj, Nk=Nk)
        arr = maps.to_array()
        self.assertEqual(np.sum(arr), self.size1)
        self.assertEqual(maps.n_m, 1)
        self.assertEqual(maps.n_v, Ni*Nj*Nk)

    def test_two_maps_template(self):
        """Test creating several maps from template."""
        maps = Maps.random(self.size2, template=template)
        arr = maps.to_array()
        n_peaks, n_maps = self.size2
        self.assertEqual(np.sum(arr), n_peaks)
        self.assertEqual(maps.n_m, n_maps)
        self.assertEqual(maps.n_v, Ni*Nj*Nk)

    def test_two_maps_manual(self):
        """Test creating several maps from Ni Nj Nk."""
        maps = Maps.random(self.size2, Ni=Ni, Nj=Nj, Nk=Nk)
        arr = maps.to_array()
        n_peaks, n_maps = self.size2
        self.assertEqual(np.sum(arr), n_peaks)
        self.assertEqual(maps.n_m, n_maps)
        self.assertEqual(maps.n_v, Ni*Nj*Nk)

    def test_two_maps_array_template(self):
        """Test creating several maps from template."""
        maps = Maps.random(self.size3, template=template)
        arr = maps.to_array()
        self.assertEqual(np.sum(arr), np.sum(self.size3))
        self.assertEqual(maps.n_m, 2)
        self.assertEqual(maps.n_v, Ni*Nj*Nk)

    def test_two_maps_array_manual(self):
        """Test creating several maps from Ni Nj Nk."""
        maps = Maps.random(self.size3, Ni=Ni, Nj=Nj, Nk=Nk)
        arr = maps.to_array()
        self.assertEqual(np.sum(arr), np.sum(self.size3))
        self.assertEqual(maps.n_m, 2)
        self.assertEqual(maps.n_v, Ni*Nj*Nk)

    def test_random_seed(self):
        """Test if obtain same results with same seeds."""
        maps1 = Maps.random(self.size3, Ni=Ni, Nj=Nj, Nk=Nk, random_state=0)
        maps2 = Maps.random(self.size3, Ni=Ni, Nj=Nj, Nk=Nk, random_state=0)

        self.assertTrue(np.array_equal(maps1.to_array(), maps2.to_array()))


class CopyHeaderTestCase(unittest.TestCase):
    """Test Maps.copy_header classmethod."""

    def test_template(self):
        """Test copy from template."""
        maps1 = Maps(template=template)
        maps2 = Maps.copy_header(maps1)
        self.assertEqual(maps1.Ni, maps2.Ni)
        self.assertEqual(maps1.Nj, maps2.Nj)
        self.assertEqual(maps1.Nk, maps2.Nk)
        self.assertTrue(np.array_equal(maps1.affine, maps2.affine))

    def test_manual(self):
        """Test copy from Ni Nj Nk."""
        maps1 = Maps(Ni=Ni, Nj=Nj, Nk=Nk)
        maps2 = Maps.copy_header(maps1)
        self.assertEqual(maps1.Ni, maps2.Ni)
        self.assertEqual(maps1.Nj, maps2.Nj)
        self.assertEqual(maps1.Nk, maps2.Nk)


class ConcatenateTestCase(unittest.TestCase):
    """Test Maps.concatenate classmethod."""

    def setUp(self):
        """Set up arrays."""
        self.Ni, self.Nj, self.Nk = 1, 1, 3
        self.arr1 = np.array([[1, 2, 3]]).T
        self.arr2 = np.array([[4, 5, 6]]).T
        self.arr3 = np.concatenate((self.arr1, self.arr2))

    def test_empty(self):
        """Test empty sequence."""
        with self.assertRaises(ValueError):
            Maps.concatenate(())

    def test_one_map(self):
        """Test one map."""
        maps1 = Maps(self.arr1, Ni=self.Ni, Nj=self.Nj, Nk=self.Nk)
        maps2 = Maps.concatenate((maps1,))

        self.assertTrue(np.array_equal(maps2.maps.toarray(), self.arr1))

    def test_two_maps(self):
        """Test two maps."""
        maps1 = Maps(self.arr1, Ni=self.Ni, Nj=self.Nj, Nk=self.Nk)
        maps2 = Maps(self.arr2, Ni=self.Ni, Nj=self.Nj, Nk=self.Nk)

        maps3 = Maps.concatenate((maps1, maps2))

        self.assertTrue(np.array_equal(maps3.maps.toarray(), self.arr3))
