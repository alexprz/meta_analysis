import unittest

from meta_analysis import Maps
from globals_test import template, Ni, Nj, Nk


class ZerosTestCase(unittest.TestCase):
    """Test zeros classmethod."""

    def test_no_kwargs(self):
        """Test without kerwords."""
        with self.assertRaises(ValueError):
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

