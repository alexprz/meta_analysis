import unittest
from hypothesis import given, settings
import hypothesis.strategies as strats
import numpy as np
import scipy
import copy

from tools import index_3D_to_1D_checked, index_1D_to_3D_checked

@strats.composite
def random_permitted_case_3D(draw):
    Ni = draw(strats.integers(min_value=1))
    Nj = draw(strats.integers(min_value=1))
    Nk = draw(strats.integers(min_value=1))

    i = draw(strats.integers(min_value=0, max_value=Ni))
    j = draw(strats.integers(min_value=0, max_value=Nj))
    k = draw(strats.integers(min_value=0, max_value=Nk))

    return i, j, k, Ni, Nj, N

@strats.composite
def random_permitted_case_1D(draw):
    Ni = draw(strats.integers(min_value=1))
    Nj = draw(strats.integers(min_value=1))
    Nk = draw(strats.integers(min_value=1))

    p = draw(strats.integers(min_value=0, max_value=Ni*Nj*Nk-1))

    return p, Ni, Nj, Nk

class TestIndexesChange(unittest.TestCase):
    def test_empty(self):
        self.assertRaises(ValueError, index_3D_to_1D_checked, 0, 0, 0, 0, 0, 0)
        self.assertRaises(ValueError, index_1D_to_3D_checked, 0, 0, 0, 0)

    def test_one_empty(self):
        self.assertRaises(ValueError, index_3D_to_1D_checked, 0, 0, 0, 0, 1, 1)
        self.assertRaises(ValueError, index_1D_to_3D_checked, 0, 0, 1, 1)

    @given(Ni=strats.integers(min_value=1),
           Nj=strats.integers(min_value=1),
           Nk=strats.integers(min_value=1))
    def test_edge_cases(self, Ni, Nj, Nk):
        self.assertEqual(index_3D_to_1D_checked(0, 0, 0, Ni, Nj, Nk), 0)
        self.assertEqual(index_1D_to_3D_checked(0, Ni, Nj, Nk), (0, 0, 0))

        self.assertEqual(index_3D_to_1D_checked(Ni-1, Nj-1, Nk-1, Ni, Nj, Nk), Ni*Nj*Nk-1)
        self.assertEqual(index_1D_to_3D_checked(Ni*Nj*Nk-1, Ni, Nj, Nk), (Ni-1, Nj-1, Nk-1))

    @given(args = random_permitted_case_3D())
    def test_main_cases(self, args):
        i, j, k, Ni, Nj, Nk = args
        p = index_3D_to_1D_checked(i, j, k, Ni, Nj, Nk)
        self.assertEqual(index_1D_to_3D_checked(p, Ni, Nj, Nk), (i, j, k))

    @given(args = random_permitted_case_1D())
    def test_main_cases(self, args):
        p, Ni, Nj, Nk = args
        i, j, k = index_1D_to_3D_checked(p, Ni, Nj, Nk)
        self.assertEqual(index_3D_to_1D_checked(i, j, k, Ni, Nj, Nk), p)

if __name__ == '__main__':
    unittest.main()