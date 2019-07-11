import unittest
from hypothesis import given, settings
import hypothesis.strategies as strats
import numpy as np
import scipy
import copy

from meta_analysis import Maps

@strats.composite
def random_maps(draw, min_maps=1):
    Ni = draw(strats.integers(min_value=1, max_value=10))
    Nj = draw(strats.integers(min_value=1, max_value=10))
    Nk = draw(strats.integers(min_value=1, max_value=10))
    n_maps = draw(strats.integers(min_value=min_maps, max_value=5))
    n_peaks = draw(strats.integers(min_value=0, max_value=100))

    return Maps(Ni=Ni, Nj=Nj, Nk=Nk).randomize(n_peaks, n_maps, inplace=True)

@strats.composite
def box_size(draw):
    Ni = draw(strats.integers(min_value=1, max_value=10))
    Nj = draw(strats.integers(min_value=1, max_value=10))
    Nk = draw(strats.integers(min_value=1, max_value=10))

    return Ni, Nj, Nk



class Test_avg(unittest.TestCase):
    @given(maps = random_maps())
    @settings(max_examples=100, deadline=1000)
    def test_random(self, maps):
        avg = maps.avg()
        self.assertTrue(np.allclose(avg.maps.toarray(), np.mean(maps.maps, axis=1)))

class Test_var(unittest.TestCase):
    @given(maps = random_maps())
    @settings(max_examples=100, deadline=1000)
    def test_random(self, maps):
        arr1 = maps.var().maps.transpose().toarray()
        arr2 = np.var(maps.maps.toarray(), axis=1)

        self.assertTrue(np.allclose(arr1, arr2))

class Test_cov(unittest.TestCase):
    def test_example(self):
        maps = Maps()
        maps.maps = scipy.sparse.csr_matrix(np.array([[1, 2, 3, 4], [2, 4, 1, 1], [3, 1, 1, 2]]))

        arr1 = np.cov(maps.maps.toarray())
        arr2 = maps.cov().toarray()

        self.assertTrue(np.allclose(arr1, arr2))

    @given(maps = random_maps(min_maps=2))
    @settings(max_examples=100, deadline=1000)
    def test_random(self, maps):
        arr1 = maps.cov().toarray()
        arr2 = np.cov(maps.maps.toarray())

        self.assertTrue(np.allclose(arr1, arr2))

class Test_iterative_smooth_avg_var(unittest.TestCase):
    @given(Ni=strats.integers(min_value=1, max_value=100),
           Nj=strats.integers(min_value=1, max_value=100),
           Nk=strats.integers(min_value=1, max_value=100),
           n_maps=strats.integers(min_value=1, max_value=10),
           )
    @settings(max_examples=100, deadline=1000)
    def test_null_maps(self, Ni, Nj, Nk, n_maps):
        maps = Maps.zeros((Ni*Nj*Nk, n_maps))
        avg, var = maps.iterative_smooth_avg_var()

        self.assertEqual(avg.maps.nnz, 0)
        self.assertEqual(var.maps.nnz, 0)


    @given(Ni=strats.integers(min_value=1, max_value=100),
           Nj=strats.integers(min_value=1, max_value=100),
           Nk=strats.integers(min_value=1, max_value=100),
           n_maps=strats.integers(min_value=1, max_value=10),
           )
    @settings(max_examples=100, deadline=1000)
    def test_constant_map(self, Ni, Nj, Nk, n_maps):
        maps = Maps.zeros((Ni*Nj*Nk, n_maps))
        maps.maps = scipy.sparse.csr_matrix(np.ones((Ni*Nj*Nk, n_maps)))
        avg, var = maps.iterative_smooth_avg_var()

        self.assertTrue(np.allclose(np.ones((Ni*Nj*Nk, 1)), avg.maps.toarray()))
        self.assertEqual(var.maps.nnz, 0)


    @given(maps = random_maps())
    @settings(max_examples=100, deadline=1000)
    def test_without_smoothing(self, maps):
        expected_avg_map = maps.avg().maps
        expected_var_map = maps.var().maps
        avg_map, var_map = maps.iterative_smooth_avg_var()

        self.assertTrue(np.allclose(expected_avg_map.toarray(), avg_map.maps.toarray()))
        self.assertTrue(np.allclose(expected_var_map.toarray(), var_map.maps.toarray()))


    @given(maps = random_maps(),
           sigma = strats.floats(min_value=0.1, max_value=10.))
    @settings(max_examples=100, deadline=1000)
    def test_with_smoothing(self, maps, sigma):
        smoothed_maps = maps.smooth(sigma=sigma)

        expected_avg = smoothed_maps.avg()
        expected_var = smoothed_maps.var()
        avg, var = maps.iterative_smooth_avg_var(sigma=sigma)

        self.assertTrue(np.allclose(expected_avg.maps.toarray(), avg.maps.toarray()))
        self.assertTrue(np.allclose(expected_var.maps.toarray(), var.maps.toarray()))
