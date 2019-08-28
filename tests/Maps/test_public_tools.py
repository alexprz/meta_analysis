import unittest
import numpy as np
import nibabel as nib

from meta_analysis import Maps
from globals_test import affine, maps


class ApplyMaskTestCase(unittest.TestCase):
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

        self.expected = np.array([[[0, 0, 3],
                                   [0, 5, 0],
                                   [0, 0, 0]]])

        self.expected2 = np.array([[[[0, 0], [0, 0], [3, 7]],
                                    [[0, 0], [5, 5], [0, 0]],
                                    [[0, 0], [0, 0], [0, 0]]]])

        self.Ni, self.Nj, self.Nk = self.mask_data.shape

    def test_mask_init_one_map(self):
        maps = Maps(self.array, Ni=self.Ni, Nj=self.Nj, Nk=self.Nk, mask=self.mask)
        self.assertTrue(np.array_equal(maps.to_array(), self.expected))

    def test_mask_init_two_maps(self):
        maps2 = Maps(self.array2, Ni=self.Ni, Nj=self.Nj, Nk=self.Nk, mask=self.mask)
        self.assertTrue(np.array_equal(maps2.to_array(0), self.expected2[:, :, :, 0]))
        self.assertTrue(np.array_equal(maps2.to_array(1), self.expected2[:, :, :, 1]))

    def test_mask_apply_one_map(self):
        maps = Maps(self.array, Ni=self.Ni, Nj=self.Nj, Nk=self.Nk)
        maps.apply_mask(self.mask)

        self.assertTrue(np.array_equal(maps.to_array(), self.expected))

    def test_mask_apply_two_maps(self):
        maps2 = Maps(self.array2, Ni=self.Ni, Nj=self.Nj, Nk=self.Nk)
        maps2.apply_mask(self.mask)

        self.assertTrue(np.array_equal(maps2.to_array(0), self.expected2[:, :, :, 0]))
        self.assertTrue(np.array_equal(maps2.to_array(1), self.expected2[:, :, :, 1]))


class NormalizeTestCase(unittest.TestCase):
    def setUp(self):
        self.array = np.array([[[1, 0, 0],
                                [0, 1, 0],
                                [0, 0, 2]]])

        self.array2 = np.array([[[[1, 0], [0, 0], [0, 0]],
                                 [[0, 0], [1, 1], [0, 0]],
                                 [[0, 0], [0, 0], [2, 0]]]])

        self.expected = np.array([[[0.25, 0, 0],
                                   [0, 0.25, 0],
                                   [0, 0, 0.5]]])

        self.expected2 = np.array([[[[0.25, 0], [0, 0], [0, 0]],
                                   [[0, 0], [0.25, 1], [0, 0]],
                                   [[0, 0], [0, 0], [0.5, 0]]]])

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

        self.assertTrue(np.array_equal(maps2.to_array(0), self.expected2[:, :, :, 0]))
        self.assertTrue(np.array_equal(maps2.to_array(1), self.expected2[:, :, :, 1]))
        self.assertTrue(np.array_equal(maps2_.to_array(0), self.expected2[:, :, :, 0]))
        self.assertTrue(np.array_equal(maps2_.to_array(1), self.expected2[:, :, :, 1]))


class RandomizeTestCase(unittest.TestCase):
    def setUp(self):
        self.p = np.array([[[0, 0.25, 0],
                            [0, 0.5, 0],
                            [0, 0, 0.25]]])

        self.mask_data = np.array([[[0, 0, 0],
                                    [0, 1, 0],
                                    [0, 0, 0]]])

        self.mask = nib.Nifti1Image(self.mask_data, affine)

    def test_one_map(self):
        Maps(Ni=1, Nj=3, Nk=3).randomize(size=10)
        Maps(Ni=1, Nj=3, Nk=3).randomize(size=(10, 1))

    def test_two_maps(self):
        Maps(Ni=1, Nj=3, Nk=3).randomize(size=(10, 2))

    def test_two_maps_p(self):
        maps = Maps(Ni=1, Nj=3, Nk=3).randomize(size=(100, 2), p=self.p)
        self.assertEqual(maps.to_array(0)[0, 0, 0], 0)

    def test_two_maps_mask(self):
        maps = Maps(Ni=1, Nj=3, Nk=3, mask=self.mask).randomize(size=(100, 2))
        self.assertEqual(maps.summed_map().to_array()[0, 1, 1], 100)

    def test_two_maps_p_mask(self):
        maps = Maps(Ni=1, Nj=3, Nk=3, mask=self.mask).randomize(size=(100, 2), p=self.p)
        self.assertEqual(maps.to_array(0)[0, 0, 0], 0)
        self.assertEqual(maps.summed_map().to_array()[0, 1, 1], 100)

    def test_two_maps_p_mask_override(self):
        maps = Maps(Ni=1, Nj=3, Nk=3, mask=self.mask).randomize(size=(100, 2), p=self.p, override_mask=True)
        self.assertEqual(maps.to_array(0)[0, 0, 0], 0)

    def test_p_forbidden(self):
        p = np.array([[[1, 0, 0],
                       [0, 1, 0],
                       [0, 0, 0]]])

        with self.assertRaises(ValueError):
            Maps(Ni=1, Nj=3, Nk=3).randomize(size=(100, 2), p=p)

        with self.assertRaises(ValueError):
            Maps(Ni=1, Nj=3, Nk=3).randomize(size=np.array([50, 50]), p=p)

    def test_array_one_map(self):
        maps = Maps(Ni=1, Nj=3, Nk=3).randomize(size=np.array([10]))
        self.assertEqual(maps.sum(), 10)

    def test_array_two_maps(self):
        maps = Maps(Ni=1, Nj=3, Nk=3).randomize(size=np.array([5, 5]))
        self.assertTrue(np.array_equal(maps.sum(axis=0), np.array([5, 5])))

    def test_array_two_maps_p(self):
        maps = Maps(Ni=1, Nj=3, Nk=3).randomize(size=np.array([50, 50]), p=self.p)
        self.assertEqual(maps.to_array(0)[0, 0, 0], 0)
        self.assertTrue(np.array_equal(maps.sum(axis=0), np.array([50, 50])))

    def test_array_two_maps_mask(self):
        maps = Maps(Ni=1, Nj=3, Nk=3, mask=self.mask).randomize(size=np.array([50, 50]))
        self.assertEqual(maps.summed_map().to_array()[0, 1, 1], 100)
        self.assertTrue(np.array_equal(maps.sum(axis=0), np.array([50, 50])))

    def test_array_two_maps_p_mask(self):
        maps = Maps(Ni=1, Nj=3, Nk=3, mask=self.mask).randomize(size=np.array([50, 50]), p=self.p)
        self.assertEqual(maps.to_array(0)[0, 0, 0], 0)
        self.assertEqual(maps.summed_map().to_array()[0, 1, 1], 100)
        self.assertTrue(np.array_equal(maps.sum(axis=0), np.array([50, 50])))

    def test_array_two_maps_p_mask_2(self):
        maps = Maps(Ni=1, Nj=3, Nk=3, mask=self.mask).randomize(size=np.array([100, 0]), p=self.p)
        self.assertEqual(maps.to_array(0)[0, 0, 0], 0)
        self.assertEqual(maps.to_array(0)[0, 1, 1], 100)
        self.assertTrue(np.array_equal(maps.sum(axis=0), np.array([100, 0])))

    def test_array_two_maps_p_mask_override(self):
        maps = Maps(Ni=1, Nj=3, Nk=3, mask=self.mask).randomize(size=np.array([50, 50]), p=self.p, override_mask=True)
        self.assertEqual(maps.to_array(0)[0, 0, 0], 0)
        self.assertTrue(np.array_equal(maps.sum(axis=0), np.array([50, 50])))


class SplitTestCase(unittest.TestCase):
    def setUp(self):
        self.array = np.array([[[1, 0, 0],
                                [0, 1, 0],
                                [0, 0, 2]]])

        self.array2 = np.array([[[[1, 0], [0, 0], [0, 0]],
                                 [[0, 0], [1, 1], [0, 0]],
                                 [[0, 0], [0, 0], [2, 0]]]])

        self.Ni, self.Nj, self.Nk = self.array.shape

    def test_max_prop_one_maps(self):
        maps = Maps(self.array)
        maps_A, maps_B = maps.split(prop=1.)

        self.assertTrue(np.array_equal(maps_A.to_array(), self.array))
        self.assertEqual(maps_B.n_maps, 0)

    def test_max_prop_two_maps(self):
        maps = Maps(self.array2)
        maps_A, maps_B = maps.split(prop=1.)

        self.assertTrue(np.array_equal(maps_A.to_array(), self.array2))
        self.assertEqual(maps_B.n_maps, 0)

    def test_min_prop_one_maps(self):
        maps = Maps(self.array)
        maps_A, maps_B = maps.split(prop=0.)

        self.assertTrue(np.array_equal(maps_B.to_array(), self.array))
        self.assertEqual(maps_A.n_maps, 0)

    def test_min_prop_two_maps(self):
        maps = Maps(self.array2)
        maps_A, maps_B = maps.split(prop=0.)

        self.assertTrue(np.array_equal(maps_B.to_array(), self.array2))
        self.assertEqual(maps_A.n_maps, 0)

    def test_half_prop_two_maps(self):
        maps = Maps(self.array2)
        maps_A, maps_B = maps.split(prop=0.5)

        self.assertTrue(np.array_equal(maps_A.to_array(), self.array2[:, :, :, 0]) or np.array_equal(maps_A.to_array(), self.array2[:, :, :, 1]))
        self.assertTrue(np.array_equal(maps_B.to_array(), self.array2[:, :, :, 0]) or np.array_equal(maps_B.to_array(), self.array2[:, :, :, 1]))
        self.assertFalse(np.array_equal(maps_A.to_array(), maps_B.to_array()))

    def test_random_state(self):
        maps_A1, maps_B1 = maps.split(prop=0.5, random_state=0)
        maps_A2, maps_B2 = maps.split(prop=0.5, random_state=0)

        print(maps_A1.to_array())
        print(maps_A2.to_array())

        self.assertTrue(np.array_equal(maps_A1.to_array(), maps_A2.to_array()))
        self.assertTrue(np.array_equal(maps_B1.to_array(), maps_B2.to_array()))


class ThresholdTestCase(unittest.TestCase):
    def setUp(self):
        self.array = np.array([[[1, 0, 0],
                                [0, 3, 0],
                                [0, 0, 2]]])

        self.array2 = np.array([[[[1, 0], [0, 0], [0, 0]],
                                 [[0, 0], [3, 3], [0, 0]],
                                 [[0, 0], [0, 0], [2, 0]]]])

        self.expected = np.array([[[0, 0, 0],
                                   [0, 3, 0],
                                   [0, 0, 2]]])

        self.expected2 = np.array([[[[0, 0], [0, 0], [0, 0]],
                                   [[0, 0], [3, 3], [0, 0]],
                                   [[0, 0], [0, 0], [2, 0]]]])

        self.Ni, self.Nj, self.Nk = self.array.shape

    def test_threshold_one_map(self):
        maps = Maps(self.array, Ni=self.Ni, Nj=self.Nj, Nk=self.Nk)
        maps_ = maps.threshold(2, inplace=False)
        maps.threshold(2, inplace=True)

        self.assertTrue(np.array_equal(maps.to_array(), self.expected))
        self.assertTrue(np.array_equal(maps_.to_array(), self.expected))

    def test_threshold_two_maps(self):
        maps2 = Maps(self.array2, Ni=self.Ni, Nj=self.Nj, Nk=self.Nk)
        maps2_ = maps2.threshold(2, inplace=False)
        maps2.threshold(2, inplace=True)

        self.assertTrue(np.array_equal(maps2.to_array(0), self.expected2[:, :, :, 0]))
        self.assertTrue(np.array_equal(maps2.to_array(1), self.expected2[:, :, :, 1]))
        self.assertTrue(np.array_equal(maps2_.to_array(0), self.expected2[:, :, :, 0]))
        self.assertTrue(np.array_equal(maps2_.to_array(1), self.expected2[:, :, :, 1]))


class ShuffleTestCase(unittest.TestCase):
    def setUp(self):
        self.array = np.array([[[1, 0, 0],
                                [0, 3, 0],
                                [0, 0, 2]]])

        self.array2 = np.array([[[[1, 0], [0, 0], [0, 0]],
                                 [[0, 0], [3, 3], [0, 0]],
                                 [[0, 0], [0, 0], [2, 0]]]])

        self.Ni, self.Nj, self.Nk = self.array.shape

    def test_one_map(self):
        maps = Maps(self.array, Ni=self.Ni, Nj=self.Nj, Nk=self.Nk)
        maps_ = maps.shuffle(inplace=False)
        maps.shuffle(inplace=True)

        self.assertTrue(np.array_equal(maps.to_array(), self.array))
        self.assertTrue(np.array_equal(maps_.to_array(), self.array))

    def test_random_state(self):
        maps = Maps(self.array2, Ni=self.Ni, Nj=self.Nj, Nk=self.Nk)
        maps_ = maps.shuffle(random_state=0, inplace=False)
        maps.shuffle(random_state=0, inplace=True)

        self.assertTrue(np.array_equal(maps.to_array(), maps_.to_array()))

    def test_avg(self):
        avg1 = maps.avg().smooth(sigma=2.)
        avg2 = maps.shuffle().avg().smooth(sigma=2.)

        self.assertTrue(np.allclose(avg1.to_array(), avg2.to_array()))
