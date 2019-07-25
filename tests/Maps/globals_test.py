import hypothesis.strategies as strats
from nilearn import datasets, masking
import pandas as pd
import numpy as np
import copy
import nibabel as nib
from nistats.datasets import fetch_spm_auditory
import nilearn
import copy

from meta_analysis import Maps

max_box_width = 10
max_maps = 5
max_peaks = 100


template = datasets.load_mni152_template()
gray_mask = masking.compute_gray_matter_mask(template)
atlas = datasets.fetch_atlas_harvard_oxford('cort-maxprob-thr25-2mm')
affine = template.affine
Ni, Nj, Nk = template.shape
gray_mask_data = gray_mask.get_fdata()
gray_mask_missmatch = nib.Nifti1Image(np.delete(gray_mask_data, 0, 0), affine)

subject_data = fetch_spm_auditory()
fmri_img = nilearn.image.concat_imgs(subject_data.func, auto_resample=True)
atlas_2 = copy.deepcopy(atlas)
atlas_2['maps'] = nilearn.image.resample_img(nilearn.image.load_img(atlas['maps']), fmri_img.affine, fmri_img.shape[:-1])
gray_mask_2 = nilearn.image.resample_img(gray_mask, fmri_img.affine, fmri_img.shape[:-1])

groupby_col = 'map__id'
df = pd.DataFrame(np.array([['mymap', -3, 42, 12, 1], ['mymap', -3, 42, 12, 1], ['mymap2', -3, 42, 12, 1]]), columns=['map__id', 'x', 'y', 'z', 'weight'])

array2D = np.random.rand(Ni*Nj*Nk, 2)
array3D = np.random.rand(Ni, Nj, Nk)
array4D_1 = np.array([np.random.rand(Ni, Nj, Nk)])
array4D_2 = np.array([np.random.rand(Ni, Nj, Nk), np.random.rand(Ni, Nj, Nk)])

array2D_missmatch = np.random.rand(Ni*Nj*Nk-1, 2)
array3D_missmatch = np.random.rand(Ni, Nj, Nk-1)
array4D_1_missmatch = np.array([np.random.rand(Ni-1, Nj, Nk)])
array4D_2_missmatch = np.array([np.random.rand(Ni, Nj-1, Nk), np.random.rand(Ni, Nj-1, Nk)])

example_maps = Maps(df, template=template, groupby_col=groupby_col)

@strats.composite
def random_permitted_case_3D(draw):
    Ni = draw(strats.integers(min_value=1, max_value=max_box_width))
    Nj = draw(strats.integers(min_value=1, max_value=max_box_width))
    Nk = draw(strats.integers(min_value=1, max_value=max_box_width))

    i = draw(strats.integers(min_value=0, max_value=Ni-1))
    j = draw(strats.integers(min_value=0, max_value=Nj-1))
    k = draw(strats.integers(min_value=0, max_value=Nk-1))

    return i, j, k, Ni, Nj, Nk

@strats.composite
def random_permitted_case_1D(draw):
    Ni = draw(strats.integers(min_value=1, max_value=max_box_width))
    Nj = draw(strats.integers(min_value=1, max_value=max_box_width))
    Nk = draw(strats.integers(min_value=1, max_value=max_box_width))

    p = draw(strats.integers(min_value=0, max_value=Ni*Nj*Nk-1))

    return p, Ni, Nj, Nk

@strats.composite
def empty_maps(draw, min_maps=1):
    Ni = draw(strats.integers(min_value=1, max_value=max_box_width))
    Nj = draw(strats.integers(min_value=1, max_value=max_box_width))
    Nk = draw(strats.integers(min_value=1, max_value=max_box_width))
    n_maps = draw(strats.integers(min_value=min_maps, max_value=max_maps))

    return Maps.zeros(Ni*Nj*Nk, n_maps)

@strats.composite
def random_maps(draw, min_maps=1):
    Ni = draw(strats.integers(min_value=1, max_value=max_box_width))
    Nj = draw(strats.integers(min_value=1, max_value=max_box_width))
    Nk = draw(strats.integers(min_value=1, max_value=max_box_width))
    n_maps = draw(strats.integers(min_value=min_maps, max_value=max_maps))
    n_peaks = draw(strats.integers(min_value=0, max_value=max_peaks))

    return Maps.random(Ni, Nj, Nk, n_peaks, n_maps)