import pytest
import matplotlib.pyplot as plt
import nilearn

from meta_analysis import Maps, plotting
from tools import build_df_from_keyword

from globals_test import template, atlas

# Parameters
keyword = 'prosopagnosia'
sigma = 2.

# Maps
df = build_df_from_keyword(keyword)
maps = Maps(df, template=template, groupby_col='pmid')
maps_atlas = Maps(df, template=template, groupby_col='pmid', atlas=atlas)
avg, var = maps.iterative_smooth_avg_var(compute_var=True, sigma=sigma, bias=False)


@pytest.mark.mpl_image_compare
def test_avg():
    avg = maps.avg()
    return plotting.plot_activity_map(avg.to_img())

@pytest.mark.mpl_image_compare
def test_var():
    var = maps.var()
    return plotting.plot_activity_map(var.to_img())

@pytest.mark.mpl_image_compare
def test_iterative_avg():
    avg, _ = maps.iterative_smooth_avg_var(compute_var=False, sigma=sigma, bias=False)
    return plotting.plot_activity_map(avg.to_img())

@pytest.mark.mpl_image_compare
def test_iterative_avg_var_1():
    return plotting.plot_activity_map(avg.to_img())


@pytest.mark.mpl_image_compare
def test_iterative_avg_var_2():
    return plotting.plot_activity_map(var.to_img())

@pytest.mark.mpl_image_compare
def test_iterative_avg_var_thresholded_1():
    return plotting.plot_activity_map(avg.to_img(), threshold=0.0007)


@pytest.mark.mpl_image_compare
def test_iterative_avg_var_thresholded_2():
    return plotting.plot_activity_map(var.to_img(), threshold=0.00002)

@pytest.mark.mpl_image_compare
def test_cov():
    cov, labels = maps_atlas.cov()
    fig = plt.figure(figsize=(20, 20))
    ax = fig.add_subplot(1,1,1)
    nilearn.plotting.plot_matrix(cov, labels=labels, figure=fig)
    return fig


