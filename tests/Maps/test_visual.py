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
maps_dense = Maps(df, template=template, groupby_col='pmid', save_memory=True)
maps_atlas = Maps(df, template=template, groupby_col='pmid', atlas=atlas)
avg, var = maps.iterative_smooth_avg_var(compute_var=True, sigma=sigma, bias=False)
avg_dense, var_dense = maps_dense.iterative_smooth_avg_var(compute_var=True, sigma=sigma, bias=False)
avg_atlas, var_atlas = maps_atlas.iterative_smooth_avg_var(compute_var=True, sigma=sigma, bias=False)
avg_biased, var_biased = maps.iterative_smooth_avg_var(compute_var=True, sigma=sigma, bias=True)
avg_dense_biased, var_dense_biased = maps_dense.iterative_smooth_avg_var(compute_var=True, sigma=sigma, bias=True)
avg_atlas_biased, var_atlas_biased = maps_atlas.iterative_smooth_avg_var(compute_var=True, sigma=sigma, bias=True)


@pytest.mark.mpl_image_compare
def test_sum():
    sum = maps.summed_map()
    return plotting.plot_activity_map(sum.to_img())

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
def test_iterative_avg_var_1_dense():
    return plotting.plot_activity_map(avg_dense.to_img())

@pytest.mark.mpl_image_compare
def test_iterative_avg_var_2_dense():
    return plotting.plot_activity_map(var_dense.to_img())

@pytest.mark.mpl_image_compare
def test_iterative_avg_var_thresholded_1_dense():
    return plotting.plot_activity_map(avg_dense.to_img(), threshold=0.0007)

@pytest.mark.mpl_image_compare
def test_iterative_avg_var_thresholded_2_dense():
    return plotting.plot_activity_map(var_dense.to_img(), threshold=0.00002)

@pytest.mark.mpl_image_compare
def test_iterative_avg_var_1_biased():
    return plotting.plot_activity_map(avg_biased.to_img())

@pytest.mark.mpl_image_compare
def test_iterative_avg_var_2_biased():
    return plotting.plot_activity_map(var_biased.to_img())

@pytest.mark.mpl_image_compare
def test_iterative_avg_var_thresholded_1_biased():
    return plotting.plot_activity_map(avg_biased.to_img(), threshold=0.0007)

@pytest.mark.mpl_image_compare
def test_iterative_avg_var_thresholded_2_biased():
    return plotting.plot_activity_map(var_biased.to_img(), threshold=0.00002)

@pytest.mark.mpl_image_compare
def test_iterative_avg_var_1_dense_biased():
    return plotting.plot_activity_map(avg_dense_biased.to_img())

@pytest.mark.mpl_image_compare
def test_iterative_avg_var_2_dense_biased():
    return plotting.plot_activity_map(var_dense_biased.to_img())

@pytest.mark.mpl_image_compare
def test_iterative_avg_var_thresholded_1_dense_biased():
    return plotting.plot_activity_map(avg_dense_biased.to_img(), threshold=0.0007)

@pytest.mark.mpl_image_compare
def test_iterative_avg_var_thresholded_2_dense_biased():
    return plotting.plot_activity_map(var_dense_biased.to_img(), threshold=0.00002)

@pytest.mark.mpl_image_compare
def test_atlas_sum():
    sum = maps_atlas.summed_map()
    return plotting.plot_activity_map(sum.to_img_atlas())

@pytest.mark.mpl_image_compare
def test_atlas_avg():
    avg = maps_atlas.avg()
    return plotting.plot_activity_map(avg.to_img_atlas())

@pytest.mark.mpl_image_compare
def test_atlas_var():
    var = maps_atlas.var()
    return plotting.plot_activity_map(var.to_img_atlas())

@pytest.mark.mpl_image_compare
def test_atlas_iterative_avg():
    avg, _ = maps_atlas.iterative_smooth_avg_var(compute_var=False, sigma=sigma, bias=False)
    return plotting.plot_activity_map(avg.to_img_atlas())

@pytest.mark.mpl_image_compare
def test_atlas_iterative_avg_var_1():
    return plotting.plot_activity_map(avg_atlas.to_img_atlas())

@pytest.mark.mpl_image_compare
def test_atlas_iterative_avg_var_2():
    return plotting.plot_activity_map(var_atlas.to_img_atlas())

@pytest.mark.mpl_image_compare
def test_atlas_iterative_avg_var_thresholded_1():
    return plotting.plot_activity_map(avg_atlas.to_img_atlas(), threshold=0.0007)

@pytest.mark.mpl_image_compare
def test_atlas_iterative_avg_var_thresholded_2():
    return plotting.plot_activity_map(var_atlas.to_img_atlas(), threshold=0.00002)

@pytest.mark.mpl_image_compare
def test_atlas_iterative_avg_var_1_biased():
    return plotting.plot_activity_map(avg_atlas_biased.to_img_atlas())

@pytest.mark.mpl_image_compare
def test_atlas_iterative_avg_var_2_biased():
    return plotting.plot_activity_map(var_atlas_biased.to_img_atlas())

@pytest.mark.mpl_image_compare
def test_atlas_iterative_avg_var_thresholded_1_biased():
    return plotting.plot_activity_map(avg_atlas_biased.to_img_atlas(), threshold=0.0007)

@pytest.mark.mpl_image_compare
def test_atlas_iterative_avg_var_thresholded_2_biased():
    return plotting.plot_activity_map(var_atlas_biased.to_img_atlas(), threshold=0.00002)

@pytest.mark.mpl_image_compare
def test_atlas_cov():
    cov, labels = maps_atlas.cov()
    fig = plt.figure(figsize=(20, 20))
    ax = fig.add_subplot(1,1,1)
    nilearn.plotting.plot_matrix(cov, labels=labels, figure=fig)
    return fig


