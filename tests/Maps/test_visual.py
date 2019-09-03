"""Visual tests on a single example."""
import pytest
import matplotlib.pyplot as plt
import nilearn

from meta_analysis import Maps, plotting

from globals_test import template, atlas, df

# Parameters
sigma = 2.

# Maps
maps = Maps(df, template=template, groupby_col='pmid')
maps_dense = Maps(df, template=template, groupby_col='pmid', save_memory=False)
maps_atlas = Maps(df, template=template, groupby_col='pmid', atlas=atlas)
avg, var = maps.iterative_smooth_avg_var(compute_var=True, sigma=sigma, bias=False)
avg_dense, var_dense = maps_dense.iterative_smooth_avg_var(compute_var=True, sigma=sigma, bias=False)
avg_atlas, var_atlas = maps_atlas.iterative_smooth_avg_var(compute_var=True, sigma=sigma, bias=False)
avg_biased, var_biased = maps.iterative_smooth_avg_var(compute_var=True, sigma=sigma, bias=True)
avg_dense_biased, var_dense_biased = maps_dense.iterative_smooth_avg_var(compute_var=True, sigma=sigma, bias=True)
avg_atlas_biased, var_atlas_biased = maps_atlas.iterative_smooth_avg_var(compute_var=True, sigma=sigma, bias=True)


@pytest.mark.mpl_image_compare
def test_sum():
    """Test sum of maps."""
    sum = maps.summed_map()
    return plotting.plot_activity_map(sum.to_img())


@pytest.mark.mpl_image_compare
def test_avg():
    """Test avg of maps."""
    avg = maps.avg()
    return plotting.plot_activity_map(avg.to_img())


@pytest.mark.mpl_image_compare
def test_var():
    """Test var of maps."""
    var = maps.var()
    return plotting.plot_activity_map(var.to_img())


@pytest.mark.mpl_image_compare
def test_iterative_avg():
    """Test iterative avg of maps."""
    avg, _ = maps.iterative_smooth_avg_var(compute_var=False, sigma=sigma, bias=False)
    return plotting.plot_activity_map(avg.to_img())


@pytest.mark.mpl_image_compare
def test_iterative_avg_var_1():
    """Test iterative avg of maps."""
    return plotting.plot_activity_map(avg.to_img())


@pytest.mark.mpl_image_compare
def test_iterative_avg_var_2():
    """Test iterative var of maps."""
    return plotting.plot_activity_map(var.to_img())


@pytest.mark.mpl_image_compare
def test_iterative_avg_var_thresholded_1():
    """Test iterative avg thresholded of maps."""
    return plotting.plot_activity_map(avg.to_img(), threshold=0.0007)


@pytest.mark.mpl_image_compare
def test_iterative_avg_var_thresholded_2():
    """Test iterative var thresholded of maps."""
    return plotting.plot_activity_map(var.to_img(), threshold=0.00002)


@pytest.mark.mpl_image_compare
def test_iterative_avg_var_1_dense():
    """Test iterative avg of dense maps."""
    return plotting.plot_activity_map(avg_dense.to_img())


@pytest.mark.mpl_image_compare
def test_iterative_avg_var_2_dense():
    """Test iterative var of dense maps."""
    return plotting.plot_activity_map(var_dense.to_img())


@pytest.mark.mpl_image_compare
def test_iterative_avg_var_thresholded_1_dense():
    """Test iterative avg of dense maps thresholded."""
    return plotting.plot_activity_map(avg_dense.to_img(), threshold=0.0007)


@pytest.mark.mpl_image_compare
def test_iterative_avg_var_thresholded_2_dense():
    """Test iterative var of maps thresholded."""
    return plotting.plot_activity_map(var_dense.to_img(), threshold=0.00002)


@pytest.mark.mpl_image_compare
def test_iterative_avg_var_1_biased():
    """Test iterative biased avg of maps."""
    return plotting.plot_activity_map(avg_biased.to_img())


@pytest.mark.mpl_image_compare
def test_iterative_avg_var_2_biased():
    """Test iterative biased var of maps."""
    return plotting.plot_activity_map(var_biased.to_img())


@pytest.mark.mpl_image_compare
def test_iterative_avg_var_thresholded_1_biased():
    """Test iterative biased avg of maps thresholded."""
    return plotting.plot_activity_map(avg_biased.to_img(), threshold=0.0007)


@pytest.mark.mpl_image_compare
def test_iterative_avg_var_thresholded_2_biased():
    """Test iterative biased var of maps thresholded."""
    return plotting.plot_activity_map(var_biased.to_img(), threshold=0.00002)


@pytest.mark.mpl_image_compare
def test_iterative_avg_var_1_dense_biased():
    """Test iterative biased avg of dense maps."""
    return plotting.plot_activity_map(avg_dense_biased.to_img())


@pytest.mark.mpl_image_compare
def test_iterative_avg_var_2_dense_biased():
    """Test iterative biased var of dense maps."""
    return plotting.plot_activity_map(var_dense_biased.to_img())


@pytest.mark.mpl_image_compare
def test_iterative_avg_var_thresholded_1_dense_biased():
    """Test iterative biased avg of dense maps thresholded."""
    return plotting.plot_activity_map(avg_dense_biased.to_img(), threshold=0.0007)


@pytest.mark.mpl_image_compare
def test_iterative_avg_var_thresholded_2_dense_biased():
    """Test iterative biased var of dense maps thresholded."""
    return plotting.plot_activity_map(var_dense_biased.to_img(), threshold=0.00002)


@pytest.mark.mpl_image_compare
def test_atlas_sum():
    """Test sum of maps on atlas."""
    sum = maps_atlas.summed_map()
    return plotting.plot_activity_map(sum.to_img_atlas(ignore_bg=True))


@pytest.mark.mpl_image_compare
def test_atlas_avg():
    """Test avg of maps on atlas."""
    avg = maps_atlas.avg()
    return plotting.plot_activity_map(avg.to_img_atlas(ignore_bg=True))


@pytest.mark.mpl_image_compare
def test_atlas_var():
    """Test var of maps on atlas."""
    var = maps_atlas.var()
    return plotting.plot_activity_map(var.to_img_atlas(ignore_bg=True))


@pytest.mark.mpl_image_compare
def test_atlas_iterative_avg():
    """Test iterative avg of maps on atlas."""
    avg, _ = maps_atlas.iterative_smooth_avg_var(compute_var=False, sigma=sigma, bias=False)
    return plotting.plot_activity_map(avg.to_img_atlas(ignore_bg=True))


@pytest.mark.mpl_image_compare
def test_atlas_iterative_avg_var_1():
    """Test iterative avg of maps on atlas."""
    return plotting.plot_activity_map(avg_atlas.to_img_atlas(ignore_bg=True))


@pytest.mark.mpl_image_compare
def test_atlas_iterative_avg_var_2():
    """Test iterative var of maps on atlas."""
    return plotting.plot_activity_map(var_atlas.to_img_atlas(ignore_bg=True))


@pytest.mark.mpl_image_compare
def test_atlas_iterative_avg_var_thresholded_1():
    """Test iterative avg of maps on atlas."""
    return plotting.plot_activity_map(avg_atlas.to_img_atlas(ignore_bg=True), threshold=0.0007)


@pytest.mark.mpl_image_compare
def test_atlas_iterative_avg_var_thresholded_2():
    """Test iterative var of maps on atlas."""
    return plotting.plot_activity_map(var_atlas.to_img_atlas(ignore_bg=True), threshold=0.00002)


@pytest.mark.mpl_image_compare
def test_atlas_iterative_avg_var_1_biased():
    """Test iterative biased avg of maps on atlas."""
    return plotting.plot_activity_map(avg_atlas_biased.to_img_atlas(ignore_bg=True))


@pytest.mark.mpl_image_compare
def test_atlas_iterative_avg_var_2_biased():
    """Test iterative biased var of maps on atlas."""
    return plotting.plot_activity_map(var_atlas_biased.to_img_atlas(ignore_bg=True))


@pytest.mark.mpl_image_compare
def test_atlas_iterative_avg_var_thresholded_1_biased():
    """Test thresholded iterative avg of maps on atlas."""
    return plotting.plot_activity_map(avg_atlas_biased.to_img_atlas(ignore_bg=True), threshold=0.0007)


@pytest.mark.mpl_image_compare
def test_atlas_iterative_avg_var_thresholded_2_biased():
    """Test thresholded iterative var of maps on atlas."""
    return plotting.plot_activity_map(var_atlas_biased.to_img_atlas(ignore_bg=True), threshold=0.00002)


@pytest.mark.mpl_image_compare
def test_atlas_cov():
    """Test cov computation on atlas."""
    cov, labels = maps_atlas.cov()
    fig = plt.figure(figsize=(20, 20))
    nilearn.plotting.plot_matrix(cov, labels=labels, figure=fig)
    return fig
