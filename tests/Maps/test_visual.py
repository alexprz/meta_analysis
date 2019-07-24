import pytest
import matplotlib.pyplot as plt

from meta_analysis import Maps, plotting
from tools import build_df_from_keyword

from globals_test import template

# Parameters
keyword = 'prosopagnosia'
sigma = 2.

# Maps
df = build_df_from_keyword(keyword)
maps = Maps(df, template=template, groupby_col='pmid')


@pytest.mark.mpl_image_compare
def test_avg():
    avg = maps.avg()
    return plotting.plot_activity_map(avg.to_img())
