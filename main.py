import numpy as np
import pandas as pd
from nilearn import datasets, plotting
from scipy.ndimage import gaussian_filter
import nibabel as nib
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')


input_path = 'minimal/'

def build_activity_map(pmid):
    '''
        Given a pmid, build its corresponding activity map
    '''

    coordinates = pd.read_csv(input_path+'coordinates.csv')

    for index, row in coordinates.loc[coordinates['pmid'] == pmid].iterrows():
        x, y, z = row['x'], row['y'], row['z']
        


if __name__ == '__main__':

    print(build_activity_map(26160289))
    motor_images = datasets.fetch_neurovault_motor_task()
    # stat_img = motor_images.images[0]
    # print(stat_img)
    # plotting.plot_glass_brain(stat_img, threshold=3)
    # plotting.plot_anat()
    # array_data = 10*np.ones(1000, dtype=np.int16).reshape((10, 10, 10))
    # print(array_data)
    # affine = np.diag([1, 1, 1, 1])
    # stat_img = nib.Nifti1Image(array_data, affine)

    # plotting.plot_stat_map(stat_img)

    # plotting.show()

    # array_data = 10*np.ones(1000, dtype=np.int16).reshape((10, 10, 10))
    # Nx, Ny, Nz = 30, 30, 30
    # array_data = np.zeros((Nx,Ny,Nz))
    # array_data[Nx//2, Ny//2, Nz//2] = 1
    # array_data = gaussian_filter(array_data, sigma=3)
    # print(array_data)
    # affine = np.diag([1, 1, 1, 1])
    # stat_img = nib.Nifti1Image(array_data, affine)

    # plotting.plot_stat_map(stat_img)

    # stat_img = datasets.load_mni152_template()
    # motor_images = datasets.fetch_neurovault_motor_task()
    # stat_img = motor_images.images[0]
    # bg_img = datasets.fetch_icbm152_2009()['t1']
    # plotting.plot_stat_map(stat_img, bg_img=bg_img)

    # plotting.show()

    # bg_img = datasets.fetch_icbm152_2009()['t1']
    bg_img = datasets.load_mni152_template()
    Ni, Nj, Nk = bg_img.shape
    affine = bg_img.affine
    array_data = np.zeros((Ni,Nj,Nk))
    array_data[Ni//2, Nj//2, Nk//2] = 1
    array_data = gaussian_filter(array_data, sigma=3)
    stat_img = nib.Nifti1Image(array_data, affine)
    plotting.plot_stat_map(stat_img, bg_img=bg_img)

    plotting.show()

