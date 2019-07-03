# from tools import build_index
from joblib import Memory
from nilearn import datasets, masking
import numpy as np
import pandas as pd
import scipy

input_path = 'minimal/'
cache_dir = 'cache_joblib'
mem = Memory(cache_dir)

# Loading MNI152 background and parameters (shape, affine...)
bg_img = datasets.load_mni152_template()
gray_mask = masking.compute_gray_matter_mask(bg_img)
Ni, Nj, Nk = bg_img.shape
affine = bg_img.affine
inv_affine = np.linalg.inv(affine)

coordinates = pd.read_csv(input_path+'coordinates.csv')
corpus_tfidf = scipy.sparse.load_npz(input_path+'corpus_tfidf.npz')
