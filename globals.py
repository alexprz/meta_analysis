from joblib import Memory
from nilearn import datasets, masking
import numpy as np
import pandas as pd
import scipy

pickle_path = 'save/'
input_path = 'minimal/'
cache_dir = 'cache_joblib'
mem = Memory(cache_dir)

# Loading MNI152 background and parameters (shape, affine...)
template = datasets.load_mni152_template()
gray_mask = masking.compute_gray_matter_mask(template)
Ni, Nj, Nk = template.shape
affine = template.affine
inv_affine = np.linalg.inv(affine)

coordinates = pd.read_csv(input_path+'coordinates.csv')
corpus_tfidf = scipy.sparse.load_npz(input_path+'corpus_tfidf.npz')

atlas = datasets.fetch_atlas_harvard_oxford('cort-maxprob-thr25-2mm')