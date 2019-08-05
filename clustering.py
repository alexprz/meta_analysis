from nilearn.decomposition import DictLearning, CanICA

from globals import pickle_path
from tools import get_dump_token, dump, load

save_dir = pickle_path+'generated_atlases/'

def fit_GroupICA(imgs, save=True, tag=None, prefix=''):
    raise NotImplementedError('')

def fit_CanICA(imgs, params, save=True, tag=None, prefix='CanICA_'):
    filepath = save_dir+get_dump_token(prefix, tag=tag)
    
    canica = load(filepath)
    if canica is not None:
        return canica

    canica = CanICA(**params)
    return dump(canica.fit(imgs), filepath, save=save)

def fit_DictLearning(imgs, params, save=True, tag=None, prefix=''):
    dictlearning = DictLearning(**params)

    return dictlearning.fit(imgs)

def fit_Kmeans(imgs, save=True, tag=None, prefix=''):
    raise NotImplementedError('')

def fit_Wards(imgs, save=True, tag=None, prefix=''):
    raise NotImplementedError('')

if __name__ == '__main__':
    pass