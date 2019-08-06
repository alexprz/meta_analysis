from nilearn.decomposition import DictLearning, CanICA

from globals import pickle_path
from tools import get_dump_token, pickle_dump, pickle_load

save_dir = pickle_path+'generated_atlases/'

def fit_GroupICA(imgs, save=True, tag=None, prefix=''):
    raise NotImplementedError('')

def fit_CanICA(imgs, params, save=True, tag=None, prefix='CanICA_'):
    filepath = save_dir+get_dump_token(prefix, tag=tag)
    
    canica = pickle_load(filepath)
    if canica is not None:
        return canica

    canica = CanICA(**params)
    return pickle_dump(canica.fit(imgs), filepath, save=save)

def fit_DictLearning(imgs, params, save=True, tag=None, prefix=''):
    dictlearning = DictLearning(**params)

    return dictlearning.fit(imgs)

def fit_Kmeans(imgs, save=True, tag=None, prefix=''):
    raise NotImplementedError('')

def fit_Wards(imgs, save=True, tag=None, prefix=''):
    raise NotImplementedError('')

def fit_Model(Model, imgs, params, load=True, save=True, tag=None):
    filepath = save_dir+get_dump_token(Model.__name__+'_', tag=tag)

    model = pickle_load(filepath, load=load)
    if model is not None:
        return model

    model = Model(**params)
    return pickle_dump(model.fit(imgs), filepath, save=save)

if __name__ == '__main__':
    pass