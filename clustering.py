from nilearn.decomposition import DictLearning, CanICA
from nilearn.regions import Parcellations

from globals import pickle_path
from tools import get_dump_token, pickle_dump, pickle_load

save_dir = pickle_path+'generated_atlases/'

def fit_GroupICA(imgs, params):
    raise NotImplementedError('')

def fit_CanICA(imgs, params):
    defaults = {
        'n_components':  10,
        'memory': "nilearn_cache",
        'memory_level': 2,
        'threshold': 3.,
        'n_init': 1,
        'verbose': 1,
        'mask_strategy': 'template',
        'n_jobs': -2
    }
    context = dict(defaults, **params)

    return CanICA(**context).fit(imgs)

def fit_DictLearning(imgs, params):
    defaults = {
        'n_components': 10,
        'memory': "nilearn_cache",
        'memory_level': 2,
        'verbose': 1,
        'random_state': 0,
        'n_epochs': 1,
        'mask_strategy': 'template'
    }
    context = dict(defaults, **params)

    return DictLearning(**context).fit(imgs)

def fit_Kmeans(imgs, params):
    raise NotImplementedError('')

def fit_Wards(imgs, params):
    defaults = {
        'method': 'ward', 
        'n_parcels': 10,
        'standardize': False,
        'smoothing_fwhm': 2.,
        'memory': 'nilearn_cache',
        'memory_level': 1,
        'verbose': 1
    }
    context = dict(defaults, **params)
    
    return Parcellations(**context).fit(imgs)

def fit_Model(Model, imgs, params=dict(), load=True, save=True, tag=None):
    filepath = save_dir+get_dump_token(Model.__name__+'_', tag=tag)

    model = pickle_load(filepath, load=load)
    if model is not None:
        return model

    model = Model(imgs, params)
    return pickle_dump(model.fit(imgs), filepath, save=save)

if __name__ == '__main__':
    pass