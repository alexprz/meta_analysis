# from tools import build_index
from joblib import Memory

input_path = 'minimal/'
cache_dir = 'cache_joblib'
mem = Memory(cache_dir)

# encode_feature, decode_feature = build_index(input_path+'feature_names.txt')
# encode_pmid, decode_pmid = build_index(input_path+'pmids.txt')
