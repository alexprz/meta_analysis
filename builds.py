from globals import input_path
from tools import build_index

encode_feature, decode_feature = build_index(input_path+'feature_names.txt')
encode_pmid, decode_pmid = build_index(input_path+'pmids.txt')
