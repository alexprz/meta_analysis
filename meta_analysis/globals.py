"""Some globals variable used in other files."""
from joblib import Memory

cache_dir = 'cache_joblib'
mem = Memory(cache_dir)
