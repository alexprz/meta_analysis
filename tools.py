from globals import mem
# import numpy as np
# import multiprocessing

# def pool_computing(function, array_to_map, concatenate_function=np.concatenate, nb_processes=multiprocessing.cpu_count(), **kwargs):
#     # Create Pool for multiprocessing
#     pool = multiprocessing.Pool(processes = nb_processes)
#     args = []

#     if isinstance(array_to_map, int):
#         sub_arg = array_to_map//nb_processes*np.ones(nb_processes).astype(int)

#     else:
#         sub_arg = np.split(array_to_map, nb_processes)

#     for i in range(nb_processes):
#         print('test')
#         print(kwargs)
#         args.append((sub_arg[i], i, kwargs))

#     # Mapping args to function, each on a different process
#     results = pool.starmap(function, args)

#     del args

#     # Closing Pool
#     pool.close()
#     pool.join()

#     # Getting the results together
#     return concatenate_function(results)


def print_percent(index, total, prefix='', rate=10000):
    if (total//rate) == 0 or index % (total//rate) == 0:
        print(prefix+str(round(100*index/total, 1))+'%')

@mem.cache
def build_index(file_path):
    '''
        Build decode & encode dictionnary of the given file_name.

        encode : dict
            key : line number
            value : string at the specified line number
        decode : dict (reverse of encode)
            key : string found in the file
            value : number of the line containing the string

        Used for the files pmids.txt & feature_names.txt
    '''
    decode = dict(enumerate(line.strip() for line in open(file_path)))
    encode = {v: k for k, v in decode.items()}
    
    return encode, decode