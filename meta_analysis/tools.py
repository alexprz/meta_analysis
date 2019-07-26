
def print_percent(index, total, string='', rate=0.1, end='\r', last_end='\n', append_last_string=' Done', flush=True, verbose=True):
    if not verbose:
        return

    period = int(rate*total/100)
    if period == 0 or index % period == 0:
        print(string.format(100*(index+1)/total, index+1, total), end=end, flush=flush)

    if index == total-1:
        print((string+append_last_string).format(100*(index+1)/total, index+1, total), end=last_end, flush=flush)


def index_3D_to_1D(i, j, k, Ni, Nj, Nk):
    '''
        Changes indexing from 3D to 1D Fortran like (first index moving fastest).
        Doesn't check bounds.
    '''
    return i + Ni*j + Ni*Nj*k

def index_1D_to_3D(p, Ni, Nj, Nk):
    '''
        Changes indexing from 1D to 3D Fortran like (first index moving fastest).
        Doesn't check bounds.
    '''
    k = p//(Ni*Nj)
    p = p%(Ni*Nj)
    j = p//Ni
    i = p%Ni
    return i, j, k

def index_3D_to_1D_checked(i, j, k, Ni, Nj, Nk):
    '''
        Equivalent to index_3D_to_1D but checks bounds.
    '''
    if i >= Ni or j >= Nj or k >= Nk or i < 0 or j < 0 or k < 0:
        raise ValueError('Indices ({}, {}, {}) are outside box of size ({}, {}, {}).'.format(i, j, k, Ni, Nj, Nk))

    if Ni == 0 or Nj == 0 or Nk == 0:
        raise ValueError('Given box of size ({}, {}, {}) should not have a null side.'.format(Ni, Nj, Nk))

    return index_3D_to_1D(i, j, k, Ni, Nj, Nk)

def index_1D_to_3D_checked(p, Ni, Nj, Nk):
    '''
        Equivalent to index_1D_to_3D but checks bounds.
    '''
    size = Ni*Nj*Nk
    if p >= size or p < 0:
        raise ValueError('Indice {} is outside vector of size {}*{}*{}={}.'.format(p, Ni, Nj, Nk, size))

    if Ni == 0 or Nj == 0 or Nk == 0:
        raise ValueError('Given box of size ({}, {}, {}) should not have a null side.'.format(Ni, Nj, Nk))

    return index_1D_to_3D(p, Ni, Nj, Nk)
