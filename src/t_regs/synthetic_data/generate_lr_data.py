import numpy as np

from ..multilinear_ops import multi_mode_product
from .qmult import qmult


def generate_low_rank_data(dim, ranks, seed=None, return_factors=False):
    '''Generates low-rank tensor data with dimensions `dim` and ranks `ranks`.
    Parameters:
        dim: Dimensions of the tensor
        ranks: Ranks of the tensor
    Outputs:
        T: Tensor of order `len(dim)`.
    '''
    rng = np.random.default_rng(seed)
    n = len(dim)
    C = rng.normal(0,1,ranks)
    U = [qmult(dim[i])[:,:ranks[i]] for i in range(n)]
    dims = [i+1 for i in range(n)]
   
    if return_factors:
        return  multi_mode_product(C, U, dims, transpose=False), C, U
    return multi_mode_product(C, U, dims, transpose=False)
