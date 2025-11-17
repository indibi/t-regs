import numpy as np

from ..multilinear_ops import multi_mode_product
from .qmult import qmult


def generate_low_rank_data(dim, ranks, seed=None, return_factors=False):
    '''Generates low-rank tensor data with dimensions `dim` and ranks `ranks`
    
    Parameters
    ----------
        dim : tuple[int]
            Dimensions of the tensor
        ranks : tuple[int]
            Ranks of the tensor
        seed : int
            Random seed for reproducibility.
        return_factors : bool
            If True, returns the core tensor and factor matrices along with
            the generated tensor. Defaults to False.
    
    Returns
    -------
        np.ndarray
            Tensor of order `len(dim)`.
    '''
    rng = np.random.default_rng(seed)
    n = len(dim)
    C = rng.normal(0,1,ranks)
    U = [qmult(dim[i])[:,:ranks[i]] for i in range(n)]
    dims = [i+1 for i in range(n)]
   
    if return_factors:
        return  multi_mode_product(C, U, dims, transpose=False), C, U
    return multi_mode_product(C, U, dims, transpose=False)


def generate_sparse_low_tucker_rank_tensor(dims: tuple[int],
                                    ranks: tuple[int],
                                    cardinalities: tuple[int],
                                    amp_factor_entries:float =0.5,
                                    seed: int = 0) -> np.ndarray:
    """Generate low-tucker rank tensor with sparse factors.

    Based on the experiment synthetic data generation procedure in [1].
    Parameters
    ----------
        dims : tuple[int]
            Dimensions of the resulting tensor.
        ranks : tuple[int]
            Ranks of the resulting tensor.
        cardinalities : tuple[int]
            Cardinality (number of non-zeros) of the factor matrices columns.
        amp_factor_entries : float
            The amplitude of the non-zero entries in  the non-zero entries of
            the factor matrices. Defaults to 0.5.
        amp_core_entries : float
            The standard deviation of the core tensor entries drawn from the
            gaussian normal distribution. Defaults to 1.0.
        seed : int
            Random seed for reproducibility.
    
    Returns
    -------
        np.ndarray
            Generated low-tucker rank tensor with sparse factors.
    
    Notes
    -----
    .. [1] Ahmed, Talal, Haroon Raja, and Waheed U. Bajwa. 'Tensor regression
        using low-rank and sparse Tucker decompositions.' SIAM Journal on 
        Mathematics of Data Science 2.4 (2020): 944-966.
    """
    rng = np.random.default_rng(seed)
    N = len(dims)
    Us = []
    C = rng.uniform(0,1, size=ranks)
    
    for dim, rank, s in zip(dims, ranks, cardinalities):
        U = np.zeros((dim, rank))
        for r in range(rank):
            zero_indices = rng.choice(dim, size=s, replace=False)
            signs = rng.choice([-1, 1], size=s)
            U[zero_indices, r] = signs*(amp_factor_entries
                                        + np.abs(
                                            rng.standard_normal(size=s)
                                            )
                                        )
        Us.append(U)
    B = multi_mode_product(C, Us, modes=list(range(1, N+1)))
    return B

# def generate_sparse_low_tucker_rank_tensor(dims: tuple[int],
#                                     ranks: tuple[int],
#                                     cardinalities: tuple[int],
#                                     amp_factor_entries:float =0.5,
#                                     seed: int = 0) -> np.ndarray:
#     """Generate low-tucker rank tensor with sparse factors.

#     Based on the experiment synthetic data generation procedure in [1].
    
#     Parameters
#     ----------
#         dims : tuple[int]
#             Dimensions of the resulting tensor.
#         ranks : tuple[int]
#             Ranks of the resulting tensor.
#         cardinalities : tuple[int]
#             Cardinality (number of non-zeros) of the factor matrices columns.
#         amp_factor_entries : float
#             The amplitude of the non-zero entries in  the non-zero entries of
#             the factor matrices. Defaults to 0.5.
#         amp_core_entries : float
#             The standard deviation of the core tensor entries drawn from the
#             gaussian normal distribution. Defaults to 1.0.
#         seed : int
#             Random seed for reproducibility.
    
#     Returns
#     -------
#         np.ndarray
#             Generated low-tucker rank tensor with sparse factors.
    
#     Notes
#     -----
#     .. [1] Ahmed, Talal, Haroon Raja, and Waheed U. Bajwa. 'Tensor regression
#         using low-rank and sparse Tucker decompositions.' SIAM Journal on 
#         Mathematics of Data Science 2.4 (2020): 944-966.
#     """
#     rng = np.random.default_rng(seed)
#     N = len(dims)
#     Us = []
#     C = rng.uniform(0,1, size=ranks)
    
#     for dim, rank, s in zip(dims, ranks, cardinalities):
#         U = np.zeros((dim, rank))
#         for r in range(rank):
#             zero_indices = rng.choice(dim, size=s, replace=False)
#             signs = rng.choice([-1, 1], size=s)
#             U[zero_indices, r] = signs*(amp_factor_entries
#                                         + np.abs(
#                                             rng.standard_normal(size=s)
#                                             )
#                                         )
#         Us.append(U)
#     B = multi_mode_product(C, Us, modes=list(range(1, N+1)))
#     return B