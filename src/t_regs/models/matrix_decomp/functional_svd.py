"""Functional PCA module
"""

from typing import Tuple

import torch


class FunctionalSVD:
    r"""Perform Functional SVD matrix decomposition, AKA Functional PCA

    The FPCA decomposition of a matrix :math:`X \in \mathbb{R}^{m\times n}`
    admits to the solution of the following optimization problem,
    .. math::
        \argmax_{U, V} \langle U V^\top , X \rangle \\
        \mathrm{subject to} U^\top S_u U = I_m \\
        V^\top S_v V = I_n
    
    Parameters
    ----------
    L1: torch.Tensor
    L2: torch.Tensor
    alpha_1

    References
    ----------
    ..  [1] Huang, J., H. Shen, and A. Buja (2009). The analysis of two-way
        functional data using two-way regularized singular value decompositions.
        Journal of the American Statistical Association 104 (488), 1609–1620.
    ..  [2] 
    """
    def __init__(
        self,
        L1,
        L2,
    ):
    pass
# def functional_svd(
#         X: torch.Tensor,
#     )->Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
