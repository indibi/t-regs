"""Generalized Least Squares Matrix Decomposition (GSVD) module.
"""
from typing import Any, Optional

import torch


# TODO: Implementing docstring for the function
# TODO: Add option to provide the eigen decomposition or Cholesky factors directly
# TODO: Add option to select the ranks based on the optimal singular value thresholding paper
# TODO: Implement a class that analyzes the degrees of freedom/BIC for different ranks
# TODO: Implement solvers for sparse Q1, Q2 such as graph Laplacians
# TODO: Figure out why cholesky method gives slightly different results than eig method

def generalized_svd(X: torch.Tensor,
                    Q1: Optional[torch.Tensor]= None,
                    Q2: Optional[torch.Tensor]= None,
                    rank: int | None = None,
                    method: str ='eig',
                    **kwargs: dict[str, Any],
                    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Perform Generalized Singular Value Decomposition (GSVD) of matrix X.
    
    Based on the formulation in:
    [1] 'Allen, Genevera I., Logan Grosenick, and Jonathan Taylor. "A
        generalized least-square matrix decomposition." Journal of the American
        Statistical Association 109.505 (2014): 145-159.'
    
    Parameters:
    -----------
    X : torch.Tensor
        Input data matrix of shape (m x n)
    Q1 : torch.Tensor
        Positive semi definite weight matrix for of shape (m x m)
    Q2 : torch.Tensor
        Positive semi definite weight matrix for of shape (n x n)
    rank : int = None
        Desired rank for the decomposition. If None, full rank is used.
    method : str = 'eig'
        Method used to compute the decomposition. Options are 'eig' or
        'cholesky'.
    **kwargs : dict
        Additional keyword arguments for specific methods.
        eig_val_threshold : float
            Threshold for eigenvalues when using 'eig' method to ensure
            numerical stability in computing matrix inverses. Default is 1e-12.
    
    Returns:
    --------
    U : torch.Tensor
        Left singular vectors of shape (m x rank)
    S : torch.Tensor
        Singular values of shape (rank,)
    Vt : torch.Tensor
        Right singular vectors transposed of shape (rank x n)
    """
    m, n = X.shape
    if rank is None:
        rank = min(m, n)
    
    if method == 'eig':
        eig_val_threshold = kwargs.get('eig_val_threshold', 1e-12)
        # Eigen decomposition method
        X_tilde = X
        if Q1 is not None:
            D1, V1 = torch.linalg.eigh(Q1)
            V1 = V1[:, D1 >= eig_val_threshold]
            D1 = D1[D1 >= eig_val_threshold]
            Q1_sqrt = V1 @ torch.diag(torch.sqrt(D1)) @ V1.T
            Q1_sqrt_inv = V1 @ torch.diag(1.0 / torch.sqrt(D1)) @ V1.T
            X_tilde = Q1_sqrt @ X_tilde
        if Q2 is not None:
            D2, V2 = torch.linalg.eigh(Q2)
            V2 = V2[:, D2 >= eig_val_threshold]
            D2 = D2[D2 >= eig_val_threshold]
            Q2_sqrt = V2 @ torch.diag(torch.sqrt(D2)) @ V2.T
            Q2_sqrt_inv = V2 @ torch.diag(1.0 / torch.sqrt(D2)) @ V2.T
            X_tilde = X_tilde @ Q2_sqrt


        U_tilde, S, Vt_tilde = torch.linalg.svd(X_tilde, full_matrices=False)
        if Q1 is None:
            U = U_tilde[:,:rank]
        else:
            Q1_sqrt_inv @ U_tilde[:, :rank]
        if Q2 is None:
            Vt = Vt_tilde[:rank, :]
        else:
            Vt_tilde[:rank, :] @ Q2_sqrt_inv
        S = S[:rank]
    elif method == 'cholesky':
        # Cholesky decomposition method
        X_tilde = X
        if Q1 is not None:
            Q1_tilde = torch.linalg.cholesky(Q1)
            X_tilde = Q1_tilde.T @ X_tilde
        if Q2 is not None:
            Q2_tilde = torch.linalg.cholesky(Q2)
            X_tilde = X_tilde @ Q2_tilde

        U_tilde, S, Vt_tilde = torch.linalg.svd(X_tilde, full_matrices=False)
        U_tilde = U_tilde[:, :rank]
        V_tilde = Vt_tilde[:rank, :].T
        if Q1 is None:
            U = U_tilde
        else:
            U = torch.linalg.solve(Q1_tilde, U_tilde)
        if Q2 is None:
            V = V_tilde
        else:
            V = torch.linalg.solve(Q2_tilde, V_tilde)
        Vt = V.T
        S = S[:rank]
    else:
        raise ValueError(f"Unknown method '{method}'."
                         "Supported methods are 'eig' and 'cholesky'.")
    return U, S, Vt

