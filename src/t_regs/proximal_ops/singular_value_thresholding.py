"""Module for the singular value thresholding proximal operators."""

import numpy as np
import torch
from ..multilinear_ops import unfold, fold

def t_soft_svt(X, tau):
    pass
    # TODO: Implement soft t-singular value thresholding

def t_sv_truncation(X, tau):
    pass
    # TODO: Implement soft t-singular value truncation

def t_hard_svt(X, tau):
    pass
    # TODO: Implement hard t-singular value thresholding

def soft_svt(T, tau): # pylint: disable=invalid-name
    """Soft threshold the singular values of matrix T with threshold tau.
    
    Args:
        T (np.ndarray or torch.Tensor): Input matrix to be thresholded.
        tau (float): threshold

    Returns:
        Xnew: Thresholded matrix.
    """
    if isinstance(T, np.ndarray):
        U, S, V = np.linalg.svd(T,  full_matrices=False) # pylint: disable=invalid-name
        s = S-tau
        smask = s > 0
        S = np.diag(s[smask])
        nuc_norm = sum(s[smask])
        X = U[:, smask]@S@V[smask, :] # pylint: disable=invalid-name
        return X, nuc_norm
    elif isinstance(T, torch.Tensor):
        try:
            U, S, V = torch.linalg.svd(T, full_matrices=False) # pylint: disable=not-callable
        except:
            U, S, V = torch.linalg.svd(T, full_matrices=False, driver='gesvda') # pylint: disable=not-callable
        s = S-tau
        smask = s > 0
        S = s[smask]
        nuc_norm = sum(s[smask])
        try:
            X = torch.einsum('ik,k,kj', U[:, smask], S, V[smask, :])
        except:
            torch.backends.opt_einsum.enabled = False
            X = torch.einsum('ik,k,kj', U[:, smask], S, V[smask, :])
        return X, nuc_norm


def mode_n_soft_svt(T, tau, n):
    """Soft thresholding of the singular values of a tensor in the n'th mode.
    
    With the thresholding parameter tau. 
    Args:
        T (np.ndarray): Tensor matricized in the n'th mode and
        thresholded in it's n'th mode singular values.
        tau (float): threshold
        n (int): mode index

    Returns:
        Tnew: Thresholded tensor.
    """
    sz = T.shape
    X, nuc_norm = soft_svt(unfold(T, n), tau)
    return fold(X, sz, n), nuc_norm

