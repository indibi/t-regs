import numpy as np
import torch
from ..multilinear_ops import unfold, fold



def soft_svt(T, tau):
    """Soft thresholding of the singular values of a matrix with the thresholding parameter tau.
    
    Args:
        T (np.ndarray or torch.Tensor): Input matrix to be thresholded.
        tau (float): threshold

    Returns:
        Xnew: Thresholded matrix.
    """
    if isinstance(X, np.ndarray):
        U, S, V = np.linalg.svd(T,  full_matrices=False)
        s = S-tau
        smask = s > 0
        S = np.diag(s[smask])
        nuc_norm = sum(s[smask])
        X = U[:, smask]@S@V[smask, :]
        return X, nuc_norm
    elif isinstance(T, torch.Tensor):
        try:
            U, S, V = torch.linalg.svd(T, full_matrices=False)
        except:
            U, S, V = torch.linalg.svd(T, full_matrices=False, driver='gesvda')
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
