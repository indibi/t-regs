"""Tensor Singular Value Thresholding


References:
-----------
..  [1] Canyi Lu, Tensor-Tensor Product Toolbox. Carnegie Mellon University.
    June, 2018. https://github.com/canyilu/tproduct.

..  [2] Canyi Lu, Jiashi Feng, Yudong Chen, Wei Liu, Zhouchen Lin and 
    Shuicheng Yan, Tensor Robust Principal Component Analysis with A New Tensor
    Nuclear Norm, arXiv preprint arXiv:1804.03728, 2018
"""
from typing import Tuple
import torch
import torch.linalg as LA       

def prox_tnn(Y:torch.Tensor,
             tau:float,
             mode:int =1) -> Tuple[torch.Tensor, float, int]:

    if Y.ndim != 3:
        raise NotImplementedError("`prox_tnn` is implemented for only 3 mode tensors")
    if mode!=1:
        raise NotImplementedError("Currently, only the first mode is supported")
    X = torch.zeros_like(Y)
    Y_f = torch.fft.fft(Y, dim=0)  # pylint: disable=not-callable
    n = Y.shape[0]
    tranks = 0
    tnn = 0

    i = 0
    U, S, Vh = LA.svd(Y_f[i,...], full_matrices=False)         # pylint: disable=not-callable
    s = S-tau
    smask = s > 0
    S = s[smask]
    tnn += sum(s[smask])
    tranks += sum(smask)
    X[i,...] = torch.einsum('ik,k,kj', U[:, smask], S, Vh[smask, :])

    for i in range((n-1)//2):
        U, S, Vh = LA.svd(Y_f[i+1,...], full_matrices=False)    # pylint: disable=not-callable
        s = S-tau
        smask = s > 0
        S = s[smask]
        tnn += sum(s[smask])
        tranks += sum(smask)
        X[i+1,...] = torch.einsum('ik,k,kj', U[:, smask], S, Vh[smask, :])
        X[-(i+1),...] = X[i+1,...].conj()

    if n%2 == 0:
        U, S, Vh = LA.svd(Y_f[i+2,...], full_matrices=False)    # pylint: disable=not-callable
        s = S-tau
        smask = s > 0
        S = s[smask]
        tnn += sum(s[smask])
        tranks += sum(smask)
        X[i+2,...] = torch.einsum('ik,k,kj', U[:, smask], S, Vh[smask, :]).conj()
        # X[-i,...] = X[i+1,...].conj()
    # TODO: There are redundancies, We do not need to multiply every face. fix that
    # X = torch.fft.ifft(X)
    return (torch.fft.ifft(X, dim=0).real, tnn, tranks) # pylint: disable=not-callable