import torch
import numpy as np

import torch.nn.functional as F

from .proximal_gradient_base import ProximalGradientBase
from ..utils import est_spectral_norm

class LassoPG(ProximalGradientBase):
    r"""Lasso solver using Proximal Gradient Method.

    The Lasso regression solves the following optimization problem:

    .. math::
        \min_{B} \frac{1}{2} ||Y - X B||_F^2 + \lambda ||B||_1

    where :math:`X` is the input data matrix, :math:`Y` is the target variables,
    :math:`B` are the coefficients to be learned, and :math:`\lambda` is the
    regularization parameter.

    Parameters
    ----------
    lda : float
        Sparsity regularization parameter :math:`\lambda`.
    
    """

    def __init__(self,
                    X: torch.Tensor | np.ndarray,
                    Y: torch.Tensor | np.ndarray,
                    lda: float,
                    regression_type: str = 'linear',
                    **kwargs):
        self.lda = lda
        self.regression_type = regression_type
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X)
        if isinstance(Y, np.ndarray):
            Y = torch.from_numpy(Y)
        if X.is_cuda:
            self.device = X.get_device()

        if kwargs.get('lipschitz_constant', None) is None:
            if self.regression_type == 'logistic':
                # For logistic regression, the Lipschitz constant is 0.25 * ||X||_2^2
                kwargs['lipschitz_constant'] = 0.25 * est_spectral_norm(X)**2
            else:# self.regression_type == 'linear':
                kwargs['lipschitz_constant'] = est_spectral_norm(X)**2
        
        super().__init__(**kwargs)
        self.X = X.to(self.device, dtype=self.dtype)    # py-lint: disable=invalid-name
        self.Y = Y.to(self.device, dtype=self.dtype)    # py-lint: disable=invalid-name

        assert self.X.dim() == 2, "X must be a 2D matrix."
        assert self.Y.dim() >= 1, "Y must be a vector or 2D matrix"
        assert self.X.size(0) == self.Y.size(0), "Number of samples in X and Y must match."

        if self.Y.dim() == 1:
            self.Y = self.Y.reshape(-1, 1)

    def func_f(self, x):
        B = x                                           # py-lint: disable=invalid-name
        if self.regression_type == 'logistic':
            logits = torch.matmul(self.X, B)
            probs = torch.sigmoid(logits)
            nll_loss = F.binary_cross_entropy(probs, self.Y, reduction='sum')
            return nll_loss
        else: # self.regression_type == 'linear':
            residual = self.Y - torch.matmul(self.X, B)
            return 0.5 * torch.norm(residual, p='fro')**2

    def func_g(self, x):
        B = x                                           # py-lint: disable=invalid-name
        return self.lda * B.abs().sum()

    def grad_f(self, x):
        B = x                                           # py-lint: disable=invalid-name
        if self.regression_type == 'logistic':
            probs = torch.sigmoid(torch.matmul(self.X, B))
            residual = probs - self.Y
        elif self.regression_type == 'linear':
            residual = torch.matmul(self.X, B) - self.Y
        else:
            msg = (f"Regression type '{self.regression_type}' not recognized."
                   f" Please use 'linear' or 'logistic'.")
            raise ValueError(msg)    
        grad = torch.matmul(self.X.T, residual)
        return grad

    def prox_step(self, x, step_size):
        thresh = self.lda * step_size
        return F.softshrink(x, lambd=thresh)
