from typing import Union, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class GeneralizedMultiLinearModel(nn.Module):
    r"""Generalized Multi-linear Model

    Parameters
    ----------
    regression_type: str
        The type of the regression model. The available options are 
        `'linear'`, `'logistic'`, 'multinomial', and 'poisson' regression.
    feature_dims: Sequence[int]
        Dimensions of the feature tensors.
    task_dims: Optional[Sequence[int]]
        The dimensions of the task for multi-task applications. :math:`(T_1,...,
        T_M). Defaults to univariate regression with value `(1,)`.
    coeff: Optional[torch.Tensor]
    """

    regression_types = ['linear', 'logistic', 'multinomial', 'poisson']
    def __init__(
        self,
        regression_type: str,
        feature_dims: Sequence[int],
        task_dims: Optional[Sequence[int]] = None,
        coeff: Optional[torch.Tensor] = None,
        ):
        super().__init__()
        task_dims = (1,) if task_dims is None else task_dims
        self.regression_type = regression_type
        self.task_dims = tuple(task_dims)
        self.feature_dims = tuple(feature_dims)
        self.dims = self.task_dims + self.feature_dims
        self._M = len(self.task_dims)
        self._N = len(self.feature_dims)
        self.order = self._M + self._N
        self.modes = [i for i in range(1, self.order + 1)]
        self.B = nn.Parameter(
            data= torch.zeros(self.dims) if coeff is None else coeff
            )


    @torch.no_grad()
    def score(self, pred:torch.Tensor, Y:torch.Tensor) -> float:    # pylint: disable=invalid-name
        # TODO: Add docstring and perhaps other options for scores.
        if self.regression_type == 'linear':
            ss_total = torch.sum((Y - torch.mean(Y, dim=0))**2)
            ss_residual = torch.sum((Y - pred)**2)
            r2_score = 1.0 - (ss_residual / ss_total)
            return r2_score.item()
        elif self.regression_type == 'logistic':
            pred = pred >0.5
            accuracy = torch.sum(Y == pred).item()
            accuracy = accuracy/ Y.shape[0]
            return accuracy
        elif self.regression_type == 'multinomial':
            pred_idx = pred.argmax(dim=1)
            accuracy = torch.sum(Y == pred_idx).item()
            accuracy = accuracy / Y.shape[0]
            return accuracy
        elif self.regression_type == 'poisson':
            raise NotImplementedError
        else:
            raise ValueError

    @torch.no_grad()
    def predict(self, # pylint: disable=unused-argument
            X:torch.Tensor, # pylint: disable=invalid-name
            y = None,
            ) -> torch.Tensor:
        b_dim = X.ndim - self.covariant_degree
        eta = torch.tensordot(X, self.B,
            dims = (
                [n for n in range(b_dim, b_dim+self._N)],
                [n for n in range(self._M, self.order)]
                )
            )
        return self.inverse_link(eta)

    def inverse_link(self, eta: torch.Tensor) -> torch.Tensor:
        """Calculate the systematic means (μ) from linear predictors (η)."""
        # TODO: improve the docstring of the inverse link functions.
        if self.regression_type == 'linear' or self.regression_type is None:
            return eta
        elif self.regression_type == 'logistic':
            return F.sigmoid(eta)
        elif self.regression_type == 'multinomial':
            return F.softmax(eta, dim=1 if eta.ndim == self._M else 0)
        elif self.regression_type == 'poisson':
            return torch.exp(eta)
        else:
            raise NotImplementedError

    def loss_fn(self,
                eta: torch.Tensor,
                Y: torch.Tensor,    # pylint: disable=invalid-name
                weights: Optional[torch.Tensor]=None) -> torch.Tensor:
        """Loss function corresponding to the Generalized Linear Model"""
        # TODO: improve the docstring of the loss functions.
        Y = Y.reshape((-1,) + self.task_dims)
        if self.regression_type == 'linear':
            return F.mse_loss(eta, Y, weight=weights, reduction='sum')
        elif self.regression_type == 'logistic':
            return F.binary_cross_entropy_with_logits(eta,
                                                      Y,
                                                      weight=weights,
                                                      reduction='sum')
        elif self.regression_type == 'multinomial':
            return F.cross_entropy(eta, Y, weight=weights, reduction='sum')
        elif self.regression_type == 'poisson':
            return  F.poisson_nll_loss(eta, Y,
                                       log_input=True,
                                       reduction='sum')
        else:
            raise NotImplementedError

    def forward(self, x: torch.Tensor):
        b_dim = x.ndim - self.covariant_degree
        eta = torch.tensordot(x, self.B,
            dims = (
                [n for n in range(b_dim, b_dim+self._N)],
                [n for n in range(self._M, self.order)]
                )
            )
        return eta

    @property
    def covariant_degree(self):
        return self._N

    @property
    def contravariant_degree(self):
        return self._M
