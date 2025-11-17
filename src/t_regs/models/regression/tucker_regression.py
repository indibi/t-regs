import torch
import torch.nn.functional as F
import numpy as np

from .regression_base import RegressionBaseClass
from ...solvers.proximal_gradient_base import ProximalGradientBase

from ..tensor_decomp.tucker.hosvd import HoSVD
from ..tensor_decomp.tucker.hooi import HoOI

from ...multilinear_ops.tensor_products import multi_mode_product as mmp


class HoSVTruncationPG(ProximalGradientBase):
    r"""Higher-Order Singular Value Truncation Proximal Gradient Method.

    Approximately solves the optimization problem:
    .. math::
        \min_{B} f(B) + I_C(B)

        where :math:`f` is a differentiable loss function and :math:`I_C`
        is the indicator function of the set
        .. math::
            C = \{ B : \text{TRank}_(B) \leq (r_1, r_2, \ldots, r_N) \}
        
    where :math:`\text{TRank}(B)` is the Tucker rank of the tensor :math:`B`.
    Here, we implement the method for tensor regression problems with
    either following loss functions:
    - Linear regression loss:
    .. math::
        f(B) = \sum_{i=1}^n \frac{1}{2}(y_i - \langle X_i, B \rangle)^2
    - Logistic regression loss:
    .. math::
        f(B) = \sum_{i=1}^n \left[ \log(1 + \exp(\langle X_i, B \rangle))
        - y_i \langle X_i, B \rangle \right]
        
    Parameters
    ----------
        X : torch.Tensor | np.ndarray
            The predictor tensor data of shape (n_samples, *dims).
        Y : torch.Tensor | np.ndarray
            The response tensor data of shape (n_samples, *task_dims).
        regression_type : str = 'linear' {'linear', 'logistic'}
            The type of regression model to fit.
            - 'linear': Linear regression loss.
            - 'logistic': Logistic regression loss.
        decomposition : str = 'HoSVD' {'HoSVD', 'HoOI'}
            The type of tensor decomposition to use for the proximal step.
            - 'HoSVD': Higher-Order Singular Value Decomposition.
            - 'HoOI': Higher-Order Orthogonal Iteration.
        ranks : tuple[int]
            The desired Tucker ranks for the coefficient tensor.
    
    """
    decomposition_options = ['hosvd', 'hooi']
    regression_options = ['linear', 'logistic']
    def __init__(self,
                X: torch.Tensor | np.ndarray,
                Y: torch.Tensor | np.ndarray,
                ranks: tuple[int, ...],
                decomposition: str = 'HoSVD',
                regression_type: str = 'linear',
                **kwargs,
                 ):
        
        self.ranks = ranks
        self.decomposition = decomposition
        self.regression_type = regression_type
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X)
        if isinstance(Y, np.ndarray):
            Y = torch.from_numpy(Y)
        super().__init__(**kwargs)
        if X.is_cuda:
            self.device = X.get_device()
        self.X = X.to(self.device, dtype=self.dtype)    # py-lint: disable=invalid-name
        self.Y = Y.to(self.device, dtype=self.dtype)    # py-lint: disable=invalid-name

        self.n_samples = X.shape[0]
        self.feature_dims = X.shape[1:]
        self.task_dims = Y.shape[1:len(self.feature_dims)+1]
        self.beta_shape = list(self.task_dims) + list(self.feature_dims)
        
        assert Y.ndim >= 1, "Response tensor Y must have at least one dimension (samples)."
        assert Y.shape[0] == self.n_samples, "Number of samples in X and Y must match."
        assert len(ranks) == len(self.feature_dims
                                )+len(self.task_dims
                        ), "Length of ranks must match number of feature dimensions."

    def func_f(self, x):
        B = x
        modes_X = list(range(1, self.X.ndim))
        modes_B = list(range(len(self.task_dims), B.ndim))
        in_products = torch.tensordot(self.X, B, 
                                    dims=(modes_X, modes_B)
        )   # shape: (n_samples, *task_dims)
        
        if self.regression_type == 'linear':
            residuals = (self.Y - in_products) # shape: (n_samples, *task_dims)
            f_val = 0.5 * torch.sum(residuals**2)
        elif self.regression_type == 'logistic':
            logits = in_products
            probs = torch.sigmoid(logits)
            f_val = F.binary_cross_entropy(probs, self.Y, reduction='sum')
        else:
            msg = (f"Regression type '{self.regression_type}' not recognized."
                   f" Please use from {self.regression_options}.")
            raise ValueError(msg)
        return f_val

    def grad_f(self, x):
        r"""Computes the gradient of the least squares loss function.

        .. math::
            \nabla f(B) = -\sum_{i=1}^n (y_i - \langle X_i, B \rangle) X_i

        Parameters
        ----------
            x : torch.Tensor
                The current estimate of the coefficient tensor.

        Returns
        -------
            torch.Tensor
                The gradient tensor of the same shape as x.
        """
        B = x
        modes_X = list(range(1, self.X.ndim))
        modes_B = list(range(len(self.task_dims), B.ndim))
        in_products = torch.tensordot(self.X, B, 
                                    dims=(modes_X, modes_B)
        )   # shape: (n_samples, *task_dims)
        
        
        exp_X_dim = [self.X.shape[0]] + [1]*len(self.task_dims) + list(self.X.shape[1:])
        if self.regression_type == 'linear':
            residuals = -(self.Y - in_products) # shape: (n_samples, *task_dims)
        elif self.regression_type == 'logistic':
            probs = torch.sigmoid(in_products)
            residuals = probs - self.Y
        else:
            msg = (f"Regression type '{self.regression_type}' not recognized."
                   f" Please use from {self.regression_options}.")
            raise ValueError(msg)
        exp_res_dim = list((*residuals.shape,))+[1]*len(self.feature_dims)
        X_exp = self.X.reshape(exp_X_dim)

        residuals = residuals.reshape(exp_res_dim)
        grad = torch.sum(residuals * X_exp, dim=0)
        return grad

    def func_g(self, x):
        # B = x
        return 0.0
    
    def prox_step(self, x, step_size=None):
        r"""Project the tensor onto low Tucker rank set using HoSVD truncation.
        
        Parameters
        ----------
            x : torch.Tensor
                The current estimate of the coefficient tensor.
            step_size : float = None
                The step size for the proximal gradient update.

        Returns
        -------
            torch.Tensor
                The projected tensor of the same shape as x.
        """

        B = x
        if self.decomposition.lower() == 'hosvd':
            hosvd = HoSVD(B, core_dims=self.ranks,
                        device=self.device,
                        dtype=self.dtype)
            C, Us = hosvd()
        elif self.decomposition.lower() == 'hooi':
            hooi = HoOI(B, n_ranks=self.ranks,
                            device=self.device,
                            dtype=self.dtype,
                            verbose=False,
                            max_it=100,
                            err_tol=1e-8)
            C, Us = hooi()
        else:
            raise ValueError((f"Decomposition method '{self.decomposition}'"
                              f" not recognized. Use {self.decomposition_options}."))
        B_proj = mmp(C, Us, modes=list(range(1, B.ndim+1)))
        return B_proj




DEFAULT_TUCKER_REGRESSION_SOLVER_PARAMS = {
    'HoSVTruncPG': {
        'decomposition': 'HoSVD',
        'step_size_strategy': 'backtracking',
        'max_it': 100,
        'err_tol': 1e-6,
    },
}


class TuckerRegression(RegressionBaseClass):
    r"""Low Tucker Rank Tensor Regression Model

    Fits a tensor regression model with a low Tucker rank constraint on 
    the coefficient tensor :math:`B`:
    .. math::
        \min_{B} \sum_{i=1}^n \frac{1}{2}(y_i - \langle X_i, B \rangle)^2 + I_C(B)

        where :math:`I_C` is the indicator function of the set
        .. math::
            C = \{ B : \text{TRank}_(B) \leq (r_1, r_2, \ldots, r_N) \}

    where :math:`\text{TRank}(B)` is the Tucker rank of the tensor :math:`B`.

    """
    solvers = ['HoSVTruncPG']
    regression_options = ['linear', 'logistic']
    def __init__(self,
                ranks: tuple[int, ...],
                regression_type: str = 'linear',
                fit_intercept: bool = False,
                solver: str = 'HoSVTruncPG',
                solver_params: dict = {},
                **kwargs,
                 ):
        self.regression_type = regression_type
        self.ranks = ranks
        self.fit_intercept = fit_intercept
        super().__init__(**kwargs)

        self.solver = solver
        default_params = DEFAULT_TUCKER_REGRESSION_SOLVER_PARAMS[solver].copy()
        self.solver_params = default_params
        self.solver_params.update(solver_params)
    
    def _fit(self, X: torch.Tensor | np.ndarray,
                    Y: torch.Tensor | np.ndarray,
                    **kwargs):

        if self.fit_intercept:
            raise NotImplementedError("Fit intercept not implemented for TuckerRegression yet.")
        
        if self.solver == 'HoSVTruncPG':
            solver = HoSVTruncationPG(X, Y,
                                    regression_type=self.regression_type,
                                    ranks=self.ranks,
                                    **self.solver_params
                                    )
            # TODO: Fix this
            self.task_dims = solver.task_dims # <-- Need to set before without using the solver
            beta0 = torch.zeros(solver.beta_shape,
                                device=solver.device,
                                dtype=solver.dtype)
            self.beta = solver(beta0)
            if hasattr(solver, 'get_run_history'):
                self.solver_results = solver.get_run_history()
            else:
                self.solver_results['objective_values'] = solver.objective
                self.solver_results['step_sizes'] = solver.step_sizes
                self.solver_results['n_iterations'] = solver.it
                self.solver_results['converged'] = solver.converged
                self.solver_results['delta_objective'] = solver.delta_objective
                self.solver_results['delta_x'] = solver.delta_x
        else:
            raise ValueError(f"Solver '{self.solver}' not recognized for TuckerRegression.")
        return self
    
    def _predict(self,
                    X: torch.Tensor | np.ndarray,
                    Y = None,
                    return_prob: bool = False,
                    **kwargs) -> torch.Tensor | np.ndarray:
        
        if self.fit_intercept:
            raise NotImplementedError("Fit intercept not implemented for TuckerRegression yet.")
        
        modes_X = list(range(1, X.ndim))
        modes_B = list(range(len(self.task_dims), self.beta.ndim))
        in_products = torch.tensordot(X, self.beta, 
                                    dims=(modes_X, modes_B)
        )
        if self.regression_type == 'logistic':
            probs = torch.sigmoid(in_products)
            if return_prob:
                return probs
            else:
                return (probs > 0.5).float()
        elif self.regression_type == 'linear':
            return in_products
        else:
            msg = (f"Regression type '{self.regression_type}' not recognized."
                   f" Please use from {self.regression_options}.")
            raise ValueError(msg)
    
    
    def _score(self, X, Y, **kwargs) -> float:
        if self.fit_intercept:
            raise NotImplementedError("Fit intercept not implemented for TuckerRegression yet.")
        if self.regression_type == 'linear':
            Y_pred = self._predict(X)
            
            ss_total = torch.sum((Y - torch.mean(Y, dim=0))**2)
            ss_residual = torch.sum((Y - Y_pred)**2)
            r2_score = 1.0 - (ss_residual / ss_total)
            return r2_score.item()
        elif self.regression_type == 'logistic':
            Y_pred = self._predict(X, return_prob=False)
            accuracy = torch.sum(Y == Y_pred).item() / Y.numel()
            return accuracy
        else:
            msg = (f"Regression type '{self.regression_type}' not recognized."
                   f" Please use from {self.regression_options}.")
            raise ValueError(msg)
        
