from collections import defaultdict

import torch
import numpy as np

from .regression_base import RegressionBaseClass
from ...solvers.lasso import LassoPG

DEFAULT_SOLVER_PARAMS = {
    'pinv': {
        'rcond': 1e-15
    },
    'lstsq': {
        'rcond': 1e-15,
        'driver': 'gels'
    }
}

class LinearRegression(RegressionBaseClass):
    """Linear Regression Model using different solvers.
    """
    solvers = ['pinv', 'lstsq']
    def __init__(self, fit_intercept: bool = True, # py-lint: disable=dangerous-default-value
                        solver: str = 'pinv',      # Not altering solver_params. No issue.
                        solver_params: dict = {},
                        **kwargs):
        """Initialize Linear Regression Model
        
        Parameters
        ----------
            fit_intercept : bool = True
                Whether to fit the intercept term.
            solver : str = 'pinv', {'pinv', 'lstsq'}
                Solver to use for fitting the model. 
                - 'pinv': Moore-Penrose Pseudoinverse method as implemented
                in torch.linalg.pinv.
                - 'lstsq': Least Squares method as implemented
                in torch.linalg.lstsq.
            solver_params (dict): Additional parameters for the solver.
                Supplied to the underlying torch.linalg solver functions.
            **kwargs: 
                dtype [torch.dtype| str]: Data type for computations.
                    Defaults to torch.float64.
                device [str]: Device to use ('cpu' or 'cuda:{i}').
                    Defaults to 'cuda' if available else 'cpu'.
        """
        super().__init__(**kwargs)
        self.fit_intercept = fit_intercept
        if solver.lower() not in self.solvers:
            raise ValueError(f"Solver '{solver}' is not supported. Choose from {self.solvers}.")


        def_solver_params = DEFAULT_SOLVER_PARAMS[solver.lower()]
        def_solver_params.update(solver_params)
        self.solver = solver
        self.solver_params = def_solver_params
        self.solver_results = defaultdict(lambda: None)
        self.beta = None
        self.n_samples = None
        self.n_features = None
        self.n_targets = None

    def _fit(self, X, y, **kwargs):
        """Fit a Linear Regression Model to the data
        
        Parameters
        ----------
            X : torch.Tensor | np.ndarray
                Input data of shape (n_samples, n_features).
            y : torch.Tensor | np.ndarray
                Target data of shape (n_samples,) or (n_samples, n_targets).
        
        Returns
        -------
            self : LinearRegression
                Fitted Linear Regression model.
        """
        assert X.ndim == 2, "Input data X must be a 2D tensor."
        assert y.ndim in [1, 2], "Target data y must be a 1D or 2D tensor."

        self.n_samples, self.n_features = X.shape
        self.n_targets = y.shape[1] if len(y.shape) > 1 else 1

        if self.fit_intercept:
            X = torch.cat([torch.ones((self.n_samples, 1),
                                        device=self.device, dtype=self.dtype
                                        ),
                            X],
                        dim=1)

        if self.solver.lower() == 'pinv':
            self.beta = torch.linalg.pinv(X, **self.solver_params) @ y     # pylint: disable=not-callable
        elif self.solver.lower() == 'lstsq':
            self.beta, res, rank, svals = torch.linalg.lstsq(X, y, **self.solver_params) # pylint: disable=not-callable
            self.solver_results['residuals'] = res
            self.solver_results['rank'] = rank
            self.solver_results['singular_values'] = svals
        return self

    def _predict(self, X, y=None, **kwargs):
        """Predict target values for given input data using the trained model"""
        if self.fit_intercept:
            n_samples = X.shape[0]
            X = torch.cat([torch.ones((n_samples, 1),
                                    device=self.device, dtype=self.dtype
                            ), X],
                        dim=1)
        return X @ self.beta

    def _score(self, X, y_true, **kwargs):
        """Score the fit of the model using R^2 score."""
        y_pred = self._predict(X)
        ss_total = torch.sum((y_true - torch.mean(y_true, dim=0))**2)
        ss_residual = torch.sum((y_true - y_pred)**2)
        r2_score = 1 - (ss_residual / ss_total)
        return r2_score.item() if self.n_targets == 1 else r2_score.cpu().numpy()


DEFAULT_LASSO_SOLVER_PARAMS = {
    'LassoPG': {
        'max_it': 1000,
        'err_tol': 1e-6,
        'lipschitz_const': None,
        'verbose': False,
        'step_size_strategy': 'fixed',
    }
}

class LassoRegression(RegressionBaseClass):
    r"""Initialize Lasso Regression Model

    Solves the optimization problem
    .. math::

        \min_{\beta} \frac{1}{2} \|Y - X \beta\|_F^2 + \lambda \|\beta\|_1

    Parameters
    ----------
    lda : float = 1.0
        Sparsity regularization parameter. :math:`\lambda` in the objective function.
    regression_type : str = 'linear', {'linear', 'logistic'}
        Type of regression model to fit.
    fit_intercept : bool = True
        Whether to fit the intercept term.
    solver : str = 'LassoPG'
        Solver to use for optimization. Currently only supports 'LassoPG' 
        (Proximal Gradient).
    solver_params : dict = {}
        Parameters for the solver.
    **kwargs
        Additional keyword arguments for the base class.
    
    Attributes
    ----------
    """
    solvers = ['LassoPG']
    def __init__(self, lda=1.0,
                        regression_type: str = 'linear',
                        fit_intercept=True,
                        solver: str = 'LassoPG',
                        solver_params: dict = {},
                        **kwargs):
        super().__init__(fit_intercept=fit_intercept, **kwargs)
        self.regression_type = regression_type
        self.lda = lda
        self.fit_intercept = fit_intercept
        
        if solver not in self.solvers:
            msg = f"Solver '{solver}' not recognized. Available solvers: {self.solvers}"
            raise ValueError(msg)
        
        self.solver = solver
        default_params = DEFAULT_LASSO_SOLVER_PARAMS[solver].copy()
        self.solver_params = default_params
        self.solver_params.update(solver_params)
    

    def _fit(self,
                X: torch.Tensor | np.ndarray,
                Y: torch.Tensor | np.ndarray,
                **kwargs):

        assert X.ndim == 2, "`X` must be a 2D matrix."
        assert Y.ndim in [1, 2], "Target data Y must be a 1D or 2D tensor."

        self.n_samples, self.n_features = X.shape
        self.n_targets = Y.shape[1] if len(Y.shape) > 1 else 1
        
        if self.fit_intercept:
            X = torch.cat([torch.ones((self.n_samples, 1),
                                        device=self.device, dtype=self.dtype
                                        ),
                            X],
                        dim=1)
            self.n_features += 1
        
        if self.solver == 'LassoPG':
            solver = LassoPG(X, Y,
                            lda=self.lda,
                            regression_type=self.regression_type,
                            **self.solver_params)
            beta0 = torch.zeros((self.n_features, self.n_targets),
                                    device=self.device,
                                    dtype=self.dtype)
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
            raise NotImplementedError(f"Solver '{self.solver}' not implemented.")
        return self
    
    def _predict(self,
                    X: torch.Tensor | np.ndarray,
                    Y = None,
                    return_probs: bool = False,
                    **kwargs) -> torch.Tensor | np.ndarray:
        assert X.ndim == 2, "`X` must be a 2D matrix."
        n_samples = X.shape[0]
        if self.fit_intercept:
            X = torch.cat([torch.ones((n_samples, 1),
                                        device=self.device, dtype=self.dtype
                                        ),
                            X],
                        dim=1)
        
        if self.regression_type == 'logistic':
            logits = X @ self.beta
            probs = torch.sigmoid(logits)
            if return_probs:
                return probs
            Y_pred = (probs >= 0.5).float()
        elif self.regression_type == 'linear':
            Y_pred = X @ self.beta
        else:
            msg = (f"Regression type '{self.regression_type}' not recognized."
                   f" Please use 'linear' or 'logistic'.")
            raise ValueError(msg)
        return Y_pred
    
    def _score(self, X, y_true, **kwargs):
        """Score the fit of the model using R^2 score."""
        if y_true.ndim == 1:
            y_true = y_true.reshape(-1, 1)
        
        if self.regression_type == 'logistic':
            Y_pred = self._predict(X)
            accuracy = torch.sum(y_true == Y_pred).item() / y_true.numel()
            return accuracy
        elif self.regression_type == 'linear':
            y_pred = self._predict(X)
            ss_total = torch.sum((y_true - torch.mean(y_true, dim=0))**2)
            ss_residual = torch.sum((y_true - y_pred)**2)
            r2_score = 1 - (ss_residual / ss_total)
            return r2_score.item() if self.n_targets == 1 else r2_score.cpu().numpy()
        else:
            msg = (f"Regression type '{self.regression_type}' not recognized."
                   f" Please use 'linear' or 'logistic'.")
            raise ValueError(msg)