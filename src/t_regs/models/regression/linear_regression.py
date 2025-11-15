

import torch
import numpy as np

from ..regression_base import RegressionBaseClass

class LinearRegression(RegressionBaseClass):

    def __init__(self, fit_intercept=True,
                        solver = 'lstsq',
                        solver_cfg=None,
                        **kwargs):
        """Initialize Linear Regression Model"""
        super().__init__(**kwargs)


    def fit(self, X, y, **kwargs):
        """Fit the Linear Regression Model to the data"""
        
        
        self.n_samples, self.n_features = X.shape
        if self.fit_intercept:
            X = torch.cat([torch.ones((self.n_samples, 1), device=self.device, dtype=self.dtype), X], dim=1)
        
        if self.solver == 'lstsq':
            self.coef_, _ = torch.linalg.lstsq(y, X, self.solver_cfg).solution.T
            self.coef_ = self.coef_[:X.shape[1]]

        self.is_fitted_ = True

    def _predict(self, X, **kwargs):
        """Predict target values for given input data using the trained model"""
        X = self._move_input(X)
        if self.fit_intercept:
            n_samples = X.shape[0]
            X = torch.cat([torch.ones((n_samples, 1), device=self.device, dtype=self.dtype), X], dim=1)
        return X @ self.coef_.T

class LassoRegression(RegressionBaseClass):

    def __init__(self, lda=1.0,
                        fit_intercept=True,
                        solver = 'proximal_gradient',
                        solver_cfg={
                            'max_iter': 1000,
                            'tol': 1e-4,
                        },
                        **kwargs):
        """Initialize Lasso Regression Model"""
        super().__init__(**kwargs)