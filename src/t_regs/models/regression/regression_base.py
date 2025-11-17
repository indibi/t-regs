"""Abstract Base Class for Regression model implementations."""

from abc import ABC, abstractmethod
from typing import Self
from collections import defaultdict

import torch
import numpy as np

class RegressionBaseClass(ABC):
    def __init__(self,
                    device: str | torch.device = 'cuda' if torch.cuda.is_available() else 'cpu',
                    dtype: torch.dtype = torch.float64,
                    verbose: int = 0,
                    **kwargs):
        """Initialize Regression Model"""
        self.is_fitted = False
        self.dtype = dtype
        self.device = device
        self.verbose = verbose
        self.solver = None
        self.model_hyperparams = {}
        self.model_params = {}
        self.solver_params = {}
        self.solver_results = defaultdict(lambda: None)

    def score(self, X, y, **kwargs):                # py-lint: disable=invalid-name
        """Score the fit of the model"""
        X = self._move_input(X)
        y = self._move_input(y)
        return self._score(X, y, **kwargs)

    def predict(self, X, y=None, **kwargs):         # py-lint: disable=invalid-name
        """Predict target values for given input data using the trained model"""
        if not self.is_fitted:
            raise RuntimeError("The model must be fitted before prediction.")
        X = self._move_input(X)
        y = self._move_input(y)
        return self._predict(X, y, **kwargs)

    def fit(self, X, y, **kwargs):                  # py-lint: disable=invalid-name
        """Fit the regression model to the data."""
        X = self._move_input(X)
        y = self._move_input(y)
        fit = self._fit(X, y, **kwargs)
        self.is_fitted = True
        return fit

    @abstractmethod
    def _fit(self, X, y, **kwargs) -> Self:         # py-lint: disable=invalid-name
        """Internal method to fit the regression model to the data."""

    @abstractmethod
    def _predict(self, X, y=None, **kwargs):        # py-lint: disable=invalid-name
        """Internal method to predict target values for given input data."""

    @abstractmethod
    def _score(self, X, y, **kwargs):               # py-lint: disable=invalid-name
        """Internal method to score the fit of the model."""

    # def get_params(self, **kwargs):
    #     """Get the model parameters"""
    #     pass

    # @abstractmethod
    # def set_params(self, **kwargs):
    #     """Set the model parameters"""
    #     pass


    def move_to_device(self, device):
        """Move model parameters to the specified device."""
        self.device = device
        for attr, value in self.__dict__.items():
            if isinstance(value, torch.Tensor):
                setattr(self, attr, value.to(device))
            if isinstance(value, np.ndarray):
                tensor_value = torch.tensor(value, device=device, dtype=self.dtype)
                setattr(self, attr, tensor_value)
        return self

    def numpy(self):
        """Convert model parameters to numpy arrays."""
        for attr, value in self.__dict__.items():
            if isinstance(value, torch.Tensor):
                setattr(self, attr, value.cpu().numpy())
        return self

    def _move_input(self, input_):
        """Move input data to the model's device and dtype."""
        if isinstance(input_, torch.Tensor):
            return input.to(self.device, dtype=self.dtype)
        elif isinstance(input_, np.ndarray):
            return torch.tensor(input_, device=self.device, dtype=self.dtype)
        else:
            return input_

