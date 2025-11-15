"""Abstract Base Class for Regression model implementations."""

from abc import ABC, abstractmethod

import torch
import numpy as np

class RegressionBaseClass(ABC):
    def __init__(self, **kwargs):
        """Initialize Regression Model"""
        self.is_fitted_ = False
        self.dtype = kwargs.get('dtype', torch.float64)
        self.device = kwargs.get('device',
                                'cuda' if torch.cuda.is_available() else 'cpu')

    @abstractmethod
    def score(self, X, y, *vargs, **kwargs):
        """Score the fit of the model."""
        pass

    @abstractmethod
    def predict(self, X, **kwargs):
        """Predict target values for given input data using the trained model."""
        pass

    @abstractmethod
    def fit(self, X, y, **kwargs):
        """Fit the regression model to the data."""
        pass

    @abstractmethod
    def get_params(self, **kwargs):
        """Get the model parameters"""
        pass

    @abstractmethod
    def set_params(self, **kwargs):
        """Set the model parameters"""
        pass

    @property
    @abstractmethod
    def is_fitted(self):
        """Check if the model has been fitted."""
        pass

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

