"""Abstract base class for PCA implementations."""
from abc import ABC, abstractmethod

import torch
import numpy as np

class PCABaseClass(ABC):
    def __init__(self, X, **kwargs):
        """Initialize PCA model
        
        Args:
            X (torch.Tensor or np.ndarray): Input data matrix.
            kwargs: Additional keyword arguments.
                dtype (torch.dtype, optional): Data type for computations. 
                    Defaults to torch.float64.
                device (str, optional): Device to use ('cpu' or 'cuda').
                    Defaults to 'cuda' if available else 'cpu'. If the input tensor is
                    already on a cuda device, it uses that device.
        """
        self.dtype = kwargs.get('dtype', torch.float64)
        if isinstance(X, torch.Tensor):
            if X.is_cuda:
                self.device = X.device
            else:
                self.device = kwargs.get('device',
                                        'cuda' if torch.cuda.is_available() else 'cpu')
                self.X = X.to(self.device, dtype=self.dtype)
        elif isinstance(X, np.ndarray):
            self.device = kwargs.get('device',
                                    'cuda' if torch.cuda.is_available() else 'cpu')
            self.X = torch.tensor(X, device=self.device, dtype=self.dtype)
        else:
            raise ValueError("Input data must be a torch.Tensor or np.ndarray.")
        
    
    @abstractmethod
    def __call__(self, *vargs, **kwargs):
        """Execute the PCA algorithm with given hyperparameters."""
        pass

    @abstractmethod
    def score(self, *vargs, **kwargs):
        """Score the fit of the model using BIC or a similar statistical measure."""
        pass

    @abstractmethod
    def fit(self, **kwargs):
        """Fit the PCA model to the data, learning the hyperparameters using fit score."""
        pass
    
    @abstractmethod
    def project(self, **kwargs):
        """Project data onto the subspace spanned by the principal components."""
        pass

    @abstractmethod
    def transform(self, **kwargs):
        """Transform data into coordinates of principal components of the PCA model."""
        pass