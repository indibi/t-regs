"""Principal Component Analysis (PCA) model."""

import numpy as np
import torch


from .base import PCABaseClass
from ...src.multilinear_ops.matricize import matricize
from ...src.multilinear_ops.tensorize import tensorize


class PCA(PCABaseClass):
    """Principal Component Analysis (PCA) model"""
    def __init__(self, X: , **kwargs):
        """Initialize the PCA model.
        
        Args:
            X (torch.Tensor or np.ndarray): Input data matrix of size (n_features, n_samples).
            feature_modes (list, optional): Modes to be treated as features. Defaults to [1].
            device (str, optional): Device to use ('cpu' or 'cuda'). Defaults to 'cuda' if available else 'cpu'.
        """
        super().__init__(X, **kwargs)

    def __call__(self, n_components):
        pass

    def fit(self, n_components=None, method='GCV', r2_threshold=0.99,
                    center=None,
                    centering='per_feature',
                    centering_func='mean'):
        """Fit the PCA model to the data.

        Args:
            method (str, optional): If the number of components is not provided, select the method
                to choose the number of components. Defaults to 'GCV'.
                'GCV' - Generalized Cross Validation
                'explained_variance_ratio' - Choose the number of components to reach the r2_threshold
            center (torch.Tensor or np.ndarray, optional): Precomputed center. If provided, it overrides centering. Defaults to None.
            centering (str, optional): Centering method. Options are 'per_feature', 'all', 'origin', 'per_cluster'. Defaults to 'per_feature'.
            centering_func (str, optional): Function to compute the center. Options are 'mean', 'median'. Defaults to 'mean'.
        """
        # Center data
        self.center = center
        self.centering = centering
        self.centering_func = centering_func
        self._center_data(center)

        # Compute SVD
        U, S, Vt = torch.linalg.svd(self.Xm, full_matrices=False)

        # Analysis for choosing number of components
        s2_cumsum = torch.cumsum(S**2, dim=0)
        self.svals = S
        self.residual = s2_cumsum[-1] - s2_cumsum
        self.explained_variance = s2_cumsum / s2_cumsum[-1]

        n = self.Xm.shape[0]
        p = self.Xm.shape[1]
        D = n*p
        ranks = torch.arange(1, len(S)+1, device=self.device)
        Ps = ranks * (n + p - ranks)
        self.degrees_of_freedom = Ps
        self.gcvs = self.residual / torch.maximum(D - Ps, torch.tensor(1, device=self.device))**2
        if n_components is None:
            if method == 'GCV':            
                n_components = ranks[torch.argmin(self.gcvs)]
            elif method == 'explained_variance_ratio':
                n_components = ranks[torch.argmax(self.explained_variance >= r2_threshold)]
        else:
            n_components = min(n_components, len(S))
        self.U = U[:, :n_components]
        self.S = S[:n_components]
        self.Vt = Vt[:n_components, :]
        self.n_components = n_components
        
    def _center_data(self, center):
        centering_funcs = { 'mean': lambda x: torch.mean(x, dim=1, keepdim=True),
                            'median': lambda x: torch.median(x, dim=1, keepdim=True)[0]}
        func = centering_funcs.get(self.centering_func, None)
        if func is None:
            raise ValueError(f"Centering function '{self.centering_func}' is not supported.")
        if center is not None:
            self.center = center
            self.Xm = self.input - self.center
        else:
            if self.centering == 'per_feature':
                self.center = func(self.input)
                self.Xm = self.input - self.center
            elif self.centering == 'all':
                self.center = func(self.input.ravel())*torch.ones((self.input.shape[0], 1), device=self.input.device)
                self.Xm = self.input - self.center
            elif self.centering == 'origin' or self.centering is None or self.centering == 'none':
                self.center = torch.zeros((self.input.shape[0], 1), device=self.device)
                self.Xm = self.input
            elif self.centering == 'per_cluster':
                raise NotImplementedError("Per-cluster centering is not implemented yet.")
            else:
                raise ValueError(f"Centering method '{self.centering}' is not supported.")

    @property
    def cov(self):
        if self._cov is None:
            self._cov = torch.cov(self.Xm)
        return self._cov
    
    #@property
    #def singular_values(self):
    #    return self.svals

    @property
    def projector(self):
        if self._projector is None:
            self._projector = self.U @ self.U.T
        return self._projector

    def project(self, X):
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, device=self.device)
        if self.center is not None:
            Xm = Xm - self.center
        return self.projector @ Xm + self.center

    def transform(self, X):
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, device=self.device)
        if self.center is not None:
            Xm = Xm - self.center
        return self.U.T @ Xm
