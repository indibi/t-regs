
import numpy as np
import torch


def generate_tensor_regression_predictors(dims : tuple[int],
                                            n_samples : int,
                                            scheme : str = 'simple',
                                            seed: int = 0,
                                            **kwargs) -> np.ndarray:
    """Generate tensor regression predictors `X`
    
    Parameters
    ----------
        dims : tuple[int]
            Dimensions of the resulting tensor predictors.
        n_samples : int
            Number of samples to generate.
        scheme : str
            Scheme to use for generating the predictors. Options are:
            'simple' - Entries of `X` are drawn from N(0, 1/n_samples).
        seed : int
            Random seed for reproducibility.
        **kwargs
            Additional keyword arguments for future extensions.
    
    Returns
    -------
        np.ndarray
            Generated tensor regression predictors of shape
            (n_samples, *dims).
    """

    rng = np.random.default_rng(seed)
    if scheme == 'simple':
        X = rng.normal(loc=0.0,
                        scale=1.0,
                        size=(n_samples, *dims))
    elif scheme == 'scaled_simple':
        X = rng.normal(loc=0.0,
                        scale=1.0/np.sqrt(n_samples),
                        size=(n_samples, *dims))
    else:
        raise ValueError(f"Unknown scheme {scheme} for generating "
                         "tensor regression predictors.")
    return X

def generate_tensor_regression_response(X: np.ndarray,
                                        B: np.ndarray,
                                        noise_type: str = None,
                                        regression_type: str = 'linear',
                                        seed: int = 0,
                                        kernel: str = 'euclidean',
                                        **kwargs) -> np.ndarray:
    """Generate tensor regression responses from predictors and coefficients.

    Parameters
    ----------
        X : np.ndarray
            Tensor regression predictors of shape (n_samples, *pred_dims).
        B : np.ndarray
            Tensor regression coefficients of shape (*task_dims, *pred_dims)
        regression_type : str = 'linear'
            Type of regression problem. Options are:
            'linear' - Linear regression.
            'logistic' - Logistic regression.
        noise_type : str
            Type of noise to add to the responses. Options are:
            'gaussian' - Add Gaussian noise.
                *For linear regression only currently*
        seed : int = 0
            Random seed for reproducibility.
        kernel : str = 'euclidean'
            Kernel type for generating responses (if applicable).
        **kwargs
            noise_std : float = 1.0
                Standard deviation of the Gaussian noise to be added.
    
    Returns
    -------
        np.ndarray
            Generated tensor regression responses of shape
            (n_samples, *task_dims).
    """
    assert X.ndim >=2, "X must have at least 2 dimensions."

    n_samples = X.shape[0]  # Number of samples
    pred_dims = X.shape[1:] # Dimensions of the predictors
    task_dims = B.shape[:B.ndim - len(pred_dims)] # Dimensions of the tasks

    rng = np.random.default_rng(seed)
    torch_rng = torch.Generator()
    torch_rng.manual_seed(seed)

    mu_noiseless = np.zeros((n_samples, *task_dims))
    if kernel == 'euclidean':
        axes_x = tuple(range(1, 1 + len(pred_dims)))
        axes_b = tuple(range(len(task_dims), B.ndim))
        mu_noiseless = np.tensordot(X, B, axes=(axes_x, axes_b))
    elif kernel == 'weighted':
        raise NotImplementedError("Weighted kernel not implemented yet.")
    else:
        raise ValueError(f"Unknown kernel type {kernel}.")
    
    if noise_type == 'gaussian':
        noise_std = kwargs.get('noise_std', 1.0)
        noise = noise_std* rng.normal(loc=0.0,
                                    scale=1.0,
                                    size=mu_noiseless.shape)
        mu = mu_noiseless + noise
    elif noise_type == 'none' or noise_type is None:
        mu = mu_noiseless
        noise = None
    else:
        raise ValueError(f"Unknown noise type {noise_type}.")
    
    if regression_type == 'linear':
        Y = mu
    elif regression_type == 'logistic':
        mu_noiseless = torch.tensor(mu_noiseless, dtype=torch.float64)
        probs = torch.sigmoid(mu_noiseless)
        Y = torch.bernoulli(probs, generator=torch_rng).numpy()
    else:
        raise ValueError(f"Unknown regression type {regression}.")

    return Y, mu_noiseless