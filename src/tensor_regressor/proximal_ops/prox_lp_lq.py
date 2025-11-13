import numpy as np
import torch
# import jax
# import jax.numpy as jnp

from ..multilinear_ops import matricize, tensorize


def prox_l21(x, alpha, axis=0):
    """Applies the proximal operator of the l21 norm with threshold parameter alpha.

    Args:
        x (torch.Tensor): input array
        alpha (float): proximal operator threshold
        axis (int): axis along which to compute the l2 norm

    Returns:
        Tensor: thresholded vectors
    """
    if isinstance(x, np.ndarray):
        norm_l2 = np.linalg.norm(x, axis=axis, keepdims=True)
        scaling_factor = np.where(norm_l2>alpha, 1-alpha/norm_l2, 0)
    elif isinstance(x, torch.Tensor):
        norm_l2 = torch.linalg.vector_norm(x, dim=axis, keepdim=True)
        scaling_factor = torch.where(norm_l2>alpha, 1-alpha/norm_l2, 0)
    return x*scaling_factor


def prox_grouped_l21(x, alpha, groups, weights=None, modes=[1], return_group_norms=False):
    """Apply the proximal operator for the (non-overlapping) grouped l21 norm.

    The proximal operator for the grouped l21 norm is defined as:
        prox_{alpha*||.||_{2,1}}(x) = argmin_{z} 0.5*||z-x||_F^2 + alpha*||z||_{2,1}
        where ||.||_{2,1} is the grouped l21 norm defined as:
            ||z||_{2,1} = sum_{i=1}^{n_groups} w_i * sqrt(sum_{j=1}^{n_features} z_{ij}^2)
    Parameters:
    -----------
        x (np.ndarray or torch.Tensor): _description_
        alpha (float, np.ndarray or torch.Tensor): Threshold parameters for the groups.
        groups (csr matrix): Compressed sparse row matrix representing the groups. 
            shape=(n_groups, n_features). Each row has 1s in the columns corresponding
            to the features belonging to the group.
        weights (np.ndarray or torch.Tensor, optional): Weights of the groups. Defaults to None.
        modes (int or lists of int, optional): Matricize the tensor. Defaults to 1.
        return_group_norms (bool, optional): Return the weighted l2 norms of the groups as well.
            Defaults to False.
    Returns:
    --------
        
    """
    if isinstance(modes, int):
        modes = [modes]
    og_shape = x.shape
    x = matricize(x, modes)
    if isinstance(x, np.ndarray):
        raise NotImplementedError("Not implemented yet for numpy arrays. Use torch.Tensor")
    elif isinstance(x, torch.Tensor):
        device = x.device
        dtype = x.dtype
        if (groups.values() != 1).any():
            # Check if the group matrix has only 1s as entries.
            raise ValueError("The group matrix should have only 1s as entries")
        if (torch.ones((1, groups.shape[0]), device=device, dtype=dtype) @ groups != 1).any():
            # Check if the groups are overlapping
            raise ValueError("The groups should not be overlapping")

        if weights is None:
            weights = torch.ones((groups.shape[0], 1), device=x.device) # size: (n_groups, 1)
        x2 = x.pow(2)
        group_norms = torch.sqrt(groups @ x2) # size: (n_groups, batch_size)
        threshold = alpha * weights
        scaling_factors = torch.where(group_norms>threshold, 1-threshold/group_norms, 0) # size: (n_groups, batch_size)
        # Torch doesn't have transpose for csr matrices. We use the transpose of the dense matrices.
        scaling_factors = (scaling_factors.T @ groups).T  # size: (n_features, batch_size)
        x = x * scaling_factors
        if return_group_norms:
            weighted_group_norms =  group_norms * weights # scaling_factors *
    x = tensorize(x, og_shape, modes)
    return x if not return_group_norms else (x, weighted_group_norms)


# @jax.jit
# def jax_prox_l21(x, alpha, axis=0):
#     norm_l2 = jnp.linalg.norm(x, axis=0).reshape((1, x.shape[1]))
#     scaling_factor = jnp.zeros(norm_l2.shape)
#     scaling_factor = jnp.where(norm_l2>alpha, 1 - alpha / norm_l2, 0)
#     return x*scaling_factor

# def torch_prox_l21(x, alpha, axis=0):
#     """Applies the proximal operator of the l21 norm with threshold parameter alpha.

#     Args:
#         x (torch.Tensor): input array
#         alpha (float): proximal operator threshold
#         axis (int, tuple of ind): axis along which to compute the l2 norm

#     Returns:
#         Tensor: thresholded vectors
#     """
#     norm_l2 = torch.vector_norm(x, dim=axis, keepdim=True)
#     scaling_factor = torch.where(norm_l2>alpha, 1-alpha/norm_l2, 0)
#     return x*scaling_factor