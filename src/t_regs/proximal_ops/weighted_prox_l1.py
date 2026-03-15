# TODO: write docstring
# TODO: add tests
import torch

from torch.nn.functional import softshrink

@torch.jit.script
def icd_weighted_prox_step_l1(y: torch.Tensor,
                              lda: float,
                              Q: torch.Tensor,
                              Q_diag: torch.Tensor | None = None,
                              x_init: torch.Tensor | None = None,
                              max_iter: int = 100,
                              tol:float = 1e-6,
                              verbose: bool =False) -> torch.Tensor:
    r"""Iterative Coordinate Descent for Weighted Lasso Proximal Operator.

    Solves the following optimization problem:

    .. math::
        \min_{x} \frac{1}{2} (y - x)^T Q (y - x) + \lambda \|x\|_1
    
    or equivalently,
    .. math::
        \min_{x} \frac{1}{2} \|y - x\|_Q^2 + \lambda \|x\|_1

    Parameters
    ----------
    y : torch.Tensor
        Input vector/tensor of shape (n,) or (n, *dims).
    lda : float
        Sparsity regularization parameter :math:`\lambda`.
    Q : torch.Tensor
        Positive semi-definite matrix of shape (n, n).
    x_init : torch.Tensor | None
        Initial guess for the solution of shape (n,) or (n, *dims).
        If None, initializes to zero.
    max_iter : int
        Maximum number of iterations.
    tol : float
        Tolerance for convergence.

    Returns
    -------
    x : torch.Tensor
        Solution vector of shape (n, *dims).
    """
    n = y.shape[0]
    if x_init is not None:
        x = x_init.clone()
    else:
        x = torch.zeros(y.shape, device=y.device, dtype=y.dtype)
    x = x.reshape(n, -1)
    
    if Q_diag is None:
        if Q.is_sparse:
            I = torch.sparse.spdiags(torch.ones(n, device='cpu', dtype=Q.dtype),
                        torch.tensor([0], device='cpu'),
                        (n, n)).to(Q.device)
            Q_diag = (Q * I).coalesce().values()
            assert Q_diag.shape[0] == n, "Q_diag shape mismatch."
        else:
            Q_diag = torch.diag(Q)
    
    assert Q_diag.min() > 0, "Diagonal elements of Q must be positive."

    Qy = Q @ y.reshape(n, -1)
    for it in range(max_iter):
        x_old = x.clone()
        # x = cd_iter_weighted_l1(x, Qy, Q, Q_diag, lda)
        for j in range(n):
            # R_j = Qy[j,:] - (Q[j, :] @ x) + Q_diag[j] * x[j, :]
            R_j = Qy[j,:] - (Q @ x)[j, :] + Q_diag[j] * x[j, :]
            x[j, :] = softshrink(R_j, lda) / Q_diag[j]
        norm_diff = torch.norm(x - x_old, p='fro')
        if verbose:
            print(f"||x_{it+1} - x_{it}||_F = {norm_diff.item()}")
        if norm_diff < tol:
            if verbose:
                print(f"Converged after {it+1} iterations.")
            break
    return x.reshape(y.shape)