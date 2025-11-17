import torch
import numpy as np

def est_spectral_norm(A: torch.Tensor | np.ndarray,
                            method: str = 'power_iteration',
                            max_it: int = 100,
                            seed: int = 0) -> float:
    """Compute the spectral norm (largest singular value) of a matrix A."""
    assert A.ndim == 2, "Input must be a 2D matrix."
    torch.manual_seed(seed)
    if isinstance(A, np.ndarray):
        A = torch.from_numpy(A)
    if method == 'svd':
        _, S, _ = torch.linalg.svd(A, full_matrices=False) # pylint: disable=not-callable
        S.sort(descending=True)
        return S[0].item()
    elif method == 'power_iteration':
        # v_k = torch.randn((A.size(1),1), device=A.device, dtype=A.dtype)
        u_k = torch.randn((A.size(0),1)).to(device=A.device, dtype=A.dtype)
        spectral_norm = 0.0
        for _ in range(max_it):
            v_k = (u_k.T @ A).T
            v_k = v_k / torch.norm(v_k)
            u_k = (A @ v_k)
            u_k = u_k / torch.norm(u_k)
            spectral_norm_new = u_k.T @ A @ v_k
            if torch.abs(spectral_norm_new - spectral_norm) < 1e-6:
                break
            spectral_norm = spectral_norm_new
        return spectral_norm.item()
